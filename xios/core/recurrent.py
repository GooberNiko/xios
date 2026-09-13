"""Gated linear recurrence — XIOS's primary sequence mixer.

Why this instead of attention
-----------------------------
Softmax attention costs O(T^2 d) compute and O(T d) KV memory.  On a laptop
that is the single reason long context is impossible.  A *gated linear
recurrence* keeps a fixed-size matrix state ``S in R^{dh x dv}`` per head:

    S_t = a_t * S_{t-1} + k_t (x) v_t
    y_t = q_t @ S_t

Cost is O(T d^2 / H) with **constant** memory per step — decoding a 100k-token
context uses exactly as much RAM as decoding token 1.  The scalar per-head
decay ``a_t in (0,1)`` is data-dependent (selective), which is what gives it
the content-based forgetting that made plain linear attention weak.

Training uses a chunked parallel form so the whole thing is matmuls (fast on
any device, no custom kernel required); inference uses the true recurrence.
"""
from __future__ import annotations

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class ShortConv(nn.Module):
    """Causal depthwise conv over time (kernel 4).

    Gives the recurrence a cheap exact-recall window for the last few tokens,
    which is where most of the benefit of local attention lives.
    """

    def __init__(self, dim: int, kernel: int = 4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=0, bias=True)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None,
                skip: int = 0):
        # x: (B, T, D) -> (B, T, D);  state: (B, D, kernel-1)
        B, T, D = x.shape
        xt = x.transpose(1, 2)                                   # (B, D, T)
        if state is None:
            state = xt.new_zeros(B, D, self.kernel - 1)
        elif skip:
            # Decode replaying a gap: at this depth, `skip` real tokens went
            # by without being processed.  The parallel path sees them as
            # zeros (inactive tokens are masked at the input), so the
            # incremental path must shift the same number of zeros through
            # the conv window, or prefill and decode would disagree.
            n = min(skip, self.kernel - 1)
            state = torch.cat([state[..., n:], state.new_zeros(B, D, n)], dim=-1)
        xt = torch.cat([state, xt], dim=-1)
        new_state = xt[..., -(self.kernel - 1):].detach() if self.kernel > 1 else state
        out = self.conv(xt)                                      # (B, D, T)
        return out.transpose(1, 2), new_state


class GatedRecurrentMixer(nn.Module):
    """Chunk-parallel gated linear recurrence with selective decay."""

    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 expand: float = 1.0, chunk: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk = chunk
        inner = int(expand * n_heads * head_dim)
        self.inner = inner

        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.gate = nn.Linear(dim, inner, bias=False)
        self.decay = nn.Linear(dim, n_heads, bias=True)
        self.conv = ShortConv(3 * inner, kernel=4)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.out_norm = RMSNorm(inner)
        self.out = nn.Linear(inner, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Bias init spreads initial half-lives across ~[2, 1024] tokens so
        # different heads start out specialised for different timescales.
        with torch.no_grad():
            hl = torch.logspace(math.log10(2.0), math.log10(1024.0), n_heads)
            a0 = torch.exp(-math.log(2.0) / hl)                 # target decay
            self.decay.bias.copy_(torch.logit(a0.clamp(1e-4, 1 - 1e-4)))
            self.decay.weight.mul_(0.1)

    # ------------------------------------------------------------------
    def _project(self, x, conv_state, delta=None, skip: int = 0):
        B, T, _ = x.shape
        qkv = self.qkv(x)
        qkv, conv_state = self.conv(qkv, conv_state, skip=skip)
        qkv = F.silu(qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        H, D = self.n_heads, self.head_dim
        q = self.q_norm(q.view(B, T, H, D)).transpose(1, 2)      # (B,H,T,D)
        k = self.k_norm(k.view(B, T, H, D)).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        log_a = F.logsigmoid(self.decay(x).float()).transpose(1, 2)   # (B,H,T) <= 0
        if delta is not None:
            # This sequence is a *gapped* subsample of the real token stream:
            # deep ponder iterations only see the tokens that reached that
            # depth.  The state must still decay across the skipped tokens,
            # or a head's forgetting timescale would silently stretch at
            # depth.  The parallel path masks inactive tokens to zero, so
            # their decay is exactly logsigmoid(bias) -- the data-independent
            # base rate.  Replaying `gap - 1` of those here makes the
            # incremental path agree with the parallel one to float precision.
            base = F.logsigmoid(self.decay.bias.float())              # (H,)
            gap = (delta - 1).clamp(min=0).unsqueeze(1).float()       # (B,1,T)
            log_a = log_a + gap * base.view(1, -1, 1)
        return q, k, v, log_a, conv_state

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, state: Optional[dict] = None,
                return_state: bool = False, delta: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                active: Optional[torch.Tensor] = None):
        """Parallel (training / prefill) path.

        state: {"S": (B,H,dh,dv), "conv": (B,3*inner,k-1)} carried across chunks
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        S = state["S"] if state else None
        conv_state = state["conv"] if state else None
        if active is not None:
            # Zero inactive tokens at the input too, otherwise the depthwise
            # short-conv would smear their content into their active
            # neighbours and the depth sparsification would not be exact.
            # Their own outputs are discarded by the caller, so nothing is lost.
            x = x * active[..., None].to(x.dtype)

        q, k, v, log_a, conv_state = self._project(x, conv_state, delta)
        q = q * (D ** -0.5)

        if active is not None:
            # Sparsified depth, done exactly.  A linear recurrence is *linear*
            # in its writes, so "run this layer over only the active
            # subsequence" is implemented exactly by zeroing the values of
            # inactive tokens -- they write nothing into the state.  Note we
            # deliberately do NOT mask the decay: the state keeps decaying at
            # every real token, so a head's half-life stays calibrated in true
            # token units rather than in subsequence steps.  Dense shapes,
            # exact sparse semantics, no gather needed during training.
            v = v * active[:, None, :, None].to(v.dtype)

        if S is None:
            S = q.new_zeros(B, H, D, D, dtype=torch.float32)

        C = self.chunk
        pad = (C - T % C) % C
        if pad:
            q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad)); log_a = F.pad(log_a, (0, pad))
        Tp = T + pad
        nC = Tp // C

        qc = q.reshape(B, H, nC, C, D).float()
        kc = k.reshape(B, H, nC, C, D).float()
        vc = v.reshape(B, H, nC, C, D).float()
        lac = log_a.reshape(B, H, nC, C)

        A = lac.cumsum(-1)                                        # (B,H,nC,C)
        # pairwise within-chunk decay, exp(A_i - A_j) for i >= j  (always <= 0)
        dec = (A.unsqueeze(-1) - A.unsqueeze(-2))                 # (B,H,nC,C,C)
        causal = torch.ones(C, C, device=x.device, dtype=torch.bool).tril()
        dec = dec.masked_fill(~causal, float("-inf")).exp()

        intra = ((qc @ kc.transpose(-1, -2)) * dec) @ vc          # (B,H,nC,C,D)

        # inter-chunk: sequentially carry the matrix state
        q_scaled = qc * A.exp().unsqueeze(-1)                     # (B,H,nC,C,D)
        kv_w = kc * (A[..., -1:, None] - A.unsqueeze(-1)).exp()   # weights for state update
        chunk_kv = kv_w.transpose(-1, -2) @ vc                    # (B,H,nC,D,D)
        a_chunk = A[..., -1].exp()                                # (B,H,nC)

        outs = []
        for c in range(nC):
            outs.append(q_scaled[:, :, c] @ S)                    # (B,H,C,D)
            S = a_chunk[:, :, c, None, None] * S + chunk_kv[:, :, c]
        inter = torch.stack(outs, dim=2)

        y = (intra + inter).reshape(B, H, Tp, D)[:, :, :T]
        y = y.to(x.dtype).transpose(1, 2).reshape(B, T, self.inner)
        y = self.out_norm(y) * F.silu(self.gate(x))
        y = self.drop(self.out(y))

        if return_state:
            return y, {"S": S, "conv": conv_state}
        return y, None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, x: torch.Tensor, state: dict,
             delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Single-token recurrent update. O(d^2), constant memory."""
        B = x.shape[0]
        skip = 0 if delta is None else max(0, int(delta.flatten()[0]) - 1)
        q, k, v, log_a, conv_state = self._project(x, state.get("conv"), delta, skip=skip)
        q = (q * (self.head_dim ** -0.5)).squeeze(2).float()      # (B,H,D)
        k = k.squeeze(2).float(); v = v.squeeze(2).float()
        a = log_a.squeeze(-1).exp()                               # (B,H)

        S = state.get("S")
        if S is None:
            S = q.new_zeros(B, self.n_heads, self.head_dim, self.head_dim)
        S = a[..., None, None] * S + k.unsqueeze(-1) @ v.unsqueeze(-2)
        y = (q.unsqueeze(-2) @ S).squeeze(-2)                     # (B,H,D)

        y = y.to(x.dtype).reshape(B, 1, self.inner)
        y = self.out_norm(y) * F.silu(self.gate(x))
        return self.out(y), {"S": S, "conv": conv_state}
