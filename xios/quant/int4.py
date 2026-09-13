"""Group-wise INT4 weight quantisation with activation-aware scaling and an
optional low-rank error corrector.

Three techniques stack here, each one cheap:

1. **Group-wise asymmetric INT4** (group = 64 input channels).  Weights drop
   from 16 bits to 4 + (16+16)/64 ~= 4.5 bits, a 3.5x shrink.  A 3.5B
   backbone goes from 7GB (fp16) to ~2GB.

2. **Activation-aware scaling (AWQ-style).**  Quantisation error costs you
   most on the weight channels that see the largest activations.  We measure
   per-input-channel activation magnitude on a calibration set, scale those
   channels up before rounding (so they get more of the grid) and divide the
   input by the same factor at runtime.  Mathematically a no-op in fp16;
   worth several points of perplexity in int4.

3. **Low-rank error correction.**  ``W - dequant(quant(W))`` is not noise; it
   has structure.  An SVD of the residual, truncated to rank r (8-16), is
   kept in fp16 and added back as ``x @ A @ B``.  Costs ~1% of the weights
   and recovers most of the remaining gap.

The dequant path below is plain PyTorch: it is *memory*-optimal but not
speed-optimal.  ``xios.quant.kernels`` swaps in a fused Triton/torch.compile
matmul when available.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# packing helpers
# ---------------------------------------------------------------------------
def pack_int4(w: torch.Tensor) -> torch.Tensor:
    """(..., N) uint8 values in [0,15] -> (..., N//8) int32."""
    assert w.shape[-1] % 8 == 0, "last dim must be a multiple of 8"
    w = w.to(torch.int32).reshape(*w.shape[:-1], -1, 8)
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    return (w << shifts).sum(-1)


def unpack_int4(p: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
    shifts = torch.arange(8, device=p.device, dtype=torch.int32) * 4
    out = (p.unsqueeze(-1) >> shifts) & 0xF
    out = out.reshape(*p.shape[:-1], -1)
    return out if n is None else out[..., :n]


def quantize_tensor(w: torch.Tensor, group: int = 64):
    """Row-wise grouped asymmetric int4. w: (out, in) -> packed, scale, zero."""
    out_f, in_f = w.shape
    pad = (-in_f) % group
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    wg = w.float().reshape(out_f, -1, group)
    lo = wg.amin(-1, keepdim=True)
    hi = wg.amax(-1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-8) / 15.0
    zero = (-lo / scale).round().clamp(0, 15)
    q = ((wg / scale) + zero).round().clamp(0, 15).to(torch.uint8)
    packed = pack_int4(q.reshape(out_f, -1))
    return packed, scale.squeeze(-1).half(), zero.squeeze(-1).to(torch.uint8), in_f + pad


def dequantize_tensor(packed, scale, zero, in_padded, group: int = 64):
    out_f = packed.shape[0]
    q = unpack_int4(packed, in_padded).reshape(out_f, -1, group).float()
    return ((q - zero.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)).reshape(out_f, in_padded)


# ---------------------------------------------------------------------------
class QuantLinear(nn.Module):
    """Drop-in int4 replacement for ``nn.Linear``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group: int = 64, lora_rank: int = 0):
        super().__init__()
        self.in_features, self.out_features, self.group = in_features, out_features, group
        in_padded = in_features + (-in_features) % group
        self.in_padded = in_padded
        self.register_buffer("packed", torch.zeros(out_features, in_padded // 8, dtype=torch.int32))
        self.register_buffer("scale", torch.zeros(out_features, in_padded // group, dtype=torch.half))
        self.register_buffer("zero", torch.zeros(out_features, in_padded // group, dtype=torch.uint8))
        self.register_buffer("act_scale", torch.ones(in_features, dtype=torch.half))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lora_rank = lora_rank
        self.awq_alpha = 0.0
        if lora_rank:
            self.register_buffer("lora_a", torch.zeros(in_features, lora_rank, dtype=torch.half))
            self.register_buffer("lora_b", torch.zeros(lora_rank, out_features, dtype=torch.half))
        self._cache: torch.Tensor | None = None

    # -- construction ---------------------------------------------------
    @staticmethod
    def _search_alpha(w: torch.Tensor, act: torch.Tensor, group: int,
                      grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)):
        """Pick the activation-scaling exponent by measuring, not by guessing.

        The published AWQ exponent (0.5) is tuned for coarse quantisation.  At
        group=64 the grouped scales already absorb most of the outlier
        problem, and forcing a fixed 0.5 measurably *hurts* -- scaling a
        high-activation channel up widens its group's range and degrades every
        other channel sharing that group.  Searching the exponent makes the
        feature monotonically safe: alpha=0 is plain int4, so the search can
        only pick something at least as good.
        """
        base = act.float().clamp(min=1e-4)
        base = base / base.mean()
        # proxy objective: activation-weighted weight error, which is what
        # actually lands in the output
        best, best_err = 0.0, None
        for a in grid:
            s = base.pow(a)
            ws = w * s[None, :]
            packed, scale, zero, in_pad = quantize_tensor(ws, group)
            deq = dequantize_tensor(packed, scale, zero, in_pad, group)[:, :w.shape[1]]
            err = (((deq / s[None, :]) - w) * base[None, :]).pow(2).sum()
            if best_err is None or err < best_err:
                best, best_err = a, err
        return best

    @classmethod
    def from_linear(cls, lin: nn.Linear, group: int = 64, act_scale: Optional[torch.Tensor] = None,
                    lora_rank: int = 0, awq_alpha: Optional[float] = None) -> "QuantLinear":
        q = cls(lin.in_features, lin.out_features, lin.bias is not None, group, lora_rank)
        w = lin.weight.data.float()

        if act_scale is not None:
            a = q._search_alpha(w, act_scale, group) if awq_alpha is None else awq_alpha
            s = act_scale.float().clamp(min=1e-4)
            s = (s / s.mean()).pow(a)
            q.awq_alpha = float(a)
            q.act_scale.copy_((1.0 / s).half())
            w = w * s[None, :]

        packed, scale, zero, in_padded = quantize_tensor(w, group)
        q.packed.copy_(packed); q.scale.copy_(scale); q.zero.copy_(zero)

        if lora_rank:
            deq = dequantize_tensor(packed, scale, zero, in_padded, group)[:, :lin.in_features]
            resid = (w - deq).float()
            # Minimise *output* error, not weight error: weight each input
            # channel by how much signal actually flows through it, factor in
            # that space, then unweight.  With uniform activations this
            # reduces to a plain SVD; with real calibration stats it puts the
            # whole rank budget on the channels that matter (LQER-style).
            if act_scale is not None:
                wgt = act_scale.float().clamp(min=1e-4)
                wgt = wgt / wgt.mean()
            else:
                wgt = torch.ones(lin.in_features)
            U, S, Vh = torch.linalg.svd(resid * wgt[None, :], full_matrices=False)
            r = min(lora_rank, S.numel())
            a = (Vh[:r].T * S[:r].sqrt()) / wgt[:, None]
            b = (U[:, :r] * S[:r].sqrt()).T
            q.lora_a.copy_(a.half()); q.lora_b.copy_(b.half())
        if lin.bias is not None:
            q.bias.data.copy_(lin.bias.data)
        return q

    # -- runtime --------------------------------------------------------
    def dequantized_weight(self) -> torch.Tensor:
        w = dequantize_tensor(self.packed, self.scale, self.zero, self.in_padded, self.group)
        return w[:, :self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.act_scale.to(x.dtype)
        w = self.dequantized_weight().to(x.dtype)
        y = torch.nn.functional.linear(x, w, self.bias)
        if self.lora_rank:
            y = y + (x @ self.lora_a.to(x.dtype)) @ self.lora_b.to(x.dtype)
        return y

    def extra_repr(self) -> str:
        bits = 4 + 32 / self.group
        return (f"in={self.in_features}, out={self.out_features}, group={self.group}, "
                f"~{bits:.2f} bits/w, lora_rank={self.lora_rank}, "
                f"awq_alpha={self.awq_alpha}")

    @property
    def nbytes(self) -> int:
        n = self.packed.numel() * 4 + self.scale.numel() * 2 + self.zero.numel()
        if self.lora_rank:
            n += (self.lora_a.numel() + self.lora_b.numel()) * 2
        return n


# ---------------------------------------------------------------------------
class QuantEmbedding(nn.Module):
    """Row-wise int8 embedding table.

    At small scale the embedding table is a large fraction of the weights
    (50M of 400M rows for the ``small`` preset), so leaving it in fp16 caps
    the achievable compression well below what the int4 layers deliver. Int8
    per-row is safe here: an embedding lookup is a single row read with no
    accumulation, so quantisation error does not compound the way it does
    through a matmul.
    """

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings, self.dim = num_embeddings, dim
        self.register_buffer("q", torch.zeros(num_embeddings, dim, dtype=torch.int8))
        self.register_buffer("scale", torch.zeros(num_embeddings, dtype=torch.half))

    @classmethod
    def from_embedding(cls, emb: nn.Embedding) -> "QuantEmbedding":
        out = cls(emb.num_embeddings, emb.embedding_dim)
        w = emb.weight.data.float()
        scale = w.abs().amax(1).clamp(min=1e-8) / 127.0
        out.q.copy_((w / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        out.scale.copy_(scale.half())
        return out

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.q[idx].to(self.scale.dtype) * self.scale[idx].unsqueeze(-1)

    def weight_for_head(self) -> torch.Tensor:
        """Dequantised table, for a tied output projection."""
        return self.q.to(self.scale.dtype) * self.scale.unsqueeze(-1)

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.scale.numel() * 2


class TiedQuantHead(nn.Module):
    """Output projection that shares the quantised embedding table.

    Zero additional storage: the logits are computed straight from the int8
    table the embedding layer already holds.
    """

    def __init__(self, emb: "QuantEmbedding"):
        super().__init__()
        self.emb = [emb]          # list -> not registered, so not double-counted

    @property
    def weight(self):
        return self.emb[0].q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb[0]
        w = e.q.to(x.dtype) * e.scale.to(x.dtype).unsqueeze(-1)
        return torch.nn.functional.linear(x, w)

    @property
    def nbytes(self) -> int:
        return 0
