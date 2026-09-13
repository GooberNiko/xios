"""Sliding-window grouped-query attention.

Only every ``attn_every``-th layer uses this.  The window bounds both compute
and the KV cache: memory is O(window * n_kv_heads * head_dim) regardless of
how long the context grows, so a 200k-token document costs the same cache as
a 2k one.  Long-range information travels through the recurrent layers
instead; the attention layers supply exact local recall (copying, in-context
lookup, verbatim quoting) which recurrences are measurably worse at.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


def build_rope(head_dim: int, max_len: int, theta: float, device=None, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D/2)
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None]
    sin = sin[None, None]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class SlidingWindowAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 window: int, rope_theta: float = 500000.0, dropout: float = 0.0,
                 causal: bool = True):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.window = window
        self.rope_theta = rope_theta
        self.causal = causal
        self.n_rep = n_heads // n_kv_heads

        self.q = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.dropout = dropout
        self._rope_cache: tuple | None = None

    def _rope_at(self, pos: torch.Tensor, device, dtype):
        """RoPE tables gathered at arbitrary (possibly non-contiguous) positions."""
        need = int(pos.max().item()) + 1 if pos.numel() else 1
        self._ensure_rope(need, device)
        _, cos, sin = self._rope_cache
        return cos[pos], sin[pos]

    def _ensure_rope(self, need: int, device):
        if self._rope_cache is None or self._rope_cache[0] < need                 or self._rope_cache[1].device != device:
            n = max(need, 1024)
            cos, sin = build_rope(self.head_dim, n, self.rope_theta, device, torch.float32)
            self._rope_cache = (n, cos, sin)

    def _rope(self, T: int, offset: int, device, dtype):
        need = offset + T
        if self._rope_cache is None or self._rope_cache[0] < need or self._rope_cache[1].device != device:
            cos, sin = build_rope(self.head_dim, max(need, 1024), self.rope_theta, device, torch.float32)
            self._rope_cache = (max(need, 1024), cos, sin)
        _, cos, sin = self._rope_cache
        return cos[offset:offset + T], sin[offset:offset + T]

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        B, H, T, D = x.shape
        return x[:, :, None].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None,
                return_state: bool = False, delta: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                active: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        offset = cache.get("pos", 0) if cache else 0

        q = self.q_norm(self.q(x).view(B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(B, T, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # `pos` carries TRUE token positions.  A ponder iteration sees a
        # gapped subsequence, so relative distance must be measured in real
        # tokens, not in subsequence index.
        if pos is None:
            pos = torch.arange(offset, offset + T, device=x.device)
        pos_flat = pos[0] if pos.dim() == 2 else pos
        cos, sin = self._rope_at(pos_flat, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None and cache.get("k") is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            # rolling window: never keep more than `window` past keys
            if k.shape[2] > self.window:
                k = k[:, :, -self.window:]
                v = v[:, :, -self.window:]

        kpos = pos_flat if (cache is None or cache.get("kpos") is None)             else torch.cat([cache["kpos"], pos_flat])
        if kpos.numel() > self.window:
            kpos = kpos[-self.window:]
        new_state = {"k": k, "v": v, "kpos": kpos,
                     "pos": int(pos_flat[-1].item()) + 1} if return_state else None

        kq, vq = self._repeat_kv(k, self.n_rep), self._repeat_kv(v, self.n_rep)
        S = kq.shape[2]

        if T == 1:
            attn_mask, is_causal = None, False
        else:
            qp = pos_flat[:, None]
            kp = kpos[None, :S] if kpos.numel() >= S else                 torch.arange(S, device=x.device)[None, :]
            allowed = (kp <= qp) if self.causal else torch.ones_like(kp <= qp)
            allowed &= (qp - kp) < self.window
            allowed = allowed[None, None].expand(B, 1, -1, -1)
            if active is not None:
                # inactive tokens are not visible as keys at this depth
                kact = active if active.shape[1] == S else                     F.pad(active, (S - active.shape[1], 0), value=True)
                allowed = allowed & kact[:, None, None, :]
                # a fully-masked row would produce NaN; let it see itself
                selfsee = torch.zeros_like(allowed)
                sr = torch.arange(T, device=x.device)
                selfsee[:, :, sr, S - T + sr] = True
                allowed = allowed | selfsee
            attn_mask, is_causal = allowed, False

        y = F.scaled_dot_product_attention(
            q, kq, vq, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.o(y), new_state

    @torch.no_grad()
    def step(self, x: torch.Tensor, cache: dict, delta=None, pos=None):
        return self.forward(x, cache=cache, return_state=True, delta=delta, pos=pos)
