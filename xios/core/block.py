"""XIOS block: (recurrent | windowed-attention) mixer + SwiGLU FFN.

Repurposed for the looped core: ``cond`` is the *iteration embedding*.  The
same weights behave differently at ponder step 1 than at ponder step 20
because AdaLN-Zero modulation shifts/scales/gates them per iteration.  That
is what lets one small block stack act like a much deeper network without
storing depth-many copies of the weights.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm, AdaLNZero, modulate
from .recurrent import GatedRecurrentMixer
from .attention import SlidingWindowAttention


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.drop(self.w_out(F.silu(a) * b))


class XiosBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int, ffn: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.kind = cfg.layer_kind(layer_idx)
        d = cfg.dim

        self.norm1 = RMSNorm(d, cfg.norm_eps)
        self.norm2 = RMSNorm(d, cfg.norm_eps)

        if self.kind == "attn":
            self.mixer = SlidingWindowAttention(
                d, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                window=cfg.attn_window, rope_theta=cfg.rope_theta, dropout=cfg.dropout)
        else:
            self.mixer = GatedRecurrentMixer(
                d, cfg.n_heads, cfg.head_dim, dropout=cfg.dropout)

        hidden = int(cfg.ffn_mult * d / 64 + 0.5) * 64
        # `ffn` may be an externally supplied module (e.g. the disk-resident
        # associative memory) that replaces the dense FFN on this layer.
        self.ffn = ffn if ffn is not None else SwiGLU(d, hidden, cfg.dropout)

        self.adaln = AdaLNZero(d, cfg.cond_dim, n_groups=2) if cfg.use_adaln else None

    def _mod(self, cond):
        if cond is None or self.adaln is None:
            return (None,) * 6
        return self.adaln(cond)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                state: Optional[dict] = None, return_state: bool = False,
                delta: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                active: Optional[torch.Tensor] = None):
        s1, c1, g1, s2, c2, g2 = self._mod(cond)
        aux = x.new_zeros(())

        h = self.norm1(x)
        if s1 is not None:
            h = modulate(h, s1, c1)
        h, new_state = self.mixer(h, state, return_state=return_state,
                                  delta=delta, pos=pos, active=active)
        x = x + (h * (1 + g1) if g1 is not None else h)

        h = self.norm2(x)
        if s2 is not None:
            h = modulate(h, s2, c2)

        if active is not None and not bool(active.all()):
            # The FFN is strictly token-wise, so halted tokens can simply be
            # gathered out.  This is where the depth budget turns into real
            # saved FLOPs during training as well as inference: the FFN is
            # ~60% of block cost and the active set shrinks geometrically
            # with iteration count.
            flat = h.reshape(-1, h.shape[-1])
            sel = active.reshape(-1).nonzero(as_tuple=True)[0]
            ffn_out = self.ffn(flat[sel])
            if isinstance(ffn_out, tuple):
                ffn_out, aux = ffn_out
            out = torch.zeros_like(flat).index_copy(0, sel, ffn_out.to(flat.dtype))
            out = out.view_as(h)
        else:
            out = self.ffn(h)
            if isinstance(out, tuple):      # memory FFNs return (y, aux)
                out, aux = out
        x = x + (out * (1 + g2) if g2 is not None else out)
        return x, new_state, aux

    @torch.no_grad()
    def step(self, x: torch.Tensor, cond, state: dict,
             delta=None, pos=None):
        s1, c1, g1, s2, c2, g2 = self._mod(cond)
        h = self.norm1(x)
        if s1 is not None:
            h = modulate(h, s1, c1)
        if self.kind == "attn":
            h, new_state = self.mixer.step(h, state or {}, delta=delta, pos=pos)
        else:
            h, new_state = self.mixer.step(h, state or {}, delta=delta)
        x = x + (h * (1 + g1) if g1 is not None else h)

        h = self.norm2(x)
        if s2 is not None:
            h = modulate(h, s2, c2)
        out = self.ffn(h)
        if isinstance(out, tuple):
            out = out[0]
        x = x + (out * (1 + g2) if g2 is not None else out)
        return x, new_state
