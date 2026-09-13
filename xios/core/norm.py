from __future__ import annotations
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square norm. Cheaper than LayerNorm (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(dtype)
        return x * self.weight if self.weight is not None else x


class AdaLNZero(nn.Module):
    """AdaLN-Zero modulation (DiT-style).

    Produces per-block (shift, scale, gate) triples from a conditioning
    vector.  Initialised to zero so a freshly built block is the identity,
    which is what lets one backbone serve both autoregressive text (cond=None
    -> pure RMSNorm) and flow-matching image/audio/video.
    """

    def __init__(self, dim: int, cond_dim: int, n_groups: int = 2):
        super().__init__()
        self.n_groups = n_groups
        self.proj = nn.Linear(cond_dim, dim * 3 * n_groups)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor):
        # cond: (B, cond_dim) or (B, T, cond_dim)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        out = self.proj(torch.nn.functional.silu(cond))
        return out.chunk(3 * self.n_groups, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift
