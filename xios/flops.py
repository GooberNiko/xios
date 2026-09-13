"""FLOP accounting, so the A/B can be run at equal compute and not just
equal parameters.

Equal-parameter is the comparison XIOS wants; equal-FLOP is the comparison
that could sink it, because a looped core spends more compute per parameter
by construction.  Reporting both is the only honest way to present this.
"""
from __future__ import annotations

import torch


def _attn_flops(d, n_heads, n_kv, head_dim, T, window=None):
    proj = 2 * d * (n_heads * head_dim) + 2 * 2 * d * (n_kv * head_dim)
    span = T if window is None else min(T, window)
    scores = 2 * 2 * span * head_dim * n_heads          # QK^T and AV
    return proj + scores


def _rec_flops(d, n_heads, head_dim):
    inner = n_heads * head_dim
    proj = 2 * (3 * d * inner) + 2 * (d * inner)        # qkv + gate
    state = 2 * 2 * n_heads * head_dim * head_dim       # k(x)v write, q@S read
    out = 2 * inner * d
    return proj + state + out


def _ffn_flops(d, hidden):
    return 2 * (2 * d * hidden) + 2 * (hidden * d)


def baseline_flops_per_token(model, T: int) -> float:
    b = model.blocks[0]
    d = model.dim
    hidden = b.w_in.out_features // 2
    per_layer = _attn_flops(d, b.n_heads, b.n_kv_heads, b.head_dim, T) \
        + _ffn_flops(d, hidden)
    return model.n_layers * per_layer + 2 * d * model.vocab_size


def xios_flops_per_token(cfg, T: int, mean_depth: float,
                         mean_active_frac: float = 1.0) -> float:
    d, hidden = cfg.dim, cfg.ffn_hidden

    def block_cost(i, ffn_share=1.0):
        if cfg.layer_kind(i) == "attn":
            mix = _attn_flops(d, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                              T, cfg.attn_window)
        else:
            mix = _rec_flops(d, cfg.n_heads, cfg.head_dim)
        return mix + ffn_share * _ffn_flops(d, hidden)

    pre = sum(block_cost(i) for i in range(cfg.prelude_blocks))
    core_once = sum(block_cost(cfg.prelude_blocks + i) for i in range(cfg.core_blocks))
    off = cfg.prelude_blocks + cfg.core_blocks
    coda = sum(block_cost(off + i) for i in range(cfg.coda_blocks))
    return pre + mean_depth * core_once + coda + 2 * d * cfg.vocab_size


@torch.no_grad()
def measure(model, ids, T: int | None = None) -> dict:
    """Actual FLOPs/token for a real batch, using observed depth."""
    T = T or ids.shape[1]
    from .baseline import BaselineTransformer
    if isinstance(model, BaselineTransformer):
        f = baseline_flops_per_token(model, T)
        return {"flops_per_token": f, "mean_depth": model.n_layers,
                "kind": "baseline"}
    out = model(ids)
    md = out.stats.mean_depth
    f = xios_flops_per_token(model.cfg, T, md)
    return {"flops_per_token": f, "mean_depth": md, "kind": "xios",
            "depth_p99": out.stats.depth_p99,
            "active_frac": out.stats.active_frac}
