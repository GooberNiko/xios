"""Bandwidth accounting, and the design rule that falls out of it.

Decoding on weak hardware is memory-bound, so bytes-read-per-token decides
speed. These tests lock in the two findings that reshaped the architecture:
a cache-resident looped core is dramatically cheaper in bandwidth than a
dense stack, and a core too large to cache is *worse* than dense.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from xios.config import get_config
from xios.bandwidth import (xios_budget, dense_budget, report,
                            design_for_bandwidth, tier_for)


def test_cache_resident_core_beats_dense():
    cfg = get_config("flagship")
    b = xios_budget(cfg)
    dn = dense_budget(3_000_000_000, cfg.vocab_size, cfg.dim)
    ratio = dn.dram_bytes / b.dram_bytes
    print(f"flagship: {b.dram_bytes/1e6:.1f} MB/token from {b.core_tier}; "
          f"3B dense: {dn.dram_bytes/1e6:.0f} MB/token -> {ratio:.0f}x fewer bytes")
    assert b.core_tier in ("L2", "L3"), "the flagship core must be cache-resident"
    assert ratio > 20, f"expected a large bandwidth win, got {ratio:.1f}x"


def test_oversized_core_is_worse_than_dense():
    """The failure mode that the original presets walked into.

    A core too big to cache is re-read from DRAM on every iteration, so depth
    multiplies bandwidth instead of being free. `base` was measured at 0.7x a
    3B dense model -- slower on exactly the hardware it targeted.
    """
    cfg = get_config("base")
    b = xios_budget(cfg)
    dn = dense_budget(3_000_000_000, cfg.vocab_size, cfg.dim)
    ratio = dn.dram_bytes / b.dram_bytes
    print(f"base (2.4GB core, uncacheable): {b.dram_bytes/1e6:.0f} MB/token "
          f"-> {ratio:.2f}x vs dense  ({'WORSE' if ratio < 1 else 'better'})")
    assert b.core_tier == "DRAM"
    assert ratio < 1.0, "this preset is supposed to demonstrate the failure"


def test_depth_is_free_when_cached():
    cfg = get_config("flagship")
    shallow = xios_budget(cfg, depth=4)
    deep = xios_budget(cfg, depth=64)
    growth = deep.dram_bytes / shallow.dram_bytes
    print(f"depth 4 -> 64 raises DRAM traffic by only {growth:.3f}x "
          f"(cache traffic {shallow.cache_bytes/1e6:.0f} -> "
          f"{deep.cache_bytes/1e6:.0f} MB)")
    assert growth < 1.01, "cached depth must not cost DRAM bandwidth"
    assert deep.cache_bytes > shallow.cache_bytes


def test_head_dominates_once_body_is_cached():
    cfg = get_config("flagship")
    b = xios_budget(cfg)
    share = b.head_bytes / b.dram_bytes
    print(f"output projection is {share:.0%} of DRAM traffic "
          f"({b.head_bytes/1e6:.1f} of {b.dram_bytes/1e6:.1f} MB)")
    assert share > 0.4, "expected the head to become the bottleneck"


def test_designer_only_returns_cacheable_cores():
    rows = design_for_bandwidth()
    assert rows, "designer found nothing"
    assert all(r["tier"] in ("L2", "L3") for r in rows)
    best = rows[0]
    print(f"best design: dim={best['dim']} core={best['core_blocks']} "
          f"depth={best['depth']:.0f} -> {best['effective_depth']:.0f} "
          f"block-applications at {best['tok_s']:.0f} tok/s "
          f"({best['dram_mb']:.1f} MB/token)")
    assert best["effective_depth"] > 100


if __name__ == "__main__":
    test_cache_resident_core_beats_dense()
    test_oversized_core_is_worse_than_dense()
    test_depth_is_free_when_cached()
    test_head_dominates_once_body_is_cached()
    test_designer_only_returns_cacheable_cores()
    print("\nbandwidth tests passed")
