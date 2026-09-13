"""Bytes-read-per-token: the metric that actually decides local speed.

Why not FLOPs
-------------
Autoregressive decoding is memory-bound, not compute-bound. Generating one
token from a dense model touches every weight exactly once, so the floor on
latency is *weights / bandwidth*, and the arithmetic units sit idle waiting.
A 7B int4 model is 3.5 GB; on a laptop at ~10 GB/s that is ~3 tokens/s no
matter how fast the CPU multiplies. Optimising FLOPs against that wall is
pointless -- the only thing that helps is reading fewer bytes.

Why a looped core is the right shape for weak hardware
------------------------------------------------------
This is the part that is usually missed, and it is not a small effect.

A dense stack must stream all of its weights from DRAM per token, because
each weight is used once and then never again for that token. A *looped* core
uses the same weights `depth` times in a row. If the core fits in a cache
tier, it is fetched from DRAM once and the remaining iterations hit cache:

    dense      DRAM bytes/token  =  all weights
    looped     DRAM bytes/token  =  core (once) + prelude + coda + head
               cache traffic     =  core x depth        (cheap, on-die)

So depth becomes nearly free in bandwidth terms, which is exactly the
resource that is scarce. The design rule that falls out is unusual and, as
far as I can tell, unexploited: **size the looped core to fit in L2/L3 and
buy capability with iterations rather than weights.**

The rule has a sting in the tail, which this module exists to expose: once
the core is cache-resident, the *output projection* becomes the bandwidth
bottleneck, because `vocab x dim` must be read every single token and cannot
be cached away. See `report()` -- it will tell you when the head dominates,
and that finding is what motivates shrinking or factorising the vocabulary
rather than the body.
"""
from __future__ import annotations

from dataclasses import dataclass

# Rough consumer-hardware tiers. Order matters, sizes are conservative.
@dataclass
class Tier:
    name: str
    bytes_: int
    gbps: float


TIERS = [
    Tier("L2", 2 << 20, 800.0),        # ~2MB, ~800 GB/s
    Tier("L3", 32 << 20, 200.0),       # ~32MB, ~200 GB/s
    Tier("DRAM", 32 << 30, 12.0),      # laptop dual-channel DDR4/5
    Tier("SSD", 2 << 40, 2.0),         # NVMe sequential
]


def tier_for(nbytes: int) -> Tier:
    for t in TIERS:
        if nbytes <= t.bytes_:
            return t
    return TIERS[-1]


def _bits_per_weight(quant: str) -> float:
    return {"fp16": 16.0, "int8": 8.0, "int4": 4.5}[quant]   # int4 + group scales


@dataclass
class Budget:
    dram_bytes: float          # per generated token, from DRAM
    cache_bytes: float         # per generated token, served from cache
    head_bytes: float          # the output projection's share of DRAM
    resident_bytes: float      # what must be held to run at all
    depth: float
    core_tier: str
    seconds_per_token: float

    @property
    def tokens_per_second(self) -> float:
        return 1.0 / max(self.seconds_per_token, 1e-12)


def _block_bytes(dim: int, ffn_hidden: int, bits: float) -> float:
    """Mixer + SwiGLU weights for one block."""
    return (4 * dim * dim + 3 * dim * ffn_hidden) * bits / 8


def xios_budget(cfg, depth: float | None = None, quant: str = "int4") -> Budget:
    bits = _bits_per_weight(quant)
    d, h = cfg.dim, cfg.ffn_hidden
    depth = cfg.target_depth if depth is None else depth

    core = cfg.core_blocks * _block_bytes(d, h, bits)
    dense_once = (cfg.prelude_blocks + cfg.coda_blocks) * _block_bytes(d, h, bits)
    # Embedding lookup is one row; the output projection is the whole table.
    head = cfg.vocab_size * d * (8.0 / 8)          # int8 table
    embed_row = d * 1.0

    tier = tier_for(int(core))
    cached = tier.name in ("L2", "L3")

    # DRAM traffic: the core is fetched once whether or not it is cache
    # resident; if it is NOT, every iteration re-fetches it.
    core_dram = core if cached else core * depth
    core_cache = core * (depth - 1) if cached else 0.0

    dram = core_dram + dense_once + head + embed_row
    seconds = (core_dram / (TIERS[2].gbps * 1e9)
               + core_cache / (tier.gbps * 1e9)
               + (dense_once + head) / (TIERS[2].gbps * 1e9))

    return Budget(dram_bytes=dram, cache_bytes=core_cache, head_bytes=head,
                  resident_bytes=core + dense_once + head, depth=depth,
                  core_tier=tier.name, seconds_per_token=seconds)


def dense_budget(n_params: int, vocab: int, dim: int,
                 quant: str = "int4") -> Budget:
    bits = _bits_per_weight(quant)
    head = vocab * dim * (8.0 / 8)
    body = max(n_params - vocab * dim, 0) * bits / 8
    dram = body + head
    seconds = dram / (TIERS[2].gbps * 1e9)
    return Budget(dram_bytes=dram, cache_bytes=0.0, head_bytes=head,
                  resident_bytes=dram, depth=1.0, core_tier="DRAM",
                  seconds_per_token=seconds)


def report(cfg, depth: float | None = None, quant: str = "int4",
           compare_dense_params: int | None = None) -> str:
    b = xios_budget(cfg, depth, quant)
    eff = cfg.prelude_blocks + cfg.core_blocks * b.depth + cfg.coda_blocks
    lines = [
        f"XIOS  dim={cfg.dim} core={cfg.core_blocks} blocks  depth={b.depth:.1f}"
        f"  ({eff:.0f} block-applications/token)",
        f"  looped core            {b.dram_bytes - b.head_bytes - cfg.dim:,.0f} B"
        f"  -> lives in {b.core_tier}",
        f"  output projection      {b.head_bytes:,.0f} B  (unavoidable, per token)",
        f"  DRAM bytes / token     {b.dram_bytes/1e6:,.1f} MB",
        f"  cache traffic / token  {b.cache_bytes/1e6:,.1f} MB  (nearly free)",
        f"  bandwidth-bound rate   {b.tokens_per_second:,.0f} tok/s",
    ]
    share = b.head_bytes / max(b.dram_bytes, 1)
    if share > 0.5:
        lines.append(
            f"  NOTE the output projection is {share:.0%} of all DRAM traffic."
            f" The body is no longer the bottleneck -- shrinking the vocabulary"
            f" or factorising the head buys more speed than shrinking the model.")
    if compare_dense_params:
        dn = dense_budget(compare_dense_params, cfg.vocab_size, cfg.dim, quant)
        lines += [
            "",
            f"dense baseline with {compare_dense_params/1e9:.1f}B params:",
            f"  DRAM bytes / token     {dn.dram_bytes/1e6:,.1f} MB",
            f"  bandwidth-bound rate   {dn.tokens_per_second:,.1f} tok/s",
            f"  XIOS reads {dn.dram_bytes / max(b.dram_bytes,1):.1f}x fewer bytes"
            f" per token and is {b.tokens_per_second/max(dn.tokens_per_second,1e-9):.1f}x"
            f" faster at the same bandwidth",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    from .config import PRESETS, get_config
    for name in PRESETS:
        cfg = get_config(name)
        print(report(cfg, compare_dense_params=3_000_000_000))
        print()


# ---------------------------------------------------------------------------
# Designing *for* the memory hierarchy instead of against it
# ---------------------------------------------------------------------------
def design_for_bandwidth(cache_budget: int = 24 << 20,
                         dram_per_token: int = 64 << 20,
                         quant: str = "int4",
                         head_dim: int = 64,
                         min_depth: float = 8.0) -> list[dict]:
    """Search configs that maximise sequential depth under a bandwidth budget.

    The measurement that motivates this: scaling the core *wider* pushes it out
    of cache, and once that happens every iteration re-reads it from DRAM, so
    the looped design becomes strictly worse than a dense one. Measured on the
    original presets, `base` (2.4 GB core, depth 8) read 0.7x as many bytes per
    token as a 3B dense model -- i.e. it was slower on the exact hardware it
    was meant for.

    So width is a constraint, not a knob: pick the largest core that still
    fits the cache, then spend everything else on iterations. Capability that
    needs raw bits goes to the disk memory instead, which costs no bandwidth
    beyond a few KB of reads.
    """
    bits = _bits_per_weight(quant)
    out = []
    for dim in (512, 640, 768, 896, 1024, 1280, 1536):
        ffn_hidden = int(2.6667 * dim / 64 + 0.5) * 64
        blk = _block_bytes(dim, ffn_hidden, bits)
        for core_blocks in (2, 3, 4, 6):
            core = core_blocks * blk
            if core > cache_budget:
                continue
            for vocab in (8192, 16384, 32768):
                head = vocab * dim                      # int8 table
                fixed = head + 2 * blk                  # + prelude/coda
                if fixed > dram_per_token:
                    continue
                # cache-resident core costs DRAM once; depth is then free of
                # DRAM traffic, so depth is bounded by latency, not bytes
                dram = core + fixed
                if dram > dram_per_token:
                    continue
                tier = tier_for(int(core))
                if tier.name not in ("L2", "L3"):
                    continue
                # how many iterations fit in the same wall-clock as the DRAM read
                budget_s = dram / (TIERS[2].gbps * 1e9)
                per_iter_s = core / (tier.gbps * 1e9)
                depth = max(min_depth, budget_s / max(per_iter_s, 1e-12))
                out.append({
                    "dim": dim, "core_blocks": core_blocks, "vocab": vocab,
                    "core_mb": core / 1e6, "head_mb": head / 1e6,
                    "dram_mb": dram / 1e6, "tier": tier.name,
                    "depth": depth,
                    "effective_depth": 2 + core_blocks * depth + 2,
                    "tok_s": 1.0 / (budget_s + depth * per_iter_s),
                })
    out.sort(key=lambda r: -r["effective_depth"])
    return out
