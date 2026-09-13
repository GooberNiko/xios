"""XIOS-Chat configuration.

Parameter budget is split three ways, deliberately:

    prelude   dense blocks, run once   -- tokenise / lift into latent space
    core      shared blocks, looped    -- ALL the reasoning capacity
    coda      dense blocks, run once   -- read out into vocabulary space

Only ``core`` is looped, and it is the only part whose *effective* depth is
decoupled from its parameter count.  A 6-block core looped 24 times performs
144 block-applications of sequential computation while occupying 6 blocks of
disk.  That ratio -- effective depth over stored depth -- is the entire bet.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal
import copy
import json
import pathlib


@dataclass
class XiosConfig:
    # ---- width -----------------------------------------------------------
    dim: int = 1024
    head_dim: int = 64
    n_heads: int = 16               # derived from dim / head_dim
    n_kv_heads: int = 4
    ffn_mult: float = 2.6667
    vocab_size: int = 32768
    max_seq_len: int = 8192

    # ---- depth -----------------------------------------------------------
    prelude_blocks: int = 2
    core_blocks: int = 6
    coda_blocks: int = 2

    # ---- adaptive depth --------------------------------------------------
    max_iters: int = 16             # hard ceiling on loop count
    min_iters: int = 2
    target_depth: float = 4.0       # average iterations the budget pays for
    ponder_eps: float = 0.02        # stop computing a token below this residual
    budget_weight: float = 0.05
    budget_mode: str = "lagrangian"   # "fixed" ignores the constraint and just nudges
    dual_lr: float = 0.002           # dual-ascent rate (normalised units)
    lambda_max: float = 1.0          # cap: the constraint must never dominate the LM loss
    depth_warmup_frac: float = 0.3  # fraction of training spent annealing the budget

    # ---- mixer layout ----------------------------------------------------
    # Every ``attn_every``-th block is windowed attention, the rest are gated
    # linear recurrences.  Recurrences carry the long range at O(1) memory;
    # attention supplies the exact local recall they are weak at.
    attn_every: int = 3
    attn_window: int = 512

    # ---- disk-resident associative memory --------------------------------
    memory_enabled: bool = False
    memory_layers: tuple = (2, 5)   # which core blocks swap FFN for memory
    memory_slots: int = 1 << 20     # 1M slots (product keys -> 2 x 1024 codes)
    memory_value_dim: int = 256
    memory_topk: int = 32
    memory_heads: int = 4
    memory_dtype: str = "int8"

    # ---- misc ------------------------------------------------------------
    use_adaln: bool = True          # iteration conditioning for the looped core
    cond_dim: int = 256
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.dim % self.head_dim == 0, "dim must be divisible by head_dim"
        self.n_heads = self.dim // self.head_dim
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.memory_layers = tuple(self.memory_layers)
        # The memory is partitioned per head and each partition is a square
        # product-key grid, so slots must factor as heads * m^2.  Round down
        # to the nearest valid size rather than failing deep in construction.
        if self.memory_enabled:
            # The store is partitioned per head and each partition is a square
            # product-key grid whose side must be a power of two, so that the
            # Morton address is pure arithmetic on two rank vectors instead of
            # a permutation table with one entry per slot.
            import math
            want = self.memory_slots
            m = int(math.isqrt(self.memory_slots // self.memory_heads))
            m = 1 << max(1, m.bit_length() - 1)
            self.memory_slots = self.memory_heads * m * m
            if self.memory_slots != want:
                import warnings
                warnings.warn(
                    f"memory_slots {want} is not heads*(2^k)^2; rounded down to "
                    f"{self.memory_slots} ({self.memory_heads} heads x {m}^2). "
                    f"Pick heads*(2^k)^2 exactly to avoid losing capacity.",
                    stacklevel=3)
            self.memory_grid = m

    # ---- layer layout ----------------------------------------------------
    def layer_kind(self, i: int) -> Literal["attn", "rec"]:
        return "attn" if (i + 1) % self.attn_every == 0 else "rec"

    @property
    def ffn_hidden(self) -> int:
        return int(self.ffn_mult * self.dim / 64 + 0.5) * 64

    @property
    def stored_blocks(self) -> int:
        return self.prelude_blocks + self.core_blocks + self.coda_blocks

    @property
    def effective_depth_max(self) -> int:
        """Block-applications on the hardest token."""
        return self.prelude_blocks + self.core_blocks * self.max_iters + self.coda_blocks

    @property
    def effective_depth_avg(self) -> float:
        return self.prelude_blocks + self.core_blocks * self.target_depth + self.coda_blocks

    @property
    def approx_params(self) -> int:
        d, h = self.dim, self.ffn_hidden
        per_block = 4 * d * d + 3 * d * h
        return self.stored_blocks * per_block + self.vocab_size * d * (1 if self.tie_embeddings else 2)

    def summary(self) -> str:
        return (f"dim={self.dim} blocks={self.prelude_blocks}+{self.core_blocks}*L+{self.coda_blocks} "
                f"params~{self.approx_params / 1e6:.0f}M  "
                f"depth avg={self.effective_depth_avg:.0f} max={self.effective_depth_max}  "
                f"(stored depth {self.stored_blocks})")

    # ---- (de)serialisation ----------------------------------------------
    def to_json(self, path) -> None:
        pathlib.Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path) -> "XiosConfig":
        return cls(**json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Presets.  ``nano`` and ``micro`` are the experiment rungs -- both train to
# convergence on a free Colab T4 in hours, which is what makes the
# architecture falsifiable before anything expensive gets committed to.
# --------------------------------------------------------------------------
PRESETS: dict[str, XiosConfig] = {
    "nano": XiosConfig(dim=384, head_dim=64, n_kv_heads=2, vocab_size=8192,
                       prelude_blocks=1, core_blocks=3, coda_blocks=1,
                       max_iters=8, target_depth=3.0, attn_every=2,
                       attn_window=256, max_seq_len=1024, cond_dim=128),
    "micro": XiosConfig(dim=768, head_dim=64, n_kv_heads=4, vocab_size=32768,
                        prelude_blocks=2, core_blocks=5, coda_blocks=2,
                        max_iters=16, target_depth=4.0, attn_every=3,
                        attn_window=512, max_seq_len=4096),
    "small": XiosConfig(dim=1536, head_dim=64, n_kv_heads=4, vocab_size=32768,
                        prelude_blocks=2, core_blocks=8, coda_blocks=2,
                        max_iters=24, target_depth=6.0, attn_every=3,
                        attn_window=768, max_seq_len=8192,
                        memory_enabled=True, memory_layers=(3, 6),
                        # 4 heads x 1024^2 -> exact, no rounding
                        memory_slots=1 << 22),
    "base": XiosConfig(dim=2048, head_dim=64, n_kv_heads=8, vocab_size=32768,
                       prelude_blocks=3, core_blocks=10, coda_blocks=3,
                       max_iters=32, target_depth=8.0, attn_every=3,
                       attn_window=1024, max_seq_len=16384,
                       memory_enabled=True, memory_layers=(4, 8),
                       # 4 heads x 2048^2 -> exact, no rounding
                       memory_slots=1 << 24, memory_value_dim=256),
}


# --------------------------------------------------------------------------
# `flagship` is designed by `xios.bandwidth.design_for_bandwidth`, not by hand.
#
# The rule it obeys, which is the opposite of how the presets above were built:
# hold the looped core inside L3 and spend everything else on ITERATIONS.
# Measured, scaling the core wider is actively harmful -- `base` above has a
# 2.4GB core that cannot be cached, so every iteration re-reads it from DRAM
# and it ends up reading 0.7x as many bytes per token as a 3B dense model,
# i.e. SLOWER on the hardware it was meant for.
#
# Width is therefore a constraint, not a knob. Capability that genuinely needs
# bits goes to the disk-resident memory, which costs no bandwidth beyond a few
# KB of reads per token. That makes mechanism 3 load-bearing rather than
# optional: with width capped, the memory is the only place knowledge can live.
# --------------------------------------------------------------------------
PRESETS["flagship"] = XiosConfig(
    dim=512, head_dim=64, n_kv_heads=2, vocab_size=32768,
    prelude_blocks=2, core_blocks=4, coda_blocks=2,
    max_iters=96, target_depth=48.0,      # ~200 block-applications per token
    attn_every=2, attn_window=1024, max_seq_len=16384, cond_dim=256,
    memory_enabled=True, memory_layers=(1, 3),
    # 4 heads x 2048^2 per layer: the knowledge lives here, not in the width
    memory_slots=1 << 24, memory_value_dim=256,
)


def get_config(name: str = "micro", **overrides) -> XiosConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = copy.deepcopy(PRESETS[name])
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg
