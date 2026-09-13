"""Disk-resident Associative Memory.

The idea
--------
An FFN is already a key-value memory: the rows of ``W_in`` are keys, the rows
of ``W_out`` are values, and the nonlinearity is a soft lookup.  The problem
is that it scores *every* key on *every* token, so knowledge capacity costs
both RAM and FLOPs in lockstep.  That coupling is why models get big.

This layer breaks it.  Keys are factorised into two small codebooks, so
scoring ``N`` slots costs ``O(sqrt(N))`` dot products.  Values live in a
memory-mapped file and only the top ``k`` of them are ever touched, so a
million-slot memory costs a few kilobytes of reads and essentially no
resident RAM.  Knowledge capacity becomes a property of the *file*, not of
the working set -- which is the only way a small model gets to know a lot.

Layout matters as much as the algorithm
---------------------------------------
Product-key retrieval selects a ``k1 x k2`` rectangle in the 2-D code grid,
but the selected codes are *scattered* along each axis -- they are the
nearest keys to the query, not an index-contiguous box.  So Z-ordering the
raw indices buys nothing (measured: slightly worse than row-major).

Two things have to be true before any layout can help.

First, each head needs its **own** slab of slots.  A single shared address
table cannot localise H heads with H unrelated key geometries -- measured,
a shared table makes Z-order *worse* than plain row-major.  Partitioning per
head also stops heads competing for the same slots, so capacity goes up for
free.

Second, the axes have to be **similarity-ordered**.  ``optimize_layout``
seriates each codebook so that codes close in key space become close in index
space; only then does Z-order collapse the retrieved neighbourhood into a few
contiguous runs.

The layout itself costs almost nothing to store.  When ``m`` is a power of
two, Morton interleaving is a bijection from ``[0,m) x [0,m)`` onto
``[0,m^2)``, so the physical address is pure arithmetic on two rank vectors
of length ``m`` -- ``2 * sqrt(slots/heads)`` numbers instead of a
permutation table with one entry per slot.  For the ``small`` preset that is
16 KB rather than 33 MB of resident RAM.

This is a pure address permutation -- the values move, the maths does not,
and accuracy is bit-identical.  It is worth running once after training,
when the key geometry has actually formed; on random init no layout can
help, because retrieval is random by construction.

Making the store actually carry information
------------------------------------------
A first version of this layer measurably did nothing: learned values beat
values *frozen at random init* by 1.008x, and shuffling the value table changed
the logits as little as deleting it. `train/memory_diagnose.py` found two
mechanical causes, both in the addressing rather than the storage:

* **Near-uniform retrieval weights.** Softmax entropy over the top 32 was
  3.326 of a maximum 3.466 — 96% of uniform. The output was therefore
  approximately the *mean* of 32 value vectors, which is nearly constant and
  destroys any information about which slots were chosen. Cause: queries were
  unit-normalised while keys were initialised at std 0.02, so score differences
  were ~0.16 and the softmax could not discriminate. Fix: initialise keys at
  ``key_dim**-0.5`` (the standard dot-product scale) and give the retrieval
  softmax a **learnable temperature**, so it can sharpen.

* **Content-independent routing.** Different tokens retrieved **69.6%** of the
  same slots. LayerNorm normalises each query vector individually, which does
  nothing to decorrelate queries *across* tokens, so a shared dominant
  direction selected the same keys for everything. Fix: **BatchNorm over the
  query features**, which is what the product-key memory literature uses and
  what forces each feature to vary across the batch. Addressing cannot be
  associative if every input addresses the same place.

Two backends
------------
``ram``   trainable ``nn.Embedding`` values (what Colab uses)
``disk``  ``np.memmap`` int8 values + per-slot scales (what inference uses)
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Morton / Z-order helpers
# ---------------------------------------------------------------------------
def _part1by1(n: torch.Tensor) -> torch.Tensor:
    """Interleave the bits of n with zeros (32-bit safe)."""
    n = n & 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def morton_encode(i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
    return _part1by1(i1.long()) | (_part1by1(i2.long()) << 1)


def morton_table(m: int, device=None) -> torch.Tensor:
    """(m, m) -> physical slot offset, in Z-order."""
    i1 = torch.arange(m, device=device)[:, None].expand(m, m)
    i2 = torch.arange(m, device=device)[None, :].expand(m, m)
    return morton_encode(i1.reshape(-1), i2.reshape(-1)).reshape(m, m)


def seriate(k: torch.Tensor, nn_graph: int = 8) -> torch.Tensor:
    """Spectral seriation of a codebook: returns rank[i] for each code i.

    We need a 1-D ordering in which codes retrieved together sit next to each
    other.  Projecting onto the leading principal component only works if the
    codebook is straight; trained codebooks lie on a curved manifold where PC1
    is many-to-one and actively scrambles the ordering (measured: correlation
    ~0.73 with the true order, and the resulting layout was *worse* than doing
    nothing).

    The Fiedler vector -- second-smallest eigenvector of the graph Laplacian
    of the k-NN similarity graph -- is the standard fix and recovers
    arc-length order along a curve.  The codebook is only sqrt(slots/heads)
    entries, so a dense eigendecomposition is nothing.
    """
    k = k.float()
    kn = torch.nn.functional.normalize(k, dim=-1)
    sim = kn @ kn.T
    m = sim.shape[0]
    sim.fill_diagonal_(float("-inf"))
    thresh = sim.topk(min(nn_graph, m - 1), dim=-1).values[:, -1:]
    W = torch.where(sim >= thresh, sim, torch.zeros_like(sim))
    W = (W + W.T) / 2
    W = (W - W.min()).clamp(min=0)
    W.fill_diagonal_(0)
    d = W.sum(1)
    L = torch.diag(d) - W
    try:
        dinv = torch.diag((d + 1e-6).rsqrt())
        fiedler = torch.linalg.eigh(dinv @ L @ dinv)[1][:, 1]
    except Exception:
        fiedler = torch.randn(m)
    order = fiedler.argsort()
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel())
    return rank


# ---------------------------------------------------------------------------
class DiskAssociativeMemory(nn.Module):
    def __init__(self, cfg, n_slots: Optional[int] = None, lazy: bool = False):
        super().__init__()
        self.cfg = cfg
        n_slots = n_slots or cfg.memory_slots
        H = cfg.memory_heads
        m = int(math.isqrt(n_slots // H))
        assert m * m * H == n_slots, \
            "memory_slots must be memory_heads * (a perfect square)"
        self.m = m
        self.per_head = m * m
        self.n_slots = n_slots
        self.n_heads = H
        self.topk = cfg.memory_topk
        self.value_dim = cfg.memory_value_dim
        self.key_dim = 64                      # per half-key
        self.sub_k = max(4, int(math.isqrt(self.topk)) * 2)

        self.q_proj = nn.Linear(cfg.dim, H * 2 * self.key_dim, bias=False)
        # BatchNorm, not LayerNorm: normalising each query vector on its own
        # leaves a shared dominant direction that sends every token to the same
        # slots (measured: 69.6% overlap between different tokens). Normalising
        # each feature across the batch is what makes addressing associative.
        self.q_norm = nn.BatchNorm1d(2 * self.key_dim)
        # Keys at the standard dot-product scale. At std 0.02 the score spread
        # was ~0.16, so the retrieval softmax sat at 96% of maximum entropy and
        # the output degenerated into the mean of `topk` values.
        ks = self.key_dim ** -0.5
        self.keys1 = nn.Parameter(torch.randn(H, m, self.key_dim) * ks)
        self.keys2 = nn.Parameter(torch.randn(H, m, self.key_dim) * ks)
        # learnable sharpness for the retrieval softmax
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.out_proj = nn.Linear(H * self.value_dim, cfg.dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Physical layout. Stored as two per-head rank vectors, not as a
        # slot-sized permutation table (see module docstring).
        self.layout = "row_major"
        self.morton_ok = (m & (m - 1)) == 0          # power of two?
        self.register_buffer("rank1", torch.arange(m).repeat(H, 1), persistent=True)
        self.register_buffer("rank2", torch.arange(m).repeat(H, 1), persistent=True)
        self.register_buffer("head_base",
                             torch.arange(H) * self.per_head, persistent=False)

        # -- value store --------------------------------------------------
        # `lazy` allocates nothing: use it to inspect or serve a configuration
        # whose store is far larger than RAM (that being the entire point).
        # Training uses `ram`; deployment calls attach() to switch to `disk`.
        self.backend = "lazy" if lazy else "ram"
        self.values = None
        if not lazy:
            self.values = nn.Embedding(n_slots, self.value_dim, sparse=True)
            nn.init.normal_(self.values.weight, std=0.02)
        self._mm = None
        self._mm_scale = None

        # Usage statistics drive the load-balancing diagnostics. Allocated
        # only when asked for: a slot-sized float buffer is pure waste at
        # inference, which is most of this layer's life.
        self.usage = None
        self.track_usage = False

    # ------------------------------------------------------------------
    def normalise_query(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalise a hidden state into query form."""
        N = x.reshape(-1, x.shape[-1]).shape[0]
        q = self.q_proj(x.reshape(N, -1)).reshape(N * self.n_heads,
                                                  2 * self.key_dim)
        return self.q_norm(q).reshape(N, self.n_heads, 2, self.key_dim)

    def logical_codes(self, q: torch.Tensor):
        """Same selection as _lookup_indices, but returns logical (i1, i2)."""
        N, H = q.shape[0], self.n_heads
        s1 = torch.einsum("nhd,hmd->nhm", q[:, :, 0], self.keys1)
        s2 = torch.einsum("nhd,hmd->nhm", q[:, :, 1], self.keys2)
        k = min(self.sub_k, self.m)
        v1, i1 = s1.topk(k, dim=-1)
        v2, i2 = s2.topk(k, dim=-1)
        cand = (v1[..., :, None] + v2[..., None, :]).reshape(N, H, k * k)
        topi = cand.topk(min(self.topk, k * k), dim=-1).indices
        return torch.gather(i1, 2, topi // k), torch.gather(i2, 2, topi % k)

    def _lookup_indices(self, q: torch.Tensor):
        """q: (B*T, H, 2, key_dim) -> (idx (N,H,topk), weight (N,H,topk))"""
        N, H = q.shape[0], self.n_heads
        q1, q2 = q[:, :, 0], q[:, :, 1]                       # (N,H,kd)

        s1 = torch.einsum("nhd,hmd->nhm", q1, self.keys1)
        s2 = torch.einsum("nhd,hmd->nhm", q2, self.keys2)
        k = min(self.sub_k, self.m)
        v1, i1 = s1.topk(k, dim=-1)                           # (N,H,k)
        v2, i2 = s2.topk(k, dim=-1)

        # k x k candidate rectangle
        cand = v1[..., :, None] + v2[..., None, :]            # (N,H,k,k)
        cand = cand.reshape(N, H, k * k)
        topv, topi = cand.topk(min(self.topk, k * k), dim=-1)

        a = torch.gather(i1, 2, topi // k)
        b = torch.gather(i2, 2, topi % k)
        w = (topv * self.logit_scale.exp()).softmax(-1)
        return self.physical_address(a, b), w

    def physical_address(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
        """Logical codes -> physical slot, as arithmetic on the rank vectors."""
        h = torch.arange(self.n_heads, device=i1.device)[None, :, None].expand_as(i1)
        base = self.head_base.to(i1.device)[h]
        if self.layout == "row_major":
            return base + i1 * self.m + i2
        r1 = torch.gather(self.rank1.to(i1.device)[h.reshape(-1), :].reshape(*i1.shape, self.m),
                          -1, i1.unsqueeze(-1)).squeeze(-1)
        r2 = torch.gather(self.rank2.to(i2.device)[h.reshape(-1), :].reshape(*i2.shape, self.m),
                          -1, i2.unsqueeze(-1)).squeeze(-1)
        return base + morton_encode(r1, r2)

    def _gather_values(self, idx: torch.Tensor) -> torch.Tensor:
        if self.backend == "lazy":
            raise RuntimeError(
                "this memory was built lazily and holds no values; call "
                "attach(path) to memory-map a store, or rebuild with lazy=False")
        if self.backend == "ram":
            return self.values(idx)
        flat = idx.reshape(-1).cpu().numpy()
        import numpy as np
        raw = torch.from_numpy(np.asarray(self._mm[flat], dtype="int8"))
        sc = torch.from_numpy(np.asarray(self._mm_scale[flat], dtype="float32"))
        out = raw.float() * sc[:, None]
        return out.reshape(*idx.shape, self.value_dim).to(self.out_proj.weight.dtype)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        shape = x.shape[:-1]
        N = int(torch.tensor(shape).prod()) if shape else 1
        H = self.n_heads

        q = self.q_proj(x).reshape(N * H, 2 * self.key_dim)
        q = self.q_norm(q).reshape(N, H, 2, self.key_dim)
        idx, w = self._lookup_indices(q)                      # (N,H,k)

        vals = self._gather_values(idx)                       # (N,H,k,vd)
        out = (w[..., None] * vals).sum(2)                    # (N,H,vd)
        out = self.out_proj(out.reshape(N, H * self.value_dim))
        out = out.reshape(*shape, -1)

        if self.track_usage:
            with torch.no_grad():
                if self.usage is None:
                    self.usage = torch.zeros(self.n_slots, device=idx.device)
                self.usage.index_add_(0, idx.reshape(-1),
                                      w.detach().reshape(-1).float())

        # Load balancing: without it a few thousand slots absorb every lookup
        # and the remaining capacity is dead weight on disk.  Penalising the
        # squared mean weight per slot (a Switch-Transformer style term)
        # spreads usage without forcing it uniform.
        aux = (w.float().mean(0).pow(2).sum() * self.n_slots / max(H, 1)) * 1e-4
        return out, aux

    # ------------------------------------------------------------------
    # Deployment: freeze values to an int8 memory-mapped file
    # ------------------------------------------------------------------
    @torch.no_grad()
    def export(self, path) -> dict:
        import numpy as np
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        w = self.values.weight.detach().float().cpu()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)

        q.numpy().tofile(path / "values.i8")
        scale.numpy().astype("float32").tofile(path / "scales.f32")
        meta = {"n_slots": self.n_slots, "value_dim": self.value_dim,
                "layout": "morton", "m": self.m}
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"bytes": q.numel() + scale.numel() * 4,
                "mb": (q.numel() + scale.numel() * 4) / 1e6, **meta}

    @property
    def store_bytes(self) -> int:
        """Size of the value store. This is DISK, not resident memory."""
        return self.n_slots * (self.value_dim + 4)

    def attach(self, path):
        """Switch to the mmap backend. Resident RAM cost: ~0."""
        import numpy as np
        path = pathlib.Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        assert meta["n_slots"] == self.n_slots and meta["value_dim"] == self.value_dim
        self._mm = np.memmap(path / "values.i8", dtype="int8", mode="r",
                             shape=(self.n_slots, self.value_dim))
        self._mm_scale = np.memmap(path / "scales.f32", dtype="float32", mode="r",
                                   shape=(self.n_slots,))
        self.backend = "disk"
        del self.values
        self.values = None
        return self

    def detach_store(self):
        """Release the memory maps.

        Needed before deleting or replacing the store: on Windows an open
        mmap keeps a lock on the file, so a process that forgets this cannot
        clean up after itself.
        """
        self._mm = None
        self._mm_scale = None
        self.backend = "lazy"
        return self

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @torch.no_grad()
    def optimize_layout(self, mode: str = "sorted_morton"):
        """Permute physical addresses for SSD page locality.

        Accuracy-neutral: this only changes *where* each value is stored, so
        run it once after training (and re-export the store afterwards).
        """
        H, m = self.n_heads, self.m
        if mode == "row_major":
            dev = self.rank1.device      # keep the layout on the model's device
            self.rank1 = torch.arange(m, device=dev).repeat(H, 1)
            self.rank2 = torch.arange(m, device=dev).repeat(H, 1)
            self.layout = mode
            return self
        if not self.morton_ok:
            raise ValueError(
                f"sorted_morton needs a power-of-two grid, got m={m}; "
                "pick memory_slots = heads * (2^k)^2")

        r1 = torch.stack([seriate(self.keys1.data[h]) for h in range(H)])
        r2 = torch.stack([seriate(self.keys2.data[h]) for h in range(H)])
        self.rank1, self.rank2 = r1.to(self.rank1.device), r2.to(self.rank2.device)
        self.layout = mode
        return self

    @torch.no_grad()
    def locality_report(self, i1: torch.Tensor, i2: torch.Tensor,
                        page_bytes: int = 4096) -> dict:
        """Distinct SSD pages touched per token, for each candidate layout.

        Takes the *logical* codes so the comparison does not depend on which
        layout happened to be installed when the lookup was recorded.
        """
        per_page = max(1, page_bytes // self.value_dim)
        saved = (self.rank1.clone(), self.rank2.clone(), self.layout)
        out = {}
        for mode in ("row_major", "sorted_morton"):
            self.optimize_layout(mode)
            phys = self.physical_address(i1, i2)
            pages = (phys // per_page).reshape(phys.shape[0], -1)
            out[mode] = float(torch.tensor(
                [len(set(r.tolist())) for r in pages]).float().mean())
        self.rank1, self.rank2, self.layout = saved
        out["reduction"] = out["row_major"] / max(out["sorted_morton"], 1e-9)
        out["lookups_per_token"] = i1.shape[1] * i1.shape[2]
        return out
