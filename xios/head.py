"""Clustered output projection: stop reading the whole vocabulary per token.

The problem this exists to solve was found by measurement, not assumed. Once
the looped core is cache-resident, `xios.bandwidth` reports that the output
projection is **54% of all DRAM traffic** on the `flagship` preset -- 16.8 MB
read for every single token. The core can be cached because it is reused
`depth` times; the head cannot, because each token needs a different part of
it and all of it is a candidate. So the head, not the body, is the bottleneck,
and shrinking the model buys nothing.

The idea
--------
A token's logit is `w_v . h`. Only tokens whose row points roughly along `h`
can win, so almost every row read is wasted. Cluster the vocabulary by row
direction, keep one centroid per cluster, and at run time:

  1. score the centroids           (n_clusters x dim  -- tiny)
  2. take the top few clusters
  3. read *only those clusters'* rows and score them exactly

Bytes read per token falls from `vocab x dim` to
`(n_clusters + selected_tokens) x dim`. On flagship that is ~0.4 MB instead
of 16.8 MB, a ~40x cut, and the surviving arithmetic is exact -- the
approximation is entirely in *which* tokens get considered, never in their
scores. That matters: a token inside a selected cluster gets its true logit,
so the distribution over the candidate set is the true conditional restricted
to that set.

This is speculative decoding's trick applied to the vocabulary axis instead
of the time axis: cheap proposal, exact rescoring of a small candidate set.

Getting it *exact* rather than merely cheap
-------------------------------------------
Clustering by direction alone is not enough, and measurement showed it:
top-1 agreement was only 78.9% at four clusters. The reason is that a logit
is `||w|| ||h|| cos(w,h)` -- a high-norm row in an unselected cluster can beat
a low-norm row in a selected one, and direction-based clustering is blind to
norm.

Storing two extra scalars per cluster fixes this and upgrades the whole scheme
from approximate to exact. For cluster `c` keep the unit centroid `u_c`, the
largest row norm `R_c`, and the widest angle `theta_c` between any member row
and `u_c`. Then for every `w` in `c`:

    w . h  <=  R_c * ||h|| * cos( max(0, angle(u_c, h) - theta_c) )

That is a genuine upper bound on the best logit the cluster could contain. So
visit clusters in descending bound order and stop as soon as the best logit
found so far beats the next cluster's bound: everything unvisited is then
provably worse, and the result is **the exact argmax** while still having read
only a fraction of the table. It is branch-and-bound over a cone
decomposition of the vocabulary, and the pruning is what buys the bandwidth.

`exact=True` uses the bound and is exact by construction; `exact=False` takes
a fixed number of clusters and is faster but approximate. Both are measured in
`evaluate_agreement`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadCost:
    exact_bytes: float
    clustered_bytes: float
    reduction: float
    n_clusters: int
    top_clusters: int
    tokens_scored: float


class ClusteredHead(nn.Module):
    """Drop-in replacement for a `dim -> vocab` output projection."""

    def __init__(self, weight: torch.Tensor, n_clusters: int = 256,
                 top_clusters: int = 4, bytes_per_weight: float = 1.0):
        """`weight` is the (vocab, dim) output matrix, already trained."""
        super().__init__()
        vocab, dim = weight.shape
        self.vocab, self.dim = vocab, dim
        self.n_clusters = n_clusters
        self.top_clusters = top_clusters
        self.bytes_per_weight = bytes_per_weight
        self.last_tokens_scored = 0.0

        assign, centroids = self._cluster(weight, n_clusters)
        # Sort rows by cluster so each cluster is one contiguous span: the
        # point is bandwidth, and a gather over scattered rows would touch
        # every page it was meant to avoid.
        order = torch.argsort(assign, stable=True)
        counts = torch.bincount(assign, minlength=n_clusters)
        starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])

        self.register_buffer("w_sorted", weight[order].contiguous())
        self.register_buffer("row_of_sorted", order)           # sorted -> vocab id
        self.register_buffer("centroids", centroids)           # (n_clusters, dim)
        self.register_buffer("starts", starts)
        self.register_buffer("counts", counts)

        # Per-cluster geometry for the bound: largest row norm, and the cosine
        # of the widest angle any member makes with the centroid.
        norms = weight.float().norm(dim=-1)
        unit = torch.nn.functional.normalize(weight.float(), dim=-1)
        max_norm = torch.zeros(n_clusters)
        min_cos = torch.ones(n_clusters)
        for c in range(n_clusters):
            m = assign == c
            if bool(m.any()):
                max_norm[c] = norms[m].max()
                min_cos[c] = (unit[m] @ centroids[c]).min().clamp(-1.0, 1.0)
        self.register_buffer("max_norm", max_norm)
        self.register_buffer("min_cos", min_cos)

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _cluster(weight: torch.Tensor, k: int, iters: int = 25):
        """Spherical k-means on row directions.

        Direction, not magnitude: a logit is a dot product, and clustering by
        raw Euclidean distance groups by norm, which is not what decides who
        wins.
        """
        w = F.normalize(weight.float(), dim=-1)
        n = w.shape[0]
        g = torch.Generator().manual_seed(0)
        cent = w[torch.randperm(n, generator=g)[:k]].clone()
        assign = torch.zeros(n, dtype=torch.long)
        for _ in range(iters):
            sim = w @ cent.T                       # (n, k)
            assign = sim.argmax(-1)
            for c in range(k):
                m = assign == c
                if bool(m.any()):
                    cent[c] = F.normalize(w[m].mean(0), dim=-1)
                else:                              # reseed an empty cluster
                    cent[c] = w[torch.randint(n, (1,), generator=g)].squeeze(0)
        return assign, cent

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor, top_clusters: Optional[int] = None,
                exact: bool = False) -> torch.Tensor:
        """Logits, with -inf outside the clusters that were visited.

        Scores inside the candidate set are exact; `exact=True` additionally
        guarantees the true argmax is inside it.
        """
        shape = h.shape[:-1]
        hf = h.reshape(-1, self.dim)
        out = torch.full((hf.shape[0], self.vocab), float("-inf"),
                         device=h.device, dtype=h.dtype)
        self.last_tokens_scored = 0.0

        cent = self.centroids.to(hf.dtype)
        csim = hf @ cent.T                                     # (N, C) cosines
        if exact:
            order, bounds = self._bound_order(hf, csim)
        else:
            k = top_clusters or self.top_clusters
            order = csim.topk(min(k, self.n_clusters), dim=-1).indices
            bounds = None

        total = 0
        for i in range(hf.shape[0]):
            best = float("-inf")
            for j in range(order.shape[1]):
                c = int(order[i, j])
                if bounds is not None and j > 0 and best >= float(bounds[i, j]):
                    break                                      # provably done
                s0, n = int(self.starts[c]), int(self.counts[c])
                if not n:
                    continue
                rows = self.w_sorted[s0:s0 + n].to(hf.dtype)    # contiguous span
                logit = rows @ hf[i]
                out[i, self.row_of_sorted[s0:s0 + n]] = logit
                best = max(best, float(logit.max()))
                total += n
        self.last_tokens_scored = total / max(hf.shape[0], 1)
        return out.reshape(*shape, self.vocab)

    def _bound_order(self, hf: torch.Tensor, csim: torch.Tensor):
        """Clusters sorted by a valid upper bound on their best logit."""
        hn = hf.float().norm(dim=-1, keepdim=True)              # (N,1)
        cos_phi = (csim.float() / hn.clamp(min=1e-9)).clamp(-1.0, 1.0)
        phi = torch.arccos(cos_phi)                            # (N,C)
        theta = torch.arccos(self.min_cos.clamp(-1.0, 1.0))[None, :]
        bound = self.max_norm[None, :] * hn * torch.cos(
            (phi - theta).clamp(min=0.0))
        order = bound.argsort(dim=-1, descending=True)
        return order, torch.gather(bound, 1, order)

    # ------------------------------------------------------------------
    def cost(self, top_clusters: Optional[int] = None,
             tokens_scored: Optional[float] = None) -> HeadCost:
        k = top_clusters or self.top_clusters
        avg = (tokens_scored if tokens_scored is not None
               else float(self.counts.float().mean()) * k)
        exact = self.vocab * self.dim * self.bytes_per_weight
        clustered = (self.n_clusters + avg) * self.dim * self.bytes_per_weight
        return HeadCost(exact_bytes=exact, clustered_bytes=clustered,
                        reduction=exact / max(clustered, 1.0),
                        n_clusters=self.n_clusters, top_clusters=k,
                        tokens_scored=avg)


# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_agreement(head: ClusteredHead, exact_w: torch.Tensor,
                       h: torch.Tensor, top_clusters_list=(1, 2, 4, 8, 16)
                       ) -> list[dict]:
    """Top-1 agreement, kept probability mass, and bytes, against the exact head.

    KL is computed over the kept support after renormalising. An earlier version
    filled masked entries with -1e9 and reported the resulting KL, which is a
    meaningless number in the hundreds of millions -- it measured the sentinel,
    not the method.
    """
    ref = h.reshape(-1, exact_w.shape[1]) @ exact_w.T.to(h.dtype)
    ref_top1 = ref.argmax(-1)
    ref_p = F.softmax(ref.float(), dim=-1)

    rows = []
    modes = [(k, False) for k in top_clusters_list] + [(None, True)]
    for k, exact in modes:
        approx = head(h, top_clusters=k, exact=exact).reshape(ref.shape)
        top1 = approx.argmax(-1)
        agree = float((top1 == ref_top1).float().mean())
        keep = torch.isfinite(approx)
        mass = float((ref_p * keep).sum(-1).mean())
        # KL(exact_restricted || approx_restricted) on the kept support
        rp = (ref_p * keep)
        rp = rp / rp.sum(-1, keepdim=True).clamp(min=1e-9)
        aq = F.softmax(approx.float().masked_fill(~keep, float("-inf")), dim=-1)
        kl = float((rp * ((rp.clamp(min=1e-12)).log()
                          - aq.clamp(min=1e-12).log())).sum(-1).mean())
        c = head.cost(k or head.n_clusters, head.last_tokens_scored)
        rows.append({"mode": "exact(bound)" if exact else f"top{k}",
                     "top1_agreement": agree, "kl": kl,
                     "prob_mass_kept": mass, "bytes": c.clustered_bytes,
                     "reduction": c.reduction,
                     "tokens_scored": head.last_tokens_scored})
    return rows
