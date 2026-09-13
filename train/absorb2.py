"""Absorption, third design: share the teacher's own block type, warm-started.

What the first two attempts got wrong
-------------------------------------
Attempts 1 and 2 asked the core to do two hard things at once:

 (a) **share weights across depth** -- the thing that actually buys the
     bandwidth win, and
 (b) **replace the mechanism** -- GPT-2's full attention, GELU MLP and
     LayerNorm swapped for gated linear recurrences, SwiGLU and RMSNorm.

Two stacked approximations, and the measurements (55% -> 69% -> 73% retained)
could not tell which one was doing the damage. The flagship argument only needs
(a). So this attempt isolates it: the shared core is built from **GPT-2's own
block class**, and initialised from **GPT-2's own weights**.

Two changes, both aimed at making the problem easier rather than the optimiser
better:

**1. Matched block type.** The core is `k` real GPT-2 blocks, shared across
`N/k` iterations. Now the only question being asked is whether one block can
stand in for several at different depths -- a question about redundancy, not
about function approximation across an architecture gap.

**2. Warm start by group-position averaging.** Shared block `j` is initialised
from the mean of the teacher's blocks at the same position in each group --
block `j` averages layers `j, k+j, 2k+j, ...`. Weight averaging is only
sensible between models that are already close, which is exactly the claim
being tested; if the layers are redundant enough to share, their weights should
be close enough to average. It starts training near the answer instead of
searching for it, which matters enormously on a CPU budget.

Per-iteration LoRA adapters then let each iteration deviate from the shared
mean, so sharing does not force every depth to behave identically.

Then `train/redundancy.py` measured what was actually going on, and it
reshaped this file completely.

**Measured on GPT-2 (12 layers):**

* Weight-space cosine between layers: **0.009**. The layers are essentially
  orthogonal. Averaging them is meaningless -- which is exactly why the warm
  start above produced a perplexity of 351,502.
* Only **13 of 132** layer pairs substitute for each other better than doing
  nothing at all. Replacing layer `i` with layer `i+1` is on average ~5x worse
  than simply skipping layer `i`.
* But the layer-drop test is stark: removing any single one of layers 1-10
  costs only **1.08-1.33x** perplexity, while removing **layer 0 costs 134x**
  and layer 11 costs 3.4x.

So GPT-2 is not uniformly redundant -- it is *mostly* redundant with two
irreplaceable layers at the ends. Every previous attempt forced one shared
block to stand in for both layer 0 (unique, 134x) and the interchangeable
middle, which is why they all plateaued.

The fix follows directly from the measurement rather than from taste: **keep
the boundary layers exactly as the teacher wrote them, and share only the
middle.** Warm-start the shared block from a single middle layer, not an
average, because averaging is now known to be meaningless.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(12)

from xios.adapters import DenseSpec, from_huggingface


# ---------------------------------------------------------------------------
class SharedTeacherCore(nn.Module):
    """`k` blocks of the teacher's own type, shared across iterations."""

    def __init__(self, teacher_blocks, core_blocks: int, dim: int,
                 adapter_rank: int = 32, warm_start: bool = True,
                 keep_first: int = 1, keep_last: int = 1):
        super().__init__()
        N = len(teacher_blocks)
        self.keep_first, self.keep_last = keep_first, keep_last

        # Boundary layers are copied verbatim and frozen. Measured: dropping
        # GPT-2's layer 0 costs 134x perplexity and layer 11 costs 3.4x, while
        # any single middle layer costs under 1.33x. Forcing a shared block to
        # cover both regimes is what capped every earlier attempt.
        self.head_layers = nn.ModuleList(
            [copy.deepcopy(teacher_blocks[i]) for i in range(keep_first)])
        self.tail_layers = nn.ModuleList(
            [copy.deepcopy(teacher_blocks[N - keep_last + i]) for i in range(keep_last)])
        for m in list(self.head_layers) + list(self.tail_layers):
            for p in m.parameters():
                p.requires_grad_(False)

        middle = teacher_blocks[keep_first:N - keep_last]
        assert len(middle) % core_blocks == 0, \
            f"{len(middle)} middle layers must divide by {core_blocks}"
        self.k = core_blocks
        self.n_iters = len(middle) // core_blocks
        self.dim = dim

        # one shared block per position within a group, warm-started from a
        # single representative middle layer (averaging is meaningless here:
        # measured weight cosine between layers is 0.009)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(middle[(len(middle) // 2) + j - core_blocks // 2])
             for j in range(core_blocks)])
        for p in self.blocks.parameters():
            p.requires_grad_(True)

        self.adapter_rank = adapter_rank
        if adapter_rank:
            self.lora_a = nn.Parameter(
                torch.randn(self.n_iters, dim, adapter_rank) * 0.02)
            self.lora_b = nn.Parameter(torch.zeros(self.n_iters, adapter_rank, dim))
        # per-iteration affine on the residual stream: cheap, and lets each
        # depth rescale what the shared block writes
        self.iter_scale = nn.Parameter(torch.ones(self.n_iters, dim))
        self.iter_shift = nn.Parameter(torch.zeros(self.n_iters, dim))

    @torch.no_grad()
    def _average_init(self, teacher_blocks):
        """Shared block j <- mean of teacher blocks at position j of each group."""
        for j, blk in enumerate(self.blocks):
            group = [teacher_blocks[g * self.k + j] for g in range(self.n_iters)]
            tgt = dict(blk.named_parameters())
            stacks: dict[str, list[torch.Tensor]] = {n: [] for n in tgt}
            for src in group:
                for n, p in src.named_parameters():
                    if n in stacks:
                        stacks[n].append(p.detach().float())
            for n, ps in stacks.items():
                if ps:
                    tgt[n].copy_(torch.stack(ps).mean(0).to(tgt[n].dtype))

    def _call_block(self, blk, h):
        for kwargs in ({}, {"attention_mask": None},
                       {"attention_mask": None, "position_ids": None}):
            try:
                out = blk(h, **kwargs)
                return out[0] if isinstance(out, tuple) else out
            except TypeError:
                continue
        raise RuntimeError("could not call block")

    def step_group(self, h: torch.Tensor, i: int) -> torch.Tensor:
        h = h * self.iter_scale[i] + self.iter_shift[i]
        for blk in self.blocks:
            h = self._call_block(blk, h)
        if self.adapter_rank:
            h = h + (h @ self.lora_a[i]) @ self.lora_b[i]
        return h

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for m in self.head_layers:
            h = self._call_block(m, h)
        for i in range(self.n_iters):
            h = self.step_group(h, i)
        for m in self.tail_layers:
            h = self._call_block(m, h)
        return h

    def waypoint_offset(self) -> int:
        """Teacher layer index where the shared region starts."""
        return self.keep_first


class Absorbed(nn.Module):
    def __init__(self, spec: DenseSpec, core: SharedTeacherCore):
        super().__init__()
        self.spec, self.core = spec, core

    def forward(self, ids):
        h = self.core(self.spec.embed(ids))
        return self.spec.head(self.spec.final_norm(h))


# ---------------------------------------------------------------------------
@torch.no_grad()
def middle_waypoints(spec, core, ids):
    """Residual stream at each shared-group boundary inside the middle region.

    The boundary layers are kept verbatim, so the shared core only has to
    reproduce the middle of the teacher -- these are its targets.
    """
    h = spec.embed(ids)
    pts = []
    start, end = core.keep_first, spec.n_layers - core.keep_last
    for i, f in enumerate(spec.layers):
        if i == start:
            pts.append(h)
        h = f(h)
        if start < i + 1 <= end and (i + 1 - start) % core.k == 0:
            pts.append(h)
    return pts


def fit(spec, core, data_fn, steps, lr, device, mix_max, mix_warmup,
        log_every=50):
    opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    n_iters = core.n_iters
    hist, t0 = [], time.time()
    for step in range(steps):
        frac = step / max(steps, 1)
        lr_now = lr * min(1.0, (step + 1) / 30) * (
            0.5 * (1 + math.cos(math.pi * min(frac, 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now
        p_self = mix_max * min(1.0, frac / max(mix_warmup, 1e-6))

        ids = data_fn(step).to(device)
        pts = middle_waypoints(spec, core, ids)

        loss, per_iter = 0.0, []
        h_self = pts[0].detach()
        for i in range(n_iters):
            inp = h_self if (i > 0 and rng.random() < p_self) else pts[i].detach()
            pred = core.step_group(inp, i)
            tgt = pts[i + 1].detach()
            r = (pred - tgt).pow(2).mean() / tgt.pow(2).mean().clamp(min=1e-8)
            loss = loss + (2.0 if i == n_iters - 1 else 1.0) * r
            per_iter.append(float(r.detach()))
            h_self = pred.detach()
        loss = loss / (n_iters + 1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in core.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "rel_mse": float(loss),
                         "per_iter": per_iter, "p_self": p_self})
            print(f"[fit] {step:5d}  rel_mse {float(loss):.5f}  "
                  f"worst {max(per_iter):.5f}  p {p_self:.2f}  "
                  f"lr {lr_now:.2e}  {time.time()-t0:.0f}s", flush=True)
    return hist


def distil(spec, model, data_fn, steps, lr, device, log_every=50):
    ps = [p for p in model.core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    hist, t0 = [], time.time()
    print(f"\ndistillation: {steps} steps end-to-end")
    for step in range(steps):
        lr_now = lr * min(1.0, (step + 1) / 20) * (
            0.5 * (1 + math.cos(math.pi * min(step / max(steps, 1), 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now
        ids = data_fn(step).to(device)
        with torch.no_grad():
            rp = F.log_softmax(spec.forward(ids).float(), -1)
        gp = F.log_softmax(model(ids).float(), -1)
        loss = F.kl_div(gp, rp, log_target=True, reduction="batchmean") / ids.shape[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "kl": float(loss)})
            print(f"[distil] {step:5d}  kl {float(loss):.4f}  lr {lr_now:.2e}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    return hist


@torch.no_grad()
def compare(spec, model, batches, device):
    agree = kl = dl = cl = 0.0
    for ids in batches:
        ids = ids.to(device)
        ref = spec.forward(ids)
        got = model(ids)
        dl += float(F.cross_entropy(ref[:, :-1].reshape(-1, ref.size(-1)).float(),
                                    ids[:, 1:].reshape(-1)))
        cl += float(F.cross_entropy(got[:, :-1].reshape(-1, got.size(-1)).float(),
                                    ids[:, 1:].reshape(-1)))
        rp = F.log_softmax(ref.float(), -1); gp = F.log_softmax(got.float(), -1)
        agree += float((ref.argmax(-1) == got.argmax(-1)).float().mean())
        kl += float((rp.exp() * (rp - gp)).sum(-1).mean())
    n = len(batches)
    return {"top1_agreement": agree / n, "kl": kl / n,
            "dense_ppl": math.exp(dl / n), "absorbed_ppl": math.exp(cl / n),
            "ppl_retained": math.exp(dl / n) / math.exp(cl / n)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--core-blocks", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--distil-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--distil-lr", type=float, default=1e-5)
    ap.add_argument("--adapter-rank", type=int, default=32)
    ap.add_argument("--mix-max", type=float, default=0.7)
    ap.add_argument("--mix-warmup", type=float, default=0.4)
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--keep-first", type=int, default=1,
                    help="teacher layers copied verbatim at the start "
                         "(GPT-2 layer 0 costs 134x perplexity if lost)")
    ap.add_argument("--keep-last", type=int, default=1)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--corpus", default="data/prose.txt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/absorb2")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    spec = from_huggingface(hf, a.model)

    core = SharedTeacherCore(hf._xios_blocks, a.core_blocks, spec.dim,
                             a.adapter_rank, not a.no_warm_start,
                             a.keep_first, a.keep_last).to(dev)
    n_core = sum(p.numel() for p in core.parameters())
    print(f"{a.model}: {spec.n_layers} layers -> "
          f"{a.keep_first} kept + {a.core_blocks} shared x {core.n_iters} iters "
          f"+ {a.keep_last} kept")
    print(f"core {n_core/1e6:.1f}M vs dense body {spec.body_params/1e6:.1f}M "
          f"({spec.body_params/max(n_core,1):.2f}x fewer)")

    text = pathlib.Path(a.corpus).read_text(encoding="utf-8", errors="replace")
    ids_all = torch.tensor(tok(text).input_ids, dtype=torch.long)
    n_val = min(40_000, ids_all.numel() // 10)
    val, tr = ids_all[:n_val], ids_all[n_val:]

    def make_fn(src, seed):
        g = torch.Generator().manual_seed(seed)
        def fn(step):
            ix = torch.randint(0, src.numel() - a.seq_len - 1,
                               (a.batch_size,), generator=g)
            return torch.stack([src[i:i + a.seq_len] for i in ix])
        return fn
    train_fn = make_fn(tr, 0)
    _vf = make_fn(val, 7)
    eval_set = [_vf(i) for i in range(a.eval_batches)]

    model = Absorbed(spec, core).to(dev)
    print("\n=== at init (warm start only, no training) ===")
    p0 = compare(spec, model, eval_set, dev)
    for k, v in p0.items():
        print(f"  {k:18s} {v:.4f}")

    h1 = fit(spec, core, train_fn, a.steps, a.lr, dev, a.mix_max, a.mix_warmup)
    print("\n=== after fitting ===")
    p1 = compare(spec, model, eval_set, dev)
    for k, v in p1.items():
        print(f"  {k:18s} {v:.4f}")

    h2 = distil(spec, model, train_fn, a.distil_steps, a.distil_lr, dev) \
        if a.distil_steps else []
    print("\n=== final ===")
    res = compare(spec, model, eval_set, dev)
    for k, v in res.items():
        print(f"  {k:18s} {v:.4f}")
    print(f"\n{a.model}: {spec.n_layers} -> {a.core_blocks} shared blocks, "
          f"perplexity retained {res['ppl_retained']:.1%} "
          f"(dense {res['dense_ppl']:.2f} -> absorbed {res['absorbed_ppl']:.2f})")

    (out / f"{a.model}.json").write_text(json.dumps(
        {"model": a.model, "at_init": p0, "after_fit": p1, "final": res,
         "fit": h1, "distil": h2, "args": a.__dict__}, indent=2),
        encoding="utf-8")
    print(f"wrote {out}/{a.model}.json")


if __name__ == "__main__":
    main()
