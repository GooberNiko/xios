"""Absorb a pretrained model into a looped core — second attempt.

The first attempt (`train/collapse_hf.py`) retained only 55-69% of GPT-2's
perplexity, against 101% on a toy teacher. The diagnosis was specific rather
than vague, which is why a second attempt is worth making:

* Phase 1 fit GPT-2's residual updates to **1.7% relative error**. Fitting the
  layers is not the problem.
* The damage appears when the loop runs on its *own* output. Phase 1 always
  feeds the true dense hidden state, so the core is never trained on the
  slightly-wrong inputs it will actually see. Errors compound over iterations.
* Phase 2 (end-to-end) fixes exactly that, and its loss was still falling
  steeply when compute ran out.

So this version changes the algorithm in three places, each aimed at a
specific part of that diagnosis.

**1. Scheduled self-feeding (the main fix).** Instead of always feeding the
true waypoint, feed the core's *own* previous output with probability `p`,
ramped from 0 upward, while still targeting the true next waypoint. That asks
the right question — "given where you actually ended up, produce where you
should be" — and trains the core to correct its own drift. It is DAgger applied
to layer collapse, and unlike full end-to-end training it keeps the per-group
target, so one forward pass still yields `n_iters` supervised signals.

**2. Per-iteration low-rank adapters.** A shared core conditioned only by
AdaLN can rescale and shift its behaviour per iteration, but it cannot change
the *direction* of the transformation. GPT-2's layers genuinely compute
different functions, which is precisely why the toy result did not transfer.
A rank-`r` additive term per iteration (`h + h A_i B_i`, `B_i` zero-init)
costs a fraction of a percent of the weights and lets each iteration
specialise. This is the smallest amount of unshared capacity that addresses
the measured failure, rather than giving up on sharing.

**3. Deeper supervision on the final group.** The last group's output feeds
the head directly, so its error is the one that reaches the logits unattenuated.
It gets extra weight.

Everything remains measured against the dense original: perplexity retained,
top-1 agreement, KL.
"""
from __future__ import annotations

import argparse
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

from xios.config import XiosConfig, get_config
from xios.adapters import DenseSpec, from_huggingface
from xios.core.block import XiosBlock
from xios.core.rcc import iteration_embedding


# ---------------------------------------------------------------------------
class AdaptedCore(nn.Module):
    """Shared blocks + per-iteration AdaLN conditioning + per-iteration LoRA."""

    def __init__(self, cfg: XiosConfig, n_iters: int, adapter_rank: int = 16):
        super().__init__()
        self.cfg, self.n_iters = cfg, n_iters
        self.blocks = nn.ModuleList(
            [XiosBlock(cfg, i) for i in range(cfg.core_blocks)])
        self.iter_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim * 2), nn.SiLU(),
            nn.Linear(cfg.cond_dim * 2, cfg.cond_dim))

        self.adapter_rank = adapter_rank
        if adapter_rank:
            d = cfg.dim
            self.lora_a = nn.Parameter(torch.randn(n_iters, d, adapter_rank) * 0.02)
            self.lora_b = nn.Parameter(torch.zeros(n_iters, adapter_rank, d))

        self.apply(self._init)
        scale = (2 * cfg.core_blocks * n_iters) ** -0.5
        for n, p in self.named_parameters():
            if n.endswith(("mixer.out.weight", "mixer.o.weight", "ffn.w_out.weight")):
                with torch.no_grad():
                    p.mul_(scale)
        if adapter_rank:                      # identity at init
            with torch.no_grad():
                self.lora_b.zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def step_group(self, h: torch.Tensor, i: int) -> torch.Tensor:
        c = self.iter_mlp(
            iteration_embedding(i, self.cfg.cond_dim, h.device).to(h.dtype)
        )[None].expand(h.shape[0], -1)
        for blk in self.blocks:
            h, _, _ = blk(h, c, None)
        if self.adapter_rank:
            h = h + (h @ self.lora_a[i]) @ self.lora_b[i]
        return h

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_iters):
            h = self.step_group(h, i)
        return h


class Absorbed(nn.Module):
    def __init__(self, spec: DenseSpec, core: AdaptedCore):
        super().__init__()
        self.spec, self.core = spec, core

    def forward(self, ids, labels=None):
        h = self.core(self.spec.embed(ids))
        logits = self.spec.head(self.spec.final_norm(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss


# ---------------------------------------------------------------------------
def fit(spec: DenseSpec, core_blocks: int, data_fn, steps: int, lr: float,
        device: str, seq_len: int, adapter_rank: int, mix_max: float,
        mix_warmup: float, log_every: int = 50):
    N = spec.n_layers
    assert N % core_blocks == 0
    n_iters = N // core_blocks

    head_dim = 64
    while spec.dim % head_dim or (spec.dim // head_dim) % 2:
        head_dim //= 2
    cfg = get_config("nano", dim=spec.dim, head_dim=head_dim,
                     n_kv_heads=max(2, (spec.dim // head_dim) // 4),
                     vocab_size=spec.vocab_size, core_blocks=core_blocks,
                     max_seq_len=max(seq_len, 256),
                     attn_window=min(256, seq_len), cond_dim=256)
    core = AdaptedCore(cfg, n_iters, adapter_rank).to(device)
    opt = torch.optim.AdamW(core.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    n_core = sum(p.numel() for p in core.parameters())
    n_ad = (core.lora_a.numel() + core.lora_b.numel()) if adapter_rank else 0
    print(f"absorbing {spec.name}: {N} layers -> {core_blocks} blocks x "
          f"{n_iters} iters   adapters rank {adapter_rank} "
          f"({n_ad/1e6:.2f}M, {100*n_ad/max(n_core,1):.1f}% of core)")
    print(f"core {n_core/1e6:.1f}M vs dense body {spec.body_params/1e6:.1f}M "
          f"({spec.body_params/max(n_core,1):.2f}x fewer)")

    rng = random.Random(0)
    hist, t0 = [], time.time()
    for step in range(steps):
        frac = step / max(steps, 1)
        lr_now = lr * min(1.0, (step + 1) / 50) * (
            0.5 * (1 + math.cos(math.pi * min(frac, 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now
        # ramp self-feeding: start from pure teacher forcing, end mostly on
        # the core's own (imperfect) inputs
        p_self = mix_max * min(1.0, frac / max(mix_warmup, 1e-6))

        ids = data_fn(step).to(device)
        pts = spec.waypoints(ids, core_blocks)

        loss, per_iter, n_self = 0.0, [], 0
        h_self = pts[0].detach()
        for i in range(n_iters):
            use_self = i > 0 and rng.random() < p_self
            inp = h_self if use_self else pts[i].detach()
            n_self += int(use_self)
            pred = core.step_group(inp, i)
            tgt = pts[i + 1].detach()
            r = (pred - tgt).pow(2).mean() / tgt.pow(2).mean().clamp(min=1e-8)
            w = 2.0 if i == n_iters - 1 else 1.0      # last group feeds the head
            loss = loss + w * r
            per_iter.append(float(r.detach()))
            h_self = pred.detach()
        loss = loss / (n_iters + 1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "rel_mse": float(loss),
                         "per_iter": per_iter, "p_self": p_self})
            print(f"[fit] {step:5d}  rel_mse {float(loss):.4f}  "
                  f"worst {max(per_iter):.4f}  self-fed {n_self}/{n_iters-1}  "
                  f"p {p_self:.2f}  lr {lr_now:.2e}  {time.time()-t0:.0f}s",
                  flush=True)
    return core, cfg, hist


def distil(spec, model: Absorbed, data_fn, steps, lr, device, log_every=50):
    opt = torch.optim.AdamW(model.core.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    hist, t0 = [], time.time()
    print(f"\ndistillation: {steps} steps end-to-end")
    for step in range(steps):
        lr_now = lr * min(1.0, (step + 1) / 30) * (
            0.5 * (1 + math.cos(math.pi * min(step / max(steps, 1), 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now
        ids = data_fn(step).to(device)
        with torch.no_grad():
            rp = F.log_softmax(spec.forward(ids).float(), -1)
        gp = F.log_softmax(model(ids)[0].float(), -1)
        loss = F.kl_div(gp, rp, log_target=True, reduction="batchmean") / ids.shape[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.core.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "kl": float(loss)})
            print(f"[distil] {step:5d}  kl {float(loss):.4f}  lr {lr_now:.2e}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    return hist


@torch.no_grad()
def compare(spec, model, batches, device):
    agree = kl = dl = cl = 0.0
    # Fixed batches, regenerated here rather than drawn from the shared
    # generator: otherwise every compare() call sees different text and the
    # teacher's own perplexity moves between runs, which makes retention
    # ratios look comparable when they are not.
    for ids in batches:
        ids = ids.to(device)
        ref = spec.forward(ids)
        got, _ = model(ids)
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
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--distil-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--distil-lr", type=float, default=3e-4)
    ap.add_argument("--adapter-rank", type=int, default=16)
    ap.add_argument("--mix-max", type=float, default=0.7,
                    help="max probability of feeding the core its own output")
    ap.add_argument("--mix-warmup", type=float, default=0.4,
                    help="fraction of training over which self-feeding ramps")
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--corpus", default="data/prose.txt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/absorb")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    spec = from_huggingface(hf, a.model)
    print(f"{a.model}: {spec.n_layers} layers, dim {spec.dim}")

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
    # One fixed evaluation set, drawn once. Reusing a live generator meant
    # every compare() call saw different text, so the teacher's own perplexity
    # moved between calls and retention ratios were not comparable across runs.
    _vf = make_fn(val, 7)
    eval_set = [_vf(i) for i in range(a.eval_batches)]

    core, cfg, h1 = fit(spec, a.core_blocks, train_fn, a.steps, a.lr, dev,
                        a.seq_len, a.adapter_rank, a.mix_max, a.mix_warmup)
    model = Absorbed(spec, core).to(dev)

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
    print(f"\n{a.model}: {spec.n_layers} layers -> {a.core_blocks} blocks, "
          f"perplexity retained {res['ppl_retained']:.1%} "
          f"(dense {res['dense_ppl']:.2f} -> absorbed {res['absorbed_ppl']:.2f})")

    torch.save({"core": core.state_dict(), "config": cfg.__dict__,
                "n_iters": spec.n_layers // a.core_blocks,
                "adapter_rank": a.adapter_rank}, out / f"{a.model}_core.pt")
    (out / f"{a.model}.json").write_text(json.dumps(
        {"model": a.model, "after_fit": p1, "final": res, "fit": h1,
         "distil": h2, "args": a.__dict__}, indent=2), encoding="utf-8")
    print(f"wrote {out}/{a.model}.json")


if __name__ == "__main__":
    main()
