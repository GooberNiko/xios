"""Collapse a real pretrained model into a looped core.

    python train/collapse_hf.py --model gpt2        --core-blocks 2
    python train/collapse_hf.py --model gpt2-medium --core-blocks 2

This is the experiment that matters for a local flagship. `train/collapse.py`
showed a 12-layer model trained *in this repo* collapses into 2 shared blocks
with 101.4% of its perplexity retained. The obvious objection is that the
teacher was a toy: 9.4M parameters on a 0.6MB corpus, so of course it was
compressible.

So here the teacher is GPT-2 -- a real model, really pretrained, that actually
knows things -- and nothing about the XIOS core matches it. GPT-2 uses learned
positional embeddings, LayerNorm and a GELU MLP; the core uses RoPE, RMSNorm,
SwiGLU and gated linear recurrences. Collapse never assumes they agree
internally; it fits only the *function* each group of layers computes on the
residual stream. If that works across a mechanism change this large, it is a
general absorption procedure.

It also tests a specific prediction I committed to before running it: deeper
models should collapse *better*, because adjacent layers in deep stacks are
more redundant. GPT-2 small (12 layers) and medium (24 layers) are the same
family at two depths, so the comparison isolates depth.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from xios.config import XiosConfig, get_config
from xios.adapters import DenseSpec, from_huggingface
from train.collapse import LoopedCore


# ---------------------------------------------------------------------------
class CollapsedSpec(nn.Module):
    """The dense model's embed/norm/head, with a looped core in the middle."""

    def __init__(self, spec: DenseSpec, core: LoopedCore):
        super().__init__()
        self.spec = spec
        self.core = core

    def forward(self, ids, labels=None):
        h = self.spec.embed(ids)
        h = self.core(h)
        logits = self.spec.head(self.spec.final_norm(h))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss


# ---------------------------------------------------------------------------
def fit_groups(spec: DenseSpec, core_blocks: int, data_fn, steps: int,
               lr: float, device: str, seq_len: int, log_every: int = 50):
    """Phase 1: per-group residual regression (teacher forced)."""
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
    core = LoopedCore(cfg, n_iters).to(device)
    opt = torch.optim.AdamW(core.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    print(f"collapsing {spec.name}: {N} layers -> {core_blocks} shared blocks "
          f"x {n_iters} iterations ({N/core_blocks:.0f}x fewer stored blocks)")
    print(f"core {sum(p.numel() for p in core.parameters())/1e6:.1f}M params "
          f"vs dense body {spec.body_params/1e6:.1f}M "
          f"({spec.body_params/max(sum(p.numel() for p in core.parameters()),1):.2f}x)")

    hist, t0 = [], time.time()
    for step in range(steps):
        lr_now = lr * min(1.0, (step + 1) / 50) * (
            0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now
        ids = data_fn(step).to(device)
        pts = spec.waypoints(ids, core_blocks)

        loss, per_iter = 0.0, []
        for i in range(n_iters):
            pred = core.step_group(pts[i].detach(), i)
            tgt = pts[i + 1].detach()
            r = (pred - tgt).pow(2).mean() / tgt.pow(2).mean().clamp(min=1e-8)
            loss = loss + r
            per_iter.append(float(r.detach()))
        loss = loss / n_iters

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "rel_mse": float(loss),
                         "per_iter": per_iter, "elapsed": time.time() - t0})
            print(f"[phase1] {step:5d}  rel_mse {float(loss):.4f}  "
                  f"worst {max(per_iter):.4f}  lr {lr_now:.2e} "
                  f"gn {float(gn):.2f}  {time.time()-t0:.0f}s", flush=True)
    return core, cfg, hist


def distil(spec: DenseSpec, collapsed: CollapsedSpec, data_fn, steps: int,
           lr: float, device: str, log_every: int = 50):
    """Phase 2: end-to-end KL to the dense model, no teacher forcing."""
    opt = torch.optim.AdamW(collapsed.core.parameters(), lr=lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    hist, t0 = [], time.time()
    print(f"\nphase 2: end-to-end distillation, {steps} steps")
    for step in range(steps):
        lr_now = lr * min(1.0, (step + 1) / 30) * (
            0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now
        ids = data_fn(step).to(device)
        with torch.no_grad():
            rp = F.log_softmax(spec.forward(ids).float(), -1)
        gp = F.log_softmax(collapsed(ids)[0].float(), -1)
        loss = F.kl_div(gp, rp, log_target=True, reduction="batchmean") / ids.shape[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(collapsed.core.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "kl": float(loss)})
            print(f"[phase2] {step:5d}  kl {float(loss):.4f}  lr {lr_now:.2e} "
                  f"gn {float(gn):.2f}  {time.time()-t0:.0f}s", flush=True)
    return hist


@torch.no_grad()
def compare(spec: DenseSpec, collapsed: CollapsedSpec, batches, device):
    agree = kl = dl = cl = 0.0
    # Fixed batches, regenerated here rather than drawn from the shared
    # generator: otherwise every compare() call sees different text and the
    # teacher's own perplexity moves between runs, which makes retention
    # ratios look comparable when they are not.
    for ids in batches:
        ids = ids.to(device)
        ref = spec.forward(ids)
        got, _ = collapsed(ids)
        rl = F.cross_entropy(ref[:, :-1].reshape(-1, ref.size(-1)).float(),
                             ids[:, 1:].reshape(-1))
        gl = F.cross_entropy(got[:, :-1].reshape(-1, got.size(-1)).float(),
                             ids[:, 1:].reshape(-1))
        rp = F.log_softmax(ref.float(), -1); gp = F.log_softmax(got.float(), -1)
        agree += float((ref.argmax(-1) == got.argmax(-1)).float().mean())
        kl += float((rp.exp() * (rp - gp)).sum(-1).mean())
        dl += float(rl); cl += float(gl)
    n = len(batches)
    return {"top1_agreement": agree / n, "kl": kl / n,
            "dense_loss": dl / n, "collapsed_loss": cl / n,
            "dense_ppl": math.exp(dl / n), "collapsed_ppl": math.exp(cl / n),
            "ppl_retained": math.exp(dl / n) / math.exp(cl / n)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--core-blocks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--distil-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--distil-lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--corpus", default="data/prose.txt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/collapse_hf")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    spec = from_huggingface(model, a.model)
    print(f"{a.model}: {spec.n_layers} layers, dim {spec.dim}, "
          f"vocab {spec.vocab_size}")

    text = pathlib.Path(a.corpus).read_text(encoding="utf-8", errors="replace")
    ids_all = torch.tensor(tok(text).input_ids, dtype=torch.long)
    n_val = min(40_000, ids_all.numel() // 10)
    val, tr = ids_all[:n_val], ids_all[n_val:]
    print(f"corpus {ids_all.numel()/1e3:.0f}k tokens "
          f"({tr.numel()/1e3:.0f}k train / {val.numel()/1e3:.0f}k val)")

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

    core, cfg, h1 = fit_groups(spec, a.core_blocks, train_fn, a.steps, a.lr,
                               dev, a.seq_len)
    collapsed = CollapsedSpec(spec, core).to(dev)

    print("\n=== after phase 1 ===")
    p1 = compare(spec, collapsed, eval_set, dev)
    for k, v in p1.items():
        print(f"  {k:18s} {v:.4f}")

    h2 = distil(spec, collapsed, train_fn, a.distil_steps, a.distil_lr, dev) \
        if a.distil_steps else []

    print("\n=== final ===")
    res = compare(spec, collapsed, eval_set, dev)
    for k, v in res.items():
        print(f"  {k:18s} {v:.4f}")
    print(f"\n{a.model}: {spec.n_layers} layers -> {a.core_blocks} blocks, "
          f"perplexity retained {res['ppl_retained']:.1%} "
          f"(dense {res['dense_ppl']:.2f} -> collapsed {res['collapsed_ppl']:.2f})")

    (out / f"{a.model.replace('/', '_')}.json").write_text(json.dumps(
        {"model": a.model, "n_layers": spec.n_layers,
         "core_blocks": a.core_blocks, "phase1": p1, "final": res,
         "h1": h1, "h2": h2, "args": a.__dict__}, indent=2), encoding="utf-8")
    print(f"wrote {out}/{a.model.replace('/', '_')}.json")


if __name__ == "__main__":
    main()
