"""Collapse a trained dense stack into a looped core.

Why this is the road to a flagship-class local model
----------------------------------------------------
The architecture here makes *inference* cheap. It does nothing for *training*:
matching a frontier model from scratch costs millions of dollars of compute
whatever shape you pick. So the realistic route is not to train the knowledge,
it is to **move knowledge that already exists into this shape**.

The cheap way to do that is not logit distillation. A dense stack hands you a
far richer signal for free: every layer's *residual update*. Layer `l`
computes `h_{l+1} = h_l + f_l(h_l)`, and all of those pairs are supervision.
Fitting a shared core to reproduce them gives `N` training targets per forward
pass instead of one, and it is plain regression rather than language
modelling -- orders of magnitude cheaper than distillation, and it never needs
the original training data, only text to push through.

Concretely, with `k` core blocks and `N` dense layers we ask iteration `i` of
the loop to reproduce what dense layers `ik .. ik+k-1` did together:

    core( h_{ik}, iteration=i )  ~=  h_{(i+1)k}

Iteration conditioning (AdaLN on the loop index) is what lets one set of
weights behave like `N/k` different layer groups. That is exactly the
mechanism the looped core already has.

What this experiment is really asking
-------------------------------------
Can a small *shared* core represent what `N` independent layers compute? Loops
reuse weights, so capacity is genuinely lower, and collapse may simply lose
too much. That is the single question standing between this project and a
credible path to a local flagship, and it is answerable in an afternoon
instead of a quarter.

Measured, not assumed: fidelity per iteration, end-to-end logit agreement, and
the actual task metric (perplexity or exact match) of the collapsed model
against the dense original.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from xios.config import XiosConfig, get_config
from xios.baseline import BaselineTransformer
from xios.core.block import XiosBlock
from xios.core.rcc import iteration_embedding
from xios.core.norm import RMSNorm


# ---------------------------------------------------------------------------
class LoopedCore(nn.Module):
    """`k` shared blocks, applied `n_iters` times with per-iteration modulation."""

    def __init__(self, cfg: XiosConfig, n_iters: int):
        super().__init__()
        self.cfg = cfg
        self.n_iters = n_iters
        self.blocks = nn.ModuleList([XiosBlock(cfg, i) for i in range(cfg.core_blocks)])
        self.iter_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim * 2), nn.SiLU(),
            nn.Linear(cfg.cond_dim * 2, cfg.cond_dim))
        self.apply(self._init)
        # depth-scale the residual writes for the *effective* depth
        scale = (2 * cfg.core_blocks * n_iters) ** -0.5
        for n, p in self.named_parameters():
            if n.endswith(("mixer.out.weight", "mixer.o.weight", "ffn.w_out.weight")):
                with torch.no_grad():
                    p.mul_(scale)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def cond(self, i: int, device, dtype, B: int):
        e = iteration_embedding(i, self.cfg.cond_dim, device).to(dtype)
        return self.iter_mlp(e)[None].expand(B, -1)

    def step_group(self, h: torch.Tensor, i: int, pos=None) -> torch.Tensor:
        """One iteration: the update that should match `k` dense layers."""
        c = self.cond(i, h.device, h.dtype, h.shape[0])
        for blk in self.blocks:
            h, _, _ = blk(h, c, None, pos=pos)
        return h

    def forward(self, h: torch.Tensor, pos=None) -> torch.Tensor:
        for i in range(self.n_iters):
            h = self.step_group(h, i, pos=pos)
        return h


# ---------------------------------------------------------------------------
class CollapsedModel(nn.Module):
    """embed (copied) -> looped core (trained) -> norm+head (copied)."""

    def __init__(self, dense: BaselineTransformer, core: LoopedCore):
        super().__init__()
        self.embed = dense.embed
        self.core = core
        self.norm_f = dense.norm_f
        self.lm_head = dense.lm_head
        for p in list(self.embed.parameters()) + list(self.norm_f.parameters()):
            p.requires_grad_(False)

    def forward(self, input_ids, labels=None, **kw):
        x = self.embed(input_ids)
        x = self.core(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        from xios.model import XiosOutput
        return XiosOutput(logits=logits, loss=loss, lm_loss=loss,
                          budget_loss=torch.zeros((), device=logits.device))


# ---------------------------------------------------------------------------
@torch.no_grad()
def dense_waypoints(dense: BaselineTransformer, ids: torch.Tensor, k: int):
    """Hidden states at every k-th layer boundary: h_0, h_k, h_2k, ... h_N."""
    x = dense.embed(ids)
    pts = [x]
    for li, blk in enumerate(dense.blocks):
        x, _ = blk(x)
        if (li + 1) % k == 0:
            pts.append(x)
    return pts


def collapse(dense: BaselineTransformer, core_blocks: int, data_fn,
             steps: int = 1500, lr: float = 1e-3, device: str = "cpu",
             log_every: int = 100, seq_len: int = 128) -> tuple:
    """Fit a looped core to the dense stack's per-group residual updates."""
    N = dense.n_layers
    assert N % core_blocks == 0, f"{N} layers must divide by {core_blocks} blocks"
    n_iters = N // core_blocks

    cfg = get_config("nano", dim=dense.dim, head_dim=dense.blocks[0].head_dim,
                     n_kv_heads=dense.blocks[0].n_kv_heads,
                     vocab_size=dense.vocab_size, core_blocks=core_blocks,
                     max_seq_len=max(seq_len, 256),
                     attn_window=min(256, seq_len), cond_dim=256)
    core = LoopedCore(cfg, n_iters).to(device)
    dense = dense.to(device).eval()

    opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],
                            lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    hist = []
    t0 = time.time()
    print(f"collapsing {N} dense layers -> {core_blocks} shared blocks x "
          f"{n_iters} iterations  ({N/core_blocks:.0f}x fewer stored blocks)")
    print(f"core params {sum(p.numel() for p in core.parameters())/1e6:.2f}M "
          f"vs dense body {sum(p.numel() for p in dense.blocks.parameters())/1e6:.2f}M")

    for step in range(steps):
        lr_now = lr * min(1.0, (step + 1) / 100) * (
            0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now

        ids = data_fn(step).to(device)
        pts = dense_waypoints(dense, ids, core_blocks)

        # Teacher forcing on hidden states: every group is an independent
        # regression target, so one forward pass yields n_iters targets.
        loss = 0.0
        per_iter = []
        for i in range(n_iters):
            pred = core.step_group(pts[i].detach(), i)
            tgt = pts[i + 1].detach()
            num = (pred - tgt).pow(2).mean()
            den = tgt.pow(2).mean().clamp(min=1e-8)
            loss = loss + num / den                   # scale-free per group
            per_iter.append(float((num / den).detach()))
        loss = loss / n_iters

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == steps - 1:
            rec = {"step": step, "rel_mse": float(loss), "lr": lr_now,
                   "grad_norm": float(gn), "per_iter": per_iter,
                   "elapsed": time.time() - t0}
            hist.append(rec)
            print(f"[collapse] {step:5d}  rel_mse {float(loss):.4f}  "
                  f"worst group {max(per_iter):.4f}  lr {lr_now:.2e}  "
                  f"gn {float(gn):.2f}", flush=True)

    return core, cfg, hist


# ---------------------------------------------------------------------------
def finetune_end_to_end(dense: BaselineTransformer, collapsed: CollapsedModel,
                        data_fn, steps: int = 400, lr: float = 3e-4,
                        device: str = "cpu", log_every: int = 100,
                        temperature: float = 1.0) -> list:
    """Phase 2: distil the dense model's own distribution, end to end.

    Phase 1 fits each group of layers in isolation, with the *true* dense
    hidden state as input. At run time the loop feeds itself, so every group
    sees its predecessor's error and the mistakes compound -- classic exposure
    bias. Fitting groups perfectly in isolation does not imply the composition
    works.

    So phase 2 runs the whole loop, no teacher forcing, and matches the dense
    model's output distribution (KL, not hard labels -- the teacher's
    uncertainty is most of the signal). Starting from the phase-1 solution
    makes this cheap: it is a correction, not a search.
    """
    dense = dense.to(device).eval()
    collapsed = collapsed.to(device).train()
    params = [p for p in collapsed.core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    hist = []
    t0 = time.time()
    print(f"\nphase 2: end-to-end distillation, {steps} steps")
    for step in range(steps):
        lr_now = lr * min(1.0, (step + 1) / 50) * (
            0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))) * 0.9 + 0.1)
        for g in opt.param_groups:
            g["lr"] = lr_now
        ids = data_fn(step).to(device)
        with torch.no_grad():
            ref = dense(ids).logits / temperature
            rp = F.log_softmax(ref.float(), -1)
        got = collapsed(ids).logits / temperature
        gp = F.log_softmax(got.float(), -1)
        loss = F.kl_div(gp, rp, log_target=True, reduction="batchmean") / ids.shape[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "kl": float(loss), "lr": lr_now})
            print(f"[finetune] {step:5d}  kl {float(loss):.4f}  "
                  f"lr {lr_now:.2e}  gn {float(gn):.2f}", flush=True)
    collapsed.eval()
    return hist


# ---------------------------------------------------------------------------
@torch.no_grad()
def compare(dense: BaselineTransformer, collapsed: CollapsedModel,
            data_fn, n_batches: int = 20, device: str = "cpu") -> dict:
    dense.eval(); collapsed.eval()
    agree = kl = dl = cl = 0.0
    for b in range(n_batches):
        ids = data_fn(10_000 + b).to(device)
        ref = dense(ids, labels=ids)
        got = collapsed(ids, labels=ids)
        rp = F.log_softmax(ref.logits.float(), -1)
        gp = F.log_softmax(got.logits.float(), -1)
        agree += float((ref.logits.argmax(-1) == got.logits.argmax(-1)).float().mean())
        kl += float((rp.exp() * (rp - gp)).sum(-1).mean())
        dl += float(ref.lm_loss); cl += float(got.lm_loss)
    n = n_batches
    return {"top1_agreement": agree / n, "kl": kl / n,
            "dense_loss": dl / n, "collapsed_loss": cl / n,
            "dense_ppl": math.exp(dl / n), "collapsed_ppl": math.exp(cl / n)}


# ---------------------------------------------------------------------------
def load_dense(path: str) -> BaselineTransformer:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = BaselineTransformer(
        dim=blob["dim"], n_layers=blob["n_layers"], head_dim=blob["head_dim"],
        n_kv_heads=blob["n_kv_heads"], vocab_size=blob["vocab_size"],
        ffn_mult=blob["ffn_mult"], max_seq_len=blob["max_seq_len"])
    m.load_state_dict(blob["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="runs/demo/baseline.pt")
    ap.add_argument("--tokenizer", default="runs/demo/tokenizer.json")
    ap.add_argument("--corpus", default="shakespeare")
    ap.add_argument("--core-blocks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume-core", default=None,
                    help="skip phase 1 and load a core.pt (saves re-fitting)")
    ap.add_argument("--finetune-steps", type=int, default=400,
                    help="phase 2 end-to-end distillation steps (0 to skip)")
    ap.add_argument("--finetune-lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/collapse")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    dense = load_dense(a.dense)
    print(f"dense: {dense.n_layers} layers, dim {dense.dim}, "
          f"{dense.n_params/1e6:.2f}M params")

    from xios.tokenizer import ByteBPE
    from train.train_text import load_texts, build_stream
    tok = ByteBPE.load(a.tokenizer)
    stream = build_stream(load_texts(a.corpus, 20000), tok)
    n_val = min(100_000, stream.numel() // 10)
    val, tr = stream[:n_val], stream[n_val:]
    print(f"corpus {stream.numel()/1e6:.2f}M tokens, vocab {tok.actual_size}")

    def make_fn(src, seed):
        g = torch.Generator().manual_seed(seed)
        def fn(step):
            ix = torch.randint(0, src.numel() - a.seq_len - 1,
                               (a.batch_size,), generator=g)
            return torch.stack([src[i:i + a.seq_len] for i in ix])
        return fn

    train_fn, val_fn = make_fn(tr, 0), make_fn(val, 7)

    if a.resume_core:
        blob = torch.load(a.resume_core, map_location="cpu", weights_only=False)
        cfg = XiosConfig(**blob["config"])
        core = LoopedCore(cfg, blob["n_iters"])
        core.load_state_dict(blob["core"])
        hist = []
        print(f"loaded phase-1 core from {a.resume_core} "
              f"({blob['n_iters']} iterations)")
    else:
        core, cfg, hist = collapse(dense, a.core_blocks, train_fn, steps=a.steps,
                                   lr=a.lr, device=dev, seq_len=a.seq_len)
    collapsed = CollapsedModel(dense, core).to(dev)

    print("\n=== after phase 1 (per-group regression, teacher forced) ===")
    p1 = compare(dense, collapsed, val_fn, n_batches=10, device=dev)
    for k, v in p1.items():
        print(f"  {k:18s} {v:.4f}")
    print(f"  perplexity retained {p1['dense_ppl']/p1['collapsed_ppl']:.1%}")

    ft = []
    if a.finetune_steps:
        ft = finetune_end_to_end(dense, collapsed, train_fn,
                                 steps=a.finetune_steps, lr=a.finetune_lr,
                                 device=dev)

    print("\n=== final (after end-to-end distillation) ===")
    res = compare(dense, collapsed, val_fn, n_batches=20, device=dev)
    for k, v in res.items():
        print(f"  {k:18s} {v:.4f}")
    keep = res["dense_ppl"] / res["collapsed_ppl"]
    print(f"\nperplexity retained: {keep:.1%}  "
          f"(dense {res['dense_ppl']:.2f} -> collapsed {res['collapsed_ppl']:.2f})")
    dense_body = sum(p.numel() for p in dense.blocks.parameters())
    core_body = sum(p.numel() for p in core.parameters())
    print(f"stored blocks: {dense.n_layers} -> {a.core_blocks} "
          f"({dense.n_layers/a.core_blocks:.0f}x fewer); "
          f"body params {dense_body/1e6:.2f}M -> {core_body/1e6:.2f}M "
          f"({dense_body/max(core_body,1):.2f}x)")

    torch.save({"core": core.state_dict(), "config": cfg.__dict__,
                "n_iters": dense.n_layers // a.core_blocks}, out / "core.pt")
    (out / "results.json").write_text(json.dumps(
        {"compare": res, "phase1": p1, "history": hist, "finetune": ft,
         "args": a.__dict__}, indent=2), encoding="utf-8")
    print(f"\nwrote {out/'results.json'}")


if __name__ == "__main__":
    main()
