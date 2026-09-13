"""Training loop, shared by XIOS and the baseline so neither gets an
accidental advantage from the optimiser.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass, asdict, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn


@dataclass
class TrainArgs:
    steps: int = 2000
    batch_size: int = 32
    seq_len: int = 128
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    log_every: int = 50
    eval_every: int = 500
    amp: bool = True
    compile: bool = False
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs/default"
    grad_checkpoint: bool = False
    depth_warmup_frac: float = 0.3


def lr_at(step: int, a: TrainArgs) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * cos)


def build_optimizer(model: nn.Module, a: TrainArgs):
    decay, no_decay, sparse = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "values.weight" in n:            # associative memory -> sparse grads
            sparse.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": a.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=a.betas, eps=1e-8)
    sopt = torch.optim.SparseAdam(sparse, lr=a.lr) if sparse else None
    return opt, sopt


def train(model, data_fn, args: TrainArgs, eval_fn=None, tag: str = "model"):
    """``data_fn(step) -> (ids, labels, mask)``; ``eval_fn(model) -> dict``."""
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    model = model.to(dev)
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing(True)

    opt, sopt = build_optimizer(model, args)
    use_amp = args.amp and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()
    running = None

    for step in range(args.steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        if sopt:
            for g in sopt.param_groups:
                g["lr"] = lr

        # budget curriculum: let the loop learn to be useful before it is
        # told to be cheap
        ponder = getattr(getattr(model, "core", None), "ponder", None)
        if ponder is not None:
            ponder.anneal(step / max(1, args.steps), args.depth_warmup_frac)

        ids, labels, mask = data_fn(step)
        ids, labels = ids.to(dev), labels.to(dev)
        mask = mask.to(dev) if mask is not None else None

        model.train()
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
            out = model(ids, labels=labels, loss_mask=mask)
            loss = out.loss

        opt.zero_grad(set_to_none=True)
        if sopt:
            sopt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        if sopt:
            sopt.step()
        scaler.update()
        if ponder is not None:
            ponder.dual_step()      # price the depth constraint

        lm = float(out.lm_loss)
        running = lm if running is None else 0.98 * running + 0.02 * lm

        if step % args.log_every == 0 or step == args.steps - 1:
            rec = {"step": step, "lm_loss": lm, "ema": running, "lr": lr,
                   "grad_norm": float(gn), "elapsed": time.time() - t0}
            st = getattr(out, "stats", None)
            if st is not None and getattr(st, "actual_depth", None) is not None:
                rec["mean_depth"] = st.mean_depth
                rec["scored_depth"] = st.scored_depth
                rec["depth_p99"] = st.depth_p99
                rec["budget_loss"] = float(out.budget_loss)
                if ponder is not None:
                    rec["depth_target"] = ponder.target_depth
                    rec["lambda"] = float(ponder.lam)
            history.append(rec)
            extra = ""
            if "mean_depth" in rec:
                extra = (f"  depth all {rec['mean_depth']:.2f}"
                         f" / scored {rec['scored_depth']:.2f}"
                         f" (tgt {rec.get('depth_target', 0):.1f}"
                         f" p99 {rec['depth_p99']:.0f}"
                         f" lam {rec.get('lambda', 0):.3f})")
            print(f"[{tag}] {step:6d}  loss {lm:.4f}  ema {running:.4f}  "
                  f"lr {lr:.2e}  gn {float(gn):.2f}{extra}", flush=True)

        if eval_fn is not None and args.eval_every and \
                (step + 1) % args.eval_every == 0:
            res = eval_fn(model)
            print(f"[{tag}] eval @ {step + 1}: {res}", flush=True)
            history.append({"step": step, "eval": res})

    # Never destroy a checkpoint silently. An orphaned run pointed at a
    # directory whose results had already been reported overwrote its model
    # hours later, and the only reason it was recoverable was an unrelated
    # copy. Keeping one generation back is cheap insurance.
    ckpt = out_dir / f"{tag}.pt"
    if ckpt.exists():
        prev = out_dir / f"{tag}.prev.pt"
        prev.unlink(missing_ok=True)
        ckpt.rename(prev)
        print(f"[{tag}] existing checkpoint moved to {prev.name}", flush=True)
    torch.save({"model": model.state_dict(), "args": asdict(args)}, ckpt)
    (out_dir / f"{tag}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    return history
