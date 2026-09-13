"""The A/B that makes XIOS falsifiable.

Two models, identical parameter count, identical data, identical optimiser,
identical seed. One is a standard transformer with all its depth baked in;
the other is XIOS, whose depth is allocated at runtime.

The decisive measurement is **extrapolation in required sequential steps**.
Both models train on problems needing 1..K steps. Both are then tested on
problems needing more than K. A fixed-depth transformer cannot compose more
operations than its depth allows, so its accuracy should fall off a cliff
just past its depth limit no matter how much it trained. If adaptive depth
is doing what it claims, XIOS should degrade more gracefully, by spending
more iterations on the harder instances -- and its measured mean depth
should *rise* with problem difficulty without anyone telling it to.

That last part is the real test. Nothing in the loss says "think longer on
hard problems". If the depth-vs-difficulty curve slopes up, the controller
discovered that on its own.

Usage:
    python train/ab_experiment.py --task perm --steps 3000 --preset nano
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from xios.config import get_config
from xios.model import XiosChat
from xios.baseline import matched_baseline, matched_flops_baseline
from xios.tasks import TaskDataset, TASK_VOCAB, exact_match, WIDTH_ARG
from xios.flops import baseline_flops_per_token, xios_flops_per_token
from train.train import TrainArgs, train


def build(preset: str, seq_len: int, max_iters: int, target_depth: float,
          match: str = "params", match_depth: float | None = None,
          **overrides):
    cfg = get_config(preset, vocab_size=TASK_VOCAB, max_seq_len=seq_len,
                     max_iters=max_iters, target_depth=target_depth,
                     attn_window=min(256, seq_len),
                     **{k: v for k, v in overrides.items() if v is not None})
    xios = XiosChat(cfg)
    if match == "flops":
        base = matched_flops_baseline(cfg, match_depth or target_depth, seq_len)
    else:
        base = matched_baseline(cfg)
    return cfg, xios, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="perm", choices=["perm", "hop", "mod", "parity"])
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-max-steps", type=int, default=6,
                    help="hardest difficulty seen during training")
    ap.add_argument("--eval-max-steps", type=int, default=14,
                    help="hardest difficulty at test time (extrapolation)")
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--target-depth", type=float, default=4.0)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--out", default="runs/ab")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, choices=["xios", "baseline"])
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load xios.pt / baseline.pt from --out "
                         "and re-run the evaluation (e.g. after a metric fix)")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--core-blocks", type=int, default=None)
    ap.add_argument("--n-kv-heads", type=int, default=None)
    ap.add_argument("--fixed-depth", action="store_true",
                    help="ablate the ponder controller: every token runs exactly "
                         "target_depth iterations, no halting, no budget loss")
    ap.add_argument("--match", default="params", choices=["params", "flops"],
                    help="size the baseline to match XIOS on parameters "
                         "(what XIOS wants) or on FLOPs/token (the harder test)")
    ap.add_argument("--match-depth", type=float, default=None,
                    help="mean depth to assume when matching FLOPs "
                         "(default: target_depth)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--curriculum-frac", type=float, default=0.5,
                    help="fraction of training over which difficulty ramps to max")
    ap.add_argument("--width", type=int, default=None,
                    help="task width (table size / group size / modulus). "
                         "Must be small enough that the model can already do "
                         "ONE step, or the depth curve measures nothing.")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg, xios, base = build(args.preset, args.seq_len, args.max_iters,
                            args.target_depth, dim=args.dim,
                            match=args.match, match_depth=args.match_depth,
                            core_blocks=args.core_blocks,
                            n_kv_heads=args.n_kv_heads,
                            adaptive_depth=not args.fixed_depth)
    print(f"task={args.task}  preset={args.preset}  device={dev}")
    print(f"XIOS     {xios.n_params/1e6:7.2f}M params | stored depth {cfg.stored_blocks}"
          f" | effective depth {cfg.effective_depth_avg:.0f}..{cfg.effective_depth_max}")
    print(f"baseline {base.n_params/1e6:7.2f}M params | depth {base.n_layers}")
    print(f"param ratio {xios.n_params / base.n_params:.3f}\n")

    train_ds = TaskDataset(args.task, 1, args.train_max_steps, args.seq_len,
                           seed=0, width=args.width)
    print(f"task width {train_ds.width} ({WIDTH_ARG.get(args.task, ('none', 0))[0] or 'n/a'}), "
          f"chance accuracy {train_ds.chance:.1%}")

    cap = train_ds.max_valid_steps
    if args.eval_max_steps > cap:
        print(f"NOTE: {args.task} at width {train_ds.width} can only label chains "
              f"up to {cap} steps; clamping --eval-max-steps from "
              f"{args.eval_max_steps} to {cap}. For deeper chains raise --width "
              f"or use --task perm.")
        args.eval_max_steps = cap
    if args.train_max_steps > cap:
        args.train_max_steps = max(1, cap - 1)
        print(f"      and --train-max-steps to {args.train_max_steps}")
    print()

    def data_fn(step):
        # Difficulty curriculum. These tasks are chains of dependent
        # operations: sampling 5-step instances from scratch gives almost no
        # learning signal, because a model that cannot do 2 steps gets every
        # 5-step instance wrong and learns nothing from the gradient. Ramping
        # the ceiling over the first `curriculum_frac` of training is what
        # makes them learnable at all -- without it, both models plateau at
        # the entropy of a random guess.
        if args.curriculum_frac > 0:
            prog = min(1.0, (step / args.steps) / args.curriculum_frac)
            ceiling = 1 + int(prog * (args.train_max_steps - 1) + 1e-9)
        else:
            ceiling = args.train_max_steps
        train_ds.max_steps = ceiling
        ids, labels, mask, _ = train_ds.batch(args.batch_size)
        return ids, labels, mask

    targs = TrainArgs(steps=args.steps, batch_size=args.batch_size,
                      seq_len=args.seq_len, lr=args.lr, device=dev,
                      out_dir=str(out), eval_every=0,
                      log_every=args.log_every)

    results = {}
    for name, model in (("xios", xios), ("baseline", base)):
        if args.only and args.only != name:
            continue
        if args.eval_only:
            ckpt = out / f"{name}.pt"
            if not ckpt.exists():
                print(f"(no {ckpt}; skipping {name})")
                continue
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(sd)
            print(f"loaded {ckpt}")
        else:
            print(f"\n=== training {name} ===")
            train(model, data_fn, targs, tag=name)
        results[name] = model

    # ---------------- evaluation -----------------------------------------
    print("\n=== evaluation: exact match vs required sequential steps ===")
    print(f"(trained on 1..{args.train_max_steps}; "
          f"{args.train_max_steps + 1}..{args.eval_max_steps} is extrapolation)\n")

    table = []
    chance = TaskDataset(args.task, 1, 1, args.seq_len, width=args.width).chance
    print(f"chance accuracy {chance:.1%}\n")
    header = f"{'steps':>6} {'baseline':>10} {'XIOS':>10} {'XIOS depth':>11}   "
    print(header)
    print("-" * len(header))
    for s in range(1, args.eval_max_steps + 1):
        ds = TaskDataset(args.task, s, s, args.seq_len, seed=1234 + s,
                         width=args.width)
        row = {"steps": s}
        for name, model in results.items():
            model.to(dev)
            r = exact_match(model, ds, s, n=args.eval_n, device=dev)
            row[name] = r["exact_match"]
            if "mean_depth" in r:
                row["xios_depth"] = r["mean_depth"]
                row["xios_depth_all"] = r.get("all_token_depth")
        mark = "" if s <= args.train_max_steps else "  <- extrapolation"
        print(f"{s:6d} {row.get('baseline', float('nan')):9.1%} "
              f"{row.get('xios', float('nan')):9.1%} "
              f"{row.get('xios_depth', float('nan')):10.2f}{mark}")
        table.append(row)

    # ---------------- compute accounting ---------------------------------
    print("\n=== compute ===")
    if "baseline" in results:
        fb = baseline_flops_per_token(results["baseline"], args.seq_len)
        print(f"baseline  {fb/1e6:8.2f} MFLOPs/token  (fixed)")
    if "xios" in results:
        depths = [r.get("xios_depth") for r in table if r.get("xios_depth")]
        md = sum(depths) / max(len(depths), 1)
        fx = xios_flops_per_token(cfg, args.seq_len, md)
        print(f"XIOS      {fx/1e6:8.2f} MFLOPs/token  (mean depth {md:.2f})")
        if "baseline" in results:
            print(f"XIOS uses {fx/fb:.2f}x the baseline's compute per token")

    # ---------------- validity checks -------------------------------------
    # Difficulty must be monotone in step count. If it is not, the task has a
    # shortcut and the depth axis is meaningless -- which is exactly how the
    # cyclic-wraparound bug in `hop` was caught.
    for name in results:
        acc = [r[name] for r in table]
        if len(acc) > 3:
            rises = sum(1 for a, b in zip(acc, acc[1:]) if b > a + 0.15)
            if rises >= 2:
                print(f"\nWARNING: {name} accuracy rises with difficulty "
                      f"{rises} times ({[f'{a:.0%}' for a in acc]}). A harder "
                      f"instance should not be easier -- this task probably has "
                      f"a shortcut, and any depth conclusion drawn from it is "
                      f"invalid.")

    one = [r for r in table if r["steps"] == 1]
    if one:
        worst = min(one[0].get(k, 0) for k in results)
        if worst < 0.5:
            print(f"\nWARNING: neither model reliably solves the ONE-step case "
                  f"({worst:.1%}). This experiment says nothing about depth -- "
                  f"lower --width or train longer before reading the table above.")

    # ---------------- verdict --------------------------------------------
    if "xios" in results and "baseline" in results:
        ext = [r for r in table if r["steps"] > args.train_max_steps]
        if ext:
            bx = sum(r["baseline"] for r in ext) / len(ext)
            xx = sum(r["xios"] for r in ext) / len(ext)
            print(f"\nextrapolation mean EM:  baseline {bx:.1%}   XIOS {xx:.1%}")
            ds_ = [r.get("xios_depth") for r in table if r.get("xios_depth")]
            if len(ds_) > 2:
                import statistics
                xs = [r["steps"] for r in table if r.get("xios_depth")]
                slope = statistics.correlation(xs, ds_) if len(set(ds_)) > 1 else 0.0
                print(f"depth-vs-difficulty correlation: {slope:+.3f}  "
                      f"({'allocating more compute to harder problems' if slope > 0.3 else 'NOT allocating adaptively'})")

    # Preserve the config that TRAINED these checkpoints. An --eval-only run
    # has its own argparse defaults, and overwriting the record with them made
    # two runs look like they used different hyperparameters when they had not.
    rec = {"config": args.__dict__, "table": table}
    prev = out / "results.json"
    if args.eval_only and prev.exists():
        try:
            old_rec = json.loads(prev.read_text(encoding="utf-8"))
            if "config" in old_rec:
                rec["config"] = old_rec["config"]
                rec["eval_config"] = args.__dict__
        except Exception:
            pass
    prev.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
