"""Can a looped model think *longer at test time* than it was trained to?

This is the question the architecture is actually built for, and Experiment 8
made it askable. With adaptive halting removed, iteration count is a free knob
at inference: the same weights can be run 2 times or 20. Nothing about the
weight file changes.

If accuracy on hard problems *rises* when the model is given more iterations
than it trained with, that is test-time compute scaling from a small local
model — the one axis where a cache-resident looped core has a structural
advantage, because extra iterations are nearly free in bandwidth (measured:
depth 4 -> 64 costs 1.000x DRAM traffic).

If accuracy instead *falls* outside the trained depth, the loop has learned a
fixed-length program rather than an iterative one, and "think longer" is not
available. That would be worth knowing before building anything on it.

The sweep is two-dimensional and costs no training at all — just forward passes
on an existing checkpoint:

    rows: problem difficulty (composition steps)
    cols: iterations granted at inference

Read the diagonal: for a problem needing `k` steps, does the best accuracy come
at more iterations as `k` grows?
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

torch.set_num_threads(12)

from xios.config import XiosConfig, get_config
from xios.model import XiosChat
from xios.tasks import TaskDataset, TASK_VOCAB, exact_match


def load(ckpt: str, depth: float, max_iters: int, seq_len: int,
         dim: int, core_blocks: int, width: int):
    """Rebuild the model with a chosen inference depth and load the weights."""
    cfg = get_config("nano", vocab_size=TASK_VOCAB, max_seq_len=seq_len,
                     dim=dim, core_blocks=core_blocks, n_kv_heads=2,
                     max_iters=max(max_iters, int(depth) + 1),
                     target_depth=float(depth), adaptive_depth=False,
                     attn_window=min(256, seq_len))
    m = XiosChat(cfg)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing = m.load_state_dict(blob["model"], strict=False)
    if getattr(missing, "unexpected_keys", None):
        # the halting head exists in an adaptive-depth checkpoint and is unused
        # here; anything else missing would be a real mismatch
        unexpected = [k for k in missing.unexpected_keys if "ponder" not in k]
        assert not unexpected, f"unexpected weights: {unexpected[:4]}"
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ablate_fixed/xios.pt")
    ap.add_argument("--task", default="perm")
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--trained-depth", type=float, default=6.0,
                    help="iterations the checkpoint was trained with")
    ap.add_argument("--depths", default="2,4,6,8,10,12,16")
    ap.add_argument("--problem-steps", default="2,4,6,8,9,10,12")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--core-blocks", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--eval-n", type=int, default=192)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/ttd")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    depths = [int(x) for x in a.depths.split(",")]
    steps = [int(x) for x in a.problem_steps.split(",")]

    print(f"checkpoint trained at depth {a.trained_depth:.0f}; "
          f"evaluating at {depths}")
    print(f"task {a.task} width {a.width}; chance "
          f"{TaskDataset(a.task, 1, 1, a.seq_len, width=a.width).chance:.1%}\n")

    grid = {}
    hdr = f"{'problem':>8} " + "".join(f"{d:>7}" for d in depths)
    print(hdr)
    print(f"{'steps':>8} " + "".join("  iters" for _ in depths))
    print("-" * len(hdr))
    for s in steps:
        ds = TaskDataset(a.task, s, s, a.seq_len, seed=1234 + s, width=a.width)
        row = []
        for d in depths:
            m, _ = load(a.ckpt, d, max(depths), a.seq_len, a.dim,
                        a.core_blocks, a.width)
            r = exact_match(m.to(dev), ds, s, n=a.eval_n, device=dev)
            row.append(r["exact_match"])
            grid[(s, d)] = r["exact_match"]
        best = max(range(len(depths)), key=lambda i: row[i])
        line = "".join(f"{100*v:6.1f}%" for v in row)
        print(f"{s:8d} {line}   best @ {depths[best]} iters")

    print()
    trained = int(a.trained_depth)
    print(f"Does more inference depth help beyond the trained {trained}?")
    for s in steps:
        at = grid.get((s, trained))
        more = [grid[(s, d)] for d in depths if d > trained]
        if at is None or not more:
            continue
        gain = max(more) - at
        verdict = ("HELPS" if gain > 0.05 else
                   "no change" if gain > -0.05 else "HURTS")
        print(f"  {s:2d} steps: {100*at:5.1f}% at {trained} iters -> "
              f"best {100*max(more):5.1f}% beyond  ({gain*100:+5.1f} pts, {verdict})")

    (out / "grid.json").write_text(json.dumps(
        {"depths": depths, "steps": steps,
         "grid": {f"{s}_{d}": v for (s, d), v in grid.items()},
         "args": a.__dict__}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/grid.json")


if __name__ == "__main__":
    main()
