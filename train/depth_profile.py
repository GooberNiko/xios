"""What does XIOS spend its thinking on?

    python train/depth_profile.py --ckpt runs/demo/xios.pt \
        --tokenizer runs/demo/tokenizer.json --prompt "ROMEO:"

Every generated token carries the number of core iterations it cost. Grouping
tokens by that number shows where the model chooses to spend compute, and it
is a far more honest probe of the adaptive mechanism than a single correlation
coefficient -- these are *generated* tokens, each with its own measured depth,
so there is no padding to average over and no way for sequence geometry to
fake a result.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import queue
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def profile(ckpt, tokenizer, prompt, n_tokens, temperature, seed, device=None):
    from app import Engine
    e = Engine.load(ckpt, tokenizer, device=device)
    if not e.trained:
        print("WARNING: no trained checkpoint -- depths will be meaningless\n")
    torch.manual_seed(seed)

    q: queue.Queue = queue.Queue()
    e.stream(prompt, n_tokens, temperature, 0.95, e.cfg.max_iters,
             threading.Event(), q)
    toks = []
    while not q.empty():
        kind, payload, depth = q.get()
        if kind == "tok":
            toks.append((payload, depth))
    if not toks:
        print("generated nothing")
        return

    text = "".join(t for t, _ in toks)
    print("--- generated ---")
    print(text.encode("ascii", "backslashreplace").decode("ascii"))

    byd = collections.defaultdict(list)
    for t, d in toks:
        byd[d].append(t)
    print(f"\n--- tokens grouped by iterations spent (ceiling {e.cfg.max_iters}) ---")
    for d in sorted(byd):
        sample = " ".join(repr(x) for x in byd[d][:12])
        pct = 100 * len(byd[d]) / len(toks)
        print(f"  depth {d:2d}  n={len(byd[d]):4d} ({pct:4.1f}%)  {sample}")

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    groups = {
        "whitespace only": [d for t, d in toks if t.strip() == ""],
        "word-initial (leading space)": [d for t, d in toks
                                         if t[:1] == " " and t.strip()],
        "word-internal": [d for t, d in toks if t[:1] not in (" ",) and t.strip()],
        "punctuation / structure": [d for t, d in toks
                                    if t.strip() and not any(c.isalnum() for c in t)],
        "contains a digit": [d for t, d in toks if any(c.isdigit() for c in t)],
    }
    print("\n--- mean depth by token kind ---")
    for k, v in groups.items():
        if v:
            print(f"  {k:30s} n={len(v):4d}  mean depth {mean(v):.2f}")

    ds = [d for _, d in toks]
    core = e.cfg.core_blocks
    print(f"\noverall mean depth {mean(ds):.2f}  min {min(ds)}  max {max(ds)}")
    print(f"core block-applications spent {sum(ds)*core} "
          f"vs {len(ds)*e.cfg.stored_blocks} for a dense stack of the same file "
          f"({sum(ds)*core/max(len(ds)*e.cfg.stored_blocks,1):.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/demo/xios.pt")
    ap.add_argument("--tokenizer", default="runs/demo/tokenizer.json")
    ap.add_argument("--prompt", default="ROMEO:\nWhat say you")
    ap.add_argument("--n-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    profile(a.ckpt, a.tokenizer, a.prompt, a.n_tokens, a.temperature,
            a.seed, a.device)


if __name__ == "__main__":
    main()
