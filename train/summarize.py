"""Compare A/B runs.

    python train/summarize.py runs/ab_perm2 runs/deep_params

Prints each run's table plus the three numbers that actually decide anything:
in-distribution accuracy at the hardest trained difficulty, extrapolation
accuracy *relative to chance* (an absolute number is meaningless without it),
and whether allocated depth tracks difficulty.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def load(d: pathlib.Path) -> dict:
    return json.loads((d / "results.json").read_text(encoding="utf-8"))


def chance_of(cfg: dict) -> float:
    from xios.tasks import TaskDataset
    ds = TaskDataset(cfg["task"], 1, 1, cfg["seq_len"], width=cfg.get("width"))
    return ds.chance


def summarize(d: pathlib.Path) -> None:
    res = load(d)
    cfg, table = res["config"], res["table"]
    chance = chance_of(cfg)
    trained = cfg["train_max_steps"]

    print(f"\n=== {d.name} ===")
    print(f"task {cfg['task']} width {cfg.get('width')}  trained 1..{trained}  "
          f"chance {chance:.1%}  max_iters {cfg['max_iters']}  "
          f"target_depth {cfg['target_depth']}  match {cfg.get('match', 'params')}")
    print(f"{'steps':>6} {'baseline':>10} {'XIOS':>10} {'depth':>8}")
    for r in table:
        mark = "" if r["steps"] <= trained else "  ext"
        print(f"{r['steps']:6d} {r.get('baseline', float('nan')):9.1%} "
              f"{r.get('xios', float('nan')):9.1%} "
              f"{r.get('xios_depth', float('nan')):8.2f}{mark}")

    ind = [r for r in table if r["steps"] == trained]
    ext = [r for r in table if r["steps"] > trained]

    if ind:
        b, x = ind[0].get("baseline"), ind[0].get("xios")
        if b is not None and x is not None:
            print(f"\nin-distribution @ {trained} steps:  baseline {b:.1%}   XIOS {x:.1%}"
                  f"   ({'XIOS' if x > b else 'baseline' if b > x else 'tie'})")

    if ext:
        b = sum(r.get("baseline", 0) for r in ext) / len(ext)
        x = sum(r.get("xios", 0) for r in ext) / len(ext)
        print(f"extrapolation mean:          baseline {b:.1%}   XIOS {x:.1%}"
              f"   (chance {chance:.1%})")
        for nm, v in (("baseline", b), ("XIOS", x)):
            verdict = ("ABOVE chance" if v > chance * 1.5 else
                       "at chance" if v > chance * 0.5 else "BELOW chance")
            print(f"    {nm:9s} {v:.1%} -> {verdict}")
        if x <= chance * 1.5:
            print("    NOTE: at or below chance means no extrapolation, whatever "
                  "the gap to the baseline is.")

    ds = [(r["steps"], r["xios_depth"]) for r in table if r.get("xios_depth")]
    if len(ds) > 2:
        import statistics
        xs = [a for a, _ in ds]
        ys = [b for _, b in ds]
        corr = statistics.correlation(xs, ys) if len(set(ys)) > 1 else 0.0
        print(f"depth vs difficulty:         r = {corr:+.3f}   "
              f"range {min(ys):.2f}..{max(ys):.2f} of {cfg['max_iters']} "
              f"({'allocating adaptively' if corr > 0.5 else 'NOT adaptive'})")
        used = max(ys) / cfg["max_iters"]
        print(f"depth headroom used:         {used:.0%} of the ceiling"
              + ("   (the deep range is going unused)" if used < 0.5 else ""))


if __name__ == "__main__":
    dirs = [pathlib.Path(a) for a in sys.argv[1:]] or [pathlib.Path("runs/ab_perm2")]
    for d in dirs:
        if (d / "results.json").exists():
            summarize(d)
        else:
            print(f"\n=== {d} === (no results.json yet)")
