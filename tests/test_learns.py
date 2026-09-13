"""Can XIOS learn at all, and as easily as a plain transformer?

Before any claim about XIOS being *better*, it has to clear the floor: on a
task a small dense transformer solves outright, XIOS must solve it too. A
looped core with adaptive halting has many ways to fail silently -- the
controller can halt before the loop does anything, the iteration conditioning
can wash out the signal, the per-depth states can desynchronise. This test is
the tripwire for all of them.

Deliberately trivial tasks (copy a character, one table lookup). Both models
should reach ~100%. If XIOS does not, nothing downstream is worth reading.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from xios.config import get_config
from xios.model import XiosChat
from xios.baseline import matched_baseline
from xios.tasks import TaskDataset, TASK_VOCAB, Sample, exact_match
import xios.tasks as T


def _copy_task(rng, steps, width=6):
    ch = "abcdefghijkl"[:width]
    body = "".join(rng.choice(ch) for _ in range(8))
    return Sample(f"c {body} = ", body[0], 1)


T.TASKS.setdefault("copy", _copy_task)
T.WIDTH_ARG.setdefault("copy", ("width", 6))


def fit(model, task, steps=800, bs=64, lr=1e-3, seq=40, width=6, device="cpu"):
    torch.manual_seed(0)
    model = model.to(device)
    ds = TaskDataset(task, 1, 1, seq, seed=0, width=width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ponder = getattr(getattr(model, "core", None), "ponder", None)
    model.train()
    for step in range(steps):
        ids, lab, mask, _ = ds.batch(bs, device=device)
        out = model(ids, labels=lab, loss_mask=mask)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ponder is not None:
            ponder.anneal(step / steps, 0.3)
            ponder.dual_step()
    r = exact_match(model, ds, 1, n=256, device=device)
    return r


def main():
    cfg = get_config("nano", vocab_size=TASK_VOCAB, dim=256, n_kv_heads=2,
                     core_blocks=2, max_seq_len=128, max_iters=6,
                     target_depth=3.0, attn_window=64)
    results = {}
    # `steps` per task: enough that a dense baseline reliably solves it.
    # Measured: 6-node single-hop needs ~700 steps at batch 64 / lr 1e-3;
    # at 500 steps the baseline itself only reaches 20%, which would make this
    # tripwire fire for the wrong reason.
    for task, seq, steps in (("copy", 40, 500), ("hop", 48, 900)):
        print(f"\n--- {task} (one step, width 6) ---")
        for name, build in (("baseline", lambda: matched_baseline(cfg)),
                            ("xios", lambda: XiosChat(cfg))):
            r = fit(build(), task, seq=seq, steps=steps)
            extra = f"  mean depth {r['mean_depth']:.2f}" if "mean_depth" in r else ""
            print(f"  {name:9s} exact match {r['exact_match']:6.1%}{extra}")
            results[(task, name)] = r["exact_match"]

    print()
    for task in ("copy", "hop"):
        b, x = results[(task, "baseline")], results[(task, "xios")]
        print(f"{task:6s} baseline {b:.1%}  XIOS {x:.1%}")
        assert b > 0.9, f"the BASELINE failed {task} ({b:.1%}) -- the harness or " \
                        f"task width is mis-set, fix that before reading anything else"
        assert x > 0.9, f"XIOS failed a task the baseline solves ({x:.1%} vs " \
                        f"{b:.1%}) -- the looped core has a bug"
    print("\nboth architectures clear the floor")


if __name__ == "__main__":
    main()
