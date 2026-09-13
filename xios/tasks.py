"""Task families that isolate *sequential computation*.

Perplexity on natural text is a blunt instrument for the question XIOS
actually asks, which is: does letting a small model think longer on hard
tokens buy reasoning a bigger fixed-depth model would otherwise be needed
for?  Natural text averages over easy and hard tokens and mostly rewards
knowledge, so it hides the effect in both directions.

These tasks do not.  Each one has a tunable number of *required sequential
steps*, and each is known to be hard for a fixed-depth transformer -- a
depth-``L`` model can only compose so many dependent operations per forward
pass, so accuracy falls off a cliff once required steps exceed what depth
allows.  If adaptive depth works at all, it shows up here as a model that
keeps solving instances past the depth cliff by spending more iterations on
them.  If it does not show up here, it does not exist.

Every task is generated, so there is no test-set contamination and
difficulty can be dialled continuously.

Calibrate before you measure
----------------------------
Each task has a *width* knob (table size, group size, modulus) separate from
its *depth* knob (number of chained steps).  These must be set so the model
can already perform ONE step reliably -- otherwise the benchmark measures
whether the model can do a single lookup, not how many it can chain, and
every curve comes out flat at chance for reasons that have nothing to do with
depth.

Measured, dense 4-layer baseline, 700 steps, single-hop pointer chase:

    4 nodes  -> 100 %      8 nodes -> 31 %
    6 nodes  -> 100 %     10 nodes -> 11 %      12 nodes -> 6 % (chance 8 %)

At 12 nodes -- the original default -- the model never learns one hop, so the
whole depth experiment reads as noise.  ``TaskDataset`` therefore takes a
``width`` argument, and ``calibrate_width`` finds the largest width a given
model actually learns, so the depth experiment starts from a working base.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Character vocabulary shared by the synthetic tasks
# ---------------------------------------------------------------------------
TASK_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz +-*=(),.:|?>#\n"
CHAR_TO_ID = {c: i + 4 for i, c in enumerate(TASK_CHARS)}
ID_TO_CHAR = {i + 4: c for i, c in enumerate(TASK_CHARS)}
PAD, BOS, EOS, SEP = 0, 1, 2, 3
TASK_VOCAB = len(TASK_CHARS) + 4


def encode_chars(s: str) -> list[int]:
    return [CHAR_TO_ID[c] for c in s if c in CHAR_TO_ID]


def decode_chars(ids) -> str:
    return "".join(ID_TO_CHAR.get(int(i), "") for i in ids)


@dataclass
class Sample:
    prompt: str
    answer: str
    steps: int          # ground-truth sequential depth required


# ---------------------------------------------------------------------------
# 1. Iterated permutation composition (S_5)
# ---------------------------------------------------------------------------
# Composing k permutations is the canonical example of a problem needing
# depth proportional to k: the group is non-abelian, so no amount of width
# lets you parallelise the chain.
def perm_compose(rng: random.Random, steps: int, n: int = 4) -> Sample:
    perms = []
    cur = list(range(n))
    for _ in range(steps):
        p = list(range(n))
        rng.shuffle(p)
        perms.append(p)
        cur = [cur[p[i]] for i in range(n)]
    body = " ".join("".join(str(x) for x in p) for p in perms)
    return Sample(f"p {body} = ", "".join(str(x) for x in cur), steps)


# ---------------------------------------------------------------------------
# 2. Multi-hop pointer chasing
# ---------------------------------------------------------------------------
# A lookup table plus a chain of dereferences. Each hop is trivial; the chain
# is not parallelisable. Tests whether depth is being spent where it matters,
# since only the final tokens need the full chain.
def pointer_chase(rng: random.Random, steps: int, n_nodes: int = 6) -> Sample:
    """Follow one pointer table `steps` times.

    The table is a **single n-cycle**, and ``steps`` must be < n_nodes.  Both
    conditions are load-bearing, and neither was obvious:

    * An arbitrary permutation has fixed points (``a -> a``), which make an
      instance solvable in zero hops whatever its declared step count.
    * Even a derangement has *short cycles*.  Following one table k times
      lands at ``k mod cycle_length``, so difficulty is not monotone in k at
      all -- a chase of 6 steps around a 2-cycle is just the identity.
      Measured on the original version, exact match went
      ``100%, 18%, 19%, 45%, 0%, 80%, 0%, 40%, 13%`` across steps 1..9: the
      80% at six steps is the model discovering that the answer is often
      simply the start node.  A benchmark whose difficulty axis is not
      monotone cannot measure depth, and a flat depth-vs-difficulty curve
      measured on it means nothing.

    A single n-cycle makes all k < n distinct, which restores monotonicity --
    at the cost of capping testable depth at ``n_nodes - 1``.  That cap is
    real: this task cannot probe deeper than its table is wide.  Use ``perm``
    for deeper chains, where every step applies a *different* permutation and
    so has no wrap-around shortcut at all.
    """
    if steps >= n_nodes:
        raise ValueError(
            f"pointer_chase: steps={steps} >= n_nodes={n_nodes}. A single "
            f"cycle wraps, so the label would be wrong. Raise n_nodes (and "
            f"check the model can still do one hop) or use the `perm` task.")
    names = [chr(ord("a") + i) for i in range(n_nodes)]
    order = list(range(n_nodes))
    rng.shuffle(order)
    vals = [0] * n_nodes
    for i in range(n_nodes):
        vals[order[i]] = order[(i + 1) % n_nodes]
    table = {names[i]: names[vals[i]] for i in range(n_nodes)}
    start = rng.choice(names)
    cur = start
    for _ in range(steps):
        cur = table[cur]
    body = " ".join(f"{k}>{v}" for k, v in table.items())
    return Sample(f"h {body} | {start} {steps} = ", cur, steps)


# ---------------------------------------------------------------------------
# 3. Modular running sum with carries
# ---------------------------------------------------------------------------
def mod_chain(rng: random.Random, steps: int, mod: int = 5) -> Sample:
    ops, cur = [], rng.randrange(mod)
    start = cur
    for _ in range(steps):
        k = rng.randrange(1, mod)
        op = rng.choice("+-*")
        ops.append(f"{op}{k}")
        cur = {"+": cur + k, "-": cur - k, "*": cur * k}[op] % mod
    return Sample(f"m {start}{''.join(ops)} = ", str(cur), steps)


# ---------------------------------------------------------------------------
# 4. Parity of a gated subset
# ---------------------------------------------------------------------------
# Parity is the standard separation example for bounded-depth circuits.
def gated_parity(rng: random.Random, steps: int) -> Sample:
    bits = [rng.randrange(2) for _ in range(steps)]
    gates = [rng.randrange(2) for _ in range(steps)]
    val = sum(b for b, g in zip(bits, gates) if g) % 2
    body = "".join(str(b) for b in bits) + "," + "".join(str(g) for g in gates)
    return Sample(f"x {body} = ", str(val), steps)


TASKS: dict[str, Callable[..., Sample]] = {
    "perm": perm_compose,
    "hop": pointer_chase,
    "mod": mod_chain,
    "parity": gated_parity,
}

# Name of each task's width knob, and its default.  Width controls how hard a
# *single* step is; steps control how many must be chained.
WIDTH_ARG: dict[str, tuple[str, int]] = {
    "perm": ("n", 4),
    "hop": ("n_nodes", 6),
    "mod": ("mod", 5),
    "parity": ("", 0),          # parity has no width knob
}


# ---------------------------------------------------------------------------
class TaskDataset:
    """Generates batches on the fly; difficulty is a curriculum knob."""

    def __init__(self, task: str = "perm", min_steps: int = 1, max_steps: int = 12,
                 seq_len: int = 256, seed: int = 0, width: int | None = None):
        assert task in TASKS, f"unknown task {task}; choose from {sorted(TASKS)}"
        self.task = task
        self.fn = TASKS[task]
        self.min_steps, self.max_steps = min_steps, max_steps
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        arg, default = WIDTH_ARG.get(task, ("", 0))
        self.width = width if width is not None else default
        self.kwargs = {arg: self.width} if arg else {}

    @property
    def max_valid_steps(self) -> int:
        """Deepest chain this task can label correctly at its current width.

        ``hop`` follows one cyclic table, so it wraps at ``width``.  ``perm``
        composes a fresh permutation every step and never wraps.
        """
        if self.task == "hop":
            return self.width - 1
        return 10 ** 6

    @property
    def chance(self) -> float:
        """Accuracy of a model that has learned the format but nothing else."""
        if self.task == "parity":
            return 0.5
        if self.task == "perm":
            import math
            return 1.0 / math.factorial(self.width)
        return 1.0 / max(self.width, 1)

    def sample(self, steps: int | None = None) -> Sample:
        s = steps if steps is not None else self.rng.randint(self.min_steps, self.max_steps)
        return self.fn(self.rng, s, **self.kwargs)

    def batch(self, batch_size: int, steps: int | None = None, device="cpu"):
        """Returns (input_ids, labels, answer_mask, steps_per_row).

        Labels are -100 everywhere except the answer, so the loss measures
        only whether the computation was carried out -- not how well the
        model memorised the prompt format.
        """
        ids, labels, masks, stepv = [], [], [], []
        for _ in range(batch_size):
            smp = self.sample(steps)
            p = encode_chars(smp.prompt)
            a = encode_chars(smp.answer)
            seq = [BOS] + p + a + [EOS]
            lab = [-100] * (1 + len(p)) + a + [EOS]
            seq = seq[:self.seq_len]
            lab = lab[:self.seq_len]
            pad = self.seq_len - len(seq)
            m = [0] * (1 + len(p)) + [1] * (len(a) + 1) + [0] * pad
            ids.append(seq + [PAD] * pad)
            labels.append(lab + [-100] * pad)
            masks.append(m[:self.seq_len])
            stepv.append(smp.steps)
        return (torch.tensor(ids, device=device),
                torch.tensor(labels, device=device),
                torch.tensor(masks, device=device, dtype=torch.bool),
                torch.tensor(stepv, device=device))


@torch.no_grad()
def exact_match(model, ds: TaskDataset, steps: int, n: int = 128,
                batch_size: int = 32, device="cpu") -> dict:
    """Fraction of instances whose answer is produced exactly.

    Teacher-forced argmax over the answer span: the model must get every
    answer token right, which is the only sensible metric for these tasks.
    """
    model.eval()
    correct = total = 0
    depth_sum, depth_n = 0.0, 0
    all_depth_sum = 0.0
    for _ in range(max(1, n // batch_size)):
        ids, labels, mask, _ = ds.batch(batch_size, steps=steps, device=device)
        out = model(ids, labels=None, loss_mask=mask)
        pred = out.logits[:, :-1].argmax(-1)
        tgt = labels[:, 1:]
        valid = tgt != -100
        hit = ((pred == tgt) | ~valid).all(dim=1)
        correct += int(hit.sum())
        total += ids.shape[0]
        st = getattr(out, "stats", None)
        if st is not None and getattr(st, "actual_depth", None) is not None:
            # Depth must be measured on the tokens that do the work.
            # Averaging over the whole sequence mixes in padding, whose
            # fraction changes with the problem size -- which silently turns
            # the depth-vs-difficulty curve into a measurement of prompt
            # length. That artefact produced a -1.000 correlation on one run
            # and a +1.000 on another, neither of which meant anything.
            m = mask.float()
            depth_sum += float((st.actual_depth.float() * m).sum()
                               / m.sum().clamp(min=1))
            all_depth_sum += float(st.actual_depth.float().mean())
            depth_n += 1
    res = {"steps": steps, "exact_match": correct / max(total, 1), "n": total}
    if depth_n:
        res["mean_depth"] = depth_sum / depth_n            # on scored tokens
        res["all_token_depth"] = all_depth_sum / depth_n   # incl. padding
    return res


@torch.no_grad()
def _em(model, ds, steps, n, batch_size, device):
    return exact_match(model, ds, steps, n=n, batch_size=batch_size,
                       device=device)["exact_match"]


def calibrate_width(model_fn, task: str, seq_len: int, widths=(4, 6, 8, 10, 12),
                    steps: int = 600, batch_size: int = 64, lr: float = 1e-3,
                    device: str = "cpu", threshold: float = 0.9, verbose=True):
    """Largest task width at which a model learns the ONE-step version.

    Run this before a depth experiment.  If the model cannot do a single step
    at the chosen width, the depth curve carries no information about depth,
    and every model will look equally bad for the wrong reason.
    """
    best = None
    for w in widths:
        model = model_fn().to(device)
        ds = TaskDataset(task, 1, 1, seq_len, seed=0, width=w)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
        for _ in range(steps):
            ids, lab, mask, _ = ds.batch(batch_size, device=device)
            out = model(ids, labels=lab, loss_mask=mask)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        em = _em(model, ds, 1, 128, batch_size, device)
        if verbose:
            print(f"  width {w:3d}: single-step EM {em:6.1%}  (chance {ds.chance:.1%})")
        if em >= threshold:
            best = w
        else:
            break
    if best is None:
        print(f"  NONE of {list(widths)} reached {threshold:.0%} on the one-step "
              f"case in {steps} steps. Train longer, widen the model, or add an "
              f"easier width -- a depth experiment run from here measures "
              f"nothing about depth.")
    return best
