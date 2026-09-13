"""The adaptive-depth mechanism itself: does it hold its budget, does it
actually skip work, and does it stay causal?"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from xios.config import get_config
from xios.model import XiosChat


def test_budget_is_two_sided():
    """A one-sided penalty is what let the loop collapse to depth 1."""
    cfg = get_config("nano", max_iters=8, target_depth=4.0)
    m = XiosChat(cfg)
    pc = m.core.ponder
    below = pc.budget_loss(torch.full((2, 16), 1.0))
    at = pc.budget_loss(torch.full((2, 16), 4.0))
    above = pc.budget_loss(torch.full((2, 16), 7.0))
    print(f"budget loss: depth 1 -> {float(below):.4f}   "
          f"depth 4 (target) -> {float(at):.4f}   depth 7 -> {float(above):.4f}")
    assert float(at) < 1e-9, "no penalty at the target"
    assert float(below) > 0, "undershooting the budget must also be penalised"
    assert float(above) > 0, "overshooting the budget must be penalised"


def test_budget_curriculum():
    cfg = get_config("nano", max_iters=12, target_depth=3.0)
    pc = XiosChat(cfg).core.ponder
    path = [round(pc.anneal(p, warmup_frac=0.3), 2) for p in
            (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)]
    print(f"depth target over training: {path}")
    assert path[0] > path[-1], "budget must tighten, not loosen"
    assert abs(path[-1] - 3.0) < 1e-6, "must land on the configured budget"
    assert path == sorted(path, reverse=True), "schedule must be monotone"


def test_halting_is_causal():
    """A token's depth must not depend on anything after it.

    This is the property that lets XIOS decode without an auxiliary router,
    and the one top-k depth routing does not have.
    """
    torch.manual_seed(0)
    cfg = get_config("nano", max_iters=8, max_seq_len=256)
    m = XiosChat(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 40))

    with torch.no_grad():
        d1 = m(ids).stats.actual_depth[0]
        alt = ids.clone()
        alt[0, 25:] = torch.randint(0, cfg.vocab_size, (15,))
        d2 = m(alt).stats.actual_depth[0]

    same = int((d1[:25] == d2[:25]).sum())
    print(f"depths for positions 0..24 unchanged after rewriting 25..39: "
          f"{same}/25")
    assert same == 25, "halting leaked information from future positions"


def test_sparsification_saves_work():
    """Deeper iterations must actually run on fewer tokens."""
    torch.manual_seed(0)
    cfg = get_config("nano", max_iters=8, target_depth=2.0)
    m = XiosChat(cfg)
    # Make halting data-dependent, as it is after training: a zero-weight head
    # gives every token the same depth and would hide a broken mechanism.
    with torch.no_grad():
        m.core.ponder.halt[-1].weight.normal_(0, 1.0)
        m.core.ponder.halt[-1].bias.fill_(0.5)
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    stats = m(ids).stats

    depths = stats.actual_depth.flatten()
    import collections
    hist = dict(sorted(collections.Counter(depths.tolist()).items()))
    print(f"depth histogram across tokens: {hist}")
    print(f"mean {stats.mean_depth:.2f}  p99 {stats.depth_p99:.0f}  "
          f"ceiling {cfg.max_iters}")
    assert len(hist) > 1, "all tokens got identical depth: allocation is not adaptive"
    assert stats.mean_depth < cfg.max_iters, "nothing exited early"

    frac = [round(a, 3) for a in stats.active_frac]
    print(f"fraction of tokens still active per iteration: {frac}")
    assert frac[-1] < frac[0], "active set must shrink with depth"

def test_lagrangian_enforces_budget():
    """A fixed penalty weight does not hold the budget.

    Measured on the modular-arithmetic task: with a constant weight of 0.05
    and a target of 3.0, mean depth settled at 5.68 and stayed there, because
    the language modelling loss simply outbids a constant. Treating the budget
    as a constraint and pricing it by dual ascent fixes that without anyone
    hand-tuning a coefficient per model size.
    """
    cfg = get_config("nano", max_iters=8, target_depth=3.0,
                     budget_mode="lagrangian", dual_lr=0.01)
    pc = XiosChat(cfg).core.ponder
    pc.target_depth = 3.0

    lam_path = []
    for _ in range(400):                     # a model insisting on depth 5.7
        pc.budget_loss(torch.full((2, 16), 5.7))
        lam_path.append(round(pc.dual_step(), 3))
    print(f"lambda while depth stays at 5.7: {lam_path[:3]} ... {lam_path[-1]}")
    assert lam_path[-1] > lam_path[0], "price must rise while over budget"

    # ...but it must never grow to the point of drowning the objective. A
    # first version reached 18 against an LM loss of 0.89 and the model simply
    # stopped learning the task.
    over = float(pc.budget_loss(torch.full((2, 16), 5.7)))
    at = float(pc.budget_loss(torch.full((2, 16), 3.0)))
    print(f"budget cost at depth 5.7: {over:.3f}   at target: {at:.3f}  "
          f"(typical LM loss ~1.0)")
    assert 0.05 < over < 2.0, \
        f"budget term {over:.2f} is not a shaping force: it must be felt but " \
        f"must not dominate a loss of order 1"
    assert at < 1e-6, "no cost exactly at target"

    for _ in range(800):                     # now it comes under budget
        pc.budget_loss(torch.full((2, 16), 2.0))
        pc.dual_step()
    print(f"lambda after sustained underspending: {float(pc.lam):.4f}")
    assert float(pc.lam) < 0.01, "price must decay when under budget"

def test_budget_prices_real_compute():
    """The budget must constrain the compute actually spent.

    `sum_i R_i` is the expected depth under *sampled* halting; execution runs
    deterministically until R < eps, which costs substantially more. Measured,
    sum(R) under-counted real iterations by ~2x, and the Lagrange multiplier
    sat at exactly 0 while depth was pinned near the ceiling -- the constraint
    was watching a number that was already under target.
    """
    torch.manual_seed(0)
    cfg = get_config("nano", max_iters=8, target_depth=3.0)
    m = XiosChat(cfg)
    with torch.no_grad():                 # data-dependent halting, as trained
        m.core.ponder.halt[-1].weight.normal_(0, 1.0)
        m.core.ponder.halt[-1].bias.fill_(0.5)
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    st = m(ids).stats

    actual = st.mean_depth
    surrogate = float(st.compute_depth.mean())
    soft = float(st.expected_depth.mean())
    print(f"actual iterations {actual:.2f} | surrogate {surrogate:.2f} | "
          f"sum(R) {soft:.2f}")
    assert abs(surrogate - actual) / actual < 0.15,         "the budgeted quantity must track the iterations actually run"
    assert soft < actual * 0.8,         "sanity: sum(R) really does under-count, which is why it cannot be used"


if __name__ == "__main__":
    test_budget_is_two_sided()
    test_budget_curriculum()
    test_halting_is_causal()
    test_sparsification_saves_work()
    test_lagrangian_enforces_budget()
    test_budget_prices_real_compute()
    print("ponder tests passed")
