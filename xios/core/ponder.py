"""Budget-allocated adaptive depth.

The premise
-----------
Every transformer in use today spends identical compute on every token.
Predicting the "the" in "one of the" and predicting the final step of a proof
cost exactly the same.  That is not a tuning inefficiency, it is a structural
one: depth is baked into the weight file.

Here depth is a *runtime allocation under a fixed budget*.  A sequence of T
tokens is given ``T * target_depth`` block-applications total.  A controller
decides how to spend them.  Trivial tokens exit after one pass and donate
their unspent budget; hard tokens loop the shared core twenty or thirty
times.  Average FLOPs per token are unchanged -- what changes is that the
hard tokens now receive sequential compute far beyond what the parameter
count would normally buy, because the loop reuses the same weights.

Halting rule
------------
Standard ACT-style geometric halting.  At iteration ``i`` an active token
emits ``p_i = sigmoid(halt(h))``.  With ``R_i`` the probability it is still
running:

    R_0 = 1,   R_{i+1} = R_i (1 - p_i)
    y    = sum_i R_i p_i h_i  +  R_N h_N        (N = max iterations)
    E[depth] = sum_i R_i

A token stops being computed once ``R_i`` falls below ``eps``.  This decision
depends only on that token's own hidden state, so it is exactly causal -- no
auxiliary predictor is needed at inference, unlike top-k routing schemes
which leak information from future positions during training.

Keeping the loop alive
----------------------
The obvious budget loss -- penalise only *exceeding* the target -- collapses.
Measured: mean depth falls from 6.8 to 1.05 inside fifty steps.  The reason
is a chicken-and-egg problem.  Early in training the later iterations are
untrained and therefore useless, so halting immediately is genuinely the
better move; but once the controller halts at depth one, those iterations
never receive gradient and can never become useful.  The loop kills itself
before it learns to be worth anything.

Two changes fix it, and both are needed:

* The budget loss is **two-sided**.  Mean depth is pinned *at* the budget
  rather than merely under it, so compute freed from easy tokens has to be
  spent somewhere rather than evaporating.  This is what makes the scheme
  compute-neutral reallocation instead of plain compute reduction.

* The budget is **annealed**.  Training starts with a generous depth target
  so the core learns to use its own iterations, then tightens to the real
  budget.  Learn to think first, learn to be efficient second.

Enforcing the budget
--------------------
A fixed penalty weight does not work.  Measured on the modular-arithmetic
task: with ``budget_weight=0.05`` and a target of 3.0, mean depth settled at
**5.68** and stayed there -- the language modelling loss simply outbids a
constant penalty, and the model happily pays it in exchange for compute.
Raising the constant by hand is guesswork that has to be redone for every
model size, task and learning rate.

So the budget is treated as a **constraint**, not a preference, and enforced
with a Lagrange multiplier updated by dual ascent:

    lambda <- clamp(lambda + eta * (ema_depth - target), 0, lambda_max)
    loss   += lambda * (mean_depth - target) / max(1, target)

If depth runs over budget the multiplier climbs until it is expensive enough
to matter; if depth undershoots it decays and the model is free to think
more.  ``lambda`` is readable as the marginal price the model puts on one
extra iteration -- a useful diagnostic in its own right.

Three details are load-bearing, all three learned the hard way:

* **The dual rate must be small and the cap low.**  First attempt used
  ``eta=0.05`` and ``lambda_max=10``.  Measured: lambda oscillated
  0 -> 0.65 -> 7.17 -> 1.77 while depth swung 3.8 -> 3.7 -> 5.5, and at
  lambda=7.17 the budget term reached ~18 against a language-modelling loss
  of 0.89 -- **twenty times the objective it was supposed to shape**.  The
  model stopped learning the task and spent its gradient on depth control.
  A constraint that dominates the objective is not a constraint, it is a
  different objective.

* **The gap is normalised by the target.**  Otherwise the same ``eta`` means
  something different at target depth 3 than at target depth 30.

* **The dual update uses a smoothed depth.**  Per-batch depth is noisy;
  feeding that noise straight into an integrator is what makes dual ascent
  ring.

Note what is *not* constrained: the distribution.  The mean is pinned; which
tokens get more than the mean is entirely up to the controller and the
language modelling loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PonderStats:
    expected_depth: torch.Tensor      # (B, T) sum of R -- the soft depth
    actual_depth: torch.Tensor        # (B, T) integer iterations actually run
    budget_loss: torch.Tensor
    compute_depth: Optional[torch.Tensor] = None   # differentiable iteration count
    halt_probs: list = field(default_factory=list)
    active_frac: list = field(default_factory=list)
    mask: Optional[torch.Tensor] = None   # the tokens the budget is measured on

    @property
    def mean_depth(self) -> float:
        """Mean over ALL positions, padding included."""
        return float(self.actual_depth.float().mean())

    @property
    def scored_depth(self) -> float:
        """Mean over the positions the budget is actually computed on.

        These two differ, sometimes wildly. When a loss mask restricts
        supervision to a few answer tokens, the budget constrains *those*
        while `mean_depth` averages over a sequence that is mostly padding --
        so a log line showing depth pinned at the ceiling can sit next to a
        Lagrange multiplier of zero, and both are correct. Reporting the
        unmasked figure while controlling the masked one is a good way to draw
        exactly the wrong conclusion, so both are exposed.
        """
        if self.mask is None:
            return self.mean_depth
        m = self.mask.float()
        return float((self.actual_depth.float() * m).sum() / m.sum().clamp(min=1))

    @property
    def depth_p99(self) -> float:
        return float(torch.quantile(self.actual_depth.float().flatten(), 0.99))


class PonderController(nn.Module):
    def __init__(self, dim: int, target_depth: float = 4.0, max_iters: int = 16,
                 min_iters: int = 2, eps: float = 0.02, budget_weight: float = 0.05,
                 budget_mode: str = "lagrangian", dual_lr: float = 0.002,
                 lambda_max: float = 1.0, depth_ema: float = 0.98):
        super().__init__()
        self.target_depth = target_depth
        self.final_target = target_depth
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.eps = eps
        self.budget_weight = budget_weight
        self.budget_mode = budget_mode
        self.dual_lr = dual_lr
        self.lambda_max = lambda_max
        self.depth_ema = depth_ema
        # dual variable: the price of one extra iteration. A buffer so it
        # survives checkpointing and is visible in logs.
        self.register_buffer("lam", torch.zeros(()), persistent=True)
        self.register_buffer("last_depth", torch.zeros(()), persistent=False)
        self.register_buffer("ema_depth", torch.zeros(()), persistent=False)
        self._ema_init = False

        self.halt = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.SiLU(), nn.Linear(dim // 4, 1))
        # Start near p=0.05 so the model begins by thinking deeply and learns
        # where it can afford to stop.  Starting shallow is a trap: the core
        # never learns to use its own later iterations.
        nn.init.zeros_(self.halt[-1].weight)
        nn.init.constant_(self.halt[-1].bias, -3.0)

    def halt_prob(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt(h).float().squeeze(-1))

    def budget_loss(self, depth: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``depth`` must be a measure of the compute actually spent.

        Passing the soft ``sum(R)`` here silently under-counts by roughly 4x
        and the constraint stops constraining anything -- see the note in
        ``RecurrentComputeCore.forward``.
        """
        if mask is not None:
            d = (depth * mask).sum() / mask.sum().clamp(min=1)
        else:
            d = depth.mean()

        scale = max(1.0, self.target_depth)
        gap = (d - self.target_depth) / scale
        with torch.no_grad():
            self.last_depth.copy_(d.detach())
            if not self._ema_init:
                self.ema_depth.copy_(d.detach())
                self._ema_init = True
            else:
                self.ema_depth.mul_(self.depth_ema).add_(
                    d.detach() * (1 - self.depth_ema))

        if self.budget_mode == "fixed":
            # Two-sided: hold the mean AT the budget. A one-sided penalty lets
            # the controller collapse to depth 1 and never recover.
            return self.budget_weight * gap.pow(2) * scale * scale

        # Lagrangian: a linear term priced by the dual variable, plus a small
        # quadratic for smooth gradients near the target. Both are in
        # normalised units so the same rates work at any target depth.
        return self.lam.detach() * gap + self.budget_weight * gap.pow(2)

    @torch.no_grad()
    def dual_step(self) -> float:
        """Dual ascent on the depth constraint. Call once per optimiser step.

        Driven by the smoothed depth, not the per-batch value: integrating raw
        batch noise is what makes this controller ring.
        """
        if self.budget_mode != "lagrangian":
            return 0.0
        gap = (float(self.ema_depth) - self.target_depth) / max(1.0, self.target_depth)
        self.lam.add_(self.dual_lr * gap).clamp_(0.0, self.lambda_max)
        return float(self.lam)

    def set_target(self, target: float) -> None:
        self.target_depth = float(target)

    def anneal(self, progress: float, warmup_frac: float = 0.3,
               start_mult: float = 0.85) -> float:
        """Budget curriculum: generous early, tightening to the real budget.

        ``progress`` is the fraction of training completed.  The target walks
        from ``start_mult * max_iters`` down to the configured budget over the
        first ``warmup_frac`` of the run, then stays there.
        """
        start = max(self.final_target, start_mult * self.max_iters)
        if warmup_frac <= 0:
            self.target_depth = self.final_target
        else:
            t = min(1.0, progress / warmup_frac)
            self.target_depth = start + (self.final_target - start) * t
        return self.target_depth
