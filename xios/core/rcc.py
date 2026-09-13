"""Recurrent Compute Core -- the looped, weight-shared reasoning stack.

One stack of ``n_blocks`` blocks is applied up to ``max_iters`` times.  Two
properties make this different from a Universal Transformer or a plain
looped decoder:

**Per-depth memory timelines.**  The recurrent mixers keep a separate state
for every iteration index.  State ``S[i]`` is the memory of "everything that
was still being thought about at depth i".  Because tokens drop out as depth
increases, ``S[0]`` sees the full token stream while ``S[12]`` sees only the
handful of tokens that were hard enough to reach depth 12.  Deep computation
therefore runs over an automatically compressed, salient history -- context
stays full at the bottom of the stack and compute falls off geometrically
towards the top.  Depth selection and history compression become the same
mechanism.

**Halting gates writes, not outputs.**  See the comment in ``forward``: the
halt probability scales how much each iteration may write to the residual
stream, rather than blending every iteration's hidden state into the output.
Measured, the blend costs real accuracy, because it mixes under-computed
states into a finished answer.

**Thinking persists.**  Those states are carried across *generation steps*,
not just within a token.  When token 40 iterates twelve times, its twelve
iterations write into twelve state timelines that token 41 then reads.
Deliberation accumulates into the sequence instead of being discarded at the
token boundary -- which is what a chain of thought buys in token space, here
bought in latent space at no token cost.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .block import XiosBlock
from .ponder import PonderController, PonderStats


def iteration_embedding(i: int, dim: int, device) -> torch.Tensor:
    """Sinusoidal embedding of the loop index."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / half)
    args = torch.tensor(float(i), device=device) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)])
    if dim % 2:
        emb = torch.cat([emb, emb.new_zeros(1)])
    return emb


class RecurrentComputeCore(nn.Module):
    def __init__(self, cfg, ffn_factory=None):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            XiosBlock(cfg, i, ffn=ffn_factory(i) if ffn_factory else None)
            for i in range(cfg.core_blocks)])
        self.iter_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim * 2), nn.SiLU(),
            nn.Linear(cfg.cond_dim * 2, cfg.cond_dim))
        self.ponder = PonderController(
            cfg.dim, target_depth=cfg.target_depth, max_iters=cfg.max_iters,
            min_iters=cfg.min_iters, eps=cfg.ponder_eps,
            budget_weight=cfg.budget_weight,
            budget_mode=getattr(cfg, "budget_mode", "lagrangian"),
            dual_lr=getattr(cfg, "dual_lr", 0.002),
            lambda_max=getattr(cfg, "lambda_max", 1.0))
        # Re-injecting the pre-loop representation at every iteration stops
        # the loop from drifting away from the actual input, which is the
        # standard failure mode of deeply looped stacks. Zero-init, so at the
        # start of training the loop is exactly an unrolled residual stack.
        self.inject = nn.Linear(cfg.dim, cfg.dim, bias=False)
        nn.init.zeros_(self.inject.weight)
        self.grad_checkpoint = False

    # ------------------------------------------------------------------
    def _cond(self, i: int, device, dtype, B: int):
        e = iteration_embedding(i, self.cfg.cond_dim, device).to(dtype)
        return self.iter_mlp(e)[None].expand(B, -1)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None,
                states: Optional[list] = None, return_states: bool = False,
                loss_mask: Optional[torch.Tensor] = None,
                max_iters: Optional[int] = None):
        """x: (B, T, D) post-prelude hidden states."""
        B, T, D = x.shape
        device = x.device
        N = max_iters or self.cfg.max_iters
        if pos is None:
            pos = torch.arange(T, device=device)

        x0 = x
        h = x
        R = x.new_ones(B, T)                       # prob still running
        expected_depth = x.new_zeros(B, T)         # sum of R: the SOFT depth
        compute_depth = x.new_zeros(B, T)          # differentiable count of
                                                   # iterations actually run
        actual_depth = torch.zeros(B, T, dtype=torch.long, device=device)
        active = torch.ones(B, T, dtype=torch.bool, device=device)

        new_states: list = [] if return_states else None
        halt_log, active_log = [], []

        for i in range(N):
            if not bool(active.any()):
                if return_states:
                    new_states.append(None)
                continue

            # How much compute this iteration really costs.
            #
            # `sum_i R_i` is the expected depth if halting were *sampled*. It
            # is not what we execute: we run deterministically until R falls
            # below eps, which for a geometric halting process takes about
            # ln(eps)/ln(1-p) iterations -- roughly 4x larger than sum(R) at
            # eps=0.02. Budgeting sum(R) therefore constrains a number that
            # is not the compute: measured, mean depth sat at 5.94 of a
            # 6-iteration ceiling while the Lagrange multiplier stayed at
            # exactly 0, because the quantity it was watching was already
            # under target. The budget has to price what is actually spent.
            #
            # This is a smooth count of "iterations where R was still above
            # the cutoff", which tracks the real iteration count and stays
            # differentiable in R, hence in the halting probabilities.
            tau = max(self.ponder.eps * 0.5, 1e-4)
            compute_depth = compute_depth + torch.sigmoid(
                (R - self.ponder.eps) / tau) * active.float()

            cond = self._cond(i, device, x.dtype, B)
            # Plain residual update. An earlier version wrote
            # `RMSNorm(h) + inject(x0)` here, renormalising the residual
            # stream on every iteration -- which throws away accumulated
            # magnitude and, measured on a trivial copy task, cost XIOS
            # 64.8% against a dense baseline's 100%. The blocks are pre-norm
            # and normalise their own inputs, so an extra norm on the stream
            # is not just redundant, it destroys information the loop needs
            # to carry.
            hi = h + self.inject(x0)

            blk_states = states[i] if (states and i < len(states) and states[i]) else None
            out_states = [] if return_states else None
            for bi, blk in enumerate(self.blocks):
                st = blk_states[bi] if blk_states else None
                if self.grad_checkpoint and self.training:
                    hi, s, _ = torch.utils.checkpoint.checkpoint(
                        blk, hi, cond, st, return_states, None, pos, active,
                        use_reentrant=False)
                else:
                    hi, s, _ = blk(hi, cond, st, return_state=return_states,
                                   pos=pos, active=active)
                if return_states:
                    out_states.append(s)
            if return_states:
                new_states.append(out_states)

            # Update-gating, not output-mixing.
            #
            # Textbook ACT accumulates `y = sum_i R_i p_i h_i`, i.e. it blends
            # the hidden states from *every* iteration into the output. That
            # blend necessarily contains under-computed intermediate states,
            # which blurs a crisp answer: measured on a trivial copy task,
            # output-mixing cost XIOS 84% against a dense baseline's 100%.
            #
            # Instead the halting probability gates how much each iteration is
            # allowed to *write* to the residual stream. A token that has
            # halted stops updating and its state is simply what it had
            # reached, never a smear across depths. Still fully
            # differentiable -- gradient reaches the halt head through the
            # gate that scales each write -- and in the hard limit it is
            # exactly "stop computing this token".
            af = active.float()
            gate = (R * af).unsqueeze(-1)
            h = h + gate * (hi - h)

            p = self.ponder.halt_prob(h)
            if i == N - 1:
                p = torch.ones_like(p)             # forced halt on the last pass
            elif i < self.ponder.min_iters - 1:
                p = torch.zeros_like(p)

            expected_depth = expected_depth + R * af
            actual_depth = actual_depth + active.long()
            R = R * (1 - p * af)

            halt_log.append(float(p.detach().mean()))
            active_log.append(float(af.detach().mean()))
            active = active & (R > self.ponder.eps)

        y = h

        stats = PonderStats(
            expected_depth=expected_depth, actual_depth=actual_depth,
            compute_depth=compute_depth,
            budget_loss=self.ponder.budget_loss(compute_depth, loss_mask),
            halt_probs=halt_log, active_frac=active_log, mask=loss_mask)
        return y, new_states, stats

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, x: torch.Tensor, pos: torch.Tensor, states: list,
             max_iters: Optional[int] = None, deltas: Optional[list] = None):
        """Single-token decode. Loops only as long as this token needs.

        ``states[i]`` is the per-depth state timeline; ``deltas[i]`` is how
        many real tokens have passed since depth ``i`` last ran, which keeps
        the recurrent decay calibrated in true token units even though the
        deep timelines are sparse.
        """
        B = x.shape[0]
        N = max_iters or self.cfg.max_iters
        device = x.device
        x0, h = x, x
        R = x.new_ones(B, 1)
        depth = 0

        for i in range(N):
            cond = self._cond(i, device, x.dtype, B)
            hi = h + self.inject(x0)
            while len(states) <= i:
                states.append(None)
            if states[i] is None:
                states[i] = [None] * len(self.blocks)
            d = deltas[i] if deltas is not None else None
            for bi, blk in enumerate(self.blocks):
                hi, s = blk.step(hi, cond, states[i][bi], delta=d, pos=pos)
                states[i][bi] = s
            h = h + R.unsqueeze(-1) * (hi - h)
            depth = i + 1

            p = self.ponder.halt_prob(h)
            if i == N - 1:
                p = torch.ones_like(p)
            elif i < self.ponder.min_iters - 1:
                p = torch.zeros_like(p)
            R = R * (1 - p)
            if float(R.max()) < self.ponder.eps:
                break

        return h, states, depth
