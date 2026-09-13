"""XiosChat -- the full chat model.

    tokens -> embed -> prelude (dense, once)
                    -> RecurrentComputeCore (shared weights, 1..N iterations)
                    -> coda (dense, once) -> logits

The prelude and coda are ordinary blocks.  Everything interesting happens in
the core, which is looped a token-dependent number of times under a global
compute budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import XiosConfig
from .core.block import XiosBlock
from .core.norm import RMSNorm
from .core.rcc import RecurrentComputeCore


@dataclass
class XiosOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    budget_loss: Optional[torch.Tensor] = None
    stats: object = None
    states: Optional[list] = None


class XiosChat(nn.Module):
    def __init__(self, cfg: XiosConfig, lazy_memory: bool = False):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.prelude = nn.ModuleList([XiosBlock(cfg, i) for i in range(cfg.prelude_blocks)])

        ffn_factory = None
        if cfg.memory_enabled:
            from .memory.dam import DiskAssociativeMemory

            def ffn_factory(i: int):
                if i in cfg.memory_layers:
                    return DiskAssociativeMemory(cfg, lazy=lazy_memory)
                return None

        self.core = RecurrentComputeCore(cfg, ffn_factory=ffn_factory)

        off = cfg.prelude_blocks + cfg.core_blocks
        self.coda = nn.ModuleList([XiosBlock(cfg, off + i) for i in range(cfg.coda_blocks)])

        self.norm_f = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init)
        self._depth_scale()

    # ------------------------------------------------------------------
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _depth_scale(self):
        """Residual-output downscaling.

        A looped core is applied up to ``max_iters`` times, so its residual
        writes compound far more than a plain stack's.  Scaling by the
        *effective* depth rather than the stored depth is what keeps the loop
        stable at high iteration counts without any warmup schedule.
        """
        eff = max(self.cfg.effective_depth_max, 1)
        scale = (2 * eff) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("mixer.out.weight", "mixer.o.weight", "ffn.w_out.weight")):
                with torch.no_grad():
                    p.mul_(scale)
        # zero-init the AdaLN projections after the generic init clobbered them
        for m in self.modules():
            from .core.norm import AdaLNZero
            if isinstance(m, AdaLNZero):
                nn.init.zeros_(m.proj.weight)
                nn.init.zeros_(m.proj.bias)
        nn.init.zeros_(self.core.inject.weight)
        nn.init.zeros_(self.core.ponder.halt[-1].weight)
        nn.init.constant_(self.core.ponder.halt[-1].bias, -3.0)

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.core.grad_checkpoint = enabled

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None, max_iters: Optional[int] = None,
                loss_mask: Optional[torch.Tensor] = None) -> XiosOutput:
        B, T = input_ids.shape
        device = input_ids.device
        if pos is None:
            pos = torch.arange(T, device=device)

        x = self.embed(input_ids)
        for blk in self.prelude:
            x, _, _ = blk(x, None, None, pos=pos)

        x, _, stats = self.core(x, pos=pos, loss_mask=loss_mask, max_iters=max_iters)

        for blk in self.coda:
            x, _, _ = blk(x, None, None, pos=pos)

        logits = self.lm_head(self.norm_f(x))

        loss = lm_loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = lm_loss + stats.budget_loss

        return XiosOutput(logits=logits, loss=loss, lm_loss=lm_loss,
                          budget_loss=stats.budget_loss, stats=stats)

    # ------------------------------------------------------------------
    # Stateful incremental runtime
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, max_iters: Optional[int] = None):
        """Run a prompt in parallel and return (last_logits, state)."""
        device = input_ids.device
        T0 = input_ids.shape[1]
        pos = torch.arange(T0, device=device)

        x = self.embed(input_ids)
        pre_states = []
        for blk in self.prelude:
            x, s, _ = blk(x, None, None, return_state=True, pos=pos)
            pre_states.append(s)

        x, core_states, stats = self.core(x, pos=pos, return_states=True, max_iters=max_iters)

        coda_states = []
        for blk in self.coda:
            x, s, _ = blk(x, None, None, return_state=True, pos=pos)
            coda_states.append(s)

        # For every depth level, the last real position that actually reached
        # it.  Deep timelines are sparse, so decode needs this to know how big
        # a gap to replay.
        reached = stats.actual_depth[0]                       # (T,)
        last_seen = []
        for i in range(self.cfg.max_iters):
            hit = (reached > i).nonzero(as_tuple=True)[0]
            last_seen.append(int(hit[-1]) if hit.numel() else -1)

        state = {"pre": pre_states, "core": core_states, "coda": coda_states,
                 "last_seen": last_seen, "pos": T0}
        return self.lm_head(self.norm_f(x[:, -1:])), state

    @torch.no_grad()
    def decode_step(self, token: torch.Tensor, state: dict,
                    max_iters: Optional[int] = None):
        """Advance one token. Returns (logits, state, depth_used)."""
        device = token.device
        B = token.shape[0]
        p = torch.tensor([state["pos"]], device=device)

        x = self.embed(token)
        for i, blk in enumerate(self.prelude):
            x, state["pre"][i] = blk.step(x, None, state["pre"][i], pos=p)

        ls = state["last_seen"]
        deltas = [torch.full((B, 1), max(1, int(p) - ls[i]), device=device,
                             dtype=torch.long)
                  for i in range(self.cfg.max_iters)]

        x, state["core"], depth = self.core.step(
            x, p, state["core"], max_iters=max_iters, deltas=deltas)
        for i in range(depth):
            ls[i] = int(p)

        for i, blk in enumerate(self.coda):
            x, state["coda"][i] = blk.step(x, None, state["coda"][i], pos=p)

        state["pos"] += 1
        return self.lm_head(self.norm_f(x)), state, depth

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.8, top_p: float = 0.95, top_k: int = 0,
                 max_iters: Optional[int] = None, eos_id: Optional[int] = None,
                 return_depths: bool = False):
        """Incremental decode.

        Per-token cost is constant in context length -- the recurrent state is
        fixed size and the attention window is bounded -- and varies only with
        how hard the token turns out to be.
        """
        self.eval()
        logits, state = self.prefill(input_ids, max_iters=max_iters)
        out, depths = input_ids, []

        for _ in range(max_new_tokens):
            nxt = _sample(logits[:, -1], temperature, top_p, top_k)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            logits, state, d = self.decode_step(nxt, state, max_iters=max_iters)
            depths.append(d)

        return (out, depths) if return_depths else out

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Size accounting.  The distinction matters: the associative memory's
    # value slots are *disk* bytes that are never resident, so folding them
    # into a parameter count would misstate what the model costs to run.
    # ------------------------------------------------------------------
    def _memory_modules(self):
        from .memory.dam import DiskAssociativeMemory
        return [m for m in self.modules() if isinstance(m, DiskAssociativeMemory)]

    @property
    def n_params(self) -> int:
        """Resident parameters: everything that must be in RAM to run."""
        skip = set()
        for mem in self._memory_modules():
            if mem.values is not None:
                skip.add(id(mem.values.weight))
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in skip or id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    @property
    def n_memory_slots(self) -> int:
        return sum(m.n_slots for m in self._memory_modules())

    @property
    def memory_store_bytes(self) -> int:
        """Disk footprint of the associative memory (int8 values + scales)."""
        return sum(m.store_bytes for m in self._memory_modules())

    def param_report(self) -> dict:
        mem_ids = set()
        for mem in self._memory_modules():
            if mem.values is not None:
                mem_ids.add(id(mem.values.weight))

        def count(mod):
            return sum(p.numel() for p in mod.parameters() if id(p) not in mem_ids)

        return {
            "embed": self.embed.weight.numel(),
            "prelude": count(self.prelude),
            "core": count(self.core),
            "coda": count(self.coda),
            "head": 0 if self.cfg.tie_embeddings else self.lm_head.weight.numel(),
            "total": self.n_params,
            "memory_slots": self.n_memory_slots,
            "memory_disk_bytes": self.memory_store_bytes,
        }


def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0 < top_p < 1:
        srt, idx = logits.sort(-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        drop = cum - srt.softmax(-1) > top_p
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)
    return torch.multinomial(logits.softmax(-1), 1)
