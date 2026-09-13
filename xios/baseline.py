"""A conventional decoder-only transformer, for the A/B that decides whether
any of this was worth it.

Deliberately ordinary: RoPE, full causal attention, GQA, SwiGLU, RMSNorm,
pre-norm residuals -- the same recipe as every strong small model. The point
is not to make a weak opponent. ``matched_baseline`` sizes it to the *same
parameter count* as a given XIOS config so the comparison isolates the
architecture rather than the budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import XiosConfig
from .core.attention import build_rope, apply_rope
from .core.norm import RMSNorm


class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, hidden, eps, rope_theta):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.n_rep = n_heads // n_kv_heads
        self.rope_theta = rope_theta
        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)
        self.q = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.w_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self._rope = None

    def _rope_tab(self, T, device):
        if self._rope is None or self._rope[0] < T or self._rope[1].device != device:
            n = max(T, 1024)
            cos, sin = build_rope(self.head_dim, n, self.rope_theta, device)
            self._rope = (n, cos, sin)
        return self._rope[1][:T], self._rope[2][:T]

    def forward(self, x, cache=None, return_state=False):
        B, T, _ = x.shape
        off = cache["pos"] if cache else 0
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope_tab(off + T, x.device)
        q = apply_rope(q, cos[off:off + T], sin[off:off + T])
        k = apply_rope(k, cos[off:off + T], sin[off:off + T])

        if cache is not None and cache.get("k") is not None:
            k = torch.cat([cache["k"], k], 2)
            v = torch.cat([cache["v"], v], 2)
        state = {"k": k, "v": v, "pos": off + T} if return_state else None

        def rep(t):
            if self.n_rep == 1:
                return t
            b, hh, tt, dd = t.shape
            return t[:, :, None].expand(b, hh, self.n_rep, tt, dd).reshape(b, hh * self.n_rep, tt, dd)

        y = F.scaled_dot_product_attention(q, rep(k), rep(v), is_causal=(T > 1))
        x = x + self.o(y.transpose(1, 2).reshape(B, T, -1))

        h = self.norm2(x)
        a, b = self.w_in(h).chunk(2, -1)
        return x + self.w_out(F.silu(a) * b), state


class BaselineTransformer(nn.Module):
    def __init__(self, dim=768, n_layers=12, head_dim=64, n_kv_heads=4,
                 vocab_size=32768, ffn_mult=2.6667, norm_eps=1e-5,
                 rope_theta=500000.0, tie_embeddings=True, max_seq_len=4096):
        super().__init__()
        n_heads = dim // head_dim
        hidden = int(ffn_mult * dim / 64 + 0.5) * 64
        self.dim, self.n_layers, self.vocab_size = dim, n_layers, vocab_size
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, head_dim, hidden, norm_eps, rope_theta)
            for _ in range(n_layers)])
        self.norm_f = RMSNorm(dim, norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.grad_checkpoint = False

        self.apply(self._init)
        scale = (2 * n_layers) ** -0.5
        for n, p in self.named_parameters():
            if n.endswith(("o.weight", "w_out.weight")):
                with torch.no_grad():
                    p.mul_(scale)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def enable_gradient_checkpointing(self, enabled=True):
        self.grad_checkpoint = enabled

    def forward(self, input_ids, labels=None, **kw):
        x = self.embed(input_ids)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(blk, x, None, False,
                                                         use_reentrant=False)
            else:
                x, _ = blk(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        from .model import XiosOutput
        return XiosOutput(logits=logits, loss=loss, lm_loss=loss,
                          budget_loss=torch.zeros((), device=logits.device))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=0.8,
                 top_p=0.95, top_k=0, eos_id=None, **kw):
        from .model import _sample
        self.eval()
        states = [None] * len(self.blocks)
        x = self.embed(input_ids)
        for i, blk in enumerate(self.blocks):
            x, states[i] = blk(x, states[i], return_state=True)
        logits = self.lm_head(self.norm_f(x[:, -1:]))
        out = input_ids
        for _ in range(max_new_tokens):
            nxt = _sample(logits[:, -1], temperature, top_p, top_k)
            out = torch.cat([out, nxt], 1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            x = self.embed(nxt)
            for i, blk in enumerate(self.blocks):
                x, states[i] = blk(x, states[i], return_state=True)
            logits = self.lm_head(self.norm_f(x))
        return out

    @property
    def n_params(self):
        seen, tot = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                tot += p.numel()
        return tot


def matched_baseline(cfg: XiosConfig, tol: float = 0.02) -> BaselineTransformer:
    """Build the deepest standard transformer with the same parameter count.

    Depth is the fair knob to spend the matched budget on: it is the thing
    XIOS claims to get for free, so the baseline is given as much real depth
    as the budget buys.

    Matching is on *resident* parameters.  If the XIOS config has its disk
    memory enabled, the comparison is no longer apples-to-apples -- XIOS
    would be carrying a knowledge store the baseline has no equivalent of --
    so the A/B experiments deliberately run with memory disabled and test the
    depth mechanism alone.
    """
    if cfg.memory_enabled:
        import warnings
        warnings.warn(
            "matched_baseline: cfg.memory_enabled=True. The baseline has no "
            "equivalent knowledge store, so this comparison is not clean. "
            "Disable memory to isolate the adaptive-depth mechanism.",
            stacklevel=2)
    target = XiosChatParams(cfg)
    best = None
    for n_layers in range(2, 129):
        m = BaselineTransformer(
            dim=cfg.dim, n_layers=n_layers, head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads, vocab_size=cfg.vocab_size,
            ffn_mult=cfg.ffn_mult, norm_eps=cfg.norm_eps,
            rope_theta=cfg.rope_theta, tie_embeddings=cfg.tie_embeddings,
            max_seq_len=cfg.max_seq_len)
        d = abs(m.n_params - target) / target
        if best is None or d < best[0]:
            best = (d, m)
        if m.n_params > target:
            break
    return best[1]


def XiosChatParams(cfg: XiosConfig) -> int:
    from .model import XiosChat
    return XiosChat(cfg).n_params


def matched_flops_baseline(cfg: XiosConfig, mean_depth: float, seq_len: int
                           ) -> BaselineTransformer:
    """Baseline sized to the same FLOPs/token as XIOS at a given mean depth.

    Matched-parameter is the comparison XIOS wants; matched-FLOP is the one
    that can sink it. A looped core spends more compute per parameter by
    construction, so a parameter-matched baseline is handed a compute
    disadvantage -- measured at 1.31x on the first sound run. Any accuracy
    win has to survive this comparison too, or it is just a win bought with
    extra arithmetic that the baseline could also have spent.
    """
    from .flops import xios_flops_per_token, baseline_flops_per_token

    target = xios_flops_per_token(cfg, seq_len, mean_depth)
    best = None
    for n_layers in range(1, 257):
        m = BaselineTransformer(
            dim=cfg.dim, n_layers=n_layers, head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads, vocab_size=cfg.vocab_size,
            ffn_mult=cfg.ffn_mult, norm_eps=cfg.norm_eps,
            rope_theta=cfg.rope_theta, tie_embeddings=cfg.tie_embeddings,
            max_seq_len=cfg.max_seq_len)
        f = baseline_flops_per_token(m, seq_len)
        d = abs(f - target) / target
        if best is None or d < best[0]:
            best = (d, m)
        if f > target:
            break
    return best[1]
