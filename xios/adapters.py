"""Wrap a real pretrained model so it can be collapsed into a looped core.

`train/collapse.py` needs four things from a dense stack, and nothing else:

    embed(ids) -> h          the input embedding, positions included
    layers[i](h) -> h        each layer as a callable on the residual stream
    final_norm(h) -> h       whatever normalisation precedes the head
    head(h) -> logits        the output projection

Every decoder-only transformer exposes these; they are just named differently
in each codebase. This module provides the mapping for HuggingFace causal LMs
so the collapse machinery can point at a genuinely pretrained model -- one that
actually knows things -- instead of only at models trained in this repo.

The architectural mismatch is the point, not a problem. GPT-2 uses learned
positional embeddings, LayerNorm and a GELU MLP; the XIOS core uses RoPE,
RMSNorm, SwiGLU and gated linear recurrences. Collapse never assumes the two
agree internally -- it only fits the *function* each group of layers computes
on the residual stream. If a different mechanism can reproduce that function,
the difference in mechanism is irrelevant. That is what makes this a general
absorption procedure rather than a weight-copying trick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class DenseSpec:
    """A dense stack, reduced to what collapse actually needs."""
    embed: Callable[[torch.Tensor], torch.Tensor]
    layers: list[Callable[[torch.Tensor], torch.Tensor]]
    final_norm: Callable[[torch.Tensor], torch.Tensor]
    head: Callable[[torch.Tensor], torch.Tensor]
    dim: int
    vocab_size: int
    n_layers: int
    name: str = "dense"
    module: Optional[nn.Module] = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        for f in self.layers:
            h = f(h)
        return self.head(self.final_norm(h))

    @torch.no_grad()
    def waypoints(self, ids: torch.Tensor, k: int) -> list[torch.Tensor]:
        """Residual stream at every k-th layer boundary."""
        h = self.embed(ids)
        pts = [h]
        for i, f in enumerate(self.layers):
            h = f(h)
            if (i + 1) % k == 0:
                pts.append(h)
        return pts

    @property
    def body_params(self) -> int:
        if self.module is None:
            return 0
        seen, tot = set(), 0
        for blk in getattr(self.module, "_xios_blocks", []):
            for p in blk.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); tot += p.numel()
        return tot


# ---------------------------------------------------------------------------
def from_huggingface(model, name: str = "hf") -> DenseSpec:
    """Adapt a HuggingFace causal LM (GPT-2, Llama, Pythia, Qwen, ...)."""
    cfg = model.config
    model.eval()

    # locate the transformer trunk under the LM wrapper
    trunk = None
    for attr in ("transformer", "model", "gpt_neox", "backbone"):
        if hasattr(model, attr):
            trunk = getattr(model, attr)
            break
    if trunk is None:
        raise ValueError(f"cannot find the trunk of {type(model).__name__}")

    blocks = None
    for attr in ("h", "layers", "blocks", "layer"):
        if hasattr(trunk, attr):
            cand = getattr(trunk, attr)
            if isinstance(cand, (nn.ModuleList, list)):
                blocks = cand
                break
    if blocks is None:
        raise ValueError(f"cannot find the layer list of {type(trunk).__name__}")

    wte = getattr(trunk, "wte", None) or getattr(trunk, "embed_tokens", None) \
        or getattr(trunk, "embed_in", None)
    wpe = getattr(trunk, "wpe", None)            # GPT-2 style learned positions
    final_norm = getattr(trunk, "ln_f", None) or getattr(trunk, "norm", None) \
        or getattr(trunk, "final_layer_norm", None)
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    if wte is None or final_norm is None or head is None:
        raise ValueError("missing embed / final norm / head")

    def embed(ids: torch.Tensor) -> torch.Tensor:
        h = wte(ids)
        if wpe is not None:
            pos = torch.arange(ids.shape[1], device=ids.device)
            h = h + wpe(pos)[None]
        return h

    def make_layer(blk):
        def f(h: torch.Tensor) -> torch.Tensor:
            # HF blocks return tuples and take assorted kwargs; different
            # versions disagree about which are required, so try the common
            # shapes rather than pinning one transformers release.
            for kwargs in ({}, {"attention_mask": None},
                           {"attention_mask": None, "position_ids": None}):
                try:
                    out = blk(h, **kwargs)
                    return out[0] if isinstance(out, tuple) else out
                except TypeError:
                    continue
            raise RuntimeError(f"could not call {type(blk).__name__}")
        return f

    spec = DenseSpec(
        embed=embed,
        layers=[make_layer(b) for b in blocks],
        final_norm=final_norm,
        head=head,
        dim=int(getattr(cfg, "n_embd", None) or cfg.hidden_size),
        vocab_size=int(cfg.vocab_size),
        n_layers=len(blocks),
        name=name,
        module=model,
    )
    model._xios_blocks = list(blocks)
    return spec


def from_baseline(dense) -> DenseSpec:
    """Adapt this repo's own BaselineTransformer."""
    def make_layer(blk):
        def f(h):
            out, _ = blk(h)
            return out
        return f
    spec = DenseSpec(
        embed=dense.embed,
        layers=[make_layer(b) for b in dense.blocks],
        final_norm=dense.norm_f,
        head=dense.lm_head,
        dim=dense.dim,
        vocab_size=dense.vocab_size,
        n_layers=dense.n_layers,
        name="baseline",
        module=dense,
    )
    dense._xios_blocks = list(dense.blocks)
    return spec
