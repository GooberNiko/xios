"""Whole-model quantisation: calibrate, swap, report."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .int4 import QuantLinear, QuantEmbedding, TiedQuantHead


@torch.no_grad()
def collect_act_scales(model: nn.Module, batches: Iterable, forward_fn=None,
                       device="cpu") -> dict[str, torch.Tensor]:
    """Record mean |activation| per input channel for every nn.Linear.

    A handful of batches (8-32) from the real data distribution is enough;
    this is the 'activation-aware' part of AWQ.
    """
    scales: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).abs().mean(0).float().cpu()
            scales[name] = x if name not in scales else torch.maximum(scales[name], x)
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    for batch in batches:
        if forward_fn is not None:
            forward_fn(model, batch)
        else:
            model(batch.to(device) if torch.is_tensor(batch) else batch)
    for h in hooks:
        h.remove()
    model.train(was_training)
    return scales


def quantize_model(model: nn.Module, group: int = 64, lora_rank: int = 0,
                   act_scales: Optional[dict] = None,
                   skip: Sequence[str] = ("lm_head", "router", "decay", "proj_out"),
                   min_features: int = 256,
                   quantize_embeddings: bool = True,
                   _root: Optional[nn.Module] = None) -> nn.Module:
    """Replace nn.Linear with QuantLinear in place.

    ``skip`` protects the layers where int4 measurably hurts: the output
    head (its errors are not averaged away by later layers), the tiny router
    and decay projections (quantising a d->1 or d->H matrix saves nothing and
    destabilises routing), and anything the caller names.
    """
    root = _root if _root is not None else model
    for name, child in list(model.named_children()):
        full = name
        if quantize_embeddings and isinstance(child, nn.Embedding)                 and child.num_embeddings >= 1024:
            qe = QuantEmbedding.from_embedding(child)
            setattr(model, name, qe)
            # Preserve a tied output projection by pointing it at the same
            # int8 table. Materialising a dequantised fp16 copy here would add
            # the whole table back -- 100MB on the `small` preset -- and
            # quietly undo most of the compression.
            head = getattr(root, "lm_head", None)
            if head is not None and isinstance(head, nn.Linear) \
                    and head.weight.data_ptr() == child.weight.data_ptr():
                setattr(root, "lm_head", TiedQuantHead(qe))
            continue
        if isinstance(child, nn.Linear):
            if any(s in full for s in skip):
                continue
            if min(child.in_features, child.out_features) < min_features:
                continue
            sc = None
            if act_scales:
                for k, v in act_scales.items():
                    if k.endswith(name) or k == name:
                        sc = v
                        break
            setattr(model, name, QuantLinear.from_linear(child, group, sc, lora_rank))
        else:
            quantize_model(child, group, lora_rank, act_scales, skip,
                           min_features, quantize_embeddings, _root=root)
    return model


def model_size_report(model: nn.Module, assume_fp16: bool = True) -> dict:
    """Deployment footprint.

    ``assume_fp16`` reports un-quantised tensors at 2 bytes, which is what
    they actually ship as; counting them at their current fp32 size
    understates the compression by a large factor at small scale.
    """
    fp, q = 0, 0
    for m in model.modules():
        if isinstance(m, (QuantLinear, QuantEmbedding)):
            q += m.nbytes
    def esize(t):
        return 2 if assume_fp16 else t.element_size()
    for n, p in model.named_parameters():
        fp += p.numel() * esize(p)
    for n, b in model.named_buffers():
        if not any(k in n for k in ("packed", "scale", "zero", "lora_", ".q")):
            fp += b.numel() * esize(b)
    total = fp + q
    return {"quantized_bytes": q, "other_bytes": fp, "total_bytes": total,
            "total_mb": total / 1e6, "total_gb": total / 1e9}
