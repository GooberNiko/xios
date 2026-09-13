"""Why doesn't the disk memory carry information?

Experiment 7 showed learned values beat frozen random values by only 1.008x.
That is not a scale problem — it says the model's output barely depends on what
is *in* the store. Before changing anything, find out which link in the chain
is broken. Four candidates, each directly measurable:

1. **Gradient starvation.** `out_proj` is zero-initialised for stability, so at
   step 0 the gradient reaching the value table is *exactly zero* (it is scaled
   by `out_proj`). If `out_proj` grows slowly, the values barely move and end up
   indistinguishable from their init — which is precisely the symptom.

2. **Averaging washout.** The output is a softmax-weighted mean of `topk=32`
   values. If those weights are near-uniform, the output is ~the mean of 32
   random vectors, which is nearly constant and carries almost no information
   about *which* slots were chosen. Then the layer degenerates into a smooth
   function of the query — exactly what a random projection gives you, and
   exactly why shuffling the values would not matter.

3. **Content-independent routing.** If retrieval does not depend on the input,
   there is no association to learn. Near-uniform slot usage (measured: entropy
   0.73) is consistent with this, and I previously mis-read it as healthy load
   balancing.

4. **Dead store.** Directly: shuffle the value table and see whether the output
   changes at all. This is the decisive test — if a trained model is indifferent
   to its own memory contents, the mechanism is doing nothing regardless of why.

Measuring all four costs a couple of minutes and tells us which fix is worth
trying, instead of a fourth blind attempt.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

torch.set_num_threads(12)

from xios.config import get_config
from xios.model import XiosChat
from xios.memory.dam import DiskAssociativeMemory
from xios.tokenizer import ByteBPE
from train.train_text import load_texts, build_stream


def mems(m):
    return [x for x in m.modules() if isinstance(x, DiskAssociativeMemory)]


def build(a, vocab):
    cfg = get_config("nano", vocab_size=vocab, max_seq_len=a.seq_len,
                     dim=a.dim, core_blocks=a.core_blocks, n_kv_heads=2,
                     max_iters=a.max_iters, target_depth=2.0,
                     attn_window=min(256, a.seq_len), memory_enabled=True,
                     memory_layers=tuple(range(a.core_blocks)),
                     memory_slots=a.heads * a.grid * a.grid,
                     memory_value_dim=a.value_dim, memory_topk=a.topk,
                     memory_heads=a.heads)
    return XiosChat(cfg)


@torch.no_grad()
def probe_weights(m, ids):
    """Candidate 2 & 3: how peaked are the retrieval weights, and does the
    selected slot set actually depend on the input?

    The queries have to come from real hidden states, not random vectors, or
    the routing statistics mean nothing. One forward pass with a hook on every
    memory layer captures all of them at once.
    """
    layers = mems(m)
    captured: dict[int, torch.Tensor] = {}
    hooks = [mod.register_forward_pre_hook(
        lambda _mod, inp, _i=li: captured.__setitem__(_i, inp[0]))
        for li, mod in enumerate(layers)]
    m(ids)
    for hk in hooks:
        hk.remove()

    out = {}
    for li, mod in enumerate(layers):
        hin = captured[li].reshape(-1, mod.cfg.dim)
        q = mod.normalise_query(hin)
        idx, w = mod._lookup_indices(q)          # (N,H,k)
        ent = float(-(w * w.clamp(min=1e-12).log()).sum(-1).mean())
        max_w = float(w.max(-1).values.mean())
        # does selection depend on the input? compare slot sets for different
        # tokens: high overlap => routing is nearly content-independent
        flat = idx.reshape(idx.shape[0], -1)
        n = min(64, flat.shape[0])
        ov = []
        for i in range(n):
            for j in range(i + 1, n):
                a_, b_ = set(flat[i].tolist()), set(flat[j].tolist())
                ov.append(len(a_ & b_) / len(a_))
        out[f"layer{li}"] = {
            "weight_entropy": ent,
            "max_entropy": math.log(w.shape[-1]),
            "mean_max_weight": max_w,
            "slot_overlap_between_tokens": sum(ov) / max(len(ov), 1),
        }
    return out


def probe_gradients(m, ids, labels):
    """Candidate 1: does any gradient reach the value table?"""
    m.zero_grad(set_to_none=True)
    out = m(ids, labels=labels)
    out.loss.backward()
    res = {}
    for li, mod in enumerate(mems(m)):
        g = mod.values.weight.grad
        gn = 0.0
        if g is not None:
            gn = float(g.coalesce().values().norm()) if g.is_sparse else float(g.norm())
        res[f"layer{li}"] = {
            "value_grad_norm": gn,
            "out_proj_weight_norm": float(mod.out_proj.weight.norm()),
            "value_weight_norm": float(mod.values.weight.norm()),
        }
    m.zero_grad(set_to_none=True)
    return res


@torch.no_grad()
def probe_shuffle(m, ids):
    """Candidate 4: is the model sensitive to its own memory contents?"""
    base = m(ids).logits.float()
    res = {}
    for li, mod in enumerate(mems(m)):
        w = mod.values.weight.data
        keep = w.clone()
        perm = torch.randperm(w.shape[0])
        w.copy_(w[perm])
        shuf = m(ids).logits.float()
        w.copy_(keep)
        w.zero_()
        zl = m(ids).logits.float()
        w.copy_(keep)
        res[f"layer{li}"] = {
            "rel_change_shuffled": float((shuf - base).norm() / base.norm()),
            "rel_change_zeroed": float((zl - base).norm() / base.norm()),
            "top1_change_shuffled": float(
                (shuf.argmax(-1) != base.argmax(-1)).float().mean()),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--core-blocks", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--value-dim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", default="runs/memq")
    a = ap.parse_args()

    texts = load_texts("shakespeare", 20000)
    tp = pathlib.Path(a.out) / "tokenizer.json"
    tok = ByteBPE.load(tp) if tp.exists() else ByteBPE(vocab_size=a.vocab).train(texts[:4000])
    stream = build_stream(texts, tok)
    g = torch.Generator().manual_seed(0)

    def batch():
        ix = torch.randint(0, stream.numel() - a.seq_len - 1,
                           (a.batch_size,), generator=g)
        ids = torch.stack([stream[i:i + a.seq_len] for i in ix])
        return ids, ids.clone()

    torch.manual_seed(0)
    m = build(a, tok.actual_size)
    ids, labels = batch()

    print("=" * 70)
    print("AT INITIALISATION")
    print("=" * 70)
    for k, v in probe_gradients(m, ids, labels).items():
        print(f"  {k}: value grad norm {v['value_grad_norm']:.3e}   "
              f"out_proj norm {v['out_proj_weight_norm']:.3e}   "
              f"values norm {v['value_weight_norm']:.2f}")
    print("  -> if the gradient is 0, the values cannot learn (candidate 1)")

    from train.train import build_optimizer, TrainArgs, lr_at
    targs = TrainArgs(steps=a.steps, lr=a.lr, device="cpu", amp=False)
    opt, sopt = build_optimizer(m, targs)
    m.train()
    for step in range(a.steps):
        lr = lr_at(step, targs)
        for gg in opt.param_groups:
            gg["lr"] = lr
        if sopt:
            for gg in sopt.param_groups:
                gg["lr"] = lr
        i2, l2 = batch()
        o = m(i2, labels=l2)
        opt.zero_grad(set_to_none=True)
        if sopt:
            sopt.zero_grad(set_to_none=True)
        o.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if sopt:
            sopt.step()
        if step % 100 == 0:
            print(f"    step {step:4d} loss {float(o.lm_loss):.4f}", flush=True)

    print()
    print("=" * 70)
    print(f"AFTER {a.steps} STEPS")
    print("=" * 70)
    m.eval()
    for k, v in probe_gradients(m, ids, labels).items():
        print(f"  {k}: value grad norm {v['value_grad_norm']:.3e}   "
              f"out_proj norm {v['out_proj_weight_norm']:.3e}   "
              f"values norm {v['value_weight_norm']:.2f}")
    print()
    for k, v in probe_weights(m, ids).items():
        print(f"  {k}: retrieval-weight entropy {v['weight_entropy']:.3f} "
              f"of max {v['max_entropy']:.3f}  "
              f"(mean top weight {v['mean_max_weight']:.3f})")
        print(f"          slot overlap between different tokens: "
              f"{v['slot_overlap_between_tokens']:.1%}")
    print("  -> entropy near max = averaging washout (candidate 2)")
    print("  -> high overlap = content-independent routing (candidate 3)")
    print()
    for k, v in probe_shuffle(m, ids).items():
        print(f"  {k}: shuffling the value table changes logits by "
              f"{v['rel_change_shuffled']:.4f} rel "
              f"({v['top1_change_shuffled']:.1%} of argmaxes)")
        print(f"          zeroing the value table changes logits by "
              f"{v['rel_change_zeroed']:.4f} rel")
    print("  -> near 0 = the model ignores its own memory (candidate 4)")


if __name__ == "__main__":
    main()
