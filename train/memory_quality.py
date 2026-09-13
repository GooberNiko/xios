"""Does the disk-resident memory actually buy quality?

This is the last unmeasured mechanism, and the crux of the whole design. The
bandwidth work showed that the looped core must stay inside L2/L3 to win, which
caps its *width*. Knowledge needs bits, and bits that cannot go in a narrow
core have to go somewhere: the product-key store on disk. If that works, width
stops being a constraint and the architecture closes. If it does not, the
`flagship` preset is a narrow reasoner with nowhere to put knowledge, and the
honest move is to say so.

Everything so far about the memory is *mechanical*: int8 export round-trips to
0.6%, the addressing is a verified bijection, page reads drop 1.83x. None of
that is evidence that a model can learn to use it.

Three variants, same data, same seed, same budget:

  dense    core FFNs as usual                      -- the control
  memory   core FFNs replaced by the product-key store
  frozen   same, but the value table is frozen at random init

The `frozen` arm is the one that makes this a real experiment rather than a
demo. A memory-equipped model could look fine simply because the surrounding
blocks compensate for a useless lookup; if `memory` beats `frozen`, the learned
values are carrying information, and the gap size says how much.

Reported per variant: validation perplexity, **resident** parameters (what must
be in RAM), **disk** parameters (what need not be), and slot utilisation --
because a store whose lookups collapse onto a few hundred slots has capacity on
paper only.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

torch.set_num_threads(12)

from xios.config import get_config
from xios.model import XiosChat
from xios.memory.dam import DiskAssociativeMemory
from xios.tokenizer import ByteBPE
from train.train import TrainArgs, train
from train.train_text import load_texts, build_stream


def build(variant: str, vocab: int, a) -> XiosChat:
    kw = dict(vocab_size=vocab, max_seq_len=a.seq_len, dim=a.dim,
              core_blocks=a.core_blocks, n_kv_heads=2, max_iters=a.max_iters,
              target_depth=a.target_depth, attn_window=min(256, a.seq_len))
    if variant == "dense":
        cfg = get_config("nano", memory_enabled=False, **kw)
    else:
        cfg = get_config("nano", memory_enabled=True,
                         memory_layers=tuple(range(a.core_blocks)),
                         memory_slots=a.heads * a.grid * a.grid,
                         memory_value_dim=a.value_dim,
                         memory_topk=a.topk, memory_heads=a.heads, **kw)
    m = XiosChat(cfg)
    if variant == "frozen":
        for mod in m.modules():
            if isinstance(mod, DiskAssociativeMemory):
                mod.values.weight.requires_grad_(False)
    for mod in m.modules():
        if isinstance(mod, DiskAssociativeMemory):
            mod.track_usage = True
    return m


def sizes(m: XiosChat) -> dict:
    mems = [x for x in m.modules() if isinstance(x, DiskAssociativeMemory)]
    disk_params = sum(x.n_slots * x.value_dim for x in mems)
    return {"resident_params": m.n_params, "disk_params": disk_params,
            "resident_mb_int4": m.n_params * 0.56 / 1e6,
            "disk_mb_int8": disk_params / 1e6,
            "n_memory_layers": len(mems),
            "slots": sum(x.n_slots for x in mems)}


def utilisation(m: XiosChat) -> dict:
    out = {}
    for i, mod in enumerate(x for x in m.modules()
                            if isinstance(x, DiskAssociativeMemory)):
        u = mod.usage
        if u is None:
            continue
        used = int((u > 0).sum())
        tot = u.numel()
        p = (u / u.sum().clamp(min=1e-9)).clamp(min=1e-12)
        ent = float(-(p * p.log()).sum() / math.log(tot))   # 1.0 = uniform
        out[f"layer{i}"] = {"slots_touched": used, "slots": tot,
                            "frac_touched": used / tot,
                            "usage_entropy": ent}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="dense,memory,frozen")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--core-blocks", type=int, default=2)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--target-depth", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=64, help="per-head code grid (m)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--value-dim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--corpus", default="shakespeare")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/memq")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    texts = load_texts(a.corpus, 20000)
    tok_path = out / "tokenizer.json"
    tok = ByteBPE.load(tok_path) if tok_path.exists() else \
        ByteBPE(vocab_size=a.vocab).train(texts[:4000])
    if not tok_path.exists():
        tok.save(tok_path)
    stream = build_stream(texts, tok)
    n_val = min(100_000, stream.numel() // 10)
    val, tr = stream[:n_val], stream[n_val:]
    print(f"corpus {stream.numel()/1e6:.2f}M tokens, vocab {tok.actual_size}\n")

    def make_fn(src, seed):
        g = torch.Generator().manual_seed(seed)
        def fn(step):
            ix = torch.randint(0, src.numel() - a.seq_len - 1,
                               (a.batch_size,), generator=g)
            ids = torch.stack([src[i:i + a.seq_len] for i in ix])
            return ids, ids.clone(), None
        return fn

    vg = torch.Generator().manual_seed(7)
    eval_set = [torch.stack([val[i:i + a.seq_len] for i in
                             torch.randint(0, val.numel() - a.seq_len - 1,
                                           (a.batch_size,), generator=vg)])
                for _ in range(24)]

    @torch.no_grad()
    def evaluate(m):
        m.eval()
        tot = 0.0
        for ids in eval_set:
            tot += float(m(ids.to(dev), labels=ids.to(dev)).lm_loss)
        return {"val_loss": tot / len(eval_set),
                "ppl": math.exp(tot / len(eval_set))}

    results = {}
    for variant in a.variants.split(","):
        print(f"\n{'='*66}\n{variant}\n{'='*66}")
        torch.manual_seed(0)
        m = build(variant, tok.actual_size, a)
        sz = sizes(m)
        print(f"resident {sz['resident_params']/1e6:.2f}M params "
              f"({sz['resident_mb_int4']:.1f} MB int4)   "
              f"disk {sz['disk_params']/1e6:.2f}M values "
              f"({sz['disk_mb_int8']:.1f} MB int8)   "
              f"slots {sz['slots']}")

        targs = TrainArgs(steps=a.steps, batch_size=a.batch_size,
                          seq_len=a.seq_len, lr=a.lr, device=dev,
                          out_dir=str(out), eval_every=0, log_every=100)
        train(m, make_fn(tr, 0), targs, tag=variant)
        r = evaluate(m.to(dev))
        r.update(sz)
        u = utilisation(m)
        if u:
            r["utilisation"] = u
            for k, v in u.items():
                print(f"  {k}: {v['slots_touched']}/{v['slots']} slots touched "
                      f"({v['frac_touched']:.1%}), usage entropy "
                      f"{v['usage_entropy']:.3f} (1.0 = uniform)")
        print(f"  val ppl {r['ppl']:.2f}")
        results[variant] = r

    print(f"\n{'='*66}\nSUMMARY\n{'='*66}")
    print(f"{'variant':10s} {'val ppl':>9s} {'resident':>11s} {'disk':>10s} "
          f"{'ppl / resident MB':>18s}")
    for k, v in results.items():
        print(f"{k:10s} {v['ppl']:9.2f} {v['resident_mb_int4']:9.1f}MB "
              f"{v['disk_mb_int8']:8.1f}MB {v['ppl']*v['resident_mb_int4']:18.1f}")

    if "memory" in results and "frozen" in results:
        g = results["frozen"]["ppl"] / results["memory"]["ppl"]
        print(f"\nlearned values vs frozen random values: {g:.3f}x better "
              f"({'the model IS using the store' if g > 1.05 else 'NO evidence the store is used'})")
    if "memory" in results and "dense" in results:
        d, mm = results["dense"], results["memory"]
        print(f"memory vs dense FFN: ppl {mm['ppl']:.2f} vs {d['ppl']:.2f} "
              f"({d['ppl']/mm['ppl']:.3f}x), resident "
              f"{mm['resident_mb_int4']:.1f}MB vs {d['resident_mb_int4']:.1f}MB "
              f"({d['resident_mb_int4']/mm['resident_mb_int4']:.2f}x less RAM)")

    (out / "results.json").write_text(json.dumps(results, indent=2),
                                      encoding="utf-8")
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
