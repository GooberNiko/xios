"""How many arbitrary facts can a model store per resident byte?

Why the previous memory experiment could not answer this
--------------------------------------------------------
`train/memory_quality.py` compared a dense FFN against the product-key store on
Shakespeare and found them equivalent (ppl 14.43 vs 14.79) -- and, damningly,
found that **freezing the value table at random init changed nothing**
(14.79 vs ~14.8). The obvious reading is that the store is useless.

That reading is probably wrong, because the benchmark cannot test the claim.
0.6 MB of Shakespeare contains almost no *knowledge*: it is a small, highly
repetitive corpus where a model does well from patterns held in its weights.
A knowledge store only earns its bytes when there are many **rare, arbitrary**
facts that cannot be compressed into a pattern. Shakespeare has none, so the
memory had nothing to remember, and a frozen random table did just as well.
Finding "no benefit" on a task with no knowledge in it is not evidence about
knowledge storage.

So this measures the thing directly. Generate `N` arbitrary key->value pairs
with no structure whatsoever -- nothing to generalise, memorisation is the only
option -- and ask how many the model can recall. Sweep `N`.

The prediction being tested: a model whose FFN is replaced by a `slots x dim`
store should recall far more facts than a dense model with the *same resident
parameters*, because its capacity lives in the (non-resident) value table. If
that holds, knowledge can live on disk and the `flagship` design closes. If a
dense FFN matches it at equal resident size, the store is decoration and should
be cut.

Arms:
  dense    ordinary FFN
  memory   product-key store, learned values
  frozen   product-key store, values frozen at random  (proves learning matters)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

torch.set_num_threads(12)

from xios.config import get_config
from xios.model import XiosChat
from xios.memory.dam import DiskAssociativeMemory


# ---------------------------------------------------------------------------
class FactStore:
    """`n_facts` arbitrary key->value pairs over a small token alphabet.

    Keys and values are random token strings, so there is no pattern to learn:
    the only way to answer is to have stored the pair. Capacity, not
    generalisation.
    """

    def __init__(self, n_facts: int, key_len: int = 3, val_len: int = 2,
                 alphabet: int = 32, seed: int = 0):
        self.rng = random.Random(seed)
        self.key_len, self.val_len, self.alphabet = key_len, val_len, alphabet
        # token ids: 0 pad, 1 bos, 2 eos, 3 '=' ; symbols start at 4
        self.PAD, self.BOS, self.EOS, self.EQ = 0, 1, 2, 3
        self.base = 4
        self.vocab = self.base + alphabet

        seen = set()
        self.facts = []
        while len(self.facts) < n_facts:
            k = tuple(self.rng.randrange(alphabet) for _ in range(key_len))
            if k in seen:
                continue
            seen.add(k)
            v = tuple(self.rng.randrange(alphabet) for _ in range(val_len))
            self.facts.append((k, v))
        self.seq_len = 1 + key_len + 1 + val_len + 1      # bos k = v eos

    def batch(self, bs: int, idx=None, device="cpu"):
        ids, labels = [], []
        for b in range(bs):
            k, v = self.facts[self.rng.randrange(len(self.facts))] \
                if idx is None else self.facts[idx[b]]
            seq = ([self.BOS] + [self.base + t for t in k] + [self.EQ]
                   + [self.base + t for t in v] + [self.EOS])
            lab = [-100] * (1 + self.key_len + 1) + \
                  [self.base + t for t in v] + [self.EOS]
            ids.append(seq); labels.append(lab)
        return (torch.tensor(ids, device=device),
                torch.tensor(labels, device=device))


def build(variant: str, store: FactStore, a):
    # head_dim is 64 in the nano preset, so a narrow model has few heads;
    # n_kv_heads must divide them
    n_heads = max(1, a.dim // 64)
    kw = dict(vocab_size=store.vocab, max_seq_len=max(store.seq_len, 32),
              dim=a.dim, core_blocks=a.core_blocks,
              n_kv_heads=1 if n_heads < 2 else 2,
              max_iters=a.max_iters, target_depth=a.target_depth,
              attn_every=2, attn_window=32, cond_dim=128,
              prelude_blocks=1, coda_blocks=1)
    if variant == "dense":
        cfg = get_config("nano", memory_enabled=False, **kw)
    else:
        cfg = get_config("nano", memory_enabled=True,
                         memory_layers=tuple(range(a.core_blocks)),
                         memory_slots=a.heads * a.grid * a.grid,
                         memory_value_dim=a.value_dim, memory_topk=a.topk,
                         memory_heads=a.heads, **kw)
    m = XiosChat(cfg)
    if variant == "frozen":
        for mod in m.modules():
            if isinstance(mod, DiskAssociativeMemory):
                mod.values.weight.requires_grad_(False)
    return m


def resident_and_disk(m):
    mems = [x for x in m.modules() if isinstance(x, DiskAssociativeMemory)]
    disk = sum(x.n_slots * x.value_dim for x in mems)
    return m.n_params, disk


@torch.no_grad()
def recall(m, store: FactStore, device, batch=64) -> float:
    """Fraction of stored facts recalled exactly (teacher-forced argmax)."""
    m.eval()
    ok = tot = 0
    for s in range(0, len(store.facts), batch):
        idx = list(range(s, min(s + batch, len(store.facts))))
        ids, labels = store.batch(len(idx), idx=idx, device=device)
        lg = m(ids).logits
        pred = lg[:, :-1].argmax(-1)
        tgt = labels[:, 1:]
        valid = tgt != -100
        ok += int((((pred == tgt) | ~valid).all(1)).sum())
        tot += len(idx)
    return ok / max(tot, 1)


def train_one(m, store, a, device, tag=""):
    from train.train import build_optimizer, TrainArgs, lr_at
    targs = TrainArgs(steps=a.steps, lr=a.lr, device=device, amp=False)
    m = m.to(device).train()
    opt, sopt = build_optimizer(m, targs)
    for step in range(a.steps):
        lr = lr_at(step, targs)
        for g in opt.param_groups:
            g["lr"] = lr
        if sopt:
            for g in sopt.param_groups:
                g["lr"] = lr
        ids, labels = store.batch(a.batch_size, device=device)
        out = m(ids, labels=labels)
        opt.zero_grad(set_to_none=True)
        if sopt:
            sopt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if sopt:
            sopt.step()
        if step % max(a.steps // 4, 1) == 0:
            print(f"    [{tag}] {step:5d}  loss {float(out.lm_loss):.4f}",
                  flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="500,2000,8000")
    ap.add_argument("--variants", default="dense,memory,frozen")
    ap.add_argument("--exposures", type=int, default=120,
                    help="times each fact is seen; steps scale with fact count "
                         "so capacity is the only variable")
    ap.add_argument("--steps", type=int, default=0,
                    help="override; 0 = derive from --exposures")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--core-blocks", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--target-depth", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--value-dim", type=int, default=32)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/factcap")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}

    for nf in [int(x) for x in a.facts.split(",")]:
        store = FactStore(nf, seed=0)
        a.steps = a.steps or 0
        steps = max(200, nf * a.exposures // a.batch_size)
        print(f"\n{'='*70}\n{nf} arbitrary facts "
              f"(key {store.key_len} symbols -> value {store.val_len}, "
              f"alphabet {store.alphabet})\n{'='*70}")
        # information content of the fact set, for reference
        bits = nf * store.val_len * math.log2(store.alphabet)
        print(f"  the fact set contains {bits/8/1e3:.1f} KB of information; "
              f"{steps} steps = {a.exposures} exposures per fact")
        for variant in a.variants.split(","):
            torch.manual_seed(0)
            m = build(variant, store, a)
            res, disk = resident_and_disk(m)
            import copy as _copy
            aa = _copy.copy(a); aa.steps = steps
            m = train_one(m, store, aa, dev, tag=f"{variant}/{nf}")
            r = recall(m, store, dev)
            print(f"  {variant:8s} recall {r:6.1%}   resident {res/1e6:.2f}M "
                  f"  disk {disk/1e6:.2f}M")
            results[f"{variant}_{nf}"] = {
                "variant": variant, "n_facts": nf, "recall": r,
                "resident_params": res, "disk_params": disk,
                "facts_per_resident_kparam": r * nf / (res / 1e3)}

    print(f"\n{'='*70}\nSUMMARY: facts recalled per 1k resident params\n{'='*70}")
    print(f"{'facts':>7} " + "".join(f"{v:>12s}" for v in a.variants.split(",")))
    for nf in [int(x) for x in a.facts.split(",")]:
        row = f"{nf:7d} "
        for v in a.variants.split(","):
            k = f"{v}_{nf}"
            row += f"{results[k]['recall']:11.1%} " if k in results else " " * 12
        print(row)
    print()
    for nf in [int(x) for x in a.facts.split(",")]:
        d, mm = results.get(f"dense_{nf}"), results.get(f"memory_{nf}")
        fz = results.get(f"frozen_{nf}")
        if d and mm:
            print(f"{nf:6d} facts: memory {mm['recall']:.1%} vs dense "
                  f"{d['recall']:.1%}"
                  + (f", frozen {fz['recall']:.1%}" if fz else ""))
    (out / "results.json").write_text(json.dumps(results, indent=2),
                                      encoding="utf-8")
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
