"""How redundant are a real model's layers, actually?

Three absorption attempts plateaued around 73% of GPT-2's perplexity, and
warm-starting a shared block from the *average* of the teacher's layers
produced a perplexity of 351,502 -- i.e. averaging GPT-2's layer weights does
not yield a working block at all. That is evidence the premise underneath the
whole approach ("adjacent layers are redundant, so they can be shared") is
weaker than assumed, at least in weight space.

Rather than guess at more compression ratios and burn another hour per guess,
measure the redundancy directly. This needs no training and answers three
questions:

1. **Weight similarity.** Are layers close in parameter space? If not,
   averaging them is meaningless -- which would explain the failed warm start.

2. **Functional similarity.** For real hidden states, does layer `j` do the
   same thing as layer `i`? This is the question that matters: two layers can
   be far apart in weight space and still compute nearly the same function.
   Sharing is viable exactly where functional similarity is high.

3. **Layer importance (drop test).** How much does removing each layer hurt?
   Layers that can be deleted outright are the ones that can certainly be
   shared, and the profile usually is not uniform -- early and late layers
   tend to matter far more than middle ones.

Together these give a *principled* sharing plan -- share where the function
repeats, keep separate where it does not -- instead of a uniform ratio picked
by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

torch.set_num_threads(12)

from xios.adapters import from_huggingface


@torch.no_grad()
def weight_similarity(blocks) -> list[list[float]]:
    """Cosine similarity between layers' flattened parameter vectors."""
    vecs = []
    for b in blocks:
        v = torch.cat([p.detach().float().flatten() for p in b.parameters()])
        vecs.append(F.normalize(v, dim=0))
    M = torch.stack(vecs)
    return (M @ M.T).tolist()


@torch.no_grad()
def functional_similarity(spec, blocks, ids, device) -> dict:
    """At each depth, how well does another layer substitute for this one?

    Relative error of `layer_j(h_i)` against `layer_i(h_i)`, where `h_i` is the
    real residual stream entering layer i. Low error means layer j is a viable
    stand-in at depth i, which is precisely what weight sharing requires.
    """
    h = spec.embed(ids.to(device))
    states = [h]
    for f in spec.layers:
        h = f(h)
        states.append(h)

    n = len(blocks)
    err = [[0.0] * n for _ in range(n)]
    for i in range(n):
        hi = states[i]
        true = states[i + 1]
        delta_norm = (true - hi).pow(2).mean().clamp(min=1e-9)
        for j in range(n):
            out = spec.layers[j](hi)
            # error measured against the true *update*, not the true output:
            # the residual stream dominates the output, so comparing outputs
            # makes every layer look interchangeable
            err[i][j] = float((out - true).pow(2).mean() / delta_norm)
    return {"rel_update_err": err}


@torch.no_grad()
def drop_test(spec, ids, device) -> list[float]:
    """Perplexity ratio when each single layer is skipped."""
    ids = ids.to(device)

    def ppl(skip=None):
        h = spec.embed(ids)
        for i, f in enumerate(spec.layers):
            if i == skip:
                continue
            h = f(h)
        lg = spec.head(spec.final_norm(h))
        return math.exp(float(F.cross_entropy(
            lg[:, :-1].reshape(-1, lg.size(-1)).float(), ids[:, 1:].reshape(-1))))

    base = ppl()
    return base, [ppl(i) / base for i in range(len(spec.layers))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--corpus", default="data/prose.txt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/redundancy")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(a.model).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    spec = from_huggingface(hf, a.model)
    blocks = hf._xios_blocks
    print(f"{a.model}: {spec.n_layers} layers, dim {spec.dim}\n")

    text = pathlib.Path(a.corpus).read_text(encoding="utf-8", errors="replace")
    ids_all = torch.tensor(tok(text).input_ids, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    ix = torch.randint(0, ids_all.numel() - a.seq_len - 1,
                       (a.batch_size,), generator=g)
    ids = torch.stack([ids_all[i:i + a.seq_len] for i in ix])

    # ---- 1. weight space -------------------------------------------------
    W = weight_similarity(blocks)
    n = len(W)
    adj = [W[i][i + 1] for i in range(n - 1)]
    off = [W[i][j] for i in range(n) for j in range(n) if i != j]
    print("1. WEIGHT-SPACE similarity (cosine between flattened layers)")
    print(f"   adjacent layers  mean {sum(adj)/len(adj):+.3f}  "
          f"min {min(adj):+.3f}  max {max(adj):+.3f}")
    print(f"   all pairs        mean {sum(off)/len(off):+.3f}")
    print(f"   -> averaging layer weights is {'plausible' if sum(adj)/len(adj) > 0.5 else 'MEANINGLESS'}"
          f", which explains the failed warm start" if sum(adj)/len(adj) <= 0.5 else "")

    # ---- 2. function space ----------------------------------------------
    fs = functional_similarity(spec, blocks, ids, dev)
    E = fs["rel_update_err"]
    print("\n2. FUNCTIONAL similarity: rel. error of layer j substituted at depth i")
    print("   (1.0 = as bad as doing nothing; <1 means j is a usable stand-in)")
    hdr = "     i\\j " + "".join(f"{j:6d}" for j in range(n))
    print(hdr)
    for i in range(n):
        row = "".join(f"{E[i][j]:6.2f}" for j in range(n))
        print(f"   {i:5d} {row}")
    nb = [E[i][i + 1] for i in range(n - 1)]
    print(f"   neighbour substitution (j=i+1): mean {sum(nb)/len(nb):.2f}")
    usable = sum(1 for i in range(n) for j in range(n)
                 if i != j and E[i][j] < 1.0)
    print(f"   pairs where a different layer beats doing nothing: "
          f"{usable}/{n*(n-1)}")

    # ---- 3. importance ---------------------------------------------------
    base, ratios = drop_test(spec, ids, dev)
    print(f"\n3. LAYER IMPORTANCE (perplexity ratio when skipped; base {base:.2f})")
    for i, r in enumerate(ratios):
        bar = "#" * min(60, int((r - 1) * 20))
        print(f"   layer {i:2d}  x{r:6.2f}  {bar}")
    order = sorted(range(n), key=lambda i: ratios[i])
    print(f"   most droppable: {order[:4]}   least: {order[-4:]}")

    (out / f"{a.model}.json").write_text(json.dumps(
        {"model": a.model, "weight_sim": W, "func_err": E,
         "drop_base_ppl": base, "drop_ratios": ratios}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {out}/{a.model}.json")


if __name__ == "__main__":
    main()
