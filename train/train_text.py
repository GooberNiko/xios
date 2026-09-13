"""Language-modelling A/B on natural text.

Perplexity is a weaker instrument for XIOS's claim than the synthetic depth
tasks -- most tokens in natural text are easy, so an average over them hides
the effect. It is still the check that matters for a chat model actually
being usable, and the depth histogram it produces is informative on its own:
if the controller is working, function words and word-continuations should
sit at minimum depth while content words and sentence starts cost more.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from xios.config import get_config
from xios.model import XiosChat
from xios.baseline import matched_baseline
from xios.tokenizer import ByteBPE
from train.train import TrainArgs, train


CORPORA = {
    # Small, public-domain, no dependencies. The standard sanity corpus:
    # ~1MB, enough for a nano model to produce recognisably English text.
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt",
}


def fetch_corpus(name: str, dest: pathlib.Path) -> pathlib.Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"downloading {name}...")
    urllib.request.urlretrieve(CORPORA[name], dest)
    print(f"  {dest} ({dest.stat().st_size/1e6:.2f} MB)")
    return dest


def load_texts(dataset: str, n_docs: int, split: str = "train") -> list[str]:
    if dataset in CORPORA:
        dataset = str(fetch_corpus(dataset, pathlib.Path("data") / f"{dataset}.txt"))
    if pathlib.Path(dataset).exists():
        text = pathlib.Path(dataset).read_text(encoding="utf-8", errors="replace")
        chunk = 2000
        return [text[i:i + chunk] for i in range(0, len(text), chunk)][:n_docs]
    from datasets import load_dataset
    ds = load_dataset(dataset, split=f"{split}[:{n_docs}]")
    key = "text" if "text" in ds.column_names else ds.column_names[0]
    return [r for r in ds[key] if r and r.strip()]


def build_stream(texts, tok: ByteBPE) -> torch.Tensor:
    ids: list[int] = []
    for t in texts:
        ids.extend(tok.encode(t, bos=True, eos=True))
    return torch.tensor(ids, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--target-depth", type=float, default=4.0)
    ap.add_argument("--out", default="runs/text")
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, choices=["xios", "baseline"],
                    help="train just one model (the UI only needs xios)")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--core-blocks", type=int, default=None)
    ap.add_argument("--n-kv-heads", type=int, default=None)
    ap.add_argument("--baseline-layers", type=int, default=None,
                    help="force the dense baseline's depth (for collapse targets)")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading text...")
    texts = load_texts(args.dataset, args.n_docs)
    print(f"{len(texts)} documents")

    tok_path = out / "tokenizer.json"
    if tok_path.exists():
        tok = ByteBPE.load(tok_path)
    else:
        print("training tokenizer...")
        tok = ByteBPE(vocab_size=args.vocab).train(texts[:4000])
        tok.save(tok_path)
    print(f"tokenizer: {tok.actual_size} tokens")

    stream = build_stream(texts, tok)
    print(f"corpus: {stream.numel()/1e6:.2f}M tokens")

    n_val = min(200_000, stream.numel() // 10)
    val, tr = stream[:n_val], stream[n_val:]

    over = {k: v for k, v in (("dim", args.dim),
                              ("core_blocks", args.core_blocks),
                              ("n_kv_heads", args.n_kv_heads)) if v is not None}
    cfg = get_config(args.preset, vocab_size=tok.actual_size,
                     max_seq_len=args.seq_len, max_iters=args.max_iters,
                     target_depth=args.target_depth,
                     attn_window=min(256, args.seq_len), **over)
    xios = XiosChat(cfg)
    if args.baseline_layers:
        from xios.baseline import BaselineTransformer
        base = BaselineTransformer(
            dim=cfg.dim, n_layers=args.baseline_layers, head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads, vocab_size=cfg.vocab_size,
            ffn_mult=cfg.ffn_mult, norm_eps=cfg.norm_eps,
            rope_theta=cfg.rope_theta, tie_embeddings=cfg.tie_embeddings,
            max_seq_len=cfg.max_seq_len)
    else:
        base = matched_baseline(cfg)
    print(f"XIOS {xios.n_params/1e6:.2f}M | baseline {base.n_params/1e6:.2f}M "
          f"x {base.n_layers} layers")

    g = torch.Generator().manual_seed(0)

    def data_fn(step):
        ix = torch.randint(0, tr.numel() - args.seq_len - 1,
                           (args.batch_size,), generator=g)
        ids = torch.stack([tr[i:i + args.seq_len] for i in ix])
        return ids, ids.clone(), None

    @torch.no_grad()
    def evaluate(model, n_batches: int = 20):
        model.eval()
        tot, cnt, depth = 0.0, 0, []
        vg = torch.Generator().manual_seed(7)
        for _ in range(n_batches):
            ix = torch.randint(0, val.numel() - args.seq_len - 1,
                               (args.batch_size,), generator=vg)
            ids = torch.stack([val[i:i + args.seq_len] for i in ix]).to(dev)
            o = model(ids, labels=ids)
            tot += float(o.lm_loss); cnt += 1
            if getattr(o, "stats", None) is not None and o.stats.actual_depth is not None:
                depth.append(o.stats.mean_depth)
        r = {"val_loss": tot / cnt, "ppl": float(torch.tensor(tot / cnt).exp())}
        if depth:
            r["mean_depth"] = sum(depth) / len(depth)
        return r

    targs = TrainArgs(steps=args.steps, batch_size=args.batch_size,
                      seq_len=args.seq_len, lr=args.lr, device=dev,
                      out_dir=str(out), eval_every=max(args.steps // 4, 1))

    results = {}
    for name, model in (("xios", xios), ("baseline", base)):
        if args.only and args.only != name:
            continue
        print(f"\n=== {name} ===")
        train(model, data_fn, targs, eval_fn=evaluate, tag=name)
        results[name] = evaluate(model.to(dev), n_batches=40)
        if name == "baseline":
            import dataclasses
            torch.save({"model": model.state_dict(),
                        "n_layers": base.n_layers, "dim": cfg.dim,
                        "vocab_size": cfg.vocab_size, "head_dim": cfg.head_dim,
                        "n_kv_heads": cfg.n_kv_heads, "ffn_mult": cfg.ffn_mult,
                        "max_seq_len": cfg.max_seq_len},
                       out / "baseline.pt")
        if name == "xios":
            # re-save with the config attached so `app.py --ckpt` can
            # reconstruct the model without being told the preset
            import dataclasses
            torch.save({"model": model.state_dict(),
                        "config": dataclasses.asdict(cfg)},
                       out / "xios.pt")
        print(f"{name}: {results[name]}")

    print("\n=== final ===")
    for k, v in results.items():
        print(f"{k:9s} val loss {v['val_loss']:.4f}  ppl {v['ppl']:.2f}"
              + (f"  mean depth {v['mean_depth']:.2f}" if "mean_depth" in v else ""))

    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
