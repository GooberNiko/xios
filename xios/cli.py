"""XIOS command line.

    python -m xios.cli info     --preset small
    python -m xios.cli chat     --ckpt runs/text/xios.pt --tokenizer runs/text/tokenizer.json
    python -m xios.cli quantize --preset small
    python -m xios.cli bench    --preset micro

``chat`` prints the depth each generated token cost, so you can watch the
compute allocation happen rather than take it on faith.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

from .config import get_config, PRESETS
from .model import XiosChat


def cmd_info(args):
    for name in ([args.preset] if args.preset else PRESETS):
        cfg = get_config(name)
        m = XiosChat(cfg, lazy_memory=True)
        rep = m.param_report()
        print(f"\n=== {name} ===")
        print(f"  {cfg.summary()}")
        print(f"  resident params: " + "  ".join(
            f"{k}={rep[k]/1e6:.1f}M" for k in
            ("embed", "prelude", "core", "coda", "head")))
        print(f"  total resident {rep['total']/1e6:.1f}M   "
              f"fp16 {rep['total']*2/1e9:.2f}GB   "
              f"int4 ~{rep['total']*0.56/1e9:.2f}GB")
        print(f"  depth: stores {cfg.stored_blocks} blocks, applies up to "
              f"{cfg.effective_depth_max} "
              f"({cfg.effective_depth_max/cfg.stored_blocks:.1f}x leverage)")
        if cfg.memory_enabled:
            print(f"  disk memory (NOT resident): {rep['memory_slots']/1e6:.1f}M slots"
                  f" x {cfg.memory_value_dim} dims int8 = "
                  f"{rep['memory_disk_bytes']/1e9:.2f}GB on disk, "
                  f"top-{cfg.memory_topk} touched per lookup")
            print(f"  total download {(rep['total']*0.56 + rep['memory_disk_bytes'])/1e9:.2f}GB"
                  f"  /  RAM needed ~{rep['total']*0.56/1e9:.2f}GB")


def cmd_quantize(args):
    from .quant import quantize_model, model_size_report
    cfg = get_config(args.preset)
    # lazy_memory: the associative store is disk-resident by design, so never
    # materialise it just to quantise the weights around it
    m = XiosChat(cfg, lazy_memory=True)
    n_before = m.n_params
    before = n_before * 2
    quantize_model(m, group=args.group, lora_rank=args.lora_rank)
    rep = model_size_report(m)
    print(f"preset {args.preset}: {n_before/1e6:.1f}M resident params")
    print(f"  fp16 {before/1e9:.3f} GB  ->  int4 {rep['total_gb']:.3f} GB "
          f"({before/rep['total_bytes']:.2f}x smaller)")
    if args.out:
        torch.save({"model": m.state_dict(), "config": cfg.__dict__}, args.out)
        print(f"  wrote {args.out}")


def cmd_bench(args):
    cfg = get_config(args.preset, max_seq_len=args.seq_len)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = XiosChat(cfg, lazy_memory=not cfg.memory_enabled).to(dev).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=dev)

    t0 = time.time()
    logits, state = m.prefill(ids)
    prefill_s = time.time() - t0

    depths, t0 = [], time.time()
    tok = ids[:, -1:]
    for _ in range(args.n_tokens):
        logits, state, d = m.decode_step(tok, state)
        depths.append(d)
        tok = logits[:, -1].argmax(-1, keepdim=True)
    dt = time.time() - t0

    print(f"preset {args.preset} on {dev}  ({m.n_params/1e6:.1f}M params)")
    print(f"  prefill {args.prompt_len} tok: {prefill_s*1000:.0f} ms")
    print(f"  decode: {args.n_tokens/dt:.1f} tok/s  ({dt/args.n_tokens*1000:.1f} ms/tok)")
    print(f"  depth used: mean {sum(depths)/len(depths):.2f}  "
          f"min {min(depths)}  max {max(depths)}  (ceiling {cfg.max_iters})")
    import collections
    hist = collections.Counter(depths)
    print("  depth histogram: " + "  ".join(
        f"{k}:{v}" for k, v in sorted(hist.items())))


def cmd_chat(args):
    from .tokenizer import ByteBPE
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tok = ByteBPE.load(args.tokenizer)
    from .config import XiosConfig
    cfg = XiosConfig(**ckpt["config"]) if "config" in ckpt else get_config(args.preset)
    m = XiosChat(cfg)
    m.load_state_dict(ckpt["model"])
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(dev).eval()

    print("XIOS chat. Ctrl-C to exit. Depth per token shown in dim text.\n")
    history = ""
    while True:
        try:
            user = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        history += f"<user>{user}<assistant>"
        ids = torch.tensor([tok.encode(history, bos=True)], device=dev)
        out, depths = m.generate(ids, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_p=args.top_p,
                                 eos_id=tok.eos_id, return_depths=True)
        new = out[0, ids.shape[1]:].tolist()
        text = tok.decode(new)
        print(text)
        if depths:
            print(f"\033[2m[depth mean {sum(depths)/len(depths):.1f}, "
                  f"max {max(depths)}/{cfg.max_iters}]\033[0m")
        history += text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xios")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info"); p.add_argument("--preset", default=None)
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("quantize")
    p.add_argument("--preset", default="small")
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_quantize)

    p = sub.add_parser("bench")
    p.add_argument("--preset", default="micro")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--n-tokens", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("chat")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--preset", default="nano")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_chat)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
