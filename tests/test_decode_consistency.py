"""Parallel training forward and incremental decode must produce the same
logits. With per-depth memory timelines this is the invariant most likely to
silently break, so it gets its own test."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from xios.config import get_config
from xios.model import XiosChat


def run(preset="nano", T=40, split=24, seed=0):
    torch.manual_seed(seed)
    cfg = get_config(preset, max_seq_len=512)
    m = XiosChat(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        full = m(ids).logits

        logits, state = m.prefill(ids[:, :split])
        got = [logits[:, -1]]
        for t in range(split, T - 1):
            logits, state, d = m.decode_step(ids[:, t:t + 1], state)
            got.append(logits[:, -1])
        inc = torch.stack(got, dim=1)

    ref = full[:, split - 1:T - 1]
    rel = (ref - inc).abs().max().item() / ref.abs().max().item()
    agree = (ref.argmax(-1) == inc.argmax(-1)).float().mean().item()
    return rel, agree


if __name__ == "__main__":
    for preset in ("nano",):
        rel, agree = run(preset)
        print(f"{preset}: max rel logit err {rel:.3e}   argmax agreement {agree:.1%}")
        assert agree == 1.0, "prefill/decode disagree on the predicted token"
        assert rel < 5e-2, f"logit drift too large: {rel}"
    print("decode consistency verified")
