"""Quantisation: does the size claim hold, and does the model still run?"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from xios.config import get_config
from xios.model import XiosChat
from xios.quant import (QuantLinear, QuantEmbedding, quantize_model,
                        model_size_report, collect_act_scales)
from xios.quant.int4 import TiedQuantHead


def test_pack_roundtrip():
    from xios.quant import pack_int4, unpack_int4
    w = torch.randint(0, 16, (7, 64), dtype=torch.uint8)
    assert (unpack_int4(pack_int4(w), 64) == w).all(), "int4 packing is lossy"
    print("int4 pack/unpack is exact")


def test_linear_accuracy():
    """On weights with realistic structure, measure each technique's effect."""
    torch.manual_seed(0)
    U, V = torch.randn(1024, 64), torch.randn(64, 1024)
    W = (U @ V) / 8 + 0.3 * torch.randn(1024, 1024)
    chan = torch.rand(1024) ** 3 * 5 + 0.1          # heavy-tailed channels
    W = W * chan[None, :]
    lin = nn.Linear(1024, 1024, bias=False)
    lin.weight.data = W
    x = torch.randn(16, 64, 1024) * chan[None, None, :]
    ref = lin(x)
    act = x.reshape(-1, 1024).abs().mean(0)

    print(f"{'variant':18s} {'rel err':>9s} {'KB':>9s} {'vs fp16':>9s}")
    for tag, kw in [("plain int4", {}),
                    ("AWQ alpha=0.5", dict(act_scale=act, awq_alpha=0.5)),
                    ("AWQ searched", dict(act_scale=act)),
                    ("+lowrank r16", dict(act_scale=act, lora_rank=16)),
                    ("+lowrank r64", dict(act_scale=act, lora_rank=64))]:
        q = QuantLinear.from_linear(lin, group=64, **kw)
        e = float((q(x) - ref).norm() / ref.norm())
        note = f"  (alpha={q.awq_alpha})" if kw.get("act_scale") is not None else ""
        print(f"{tag:18s} {e:9.4f} {q.nbytes/1e3:9.1f} "
              f"{lin.weight.numel()*2/q.nbytes:8.2f}x{note}")


def test_model_quantization():
    torch.manual_seed(0)
    cfg = get_config("nano")
    m = XiosChat(cfg, lazy_memory=True).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        ref = m(ids).logits
    before = m.n_params * 2

    quantize_model(m)
    assert isinstance(m.embed, QuantEmbedding)
    assert isinstance(m.lm_head, TiedQuantHead), "tied head must stay tied"
    assert m.lm_head.nbytes == 0, "tied head must cost no extra storage"

    with torch.no_grad():
        got = m(ids).logits
    rel = float((got - ref).norm() / ref.norm())
    rep = model_size_report(m)
    print(f"nano: fp16 {before/1e6:.1f} MB -> int4 {rep['total_mb']:.1f} MB "
          f"({before/rep['total_bytes']:.2f}x)  logit rel err {rel:.4f}")
    assert before / rep["total_bytes"] > 2.0

    # and it must still decode
    out = m.generate(ids, max_new_tokens=4, temperature=0.0)
    assert out.shape[1] == ids.shape[1] + 4
    print("quantised model still decodes")


def test_size_across_presets():
    print(f"\n{'preset':8s} {'resident':>10s} {'fp16':>9s} {'int4':>9s} {'ratio':>7s}")
    for name in ("nano", "micro", "small"):
        cfg = get_config(name)
        m = XiosChat(cfg, lazy_memory=True)
        n_resident = m.n_params          # capture BEFORE weights become buffers
        before = n_resident * 2
        quantize_model(m)
        rep = model_size_report(m)
        print(f"{name:8s} {n_resident/1e6:9.1f}M {before/1e9:8.3f}G "
              f"{rep['total_gb']:8.3f}G {before/rep['total_bytes']:6.2f}x")


if __name__ == "__main__":
    test_pack_roundtrip()
    print()
    test_linear_accuracy()
    print()
    test_model_quantization()
    test_size_across_presets()
    print("\nquantisation tests passed")
