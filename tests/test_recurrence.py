"""The parallel (training) form of the gated recurrence must be numerically
identical to the sequential (inference) form, otherwise a model trains under
one set of semantics and generates under another."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from xios.core.recurrent import GatedRecurrentMixer


def test_parallel_matches_sequential():
    torch.manual_seed(0)
    B, T, D, H, dh = 2, 77, 256, 4, 64
    mix = GatedRecurrentMixer(D, H, dh, chunk=16).eval()
    x = torch.randn(B, T, D)

    with torch.no_grad():
        par, _ = mix(x, return_state=True)
        state = {"S": None, "conv": None}
        seq = []
        for t in range(T):
            y, state = mix.step(x[:, t:t + 1], state)
            seq.append(y)
        seq = torch.cat(seq, dim=1)

    err = (par - seq).abs().max().item()
    rel = err / par.abs().max().item()
    print(f"max abs err {err:.3e}  rel {rel:.3e}")
    assert rel < 2e-3, f"parallel/sequential mismatch: rel={rel}"


def test_chunk_size_invariance():
    torch.manual_seed(1)
    x = torch.randn(2, 100, 128)
    outs = []
    for c in (16, 32, 64):
        torch.manual_seed(7)
        m = GatedRecurrentMixer(128, 4, 32, chunk=c).eval()
        with torch.no_grad():
            outs.append(m(x)[0])
    for o in outs[1:]:
        assert (o - outs[0]).abs().max() < 1e-3


def test_prefill_then_decode():
    """Prefill a prompt in parallel, then decode token-by-token from state."""
    torch.manual_seed(2)
    m = GatedRecurrentMixer(128, 4, 32, chunk=16).eval()
    x = torch.randn(1, 50, 128)
    with torch.no_grad():
        full, _ = m(x, return_state=True)
        pre, st = m(x[:, :30], return_state=True)
        rest = []
        for t in range(30, 50):
            y, st = m.step(x[:, t:t + 1], st)
            rest.append(y)
        joined = torch.cat([pre] + rest, dim=1)
    rel = (full - joined).abs().max().item() / full.abs().max().item()
    print(f"prefill+decode rel err {rel:.3e}")
    assert rel < 2e-3


if __name__ == "__main__":
    test_parallel_matches_sequential()
    test_chunk_size_invariance()
    test_prefill_then_decode()
    print("all recurrence tests passed")
