"""The disk path, end to end: train in RAM, export to int8, memory-map it
back, and check the model still produces the same answers with ~zero resident
memory for the store.

This is the claim that knowledge can live on the SSD. If the round trip is
lossy or the mmap backend disagrees with the RAM backend, the claim is empty.
"""
import sys, pathlib, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from xios.config import get_config
from xios.memory.dam import DiskAssociativeMemory


def test_export_attach_roundtrip():
    torch.manual_seed(0)
    cfg = get_config("nano", memory_enabled=True, memory_slots=4 * (1 << 12),
                    memory_value_dim=64, memory_topk=16, memory_heads=4)
    dam = DiskAssociativeMemory(cfg).eval()
    # give the values some structure so int8 error is meaningful
    with torch.no_grad():
        dam.values.weight.normal_(0, 0.05)
        dam.out_proj.weight.normal_(0, 0.02)

    x = torch.randn(2, 16, cfg.dim)
    with torch.no_grad():
        ram_out, _ = dam(x)

    with tempfile.TemporaryDirectory() as td:
        info = dam.export(td)
        files = sorted(os.listdir(td))
        on_disk = sum(os.path.getsize(os.path.join(td, f)) for f in files)
        print(f"exported {files} -> {on_disk/1e6:.3f} MB "
              f"({info['n_slots']} slots x {info['value_dim']} dims)")

        dam.attach(td)
        assert dam.backend == "disk"
        assert dam.values is None, "RAM copy must be released after attach"

        with torch.no_grad():
            disk_out, _ = dam(x)
        dam.detach_store()          # release the mmap so the dir can be removed

    rel = float((disk_out - ram_out).norm() / ram_out.norm())
    print(f"mmap int8 vs fp32 RAM backend: rel err {rel:.4f}")
    assert rel < 0.05, f"disk backend disagrees with RAM backend: {rel}"

    # int8 + one fp32 scale per slot
    expected = info["n_slots"] * (info["value_dim"] + 4)
    assert abs(on_disk - expected) < 1024, "unexpected on-disk size"
    print(f"bytes per slot: {on_disk/info['n_slots']:.1f} "
          f"({info['value_dim']} int8 + 4 scale)")


def test_layout_survives_export():
    """Optimising the layout then exporting must keep values and addresses
    in agreement -- otherwise every lookup silently reads the wrong slot."""
    torch.manual_seed(1)
    cfg = get_config("nano", memory_enabled=True, memory_slots=4 * (1 << 12),
                    memory_value_dim=64, memory_topk=16, memory_heads=4)
    dam = DiskAssociativeMemory(cfg).eval()
    with torch.no_grad():
        dam.values.weight.normal_(0, 0.05)
        dam.out_proj.weight.normal_(0, 0.02)

    x = torch.randn(1, 8, cfg.dim)
    dam.optimize_layout("sorted_morton")
    with torch.no_grad():
        before, _ = dam(x)

    with tempfile.TemporaryDirectory() as td:
        dam.export(td)
        dam.attach(td)
        with torch.no_grad():
            after, _ = dam(x)
        dam.detach_store()

    rel = float((after - before).norm() / before.norm())
    print(f"seriated layout preserved across export/attach: rel err {rel:.4f}")
    assert rel < 0.05, "layout and value store fell out of sync"


def test_resident_cost():
    """A large store must cost essentially nothing to hold open."""
    cfg = get_config("base")
    from xios.model import XiosChat
    m = XiosChat(cfg, lazy_memory=True)
    rep = m.param_report()
    ratio = rep["memory_disk_bytes"] / (rep["total"] * 2)
    print(f"base preset: {rep['total']/1e6:.0f}M resident params "
          f"({rep['total']*2/1e9:.2f} GB fp16), "
          f"{rep['memory_disk_bytes']/1e9:.2f} GB on disk "
          f"({ratio:.1f}x the resident weights, none of it in RAM)")
    assert rep["memory_disk_bytes"] > rep["total"] * 2, \
        "the store should be able to exceed the weights it serves"


if __name__ == "__main__":
    test_export_attach_roundtrip()
    test_layout_survives_export()
    test_resident_cost()
    print("\ndisk memory tests passed")
