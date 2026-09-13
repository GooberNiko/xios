"""Does the address layout actually reduce SSD page reads?

Measured, not assumed. On random init no layout can help (retrieval is
random), so the interesting number is the one under trained-like key
geometry, where codes lie on a low-dimensional manifold the way they do
after training.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from xios.config import get_config
from xios.memory.dam import DiskAssociativeMemory


def make(slots=4 * (1 << 14), heads=4, vdim=128, topk=32):
    cfg = get_config("nano", memory_slots=slots, memory_value_dim=vdim,
                     memory_topk=topk, memory_heads=heads)
    return cfg, DiskAssociativeMemory(cfg)


def structure_keys(dam, seed=0, shuffle=True):
    """Put the codebooks on a smooth low-dim manifold, as training does.

    ``shuffle`` is essential for an honest test: a trained codebook has no
    relationship between a code's *index* and its position on the manifold.
    Leaving the codes in manifold order hands row-major a locality it would
    never have in practice -- and that artefact is exactly what made an
    earlier version of this measurement look negative.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for kb in (dam.keys1, dam.keys2):
            H, m, d = kb.shape
            t = torch.linspace(0, 1, m)
            basis = torch.randn(H, 4, d, generator=g)
            coef = torch.stack([t, t ** 2, torch.sin(6 * t), torch.cos(6 * t)], 1)
            k = (torch.einsum("mc,hcd->hmd", coef, basis) * 0.3
                 + 0.02 * torch.randn(H, m, d, generator=g))
            if shuffle:
                for h in range(H):
                    k[h] = k[h][torch.randperm(m, generator=g)]
            kb.copy_(k)


def probe(cfg, dam, tag):
    x = torch.randn(1, 256, cfg.dim)
    q = dam.normalise_query(x)
    i1, i2 = dam.logical_codes(q)
    r = dam.locality_report(i1, i2)
    print(f"{tag:24s} row_major {r['row_major']:6.1f} | sorted_morton "
          f"{r['sorted_morton']:6.1f} | {r['reduction']:.2f}x  "
          f"({r['lookups_per_token']} lookups/token)")
    return r


if __name__ == "__main__":
    cfg, dam = make()
    probe(cfg, dam, "random-init keys")

    cfg, dam = make()
    structure_keys(dam, shuffle=False)
    probe(cfg, dam, "manifold, index-sorted")

    cfg, dam = make()
    structure_keys(dam, shuffle=True)
    r = probe(cfg, dam, "manifold, shuffled")

    # the layout must be a bijection: every slot addressed exactly once
    for mode in ("row_major", "sorted_morton"):
        dam.optimize_layout(mode)
        m = dam.m
        g1 = torch.arange(m)[None, :, None].expand(1, m, m).reshape(1, 1, -1)
        g2 = torch.arange(m)[None, None, :].expand(1, m, m).reshape(1, 1, -1)
        g1 = g1.expand(1, dam.n_heads, -1)
        g2 = g2.expand(1, dam.n_heads, -1)
        phys = dam.physical_address(g1, g2)
        uniq = phys.reshape(-1).unique().numel()
        assert uniq == dam.n_slots, f"{mode}: {uniq} != {dam.n_slots} slots addressed"
        print(f"{mode:14s} addressing is a bijection over all {dam.n_slots} slots")

    # the rank vectors are all the layout costs to store
    layout_bytes = (dam.rank1.numel() + dam.rank2.numel()) * 8
    print(f"layout table: {layout_bytes/1e3:.1f} KB resident "
          f"(a per-slot permutation would be {dam.n_slots*8/1e6:.1f} MB)")

    x = torch.randn(2, 16, cfg.dim)
    y, aux = dam(x)
    y.sum().backward()
    print(f"forward/backward ok  out={tuple(y.shape)}  aux={float(aux):.5f}")
    print(f"VERDICT: sorted-Morton gives {r['reduction']:.2f}x fewer page reads "
          f"under realistic key geometry")
