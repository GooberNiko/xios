"""Thirty-second GPU sanity check. Run this FIRST in Colab.

Everything in this repo was developed on a CPU. The things most likely to break
on a GPU are not the science, they are dtype and device plumbing — and they
break twenty minutes into a training run rather than at the start. This checks
the specific risks up front:

* **fp16 autocast through the recurrent scan.** The gated linear recurrence
  computes its cumulative decay in float32 on purpose; under autocast the
  surrounding matmuls become fp16, and a mismatch or an overflow there would be
  silent until the loss went to NaN. A T4 is Turing, so it has no bf16 and
  *will* take the fp16 path.
* **Device consistency.** Buffers, RoPE tables and the memory's address vectors
  all have to follow `.to(device)`.
* **Train/decode equivalence on device**, which is the invariant the whole
  architecture rests on.

It also prints measured step times so runs can be sized honestly instead of
guessed at.
"""
from __future__ import annotations

import sys
import time

import torch

from xios.config import get_config
from xios.model import XiosChat
from xios.tasks import TaskDataset, TASK_VOCAB


def hr(t):
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


def main() -> int:
    hr("device")
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    print(f"torch {torch.__version__}  cuda {cuda}")
    if cuda:
        print(f"gpu   {torch.cuda.get_device_name(0)}")
        print(f"bf16  {torch.cuda.is_bf16_supported()}  "
              f"-> autocast will use "
              f"{'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram  {free/1e9:.1f} of {total/1e9:.1f} GB free")
    else:
        print("NOTE: no GPU. Change Runtime -> Change runtime type -> T4 GPU.")

    fails = []

    # ---- forward/backward, plain and autocast ---------------------------
    hr("forward / backward")
    cfg = get_config("nano", vocab_size=TASK_VOCAB, dim=256, n_kv_heads=2,
                     core_blocks=2, max_seq_len=128, max_iters=8,
                     target_depth=6.0, adaptive_depth=False, attn_window=64)
    m = XiosChat(cfg).to(dev)
    ids = torch.randint(0, cfg.vocab_size, (8, 64), device=dev)

    for label, amp_dtype in (("fp32", None),
                             ("autocast", torch.bfloat16
                              if (cuda and torch.cuda.is_bf16_supported())
                              else torch.float16)):
        try:
            m.zero_grad(set_to_none=True)
            if amp_dtype is None or not cuda:
                out = m(ids, labels=ids)
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    out = m(ids, labels=ids)
            loss = float(out.lm_loss)
            out.loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
            ok = torch.isfinite(torch.tensor(loss)) and torch.isfinite(gn)
            print(f"  {label:9s} loss {loss:.4f}  grad norm {float(gn):.3f}  "
                  f"{'ok' if ok else 'NON-FINITE'}")
            if not ok:
                fails.append(f"{label}: non-finite")
        except Exception as e:
            print(f"  {label:9s} FAILED: {type(e).__name__}: {e}")
            fails.append(f"{label}: {e}")

    # ---- the invariant the architecture rests on ------------------------
    hr("train forward == incremental decode (on device)")
    try:
        m.eval()
        with torch.no_grad():
            full = m(ids[:1, :40]).logits
            logits, st = m.prefill(ids[:1, :24])
            got = [logits[:, -1]]
            for t in range(24, 39):
                logits, st, _ = m.decode_step(ids[:1, t:t + 1], st)
                got.append(logits[:, -1])
            inc = torch.stack(got, 1)
        ref = full[:, 23:39]
        agree = float((ref.argmax(-1) == inc.argmax(-1)).float().mean())
        rel = float((ref - inc).abs().max() / ref.abs().max())
        print(f"  argmax agreement {agree:.1%}   max rel logit err {rel:.2e}")
        if agree < 1.0:
            fails.append(f"decode disagreement ({agree:.1%})")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        fails.append(f"decode: {e}")

    # ---- test-time depth knob still works on device ---------------------
    hr("test-time depth is a live knob")
    try:
        with torch.no_grad():
            depths = []
            for d in (2, 6, 12):
                c2 = get_config("nano", vocab_size=TASK_VOCAB, dim=256,
                                n_kv_heads=2, core_blocks=2, max_seq_len=128,
                                max_iters=16, target_depth=float(d),
                                adaptive_depth=False, attn_window=64)
                m2 = XiosChat(c2).to(dev)
                m2.load_state_dict(m.state_dict(), strict=False)
                st = m2(ids).stats
                depths.append((d, st.mean_depth))
        print("  requested -> actual iterations: " +
              ", ".join(f"{a}->{b:.0f}" for a, b in depths))
        if any(abs(a - b) > 0.01 for a, b in depths):
            fails.append("depth knob not honoured")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        fails.append(f"depth knob: {e}")

    # ---- measured speed, so runs can be sized --------------------------
    hr("throughput (for sizing runs)")
    try:
        ds = TaskDataset("perm", 1, 4, 64, seed=0, width=4)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        amp = cuda
        adt = (torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported())
               else torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=amp and adt == torch.float16)
        for bs in (32, 128):
            i2, l2, msk, _ = ds.batch(bs, device=dev)
            for _ in range(3):                       # warm up
                with torch.autocast("cuda", dtype=adt, enabled=amp):
                    o = m(i2, labels=l2, loss_mask=msk)
                opt.zero_grad(set_to_none=True)
                scaler.scale(o.loss).backward()
                scaler.step(opt); scaler.update()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            n = 10
            for _ in range(n):
                with torch.autocast("cuda", dtype=adt, enabled=amp):
                    o = m(i2, labels=l2, loss_mask=msk)
                opt.zero_grad(set_to_none=True)
                scaler.scale(o.loss).backward()
                scaler.step(opt); scaler.update()
            if cuda:
                torch.cuda.synchronize()
            dt = (time.time() - t0) / n
            print(f"  batch {bs:4d}: {dt*1000:7.1f} ms/step  "
                  f"-> 8000 steps in {8000*dt/60:5.1f} min")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        fails.append(f"throughput: {e}")

    hr("verdict")
    if fails:
        print("NOT READY:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("all checks passed — safe to start the real runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
