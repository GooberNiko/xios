"""XIOS desktop UI.

    python app.py                                   # random weights, demo only
    python app.py --ckpt runs/text/xios.pt --tokenizer runs/text/tokenizer.json

Two things make this more than a text box:

**Thinking is literal here.** XIOS decides per token how many times to loop
its core, so "thinking" is not a metaphor or a hidden scratchpad -- it is an
integer the model produces. Every generated token is shaded by the depth it
cost, and the panel on the right reports the distribution live. You can watch
the model spend one iteration on a space and twelve on a word it finds hard.

**Web search is retrieval, not magic.** Results are fetched, packed into a
context preamble and shown to you verbatim, so you can see exactly what the
model was given rather than guessing what it "knew".

Tkinter only; no dependencies beyond torch.
"""
from __future__ import annotations

import argparse
import pathlib
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from xios.config import XiosConfig, get_config
from xios.model import XiosChat, _sample
from xios.tokenizer import ByteBPE
from xios import search as websearch

# --------------------------------------------------------------------------
# depth -> colour. Cool = cheap, warm = the model worked for it.
# --------------------------------------------------------------------------
DEPTH_COLORS = ["#1b3a4b", "#1f5673", "#227c9d", "#17a398", "#57cc99",
                "#c7f9cc", "#ffe066", "#ffb703", "#fb8500", "#f4743b",
                "#e63946", "#c1121f"]


def depth_color(depth: int, ceiling: int) -> str:
    if ceiling <= 1:
        return DEPTH_COLORS[0]
    f = min(max((depth - 1) / max(ceiling - 1, 1), 0.0), 1.0)
    return DEPTH_COLORS[min(int(f * (len(DEPTH_COLORS) - 1)), len(DEPTH_COLORS) - 1)]


# --------------------------------------------------------------------------
@dataclass
class Engine:
    """Wraps a XiosChat for streaming generation with per-token depth."""
    model: XiosChat
    tok: ByteBPE
    cfg: XiosConfig
    trained: bool
    device: str = "cpu"

    @classmethod
    def load(cls, ckpt: Optional[str], tokenizer: Optional[str],
             preset: str = "nano", device: Optional[str] = None) -> "Engine":
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tok = None
        if tokenizer and pathlib.Path(tokenizer).exists():
            tok = ByteBPE.load(tokenizer)
        elif ckpt:
            guess = pathlib.Path(ckpt).with_name("tokenizer.json")
            if guess.exists():
                tok = ByteBPE.load(guess)
        if tok is None:
            # Byte-level fallback: no merges, so every byte is its own token.
            # Round-trips any text, which is what a demo needs.
            tok = ByteBPE(merges=[], vocab_size=256 + 5)

        trained = False
        if ckpt and pathlib.Path(ckpt).exists():
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            cfg = XiosConfig(**blob["config"]) if "config" in blob else \
                get_config(preset, vocab_size=tok.actual_size)
            model = XiosChat(cfg)
            missing = model.load_state_dict(blob["model"], strict=False)
            trained = True
            if getattr(missing, "missing_keys", None):
                print(f"[warn] {len(missing.missing_keys)} missing keys in checkpoint")
        else:
            cfg = get_config(preset, vocab_size=tok.actual_size,
                             max_seq_len=1024)
            model = XiosChat(cfg)

        model.to(dev).eval()
        return cls(model=model, tok=tok, cfg=cfg, trained=trained, device=dev)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def stream(self, prompt: str, max_new_tokens: int, temperature: float,
               top_p: float, max_iters: Optional[int], stop_flag,
               out_q: "queue.Queue"):
        """Generate, pushing ('tok', text, depth) events onto out_q."""
        # Reserve room for the reply inside the context window, and keep the
        # most recent prompt tokens. The obvious one-liner
        # `ids[:, -max_seq_len + max_new_tokens:]` is wrong: with
        # max_new_tokens > max_seq_len it slices from a positive index and
        # hands the model an EMPTY prompt, which then crashes in the
        # depthwise conv rather than failing anywhere informative.
        max_new_tokens = max(1, min(max_new_tokens, self.cfg.max_seq_len - 8))
        keep = max(1, self.cfg.max_seq_len - max_new_tokens)
        ids = torch.tensor([self.tok.encode(prompt, bos=True)],
                           device=self.device)[:, -keep:]

        t0 = time.time()
        logits, state = self.model.prefill(ids, max_iters=max_iters)
        out_q.put(("status", f"prefilled {ids.shape[1]} tokens "
                             f"in {time.time() - t0:.2f}s", 0))

        produced = []
        for _ in range(max_new_tokens):
            if stop_flag.is_set():
                out_q.put(("status", "stopped", 0))
                break
            nxt = _sample(logits[:, -1], temperature, top_p, 0)
            tid = int(nxt)
            if tid == self.tok.eos_id:
                break
            produced.append(tid)
            # decode incrementally so multi-byte characters render correctly
            text = self.tok.decode(produced)
            piece = text[len(self.tok.decode(produced[:-1])):] if len(produced) > 1 \
                else text
            logits, state, depth = self.model.decode_step(
                nxt, state, max_iters=max_iters)
            out_q.put(("tok", piece, depth))

        out_q.put(("done", f"{len(produced)} tokens in {time.time() - t0:.2f}s",
                   0))


# --------------------------------------------------------------------------
class App:
    def __init__(self, root, engine: Engine, args):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = root
        self.engine = engine
        self.args = args
        self.q: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.busy = False
        self.history = ""
        self.reply = ""          # accumulated text of the reply in flight
        self.depths: list[int] = []

        root.title("XIOS")
        root.geometry("1100x740")
        root.minsize(820, 560)

        self._build_header()
        self._build_body()
        self._build_input()
        self._poll()

        if not engine.trained:
            self._sys(
                "No trained checkpoint loaded, so this model has RANDOM weights "
                "and its text will be nonsense. The depth visualisation and the "
                "search pipeline are still real and worth watching.\n\n"
                "To train a small language model (~20 min on a GPU, longer on CPU):\n"
                "    python train/train_text.py --dataset roneneldan/TinyStories \\\n"
                "        --preset nano --steps 4000 --out runs/text\n"
                "then relaunch with:\n"
                "    python app.py --ckpt runs/text/xios.pt "
                "--tokenizer runs/text/tokenizer.json\n")

    # ------------------------------------------------------------------
    def _build_header(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")

        e = self.engine
        state = "trained" if e.trained else "RANDOM WEIGHTS"
        ttk.Label(bar, text="XIOS", font=("Segoe UI", 15, "bold")).pack(side="left")
        ttk.Label(bar, text=f"  {e.cfg.dim}d · {e.model.n_params/1e6:.1f}M params · "
                            f"{e.cfg.stored_blocks} stored blocks · up to "
                            f"{e.cfg.max_iters} iterations · {e.device} · {state}",
                  foreground="#555").pack(side="left")

        self.search_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="web search", variable=self.search_var
                        ).pack(side="right", padx=6)

    def _build_body(self):
        tk, ttk = self.tk, self.ttk
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Conversation", font=("Segoe UI", 10, "bold")
                  ).pack(anchor="w")
        self.chat = tk.Text(left, wrap="word", font=("Consolas", 11),
                            bg="#0f1419", fg="#e6e6e6", insertbackground="#fff",
                            relief="flat", padx=10, pady=10)
        self.chat.pack(fill="both", expand=True, pady=(2, 8))
        self.chat.tag_configure("user", foreground="#7fd1ff",
                                font=("Consolas", 11, "bold"))
        self.chat.tag_configure("sys", foreground="#ffb703")
        self.chat.tag_configure("src", foreground="#8a8a8a",
                                font=("Consolas", 9))
        for i, c in enumerate(DEPTH_COLORS):
            self.chat.tag_configure(f"d{i}", foreground=c)
        self.chat.configure(state="disabled")

        right = ttk.Frame(body, width=290)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        ttk.Label(right, text="Thinking", font=("Segoe UI", 10, "bold")
                  ).pack(anchor="w")
        ttk.Label(right, text="iterations the core ran per token",
                  foreground="#777", font=("Segoe UI", 8)).pack(anchor="w")

        self.canvas = tk.Canvas(right, height=150, bg="#0f1419",
                                highlightthickness=0)
        self.canvas.pack(fill="x", pady=6)

        self.stats = tk.Text(right, height=13, font=("Consolas", 9),
                             bg="#0f1419", fg="#c9c9c9", relief="flat",
                             padx=8, pady=8)
        self.stats.pack(fill="both", expand=True)
        self.stats.configure(state="disabled")
        self._legend(right)

    def _legend(self, parent):
        tk = self.tk
        c = tk.Canvas(parent, height=26, bg="#0f1419", highlightthickness=0)
        c.pack(fill="x")
        n = len(DEPTH_COLORS)
        for i, col in enumerate(DEPTH_COLORS):
            c.create_rectangle(i * 20 + 4, 4, i * 20 + 22, 16, fill=col, width=0)
        c.create_text(6, 21, text="1 iter", anchor="w", fill="#777",
                      font=("Segoe UI", 7))
        c.create_text(n * 20, 21, text=f"{self.engine.cfg.max_iters}", anchor="e",
                      fill="#777", font=("Segoe UI", 7))

    def _build_input(self):
        tk, ttk = self.tk, self.ttk
        bot = ttk.Frame(self.root, padding=(10, 8))
        bot.pack(fill="x")

        self.entry = tk.Text(bot, height=3, font=("Consolas", 11), wrap="word")
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)

        col = ttk.Frame(bot)
        col.pack(side="right", padx=(8, 0))
        self.send_btn = ttk.Button(col, text="Send", command=self.send, width=10)
        self.send_btn.pack()
        self.stop_btn = ttk.Button(col, text="Stop", command=self.stop,
                                   width=10, state="disabled")
        self.stop_btn.pack(pady=(4, 0))

        opts = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        opts.pack(fill="x")
        self.temp = tk.DoubleVar(value=0.8)
        self.toks = tk.IntVar(value=96)
        self.iters = tk.IntVar(value=self.engine.cfg.max_iters)
        for label, var, frm, to, res in (
                ("temp", self.temp, 0.0, 1.5, 0.05),
                ("max tokens", self.toks, 8, 512, 8),
                ("max iterations", self.iters, 1, self.engine.cfg.max_iters, 1)):
            f = ttk.Frame(opts)
            f.pack(side="left", padx=(0, 18))
            ttk.Label(f, text=label, foreground="#666",
                      font=("Segoe UI", 8)).pack(anchor="w")
            ttk.Scale(f, from_=frm, to=to, variable=var, length=140,
                      command=lambda *_: self._refresh_opts()).pack()
            lbl = ttk.Label(f, text="", foreground="#888", font=("Segoe UI", 8))
            lbl.pack(anchor="w")
            f._lbl, f._var = lbl, var
            setattr(self, f"_opt_{label.replace(' ', '_')}", f)
        self._refresh_opts()

    def _refresh_opts(self):
        for name in ("temp", "max_tokens", "max_iterations"):
            f = getattr(self, f"_opt_{name}", None)
            if f is not None:
                v = f._var.get()
                f._lbl.configure(text=f"{v:.2f}" if isinstance(v, float) else str(int(v)))

    # ------------------------------------------------------------------
    def _write(self, text, *tags):
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tags)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _sys(self, text):
        self._write("\n" + text + "\n", "sys")

    def _on_return(self, event):
        if event.state & 0x0001:        # shift held -> newline
            return
        self.send()
        return "break"

    def stop(self):
        self.stop_flag.set()

    def send(self):
        if self.busy:
            return
        msg = self.entry.get("1.0", "end").strip()
        if not msg:
            return
        self.entry.delete("1.0", "end")
        self._write(f"\n> {msg}\n", "user")

        self.busy = True
        self.stop_flag.clear()
        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.depths = []
        self.reply = ""
        threading.Thread(target=self._worker, args=(msg,), daemon=True).start()

    def _worker(self, msg):
        prefix = ""
        if self.search_var.get():
            self.q.put(("status", "searching...", 0))
            results, err = websearch.search(msg, n=5)
            if err:
                self.q.put(("search_err", err, 0))
            else:
                self.q.put(("search", results, 0))
                prefix = websearch.build_context(results)

        self.history += f"<user>{msg}<assistant>"
        # keep the transcript inside the model's context; trim from the front
        budget = self.engine.cfg.max_seq_len * 3       # rough chars-per-token
        if len(self.history) > budget:
            self.history = self.history[-budget:]
        prompt = prefix + self.history
        try:
            self.engine.stream(
                prompt, int(self.toks.get()), float(self.temp.get()),
                0.95, int(self.iters.get()), self.stop_flag, self.q)
        except Exception as e:
            import traceback
            self.q.put(("error", f"{type(e).__name__}: {e}\n"
                                 f"{traceback.format_exc(limit=3)}", 0))

    # ------------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                kind, payload, depth = self.q.get_nowait()
                if kind == "tok":
                    ceiling = max(int(self.iters.get()), 1)
                    idx = DEPTH_COLORS.index(depth_color(depth, ceiling))
                    self._write(payload, f"d{idx}")
                    self.reply += payload
                    self.depths.append(depth)
                    self._draw()
                elif kind == "search":
                    self._write("\n  searched the web:\n", "src")
                    for i, r in enumerate(payload, 1):
                        self._write(f"    [{i}] {r.title}\n         {r.url}\n", "src")
                    self._write("\n")
                elif kind == "search_err":
                    self._sys(f"search failed ({payload}) - answering without it")
                elif kind == "status":
                    pass
                elif kind == "error":
                    self._sys(payload)
                    self._finish()
                elif kind == "done":
                    self.history += self.chat.get("end-2l", "end-1c")
                    self._write(f"\n  [{payload}]\n", "src")
                    self._finish()
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    def _finish(self):
        self.busy = False
        self.send_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._draw()

    def _draw(self):
        c = self.canvas
        c.delete("all")
        d = self.depths
        if not d:
            return
        w = int(c.winfo_width()) or 280
        h = 150
        ceiling = max(int(self.iters.get()), 1)
        bw = max(1.0, min(8.0, w / max(len(d), 1)))
        for i, v in enumerate(d[-int(w / bw):]):
            x = i * bw
            bh = (v / ceiling) * (h - 18)
            c.create_rectangle(x, h - bh - 4, x + max(bw - 1, 1), h - 4,
                               fill=depth_color(v, ceiling), width=0)
        c.create_text(4, 8, text=f"ceiling {ceiling}", anchor="w",
                      fill="#555", font=("Segoe UI", 7))

        import collections
        hist = collections.Counter(d)
        mean = sum(d) / len(d)
        core = self.engine.cfg.core_blocks
        spent = sum(d) * core
        dense = len(d) * self.engine.cfg.stored_blocks
        lines = [
            f"tokens        {len(d)}",
            f"mean depth    {mean:.2f}",
            f"min / max     {min(d)} / {max(d)}",
            f"at ceiling    {100*hist[ceiling]/len(d):.0f}%",
            "",
            "distribution",
        ]
        for k in sorted(hist):
            bar = "#" * max(1, int(24 * hist[k] / len(d)))
            lines.append(f"  {k:2d} {bar} {hist[k]}")
        lines += [
            "",
            f"core block-applications",
            f"  spent       {spent}",
            f"  dense equiv {dense}",
            f"  ratio       {spent/max(dense,1):.2f}x",
        ]
        self.stats.configure(state="normal")
        self.stats.delete("1.0", "end")
        self.stats.insert("1.0", "\n".join(lines))
        self.stats.configure(state="disabled")


# --------------------------------------------------------------------------
def selftest(engine: Engine) -> int:
    """Exercise the generation path headlessly (no display needed)."""
    print(f"engine: {engine.cfg.dim}d, {engine.model.n_params/1e6:.2f}M params, "
          f"trained={engine.trained}, vocab={engine.cfg.vocab_size}, "
          f"tok={engine.tok.actual_size}")
    q: queue.Queue = queue.Queue()
    engine.stream("hello there", 16, 0.8, 0.95, engine.cfg.max_iters,
                  threading.Event(), q)
    toks, depths = [], []
    while not q.empty():
        kind, payload, depth = q.get()
        if kind == "tok":
            toks.append(payload)
            depths.append(depth)
        elif kind in ("done", "status"):
            print(f"  [{kind}] {payload}")
    print(f"generated {len(toks)} tokens; depths {depths}")
    # ascii-safe: an untrained model emits arbitrary bytes, and a Windows
    # console in a legacy codepage will refuse to print them
    text = "".join(toks)
    print("text: " + text.encode("ascii", "backslashreplace").decode("ascii"))
    assert toks, "generated nothing"
    assert all(1 <= d <= engine.cfg.max_iters for d in depths), "depth out of range"

    res, err = websearch.search("xios adaptive depth", n=3)
    print(f"search: {len(res)} results" + (f" (error: {err})" if err else ""))
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description="XIOS chat UI")
    ap.add_argument("--ckpt", default=None, help="path to a trained .pt")
    ap.add_argument("--tokenizer", default=None, help="path to tokenizer.json")
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="run the generation path without opening a window")
    args = ap.parse_args()

    engine = Engine.load(args.ckpt, args.tokenizer, args.preset, args.device)
    if args.selftest:
        return selftest(engine)

    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    App(root, engine, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
