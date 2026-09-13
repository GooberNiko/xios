"""Byte-level BPE. Self-contained, no dependencies.

Deliberately simple: the architecture is the variable under test, so the
tokenizer needs to be identical for XIOS and the baseline and not much else.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Iterable


class ByteBPE:
    def __init__(self, merges: list[tuple[int, int]] | None = None,
                 vocab_size: int = 8192, specials: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<user>", "<assistant>")):
        self.merges = merges or []
        self.vocab_size = vocab_size
        self.specials = specials
        self._rebuild()

    # ------------------------------------------------------------------
    def _rebuild(self):
        self.n_special = len(self.specials)
        self.special_ids = {s: 256 + i for i, s in enumerate(self.specials)}
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        base = 256 + self.n_special
        self.merge_id = {pair: base + i for i, pair in enumerate(self.merges)}
        self.id_to_pair = {v: k for k, v in self.merge_id.items()}
        self.actual_size = base + len(self.merges)

    @property
    def pad_id(self): return self.special_ids["<pad>"]

    @property
    def bos_id(self): return self.special_ids["<bos>"]

    @property
    def eos_id(self): return self.special_ids["<eos>"]

    # ------------------------------------------------------------------
    def train(self, texts: Iterable[str], verbose: bool = True) -> "ByteBPE":
        target = self.vocab_size - 256 - self.n_special
        words: Counter = Counter()
        for t in texts:
            for w in t.replace("\n", " \n ").split(" "):
                if w:
                    words[w] += 1
        seqs = {w: list(w.encode("utf-8")) for w in words}
        freqs = dict(words)

        merges: list[tuple[int, int]] = []
        for step in range(target):
            pairs: Counter = Counter()
            for w, s in seqs.items():
                f = freqs[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best, cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            new_id = 256 + self.n_special + len(merges)
            merges.append(best)
            for w, s in list(seqs.items()):
                if len(s) < 2:
                    continue
                out, i = [], 0
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                        out.append(new_id); i += 2
                    else:
                        out.append(s[i]); i += 1
                seqs[w] = out
            if verbose and (step + 1) % 500 == 0:
                print(f"  bpe merge {step + 1}/{target}  (top pair count {cnt})")
        self.merges = merges
        self._rebuild()
        return self

    # ------------------------------------------------------------------
    def _encode_word(self, word: str) -> list[int]:
        s = list(word.encode("utf-8"))
        while len(s) >= 2:
            best, bi = None, None
            for i, pair in enumerate(zip(s, s[1:])):
                r = self.merge_rank.get(pair)
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            pair = (s[bi], s[bi + 1])
            s[bi:bi + 2] = [self.merge_id[pair]]
        return s

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        out: list[int] = [self.bos_id] if bos else []
        parts = text.replace("\n", " \n ").split(" ")
        for i, w in enumerate(parts):
            if not w:
                continue
            out.extend(self._encode_word(w if i == 0 else " " + w))
        if eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        out = bytearray()
        rev_special = {v: k for k, v in self.special_ids.items()}

        def expand(tok: int, acc: bytearray):
            if tok < 256:
                acc.append(tok)
            elif tok in rev_special:
                acc.extend(rev_special[tok].encode())
            else:
                a, b = self.id_to_pair[tok]
                expand(a, acc); expand(b, acc)

        for t in ids:
            expand(int(t), out)
        return out.decode("utf-8", errors="replace").replace(" \n ", "\n")

    # ------------------------------------------------------------------
    def save(self, path):
        pathlib.Path(path).write_text(json.dumps({
            "vocab_size": self.vocab_size, "specials": list(self.specials),
            "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ByteBPE":
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(merges=[tuple(m) for m in d["merges"]],
                   vocab_size=d["vocab_size"], specials=tuple(d["specials"]))
