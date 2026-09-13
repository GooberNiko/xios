"""Minimal web search, no API key and no dependencies.

Scrapes DuckDuckGo's HTML endpoint. This is deliberately small and will
break whenever they change their markup -- it is a convenience for the demo
UI, not infrastructure. Every failure mode returns an empty list with a
reason rather than raising, because a chat UI should degrade to "no results"
rather than fall over.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Two independent passes rather than one combined pattern: attribute order
# differs between the title and snippet anchors, and a single regex spanning
# both is brittle enough that it silently matched nothing the first time.
_TITLE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAGS = re.compile(r"<[^>]+>")


@dataclass
class Result:
    title: str
    snippet: str
    url: str

    def as_context(self) -> str:
        return f"{self.title}: {self.snippet}"


def _clean(s: str) -> str:
    return html.unescape(_TAGS.sub("", s)).strip()


def _unwrap(u: str) -> str:
    """DuckDuckGo wraps outbound links in a redirector."""
    if "uddg=" in u:
        q = urllib.parse.urlparse(u).query
        v = urllib.parse.parse_qs(q).get("uddg")
        if v:
            return v[0]
    return u


def search(query: str, n: int = 5, timeout: float = 10.0
           ) -> tuple[list[Result], Optional[str]]:
    """Returns (results, error). Never raises."""
    if not query.strip():
        return [], "empty query"
    try:
        data = urllib.parse.urlencode({"q": query, "kl": "wt-wt"}).encode()
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/", data=data,
            headers={"User-Agent": UA,
                     "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            page = r.read().decode("utf-8", errors="replace")
    except Exception as e:                      # offline, blocked, DNS, TLS...
        return [], f"{type(e).__name__}: {e}"

    titles = _TITLE.findall(page)
    snippets = _SNIPPET.findall(page)
    out: list[Result] = []
    for (url, title), snip in zip(titles, snippets):
        t, sn = _clean(title), _clean(snip)
        if t and sn:
            out.append(Result(t, sn, _unwrap(url)))
        if len(out) >= n:
            break
    if not out:
        hint = ("the page layout may have changed"
                if "result__a" not in page else "titles found but no snippets")
        return [], f"no results parsed ({hint})"
    return out, None


def build_context(results: list[Result], max_chars: int = 1200) -> str:
    """Pack results into a prompt preamble, truncated to a budget."""
    lines, used = [], 0
    for i, r in enumerate(results, 1):
        line = f"[{i}] {r.as_context()}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return "Search results:\n" + "\n".join(lines) + "\n\n"


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "what is a transformer model"
    res, err = search(q)
    print(f"query: {q!r}")
    if err:
        print(f"error: {err}")
    for r in res:
        print(f"  - {r.title}\n    {r.snippet[:120]}\n    {r.url}")
