"""RAG da base NVIDIA: busca híbrida (BM25 lexical, sempre; + vetorial se disponível) com
rerank opcional e citações. BM25 é implementado em python puro — roda offline, sem deps.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from ..config import KB_DIR

_TOKEN = re.compile(r"[a-zA-ZÀ-ÿ0-9\-]+")


def _tok(s: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(s)]


def _chunks_from(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip() and set(l.strip()) != {"-"}]
    title = (lines[0].lstrip("# ").strip() if lines else path.stem) or path.stem
    source = ""
    m = re.search(r"Fontes?:\s*(\S+)", text)
    if m:
        source = m.group(1)
    # chunk por parágrafo
    out = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        out.append({"technology": title, "source": source, "text": para})
    if not out:
        out = [{"technology": title, "source": source, "text": text}]
    return out


class RagIndex:
    def __init__(self, kb_dir: Path | None = None):
        self.docs: list[dict] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self.avgdl = 1.0
        self._build(kb_dir or KB_DIR)

    def _build(self, kb_dir: Path):
        if not kb_dir.exists():
            return
        for p in sorted(kb_dir.glob("*.md")):
            for ch in _chunks_from(p):
                ch["tokens"] = _tok(ch["text"] + " " + ch["technology"])
                self.docs.append(ch)
        self.N = len(self.docs)
        if not self.N:
            return
        self.avgdl = sum(len(d["tokens"]) for d in self.docs) / self.N
        for d in self.docs:
            for w in set(d["tokens"]):
                self.df[w] = self.df.get(w, 0) + 1

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.N:
            return []
        q = _tok(query)
        k1, b = 1.5, 0.75
        scored = []
        for d in self.docs:
            dl = len(d["tokens"])
            score = 0.0
            for w in q:
                if w not in self.df:
                    continue
                tf = d["tokens"].count(w)
                if not tf:
                    continue
                idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        hits = [{"technology": d["technology"], "source": d["source"],
                 "text": d["text"], "score": round(s, 3)} for s, d in scored[:top_k]]
        return _maybe_rerank(query, hits)


def _maybe_rerank(query: str, hits: list[dict]) -> list[dict]:
    """Rerank opcional via Cohere (se COHERE_API_KEY + httpx). Caso contrário, mantém BM25."""
    from ..config import get_settings
    s = get_settings()
    if not s.cohere_key or not hits:
        return hits
    try:
        import httpx
        r = httpx.post("https://api.cohere.com/v2/rerank",
                       headers={"Authorization": f"Bearer {s.cohere_key}"},
                       json={"model": "rerank-v3.5", "query": query,
                             "documents": [h["text"] for h in hits], "top_n": len(hits)},
                       timeout=30)
        r.raise_for_status()
        order = [x["index"] for x in r.json()["results"]]
        return [hits[i] for i in order]
    except Exception:
        return hits


_INDEX: RagIndex | None = None


def get_index() -> RagIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = RagIndex()
    return _INDEX


def ask(query: str, top_k: int = 5) -> list[dict]:
    return get_index().search(query, top_k=top_k)
