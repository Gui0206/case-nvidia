"""Lexical BM25 index — the keyword half of hybrid retrieval.

Persisted as a single pickle holding both the BM25 model and the canonical chunk
list (so chunk ids line up exactly with the Qdrant point ids).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from ..config import BM25_DIR

_INDEX_PATH = BM25_DIR / "index.pkl"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build(chunks: list[dict]) -> None:
    """Build and persist a BM25 index over ``chunks`` (each must have a 'text')."""
    from rank_bm25 import BM25Okapi

    BM25_DIR.mkdir(parents=True, exist_ok=True)
    corpus = [tokenize(c["text"] + " " + c.get("technology", "")) for c in chunks]
    bm25 = BM25Okapi(corpus)
    with open(_INDEX_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": chunks}, fh)


class Bm25Index:
    def __init__(self, bm25, chunks: list[dict]):
        self.bm25 = bm25
        self.chunks = chunks

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:limit] if s > 0]


def load() -> Bm25Index | None:
    if not _INDEX_PATH.exists():
        return None
    with open(_INDEX_PATH, "rb") as fh:
        data = pickle.load(fh)
    return Bm25Index(data["bm25"], data["chunks"])


def index_path() -> Path:
    return _INDEX_PATH
