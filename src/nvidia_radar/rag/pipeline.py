"""Hybrid retrieval + reranking pipeline with citations."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from ..config import get_settings
from . import bm25 as bm25mod
from . import vectorstore
from .embeddings import get_embedder
from .reranker import get_reranker

logger = logging.getLogger("nvidia_radar.rag")


@dataclass
class RetrievedChunk:
    technology: str
    text: str
    source: str
    section: str
    score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None

    def citation(self) -> dict:
        snippet = self.text.split("\n", 1)[-1][:280].strip()
        return {
            "technology": self.technology,
            "source": self.source,
            "snippet": snippet,
            "score": round(self.score, 4),
        }


def _rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    fused: dict[int, float] = {}
    for ids in rank_lists:
        for rank, cid in enumerate(ids):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


class RagPipeline:
    def __init__(self):
        self.bm25 = bm25mod.load()
        if self.bm25 is None:
            raise RuntimeError(
                "Knowledge base index not found. Run `radar build-kb` first."
            )
        self.chunks = self.bm25.chunks
        self.embedder = get_embedder()
        self.reranker = get_reranker()

    def search(self, query: str, top_k: int | None = None,
               candidates: int | None = None) -> list[RetrievedChunk]:
        s = get_settings()
        top_k = top_k or s.radar_rag_top_k
        candidates = candidates or s.radar_rag_candidates

        # --- dense (vector) ---
        dense = vectorstore.search(self.embedder.embed_query(query), limit=candidates)
        dense_ids = [cid for cid, _ in dense]
        dense_rank = {cid: r for r, cid in enumerate(dense_ids)}

        # --- lexical (BM25) ---
        lexical = self.bm25.search(query, limit=candidates)
        lexical_ids = [cid for cid, _ in lexical]
        lexical_rank = {cid: r for r, cid in enumerate(lexical_ids)}

        # --- fuse ---
        fused = _rrf([dense_ids, lexical_ids])
        if not fused:  # nothing matched either way
            return []
        fused_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:candidates]

        # --- rerank ---
        docs = [self.chunks[cid]["text"] for cid in fused_ids]
        reranked = self.reranker.rerank(query, docs, top_n=min(top_k, len(docs)))

        results: list[RetrievedChunk] = []
        for local_idx, score in reranked:
            cid = fused_ids[local_idx]
            c = self.chunks[cid]
            results.append(
                RetrievedChunk(
                    technology=c["technology"],
                    text=c["text"],
                    source=c["source"],
                    section=c["section"],
                    score=score,
                    dense_rank=dense_rank.get(cid),
                    lexical_rank=lexical_rank.get(cid),
                )
            )
        return results

    def reranker_name(self) -> str:
        return getattr(self.reranker, "name", "unknown")


@lru_cache
def get_pipeline() -> RagPipeline:
    return RagPipeline()
