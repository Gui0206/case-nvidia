"""Reranking stage.

Uses Cohere Rerank when ``COHERE_API_KEY`` is set; otherwise a local ONNX
cross-encoder (``fastembed.rerank``). Both expose the same interface and return
``(original_index, score)`` pairs sorted best-first.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from ..config import get_settings

logger = logging.getLogger("nvidia_radar.reranker")


class _CohereReranker:
    def __init__(self, api_key: str, model: str):
        import cohere

        self.client = cohere.Client(api_key)
        self.model = model
        self.name = f"cohere:{model}"

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        res = self.client.rerank(query=query, documents=docs, model=self.model, top_n=top_n)
        return [(r.index, float(r.relevance_score)) for r in res.results]


class _LocalReranker:
    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model)
        self.name = f"local:{model}"

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        scores = list(self.model.rerank(query, docs))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, float(s)) for i, s in ranked[:top_n]]


@lru_cache
def get_reranker():
    s = get_settings()
    if s.has_cohere:
        try:
            return _CohereReranker(s.cohere_api_key, s.radar_cohere_rerank_model)
        except Exception as err:
            logger.warning("Cohere reranker unavailable (%s); using local cross-encoder", err)
    return _LocalReranker(s.radar_rerank_local_model)
