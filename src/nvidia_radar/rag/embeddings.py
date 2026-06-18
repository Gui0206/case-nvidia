"""Local embeddings via fastembed (ONNX — no torch, CPU-friendly, multilingual-ish)."""
from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


class Embedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # Probe dimensionality once.
        self._dim = len(next(iter(self._model.embed(["dimension probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models benefit from the query-specific instruction; fall back gracefully.
        try:
            return list(map(float, next(iter(self._model.query_embed([text])))))
        except Exception:
            return list(map(float, next(iter(self._model.embed([text])))))


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings().radar_embed_model)
