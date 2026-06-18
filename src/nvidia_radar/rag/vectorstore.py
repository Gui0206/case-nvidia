"""Qdrant (embedded / local file mode) vector store.

A single client per process — embedded Qdrant locks its storage directory, so the
ingestion script and the API server must not run simultaneously.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import QDRANT_DIR, get_settings


@lru_cache
def get_client():
    from qdrant_client import QdrantClient

    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_DIR))


def recreate_collection(dim: int) -> None:
    from qdrant_client.models import Distance, VectorParams

    client = get_client()
    name = get_settings().qdrant_collection
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert(ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
    from qdrant_client.models import PointStruct

    client = get_client()
    points = [
        PointStruct(id=i, vector=v, payload=p)
        for i, v, p in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=get_settings().qdrant_collection, points=points)


def search(vector: list[float], limit: int) -> list[tuple[int, float]]:
    """Return (chunk_id, cosine_score) pairs."""
    client = get_client()
    resp = client.query_points(
        collection_name=get_settings().qdrant_collection,
        query=vector,
        limit=limit,
        with_payload=False,
    )
    return [(int(p.id), float(p.score)) for p in resp.points]


def close() -> None:
    """Release the embedded-Qdrant lock (call when a short-lived process exits)."""
    try:
        get_client().close()
        get_client.cache_clear()
    except Exception:
        pass


def collection_size() -> int:
    try:
        return get_client().count(get_settings().qdrant_collection).count
    except Exception:
        return 0
