"""Retrieval-Augmented Generation over the NVIDIA knowledge base.

Pipeline: hybrid retrieval (dense vectors via Qdrant + lexical BM25, fused with
Reciprocal Rank Fusion) followed by reranking (Cohere Rerank if configured, else a
local ONNX cross-encoder) and citation-carrying results.
"""
from .pipeline import RagPipeline, RetrievedChunk, get_pipeline

__all__ = ["RagPipeline", "RetrievedChunk", "get_pipeline"]
