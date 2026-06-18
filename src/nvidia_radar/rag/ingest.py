"""Ingest the NVIDIA knowledge base: parse -> chunk -> embed -> index.

Builds both stores from ``data/nvidia_kb/*.md``:
  * Qdrant collection (dense vectors)
  * BM25 pickle (lexical) — chunk ids are shared between the two.
"""
from __future__ import annotations

import logging
import re

from ..config import KB_DIR, ensure_dirs
from . import bm25, vectorstore
from .embeddings import get_embedder

logger = logging.getLogger("nvidia_radar.ingest")

_LIST_FIELDS = {"aliases", "use_cases"}
_MAX_CHUNK_CHARS = 1400


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    meta: dict = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            block, body = parts[1], parts[2]
            for line in block.strip().splitlines():
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if key in _LIST_FIELDS:
                    meta[key] = [v.strip() for v in val.split(",") if v.strip()]
                else:
                    meta[key] = val
    return meta, body.strip()


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (section_title, section_text) on '## ' headers."""
    sections: list[tuple[str, str]] = []
    current_title = "Visão geral"
    buffer: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if buffer:
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title = line[3:].strip()
            buffer = []
        elif line.startswith("# "):
            continue  # the document H1 (technology name) is already in metadata
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_title, "\n".join(buffer).strip()))
    return [(t, c) for t, c in sections if c]


def _chunk_text(text: str) -> list[str]:
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text]
    out, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) > _MAX_CHUNK_CHARS and buf:
            out.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        out.append(buf.strip())
    return out


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(KB_DIR.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        tech = meta.get("technology", path.stem)
        for section_title, section_text in _split_sections(body):
            for piece in _chunk_text(section_text):
                # Prefix with tech + section so the embedding is self-describing.
                text = f"{tech} — {section_title}\n{piece}"
                chunks.append(
                    {
                        "id": len(chunks),
                        "text": text,
                        "technology": tech,
                        "category": meta.get("category", ""),
                        "source": meta.get("source", ""),
                        "aliases": meta.get("aliases", []),
                        "use_cases": meta.get("use_cases", []),
                        "section": section_title,
                        "doc": path.stem,
                    }
                )
    return chunks


def ingest(verbose: bool = True) -> int:
    ensure_dirs()
    chunks = build_chunks()
    if not chunks:
        raise RuntimeError(f"No knowledge-base documents found in {KB_DIR}")

    embedder = get_embedder()
    vectors = embedder.embed_passages([c["text"] for c in chunks])

    vectorstore.recreate_collection(embedder.dim)
    vectorstore.upsert(
        ids=[c["id"] for c in chunks],
        vectors=vectors,
        payloads=[
            {"technology": c["technology"], "source": c["source"],
             "section": c["section"], "text": c["text"]}
            for c in chunks
        ],
    )
    bm25.build(chunks)

    if verbose:
        techs = sorted({c["technology"] for c in chunks})
        logger.info("Ingested %d chunks across %d technologies", len(chunks), len(techs))
        print(f"✓ Indexed {len(chunks)} chunks from {len(techs)} NVIDIA technologies")
        print(f"  embedding: {embedder.model_name} ({embedder.dim}d) · vector + BM25 stores built")
    return len(chunks)
