"""NVIDIA RAG Agent — retrieves grounding from the NVIDIA knowledge base.

For each startup it issues one query per diagnosed gap (plus a sector query), fuses
and reranks results, and stores citation-bearing chunks for the Recommender. Results
are stored as plain dicts so they survive checkpoint serialization.
"""
from __future__ import annotations

import logging

from ..rag import get_pipeline
from ..state import RadarState, emit
from ..models import StartupProfile

logger = logging.getLogger("nvidia_radar.rag_agent")


def _queries_for(p: StartupProfile) -> list[str]:
    qs: list[str] = []
    base = f"{p.sector or ''} {' '.join(p.ai_technologies[:6])}".strip()
    for g in p.gaps[:4]:
        qs.append(f"{g.area}: {g.description} ({base})".strip())
    if base:
        qs.append(f"Startup {p.classification}: {base}")
    if not qs:
        qs.append(p.one_liner or p.description or p.name)
    return qs


def _retrieve(pipeline, p: StartupProfile, per_query: int = 3, cap: int = 8) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    hits: list[dict] = []
    for q in _queries_for(p):
        try:
            for ch in pipeline.search(q, top_k=per_query):
                key = (ch.technology, ch.section)
                if key in seen:
                    continue
                seen.add(key)
                hits.append({
                    "technology": ch.technology,
                    "source": ch.source,
                    "section": ch.section,
                    "text": ch.text,
                    "snippet": ch.citation()["snippet"],
                    "score": ch.score,
                })
        except Exception as err:
            logger.warning("RAG query failed (%s): %s", q, err)
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:cap]


def node(state: RadarState, config=None) -> dict:
    startups: list[StartupProfile] = state.get("startups", [])
    emit(config, "rag", "Consultando base de conhecimento NVIDIA (híbrido + rerank)…")
    errors: list[str] = []
    rag_hits: dict[str, list[dict]] = {}

    try:
        pipeline = get_pipeline()
    except Exception as err:
        emit(config, "rag", f"RAG indisponível: {err}", status="error")
        return {"errors": [f"RAG pipeline unavailable: {err}"],
                "rag_hits": {},
                "progress": [{"stage": "rag", "status": "error", "message": str(err)}]}

    for p in startups:
        try:
            hits = _retrieve(pipeline, p)
            rag_hits[p.name] = hits
            techs = sorted({h["technology"] for h in hits})
            emit(config, "rag", f"{p.name}: {len(hits)} trechos · {', '.join(techs[:5])}",
                 data={"name": p.name, "technologies": techs})
        except Exception as err:
            errors.append(f"rag failed for {p.name}: {err}")

    emit(config, "rag", f"Recuperação concluída via {pipeline.reranker_name()}", status="done")
    return {
        "rag_hits": rag_hits,
        "errors": errors,
        "progress": [{"stage": "rag", "status": "done",
                      "message": "Base NVIDIA consultada com reranking"}],
    }
