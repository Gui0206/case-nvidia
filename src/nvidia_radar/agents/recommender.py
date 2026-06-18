"""Recommendation Agent — maps startup gaps to NVIDIA technologies.

Strictly grounded in the RAG context: it may only recommend technologies present in
the retrieved knowledge-base chunks, and every recommendation must cite its source.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..llm import complete_structured
from ..models import Citation, Recommendation, StartupProfile
from ..state import RadarState, emit
from ._common import profile_brief, truncate

logger = logging.getLogger("nvidia_radar.recommender")

_SYSTEM = """Você é o Recommendation Agent da NVIDIA. Cruze o perfil e os GAPS técnicos da
startup com as tecnologias NVIDIA recuperadas da base de conhecimento (contexto RAG).

Regras:
- Recomende SOMENTE tecnologias presentes no contexto RAG fornecido. Não invente produtos.
- Gere de 2 a 5 recomendações, priorizando o que tem maior impacto para os gaps.
- Para cada recomendação:
  - technology, category
  - fit_score (0-100): aderência entre o gap e a tecnologia
  - priority em {alta, média, baixa}; complexity em {baixa, média, alta}
  - technical_justification: por que tecnicamente faz sentido
  - business_justification: impacto de negócio (custo, latência, defensibilidade, GTM)
  - next_action: próxima ação concreta para o time NVIDIA Inception
  - addresses_gaps: quais gaps (pelo nome da área) a recomendação resolve
  - citations: itens com technology, source (URL da base) e snippet usados como fundamento
- Sempre que fizer sentido, inclua o NVIDIA Inception como porta de entrada (créditos, suporte).
"""


class RecommendationResult(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list)


def _rag_block(hits: list[dict]) -> str:
    if not hits:
        return "—"
    out = []
    for h in hits:
        out.append(
            f"### {h['technology']} (fonte: {h.get('source','')})\n"
            f"{truncate(h.get('text',''), 700)}"
        )
    return "\n\n".join(out)


def _gaps_block(p: StartupProfile) -> str:
    if not p.gaps:
        return "—"
    return "\n".join(f"- [{g.severity}] {g.area}: {g.description}" for g in p.gaps)


def _recommend_one(p: StartupProfile, hits: list[dict]) -> list[Recommendation]:
    threat = f"{p.lab_threat.risk_score} ({p.lab_threat.level})" if p.lab_threat else "—"
    maturity = p.maturity.overall if p.maturity else "—"
    user = (
        f"{profile_brief(p)}\n\n"
        f"Maturidade AI-native: {maturity}/100 · Risco de deslocamento por labs: {threat}\n\n"
        f"GAPS técnicos:\n{_gaps_block(p)}\n\n"
        f"CONTEXTO RAG (tecnologias NVIDIA recuperadas):\n{_rag_block(hits)}"
    )
    res = complete_structured(RecommendationResult, _SYSTEM, user, fast=False)
    # Backfill citation sources from the RAG hits when the model omits them.
    src_by_tech = {h["technology"]: h.get("source", "") for h in hits}
    for rec in res.recommendations:
        if not rec.citations and rec.technology in src_by_tech:
            rec.citations = [Citation(technology=rec.technology,
                                      source=src_by_tech[rec.technology])]
    # Rank by fit then priority.
    order = {"alta": 0, "média": 1, "baixa": 2}
    res.recommendations.sort(key=lambda r: (order.get(r.priority, 1), -r.fit_score))
    return res.recommendations


def node(state: RadarState, config=None) -> dict:
    startups: list[StartupProfile] = state.get("startups", [])
    rag_hits: dict[str, list[dict]] = state.get("rag_hits", {})
    emit(config, "recommender", f"Gerando recomendações NVIDIA para {len(startups)} startups…")
    errors: list[str] = []

    for p in startups:
        try:
            recs = _recommend_one(p, rag_hits.get(p.name, []))
            p.recommendations = recs
            emit(config, "recommender",
                 f"{p.name}: {len(recs)} recomendações ({', '.join(r.technology for r in recs[:4])})",
                 data={"name": p.name, "count": len(recs)})
        except Exception as err:
            errors.append(f"recommend failed for {p.name}: {err}")
            logger.warning("recommendation failed for %s: %s", p.name, err)

    emit(config, "recommender", "Motor de recomendação concluído", status="done")
    return {
        "startups": startups,
        "errors": errors,
        "progress": [{"stage": "recommender", "status": "done",
                      "message": "Recomendações NVIDIA geradas"}],
    }
