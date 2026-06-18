"""Briefing Agent — executive briefings for the NVIDIA Startups & VCs manager."""
from __future__ import annotations

import logging

from ..llm import complete_text
from ..models import StartupProfile
from ..state import RadarState, emit
from ._common import profile_brief, sources_block

logger = logging.getLogger("nvidia_radar.briefing")

_SYSTEM = """Você é o Briefing Agent da NVIDIA. Escreva um briefing executivo em português,
claro e acionável, para o gerente de Startups & VCs da NVIDIA no Brasil (programa Inception).
Use markdown enxuto. Não invente fatos além dos fornecidos. Seja direto e estratégico."""

_PORTFOLIO_SYSTEM = """Você é o Briefing Agent da NVIDIA. Sintetize um panorama de portfólio em
português (markdown) para o gerente de Startups & VCs, respondendo: quais startups são
AI-native e por quê, quais são apenas AI-enabled (e o risco de virarem irrelevantes frente aos
grandes labs), e a ordem de priorização para abordagem pelo NVIDIA Inception."""


def _recs_text(p: StartupProfile) -> str:
    if not p.recommendations:
        return "—"
    out = []
    for r in p.recommendations:
        cites = ", ".join(c.source for c in r.citations if c.source) or "base NVIDIA"
        out.append(
            f"- **{r.technology}** (prioridade {r.priority}, fit {r.fit_score}, "
            f"complexidade {r.complexity})\n"
            f"  - Técnico: {r.technical_justification}\n"
            f"  - Negócio: {r.business_justification}\n"
            f"  - Próxima ação: {r.next_action}\n"
            f"  - Fontes: {cites}"
        )
    return "\n".join(out)


def _maturity_text(p: StartupProfile) -> str:
    if not p.maturity:
        return "—"
    dims = " · ".join(f"{k}: {v}" for k, v in p.maturity.radar().items())
    return f"Overall {p.maturity.overall}/100 — {dims}"


def _brief_one(p: StartupProfile) -> str:
    threat = (
        f"{p.lab_threat.risk_score}/100 ({p.lab_threat.level}); "
        f"vetores: {', '.join(p.lab_threat.threat_vectors[:4]) or '—'}; "
        f"moats: {', '.join(p.lab_threat.moats[:4]) or '—'}"
        if p.lab_threat else "—"
    )
    gaps = "\n".join(f"- [{g.severity}] {g.area}: {g.description}" for g in p.gaps) or "—"
    user = f"""Gere o briefing executivo da startup abaixo.

{profile_brief(p, include_gaps=False)}

Classificação: {p.classification}
Racional: {p.classification_rationale or '—'}
Maturidade AI-native: {_maturity_text(p)}
Lab Displacement Risk: {threat}
Inception fit: {p.inception_fit if p.inception_fit is not None else '—'}/100
Qualidade da evidência: {p.evidence_quality or '—'} (confiança {p.confidence if p.confidence is not None else '—'})

GAPS:
{gaps}

RECOMENDAÇÕES NVIDIA:
{_recs_text(p)}

FONTES:
{sources_block(p)}

Estruture com: Resumo executivo · Por que é (ou não) AI-native · Ameaça dos grandes labs ·
Recomendações NVIDIA priorizadas · Abordagem sugerida (técnica, comercial e de comunidade via
Inception) · Fontes."""
    return complete_text(_SYSTEM, user, fast=False)


def _portfolio(query: str, startups: list[StartupProfile]) -> str:
    rows = []
    for p in sorted(startups, key=lambda x: (x.inception_fit or 0), reverse=True):
        rows.append(
            f"- {p.name} | classe: {p.classification} | "
            f"maturidade: {p.maturity.overall if p.maturity else '—'} | "
            f"risco labs: {p.lab_threat.risk_score if p.lab_threat else '—'} | "
            f"inception_fit: {p.inception_fit if p.inception_fit is not None else '—'} | "
            f"setor: {p.sector or '—'}"
        )
    user = (
        f"Consulta original: {query}\n\nStartups analisadas (ordenadas por inception_fit):\n"
        + "\n".join(rows)
        + "\n\nProduza: (1) Top picks para abordar primeiro e por quê; (2) quem é AI-native vs "
        "AI-enabled e a leitura de risco; (3) padrões/insights do conjunto; (4) próximos passos "
        "para o time NVIDIA. Markdown enxuto."
    )
    return complete_text(_PORTFOLIO_SYSTEM, user, fast=False)


def node(state: RadarState, config=None) -> dict:
    startups: list[StartupProfile] = state.get("startups", [])
    emit(config, "briefing", f"Redigindo briefings executivos ({len(startups)})…")
    errors: list[str] = []

    for p in startups:
        try:
            p.briefing = _brief_one(p)
            emit(config, "briefing", f"Briefing pronto: {p.name}", data={"name": p.name})
        except Exception as err:
            errors.append(f"briefing failed for {p.name}: {err}")
            logger.warning("briefing failed for %s: %s", p.name, err)

    portfolio = None
    if startups:
        try:
            portfolio = _portfolio(state.get("query", ""), startups)
        except Exception as err:
            errors.append(f"portfolio briefing failed: {err}")

    emit(config, "briefing", "Briefings concluídos", status="done")
    return {
        "startups": startups,
        "portfolio_summary": portfolio,
        "errors": errors,
        "progress": [{"stage": "briefing", "status": "done", "message": "Briefings executivos prontos"}],
    }
