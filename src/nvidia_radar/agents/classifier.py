"""Startup Classifier Agent — classification + AI-native maturity scoring.

This is the project's differentiator: a transparent, evidence-grounded rubric that
answers "why isn't every startup AI-native like the best ones?" and quantifies the
risk of being displaced by foundational labs (the "wrapper risk").
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..llm import complete_structured
from ..models import Classification, Gap, LabThreat, MaturityScore, StartupProfile
from ..state import RadarState, emit
from ._common import profile_brief, sources_block, truncate

logger = logging.getLogger("nvidia_radar.classifier")

_SYSTEM = """Você é o Startup Classifier de uma plataforma da NVIDIA que qualifica startups
brasileiras para o programa NVIDIA Inception.

Classifique a empresa e pontue sua maturidade AI-native com base nas EVIDÊNCIAS fornecidas.
Seja rigoroso e calibrado: se a evidência é fraca, seja conservador e aponte isso nos gaps.

## Classificação
- "AI-native": IA é o núcleo do valor. A empresa PRODUZ IA (treina/fine-tuna/post-traina
  modelos) e/ou tem dados proprietários + workflow profundo + defensibilidade real.
- "AI-enabled": usa IA de forma relevante, mas majoritariamente via APIs externas, com
  menos dados proprietários e pouca otimização técnica da stack.
- "non-AI": IA é periférica ou inexistente.
- "unknown": evidência insuficiente para decidir.

## Maturity score (0-100 cada dimensão)
- proprietary_data: possui dados únicos / data flywheel.
- model_engineering: treina, fine-tuna ou faz post-training de modelos próprios (sinal forte).
- inference_optimization: faz engenharia de custo/latência (não apenas chama API).
- workflow_depth: entrega o RESULTADO de ponta a ponta, não só uma ferramenta fina.
- ai_in_product: o quão central a IA é para o valor entregue.
- defensibility: moat frente à comoditização pelos grandes labs (OpenAI, Anthropic, Google).
- overall: visão holística (não precisa ser média exata).

## Lab Displacement Risk (lab_threat.risk_score 0-100; maior = mais ameaçada)
Avalie o risco de a empresa ser substituída por funcionalidades nativas dos grandes labs.
Wrappers finos de LLM, sem dados próprios nem otimização, têm risco ALTO. Liste
`threat_vectors` (o que a ameaça) e `moats` (o que a protege). Defina `level` em
{baixo, médio, alto, crítico} coerente com o score.

## Gaps (2 a 5)
Lacunas TÉCNICAS de stack que a NVIDIA poderia endereçar (custo/latência de inferência,
governança de agentes, processamento de dados em escala, voz, infra de treino, etc.),
cada uma com severity em {baixa, média, alta}.

## inception_fit (0-100)
O quão atraente a empresa é como alvo do NVIDIA Inception: combine maturidade/potencial,
gaps endereçáveis pela stack NVIDIA e o fato de ser brasileira. Uma empresa promissora mas
ameaçada é uma ÓTIMA oportunidade de nutrição.
"""


class ClassificationResult(BaseModel):
    classification: Classification
    classification_rationale: str
    maturity: MaturityScore
    lab_threat: LabThreat
    gaps: list[Gap] = Field(default_factory=list)
    inception_fit: int = 50


def _evidence_block(p: StartupProfile, limit: int = 12) -> str:
    if not p.evidence:
        return "—"
    return "\n".join(
        f"- {truncate(e.claim, 160)} (fonte: {e.source_url or '?'})" for e in p.evidence[:limit]
    )


def _classify_one(p: StartupProfile) -> ClassificationResult:
    user = (
        f"{profile_brief(p, include_gaps=False)}\n\n"
        f"Evidências coletadas:\n{_evidence_block(p)}\n\n"
        f"Fontes:\n{sources_block(p)}"
    )
    return complete_structured(ClassificationResult, _SYSTEM, user, fast=False)


def node(state: RadarState, config=None) -> dict:
    startups: list[StartupProfile] = state.get("startups", [])
    emit(config, "classifier", f"Classificando maturidade AI-native de {len(startups)} startups…")
    errors: list[str] = []

    for p in startups:
        try:
            res = _classify_one(p)
            p.classification = res.classification
            p.classification_rationale = res.classification_rationale
            p.maturity = res.maturity
            p.lab_threat = res.lab_threat
            p.gaps = res.gaps
            p.inception_fit = res.inception_fit
            emit(config, "classifier",
                 f"{p.name}: {p.classification} · maturidade {res.maturity.overall} · "
                 f"risco labs {res.lab_threat.risk_score}",
                 data={"name": p.name, "classification": p.classification,
                       "maturity": res.maturity.overall, "threat": res.lab_threat.risk_score})
        except Exception as err:
            errors.append(f"classify failed for {p.name}: {err}")
            logger.warning("classification failed for %s: %s", p.name, err)

    emit(config, "classifier", "Diagnóstico de maturidade concluído", status="done")
    return {
        "startups": startups,
        "errors": errors,
        "progress": [{"stage": "classifier", "status": "done",
                      "message": "Maturidade AI-native diagnosticada"}],
    }
