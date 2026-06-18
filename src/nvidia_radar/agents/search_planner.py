"""Search Planner Agent — turns a user query into a concrete discovery plan."""
from __future__ import annotations

from ..llm import complete_structured
from ..models import SearchPlan
from ..state import RadarState, emit

_SYSTEM = """Você é o Search Planner de uma plataforma da NVIDIA que mapeia startups
brasileiras AI-native para o programa NVIDIA Inception.

Seu trabalho: transformar a consulta do usuário em um plano de descoberta acionável.
Você conhece bem o ecossistema de startups de IA do Brasil.

Produza:
- interpreted_query: a intenção reformulada com clareza.
- sector_focus: setores/verticais relevantes à consulta.
- queries: 4 a 6 buscas web em português, específicas, para descobrir empresas
  (use fontes como sites oficiais, NeoFeed, Brazil Journal, Exame, StartSe, Distrito).
- candidate_startups: 5 a 10 NOMES concretos de startups brasileiras reais e plausíveis
  que se encaixem na consulta (priorize empresas com sinais de uso intensivo de IA).
  Estes nomes serão VERIFICADOS por scraping — não invente nomes implausíveis.
- priority_sources: domínios/fontes prioritárias para buscar.
- notes: observações úteis para os próximos agentes.
"""


def node(state: RadarState, config=None) -> dict:
    query = state["query"]
    emit(config, "planner", f"Planejando descoberta para: “{query}”")
    plan = complete_structured(
        SearchPlan,
        _SYSTEM,
        f"Consulta do usuário: {query}\n"
        f"Número-alvo de startups a analisar: {state.get('max_startups', 3)}",
        fast=False,
    )
    emit(
        config,
        "planner",
        f"{len(plan.candidate_startups)} candidatos · {len(plan.queries)} buscas planejadas",
        status="done",
        data={"candidates": plan.candidate_startups, "queries": plan.queries,
              "sectors": plan.sector_focus},
    )
    return {
        "plan": plan,
        "progress": [
            {"stage": "planner", "status": "done",
             "message": f"Plano pronto: {len(plan.candidate_startups)} candidatos"}
        ],
    }
