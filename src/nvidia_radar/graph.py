"""Orquestração multi-agente.

Monta um StateGraph do LangGraph se a lib estiver instalada; caso contrário, roda um
pipeline sequencial equivalente (mesmos nós, mesma ordem). Ambos produzem um Diagnosis +
briefing. Para demo offline, o 'extractor' resolve sinais a partir do gold set.
"""
from __future__ import annotations

from .models import Evidence, SignalSet
from .scoring.engine import diagnose
from .agents.briefing import build_briefing
from .agents.recommender import recommend
from .eval.gold_eval import load_gold


def _gold_lookup(name: str) -> dict | None:
    name_l = name.lower().strip()
    for s in load_gold():
        if name_l in s["name"].lower() or s["name"].lower().startswith(name_l):
            return s
    return None


# ── nós (funções de estado) ──────────────────────────────────────────────────
def node_extractor(state: dict) -> dict:
    """Resolve sinais. Offline: usa gold. (Com LLM+scraping, extrairia de páginas reais.)"""
    if state.get("signals"):
        return state
    g = _gold_lookup(state.get("name") or state.get("query", ""))
    if g:
        state.update(name=g["name"], signals=g["signals"], modality=g.get("modality", "text"),
                     model_class=g.get("model_class", "none"), vertical=g.get("vertical", ""),
                     discovery_edge=g.get("discovery_edge", 0.5),
                     evidence=[{"source_url": u} for u in g.get("evidence", [])])
    else:
        state.setdefault("errors", []).append(
            f"'{state.get('name') or state.get('query')}' não está no gold set e não há LLM/scraping configurado.")
        state.setdefault("signals", [])
    return state


def node_diagnose(state: dict) -> dict:
    sig = SignalSet(active=set(state.get("signals", [])),
                    modality=state.get("modality", "text"),
                    model_class=state.get("model_class", "none"),
                    evidence=[Evidence(claim="", source_url=e.get("source_url", ""))
                              for e in state.get("evidence", [])])
    state["_sig"] = sig
    state["diagnosis"] = diagnose(state.get("name", "?"), sig, state.get("vertical", ""),
                                  discovery_edge=state.get("discovery_edge", 0.5))
    return state


def node_recommend(state: dict) -> dict:
    from .rag.pipeline import get_index
    rag = get_index()
    state["recommendations"] = recommend(state["diagnosis"], state["_sig"], rag=rag)
    return state


def node_briefing(state: dict) -> dict:
    from .rag.pipeline import get_index
    state["briefing"] = build_briefing(state["diagnosis"], state["_sig"], rag=get_index())
    return state


def build_langgraph():
    """Retorna um app LangGraph compilado, se a lib estiver disponível."""
    try:
        from langgraph.graph import StateGraph, START, END
        from .state import RadarState
    except Exception:
        return None
    g = StateGraph(RadarState)
    g.add_node("extractor", node_extractor)
    g.add_node("diagnose", node_diagnose)
    g.add_node("recommend", node_recommend)
    g.add_node("briefing", node_briefing)
    g.add_edge(START, "extractor")
    g.add_edge("extractor", "diagnose")
    g.add_edge("diagnose", "recommend")
    g.add_edge("recommend", "briefing")
    g.add_edge("briefing", END)
    return g.compile()


def run(name_or_query: str) -> dict:
    """Executa o pipeline (LangGraph se disponível; senão sequencial)."""
    state = {"name": name_or_query, "query": name_or_query}
    app = build_langgraph()
    if app is not None:
        state = app.invoke(state)
    else:
        for node in (node_extractor, node_diagnose, node_recommend, node_briefing):
            state = node(state)
    return state
