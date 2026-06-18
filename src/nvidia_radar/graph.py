"""LangGraph assembly + run orchestration.

Flow:
    START -> planner -> scraper -> [has sources?] -> extractor -> classifier
          -> validator -> [needs more evidence & retries left?]
                              ├─ yes -> scraper (deepening pass)
                              └─ no  -> rag_agent -> recommender -> briefing -> END

Demonstrates the LangGraph features called for in the brief: typed state, conditional
transitions, a bounded retry loop, and a checkpointer.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import (
    briefing,
    classifier,
    evidence_validator,
    extractor,
    rag_agent,
    recommender,
    scraper,
    search_planner,
)
from .config import get_settings
from .models import RadarRun
from .state import RadarState

logger = logging.getLogger("nvidia_radar.graph")


def _route_after_scraper(state: RadarState) -> str:
    return "extractor" if state.get("raw_sources") else "end"


def _route_after_validation(state: RadarState) -> str:
    scrape_round = state.get("scrape_round", 1)
    flagged = any(getattr(p, "needs_more_evidence", False) for p in state.get("startups", []))
    if scrape_round < 2 and flagged:
        return "scraper"
    return "rag_agent"


def build_graph(checkpointer=None):
    g = StateGraph(RadarState)
    g.add_node("planner", search_planner.node)
    g.add_node("scraper", scraper.node)
    g.add_node("extractor", extractor.node)
    g.add_node("classifier", classifier.node)
    g.add_node("validator", evidence_validator.node)
    g.add_node("rag_agent", rag_agent.node)
    g.add_node("recommender", recommender.node)
    g.add_node("briefing", briefing.node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "scraper")
    g.add_conditional_edges("scraper", _route_after_scraper,
                            {"extractor": "extractor", "end": END})
    g.add_edge("extractor", "classifier")
    g.add_edge("classifier", "validator")
    g.add_conditional_edges("validator", _route_after_validation,
                            {"scraper": "scraper", "rag_agent": "rag_agent"})
    g.add_edge("rag_agent", "recommender")
    g.add_edge("recommender", "briefing")
    g.add_edge("briefing", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stats(state: RadarState) -> dict:
    startups = state.get("startups", [])
    native = sum(1 for p in startups if p.classification == "AI-native")
    enabled = sum(1 for p in startups if p.classification == "AI-enabled")
    maturities = [p.maturity.overall for p in startups if p.maturity]
    return {
        "startups": len(startups),
        "ai_native": native,
        "ai_enabled": enabled,
        "pages_scraped": len(state.get("raw_sources", [])),
        "scrape_rounds": state.get("scrape_round", 1),
        "avg_maturity": round(sum(maturities) / len(maturities), 1) if maturities else None,
        "errors": len(state.get("errors", [])),
    }


def run_radar(
    query: str,
    max_startups: int | None = None,
    progress_cb=None,
    thread_id: str | None = None,
    persist: bool = True,
) -> RadarRun:
    """Execute the full multi-agent pipeline for a query and return a RadarRun."""
    s = get_settings()
    if not s.has_llm:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot run the pipeline.")

    run_id = thread_id or uuid.uuid4().hex[:12]
    max_startups = max_startups or s.default_max_startups
    graph = build_graph()
    config = {
        "configurable": {"thread_id": run_id, "progress_cb": progress_cb},
        "recursion_limit": 50,
    }
    init: RadarState = {"query": query, "max_startups": max_startups, "run_id": run_id}

    logger.info("Radar run %s started: %r (max %d)", run_id, query, max_startups)
    final = graph.invoke(init, config=config)

    run = RadarRun(
        run_id=run_id,
        query=query,
        created_at=_now_iso(),
        status="completed",
        startups=final.get("startups", []),
        portfolio_summary=final.get("portfolio_summary"),
        errors=final.get("errors", []),
        stats=_stats(final),
    )
    if persist:
        try:
            from .db.store import save_run

            save_run(run)
        except Exception as err:  # persistence must never fail a run
            logger.warning("could not persist run %s: %s", run_id, err)
    return run
