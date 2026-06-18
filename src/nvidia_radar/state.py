"""LangGraph state definition and progress-event helper.

The state is a ``TypedDict`` that flows through the graph. Domain objects are kept
as Pydantic instances for ergonomics; ``errors`` and ``progress`` use additive
reducers so every node can append without clobbering earlier entries.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, Optional, TypedDict

from .models import RawSource, SearchPlan, StartupProfile

# A callback the API/CLI can pass via ``config["configurable"]["progress_cb"]``
ProgressCallback = Callable[[dict], None]


class RadarState(TypedDict, total=False):
    # ---- inputs ----
    query: str
    max_startups: int
    run_id: str

    # ---- planner ----
    plan: Optional[SearchPlan]

    # ---- scraper ----
    raw_sources: list[RawSource]
    scrape_round: int
    needs_rescrape: bool

    # ---- per-startup records (accumulated by extractor onwards) ----
    startups: list[StartupProfile]

    # ---- RAG grounding per startup (name -> list of chunk dicts) ----
    rag_hits: dict[str, list[dict]]

    # ---- outputs ----
    portfolio_summary: Optional[str]
    stats: dict

    # ---- bookkeeping (additive) ----
    errors: Annotated[list[str], operator.add]
    progress: Annotated[list[dict], operator.add]


def emit(
    config: Optional[dict],
    stage: str,
    message: str,
    status: str = "running",
    data: Optional[dict[str, Any]] = None,
) -> dict:
    """Build a progress event and push it to the live callback if present.

    Returns the event so the calling node can also append it to ``progress``.
    """
    event = {"stage": stage, "message": message, "status": status, "data": data or {}}
    cb: Optional[ProgressCallback] = None
    if config:
        cb = config.get("configurable", {}).get("progress_cb")
    if cb:
        try:
            cb(event)
        except Exception:  # never let UI plumbing break the pipeline
            pass
    return event
