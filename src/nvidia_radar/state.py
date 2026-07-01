"""Estado compartilhado do pipeline (compatível com LangGraph; funciona sem ele)."""
from __future__ import annotations

from typing import Any, TypedDict


class RadarState(TypedDict, total=False):
    query: str
    name: str
    signals: list[str]
    modality: str
    model_class: str
    vertical: str
    evidence: list[dict]
    discovery_edge: float
    diagnosis: Any        # Diagnosis
    briefing: str
    recommendations: list[dict]
    errors: list[str]
