"""Confiança = quão coberta por evidência está a decisão. Nunca fingir certeza."""
from __future__ import annotations


def label(ratio: float) -> str:
    if ratio >= 0.6:
        return "alta"
    if ratio >= 0.3:
        return "média"
    return "baixa"


def worst(*labels: str) -> str:
    order = {"baixa": 0, "média": 1, "alta": 2}
    return min(labels, key=lambda x: order.get(x, 1)) if labels else "média"
