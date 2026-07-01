"""IPI — Índice de Prioridade Inception. Funde os 4 componentes; decomposto e auditável."""
from __future__ import annotations

from ..models import ComputeScore, IPI, LeverageScore, MaturityScore
from .confidence import worst

DEFAULT_WEIGHTS = {"R": 0.30, "L": 0.30, "C": 0.30, "D": 0.10}


def compute_ipi(aims: MaturityScore, lev: LeverageScore, cds: ComputeScore,
                discovery_edge: float = 0.5, weights: dict | None = None) -> IPI:
    w = weights or DEFAULT_WEIGHTS
    realness = aims.overall
    leverage = lev.leverage
    compute_urg = cds.score
    breakdown = {"realness": realness, "leverage": leverage,
                 "compute_urgency": compute_urg, "discovery": discovery_edge}
    raw = (w["R"] * realness + w["L"] * leverage +
           w["C"] * compute_urg + w["D"] * discovery_edge)
    conf = worst(aims.confidence, lev.confidence, cds.confidence)
    return IPI(ipi=round(100 * raw), confidence=conf, breakdown=breakdown,
               weights=dict(w), assumptions=cds.assumptions)
