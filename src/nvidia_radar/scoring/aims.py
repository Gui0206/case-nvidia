"""AIMS — AI-Native Maturity Score (6 dimensões, determinístico e rastreável).

Os limiares de tier são CALIBRÁVEIS a partir do gold set (flywheel). Se
data/gold/calibration.json existir, usa-os; senão usa os defaults do Kit (0.70/0.40).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..models import DimensionScore, MaturityScore, SignalSet, clamp
from . import features as F
from .confidence import label

_DEFAULT_THRESHOLDS = {"native": 0.70, "enabled": 0.40}
_CALIB_PATH = Path(__file__).resolve().parents[3] / "data" / "gold" / "calibration.json"


def _thresholds() -> dict:
    if _CALIB_PATH.exists():
        try:
            c = json.loads(_CALIB_PATH.read_text(encoding="utf-8"))
            return {"native": c["native"], "enabled": c["enabled"]}
        except Exception:
            pass
    return dict(_DEFAULT_THRESHOLDS)


def _score_dimension(name: str, sig: SignalSet) -> DimensionScore:
    _w, base, feats = F.AIMS_DIMENSIONS[name]
    total = base
    fired: list[str] = []
    for feat, delta in feats:
        if sig.has(feat):
            total += delta
            fired.append(feat)
    return DimensionScore(name=name, score=clamp(total), fired=fired, has_evidence=bool(fired))


def tier_of(overall: float, thresholds: dict | None = None) -> str:
    t = thresholds or _thresholds()
    if overall >= t["native"]:
        return "AI-native"
    if overall >= t["enabled"]:
        return "AI-enabled"
    return "non-AI"


def score_aims(sig: SignalSet) -> MaturityScore:
    dims: dict[str, DimensionScore] = {}
    overall = 0.0
    for name, (weight, _b, _f) in F.AIMS_DIMENSIONS.items():
        ds = _score_dimension(name, sig)
        dims[name] = ds
        overall += weight * ds.score
    covered = sum(1 for d in dims.values() if d.fired)
    return MaturityScore(overall=overall, dimensions=dims,
                         tier=tier_of(overall), confidence=label(covered / len(dims)))
