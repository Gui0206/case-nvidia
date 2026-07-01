"""CDS — Compute Demand Score. Estimativa DIRECIONAL, com premissas visíveis. Sem falsa precisão."""
from __future__ import annotations

from ..models import ComputeScore, SignalSet, clamp
from . import features as F
from .confidence import label

_HUMAN = {
    "trains_models": "treina/pós-treina modelos",
    "heavy_modality": "modalidade pesada (voz/visão/robótica)",
    "large_model_class": "usa modelos de classe grande (70B+)",
    "high_inference_volume": "alto volume de inferência",
    "realtime_critical": "latência crítica ao produto",
    "external_api_plus_growth": "depende de API externa e está crescendo",
    "cost_reduction_jobs": "vagas de redução de custo de inferência",
    "public_cost_complaints": "menções públicas de custo de API",
    "latency_sensitive_product": "produto sensível a latência",
}


def _derive(sig: SignalSet) -> set[str]:
    """Deriva features de compute a partir de modality/model_class."""
    extra = set()
    if sig.modality in F.HEAVY_MODALITIES:
        extra.add("heavy_modality")
    if sig.model_class in F.LARGE_MODEL_CLASSES:
        extra.add("large_model_class")
    if sig.has("trains_or_finetunes") or sig.has("publishes_hf_models"):
        extra.add("trains_models")
    return extra


def _accumulate(active: set[str], feats):
    total, fired = 0.0, []
    for feat, delta in feats:
        if feat in active:
            total += delta
            fired.append(feat)
    return clamp(total), fired


def score_cds(sig: SignalSet) -> ComputeScore:
    active = set(sig.active) | _derive(sig)
    magnitude, mag_fired = _accumulate(active, F.CDS_MAGNITUDE)
    pain, pain_fired = _accumulate(active, F.CDS_PAIN)
    score = clamp(0.55 * magnitude + 0.45 * pain)
    n = len(F.CDS_MAGNITUDE) + len(F.CDS_PAIN)
    conf = label((len(mag_fired) + len(pain_fired)) / n)
    assumptions = [_HUMAN.get(f, f) for f in (mag_fired + pain_fired)]
    return ComputeScore(score=score, magnitude=magnitude, pain=pain,
                        confidence=conf, assumptions=assumptions)
