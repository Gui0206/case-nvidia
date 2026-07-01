"""Motor de diagnóstico: SignalSet -> Diagnosis completo (AIMS, Alavancagem, CDS, IPI, AI-washing)."""
from __future__ import annotations

from ..models import Diagnosis, SignalSet, clamp
from .aims import score_aims
from .leverage import score_leverage
from .cds import score_cds
from .ipi import compute_ipi


def detect_ai_washing(sig: SignalSet, aims_overall: float) -> tuple[bool, float]:
    """Adversarial e CONSERVADOR: só marca washing quando há afirmação de IA de marketing
    E sinais de wrapper puro E evidência técnica fraca. Evita falso-positivo em AI-enabled real."""
    technical = 0.0
    for f in ("publishes_hf_models", "trains_or_finetunes", "ml_repos_github",
              "publishes_papers", "own_serving_infra", "unique_data_source"):
        if sig.has(f):
            technical += 0.25
    technical = clamp(technical)
    marketing = 0.0
    if sig.has("marketing_only_ai_claim"):
        marketing += 0.6
    if sig.has("ai_is_primary_feature") or sig.has("ai_is_core_value"):
        marketing += 0.4
    marketing = clamp(marketing)
    wrapper = sig.has("thin_llm_wrapper") or sig.has("pure_api_no_optimization")
    gap = marketing - technical
    is_washing = bool(marketing >= 0.6 and wrapper and technical < 0.25 and gap > 0.5)
    return (is_washing, gap)


def diagnose(name: str, sig: SignalSet, vertical: str = "",
             discovery_edge: float = 0.5, weights: dict | None = None) -> Diagnosis:
    aims = score_aims(sig)
    lev = score_leverage(sig)
    cds = score_cds(sig)
    ipi = compute_ipi(aims, lev, cds, discovery_edge=discovery_edge, weights=weights)
    washing, gap = detect_ai_washing(sig, aims.overall)
    return Diagnosis(name=name, vertical=vertical, aims=aims, leverage=lev,
                     compute=cds, ipi=ipi, ai_washing=washing, washing_gap=gap)
