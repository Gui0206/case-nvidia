"""Technical Wedge — hipótese de valor quantificada, com premissas DECLARADAS (sem falsa precisão)."""
from __future__ import annotations

from ..models import Diagnosis, SignalSet

_MODEL_COST = {"small": ("classe pequena", 10), "7b": ("classe 7B", 15),
               "70b": ("classe 70B", 45), "405b+": ("classe 405B+", 70), "none": ("modelo próprio", 30)}


def wedge(diag: Diagnosis, sig: SignalSet) -> dict | None:
    if diag.compute.score < 0.35:
        return None
    label, cost_idx = _MODEL_COST.get(sig.model_class, ("modelo atual", 30))
    api = sig.has("uses_only_external_api") or sig.has("external_api_plus_growth")
    cost_cut = 40 if api else 25
    lat_cut = 45 if (sig.has("latency_sensitive_product") or sig.has("realtime_critical")) else 25
    premises = ["estimativa de ORDEM DE GRANDEZA, não medição",
                f"modelo assumido: {label}",
                f"consumo inferido de: {', '.join(diag.compute.assumptions[:3]) or 'sinais gerais'}"]
    text = (f"A {diag.name} opera com {label}"
            + (" via API externa" if api else "")
            + f" (confiança {diag.compute.confidence}). "
            f"Migrando para NIM + TensorRT-LLM (GPU L40S/H100), estimativa direcional: "
            f"−{cost_cut}% custo/token e −{lat_cut}% latência p95. "
            f"PoC viável em ~2 semanas com créditos Inception.")
    return {"hypothesis": text, "est_cost_reduction_pct": cost_cut,
            "est_latency_reduction_pct": lat_cut, "premises": premises}
