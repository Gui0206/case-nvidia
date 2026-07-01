"""Alavancagem = LDR × Resgatabilidade. O coração da tese do Ponto de Alavancagem."""
from __future__ import annotations

from ..models import LeverageScore, SignalSet, clamp
from . import features as F
from .confidence import label


def _accumulate(sig: SignalSet, base: float, feats):
    total = base
    fired = []
    for feat, delta in feats:
        if sig.has(feat):
            total += delta
            fired.append(feat)
    return clamp(total), fired


def quadrant(ldr: float, res: float) -> str:
    hi_ldr, hi_res = ldr >= 0.55, res >= 0.55
    if hi_ldr and hi_res:
        return "PONTO DE ALAVANCAGEM — ameaçada pelos labs e resgatável pela stack NVIDIA (prioridade máxima)"
    if hi_ldr and not hi_res:
        return "ameaçada, mas a solução está fora do alcance da NVIDIA (produto/distribuição) — despriorizar"
    if not hi_ldr and hi_res:
        return "defensável e de fácil adoção da stack NVIDIA — nutrir com créditos"
    return "baixo fit estratégico — prioridade baixa"


def score_leverage(sig: SignalSet) -> LeverageScore:
    ldr, ldr_fired = _accumulate(sig, F.LDR_BASE, F.LDR_FEATURES)
    res, res_fired = _accumulate(sig, F.RES_BASE, F.RES_FEATURES)
    n = len(F.LDR_FEATURES) + len(F.RES_FEATURES)
    conf = label((len(ldr_fired) + len(res_fired)) / n)
    return LeverageScore(ldr=ldr, res=res, leverage=ldr * res,
                         ldr_fired=ldr_fired, res_fired=res_fired,
                         confidence=conf, quadrant=quadrant(ldr, res))
