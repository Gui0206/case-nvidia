"""Calibra os limiares de tier do AIMS a partir do gold set (materializa o flywheel).

Busca em grade os limiares (native, enabled) que maximizam a acurácia de tier sobre os
rótulos de ground-truth, e grava em data/gold/calibration.json. O AIMS passa a usá-los.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import GOLD_PATH
from ..scoring.aims import tier_of
from .gold_eval import diagnose_gold, load_gold

OUT = GOLD_PATH.parent / "calibration.json"


def calibrate(write: bool = True) -> dict:
    gold = load_gold()
    scored = [(diagnose_gold(s).aims.overall, s["expected"]) for s in gold]
    best = None
    grid = [i / 100 for i in range(5, 95)]
    for enabled in grid:
        for native in grid:
            if native <= enabled:
                continue
            acc = sum(1 for a, exp in scored
                      if tier_of(a, {"native": native, "enabled": enabled}) == exp)
            if best is None or acc > best[0]:
                best = (acc, native, enabled)
    acc, native, enabled = best
    result = {"native": round(native, 3), "enabled": round(enabled, 3),
              "tier_accuracy": round(acc / len(scored), 3), "n": len(scored),
              "note": "limiares calibrados por busca em grade sobre o gold set (flywheel)"}
    if write:
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = calibrate()
    print(f"Calibração: native≥{r['native']}  enabled≥{r['enabled']}  "
          f"→ acurácia de tier {r['tier_accuracy']*100:.0f}%  (escrito em {OUT.name})")
