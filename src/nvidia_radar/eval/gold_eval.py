"""Avaliação contra o conjunto gold: roda o scoring determinístico em todas as startups,
ranqueia por IPI e por AIMS, e mede quão bem o AIMS separa os rótulos esperados.

Métricas (stdlib, sem numpy/scipy):
- Spearman ρ entre AIMS e ordem-alvo (AI-native > AI-enabled > non-AI)
- acurácia de tier (AIMS.tier == rótulo esperado)
- precision/recall do detector de AI-washing sobre os rótulos non-AI
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import GOLD_PATH
from ..models import Evidence, SignalSet
from ..scoring.engine import diagnose

TIER_RANK = {"AI-native": 2, "AI-enabled": 1, "non-AI": 0, "unknown": 0}


def load_gold(path: Path | None = None) -> list[dict]:
    p = path or GOLD_PATH
    return json.loads(p.read_text(encoding="utf-8"))["startups"]


def diagnose_gold(s: dict):
    sig = SignalSet(active=set(s["signals"]), modality=s.get("modality", "text"),
                    model_class=s.get("model_class", "none"),
                    evidence=[Evidence(claim="", source_url=u, source_type="web")
                              for u in s.get("evidence", [])])
    return diagnose(s["name"], sig, s.get("vertical", ""),
                    discovery_edge=s.get("discovery_edge", 0.5))


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n * n - 1))


def evaluate(path: Path | None = None) -> dict:
    gold = load_gold(path)
    rows = []
    for s in gold:
        d = diagnose_gold(s)
        rows.append({"name": s["name"], "vertical": s.get("vertical", ""),
                     "expected": s["expected"], "aims": d.aims.overall,
                     "tier": d.aims.tier, "ipi": d.ipi.ipi, "conf": d.ipi.confidence,
                     "leverage": d.leverage.leverage, "quadrant_hot": d.leverage.ldr >= 0.55 and d.leverage.res >= 0.55,
                     "washing": d.ai_washing, "diag": d})
    aims_vals = [r["aims"] for r in rows]
    target = [TIER_RANK[r["expected"]] for r in rows]
    rho = _spearman(aims_vals, target)
    tier_acc = sum(1 for r in rows if r["tier"] == r["expected"]) / len(rows)
    # washing: positivos esperados = non-AI
    tp = sum(1 for r in rows if r["washing"] and r["expected"] == "non-AI")
    fp = sum(1 for r in rows if r["washing"] and r["expected"] != "non-AI")
    fn = sum(1 for r in rows if not r["washing"] and r["expected"] == "non-AI")
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {"rows": rows, "spearman": rho, "tier_accuracy": tier_acc,
            "washing_precision": prec, "washing_recall": rec}


def print_report(path: Path | None = None) -> dict:
    r = evaluate(path)
    rows = sorted(r["rows"], key=lambda x: -x["ipi"])
    print("\n=== NVIDIA Startup AI Radar — avaliação do gold set ===\n")
    print(f"{'#':>2}  {'IPI':>3} {'AIMS':>4} {'conf':>5} {'tier':>10} {'esperado':>10} {'washing':>7}  startup")
    print("-" * 88)
    for i, x in enumerate(rows, 1):
        hot = "🔥" if x["quadrant_hot"] else "  "
        wsh = "WASH" if x["washing"] else " -  "
        ok = "" if x["tier"] == x["expected"] else "  ⚠tier"
        print(f"{i:>2}  {x['ipi']:>3} {x['aims']*100:>4.0f} {x['conf']:>5} {x['tier']:>10} "
              f"{x['expected']:>10} {wsh:>7} {hot} {x['name']}{ok}")
    print("-" * 88)
    print(f"Spearman ρ (AIMS vs rótulo):  {r['spearman']:.3f}   (meta ≥ 0.80)")
    print(f"Acurácia de tier:             {r['tier_accuracy']*100:.0f}%")
    print(f"AI-washing precision/recall:  {r['washing_precision']:.2f} / {r['washing_recall']:.2f}")
    print()
    return r


if __name__ == "__main__":
    print_report()


def map_data(path: Path | None = None, with_briefing: bool = True) -> list[dict]:
    """Dado do Mapa de Alavancagem, produzido pelo motor de scoring real (não hardcoded).

    Se with_briefing, embute o briefing de cada startup (mapa auto-suficiente, sem servidor).
    """
    from ..models import Evidence, SignalSet
    briefing_fn = None
    if with_briefing:
        try:
            from ..agents.briefing import build_briefing
            from ..rag.pipeline import get_index
            rag = get_index()
            briefing_fn = lambda sig, dg: build_briefing(dg, sig, rag=rag)
        except Exception:
            briefing_fn = None
    rows = []
    for s in load_gold(path):
        d = diagnose_gold(s)
        brief = ""
        if briefing_fn:
            sig = SignalSet(active=set(s["signals"]), modality=s.get("modality", "text"),
                            model_class=s.get("model_class", "none"),
                            evidence=[Evidence(claim="", source_url=u) for u in s.get("evidence", [])])
            try:
                brief = briefing_fn(sig, d)
            except Exception:
                brief = ""
        rows.append({
            "briefing": brief,
            "name": s["name"], "vertical": s.get("vertical", ""),
            "one_liner": s.get("one_liner", ""), "expected": s["expected"],
            "res": round(d.leverage.res * 100), "cds": round(d.compute.score * 100),
            "aims": round(d.aims.overall * 100), "ldr": round(d.leverage.ldr * 100),
            "tier": d.aims.tier, "ipi": d.ipi.ipi, "confidence": d.ipi.confidence,
            "leverage": round(d.leverage.leverage * 100), "quadrant": d.leverage.quadrant,
            "washing": d.ai_washing, "note": s.get("note", ""),
        })
    return rows
