"""Briefing Agent — gera o briefing executivo em Markdown a partir do diagnóstico completo.

100% offline (template determinístico). Se LLM disponível, pode refinar o texto (opcional).
"""
from __future__ import annotations

from ..models import Diagnosis, SignalSet
from .recommender import recommend
from .wedge import wedge

_DIM_PT = {"model_engineering": "Engenharia de modelos", "proprietary_data": "Dado proprietário",
           "workflow_depth": "Profundidade de workflow", "inference_optimization": "Otimização de inferência",
           "ai_in_product": "IA no produto", "defensibility": "Defensibilidade"}


def build_briefing(diag: Diagnosis, sig: SignalSet, rag=None) -> str:
    a, l, c, ipi = diag.aims, diag.leverage, diag.compute, diag.ipi
    recs = recommend(diag, sig, rag=rag)
    w = wedge(diag, sig)
    L = []
    L.append(f"# Briefing — {diag.name}")
    L.append(f"*{diag.vertical} · IPI **{ipi.ipi}** (confiança {ipi.confidence}) · "
             f"tier **{a.tier}** (AIMS {a.overall*100:.0f})*\n")
    if diag.ai_washing:
        L.append("> ⚠️ **AI-washing detectado** — narrativa de IA sem evidência técnica. Descartar/monitorar.\n")

    L.append("## Por que agora (Ponto de Alavancagem)")
    L.append(f"- **Alavancagem** {l.leverage*100:.0f}/100 — {l.quadrant}")
    L.append(f"- **Lab Displacement Risk** {l.ldr*100:.0f} · **Resgatabilidade NVIDIA** {l.res*100:.0f}")
    L.append(f"- **Urgência de compute (CDS)** {c.score*100:.0f} (confiança {c.confidence})")
    if c.assumptions:
        L.append(f"  - inferido de: {', '.join(c.assumptions)}")
    L.append("")

    L.append("## Diagnóstico de maturidade (AIMS)")
    for k, ds in a.dimensions.items():
        bar = "█" * round(ds.score * 10) + "░" * (10 - round(ds.score * 10))
        L.append(f"- {_DIM_PT[k]:26} `{bar}` {ds.score*100:>3.0f}  "
                 + (f"({', '.join(ds.fired)})" if ds.fired else "(sem evidência)"))
    L.append("")

    L.append("## Recomendações NVIDIA")
    if not recs:
        L.append("_Sem gaps endereçáveis pela stack NVIDIA no momento._")
    for r in recs:
        L.append(f"### {r['tech']}  ·  prioridade {r['priority']} · complexidade {r['complexity']}")
        L.append(f"- **Técnica:** {r['tech_just']}")
        L.append(f"- **Negócio:** {r['biz_just']}")
        L.append(f"- **Próxima ação:** {r['next_action']}")
        L.append(f"- **Hook Inception:** {r['hook']}")
        if r.get("citations"):
            L.append(f"- **Fonte:** {r['citations'][0]['source']}")
    L.append("")

    if w:
        L.append("## Cunha técnica (hipótese quantificada)")
        L.append(f"> {w['hypothesis']}")
        L.append(f"- Premissas: {'; '.join(w['premises'])}")
        L.append("")

    if sig.evidence:
        L.append("## Fontes")
        for e in sig.evidence:
            if e.source_url:
                L.append(f"- {e.source_url}")
    return "\n".join(L)
