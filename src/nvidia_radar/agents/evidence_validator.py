"""Evidence Validator Agent — is the profile sufficiently sourced to trust?

Sets ``evidence_quality``, ``confidence`` and ``needs_more_evidence``. The last flag
drives the conditional edge back to the Scraper for a deepening pass (bounded retry).
"""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from ..llm import complete_structured
from ..models import StartupProfile
from ..state import RadarState, emit
from ._common import profile_brief, sources_block

logger = logging.getLogger("nvidia_radar.validator")

_SYSTEM = """Você é o Evidence Validator. Avalie se o perfil de uma startup está
suficientemente sustentado por fontes públicas para apoiar uma decisão de negócio.

- evidence_quality: "forte" | "média" | "fraca".
- confidence: 0.0 a 1.0, calibrada à qualidade e quantidade das fontes.
- needs_more_evidence: marque True APENAS se houver menos de 2 fontes distintas
  OU se afirmações centrais (o que a empresa faz e como usa IA) não tiverem suporte.
- notes: o que falta confirmar (ex.: founders, funding, uso real de IA).
"""


class ValidationResult(BaseModel):
    evidence_quality: Literal["forte", "média", "fraca"]
    confidence: float
    needs_more_evidence: bool
    notes: str = ""


def node(state: RadarState, config=None) -> dict:
    startups: list[StartupProfile] = state.get("startups", [])
    scrape_round = state.get("scrape_round", 1)
    emit(config, "validator", f"Validando evidências de {len(startups)} perfis…")
    errors: list[str] = []
    flagged = 0

    for p in startups:
        try:
            user = (
                f"{profile_brief(p, include_gaps=False)}\n\n"
                f"Nº de fontes: {len(p.sources)}\nFontes:\n{sources_block(p)}\n\n"
                f"Nº de evidências registradas: {len(p.evidence)}"
            )
            res = complete_structured(ValidationResult, _SYSTEM, user, fast=True)
            p.evidence_quality = res.evidence_quality
            p.confidence = max(0.0, min(1.0, res.confidence))
            # Only allow another scrape pass while we still have retries left.
            p.needs_more_evidence = bool(res.needs_more_evidence) and scrape_round < 2
            flagged += int(p.needs_more_evidence)
        except Exception as err:
            errors.append(f"validate failed for {p.name}: {err}")
            logger.warning("validation failed for %s: %s", p.name, err)

    emit(config, "validator",
         f"Validação concluída · {flagged} perfil(is) precisam de reforço",
         status="done", data={"flagged": flagged})
    return {
        "startups": startups,
        "errors": errors,
        "progress": [{"stage": "validator", "status": "done",
                      "message": f"Evidências validadas ({flagged} p/ reforço)"}],
    }
