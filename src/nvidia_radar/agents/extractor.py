"""Extractor Agent — unstructured pages -> structured StartupProfile facts."""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..llm import complete_structured
from ..models import Evidence, Founder, RawSource, StartupProfile
from ..state import RadarState, emit
from ._common import truncate

logger = logging.getLogger("nvidia_radar.extractor")

_PER_SOURCE_CHARS = 3500
_TOTAL_CHARS = 11000

_SYSTEM = """Você é o Extractor de uma plataforma de inteligência da NVIDIA.
A partir de páginas públicas (site oficial, notícias, blog) de UMA startup, extraia
fatos estruturados.

Regras:
- Extraia APENAS o que estiver sustentado pelo texto. NÃO invente.
- Se um campo for desconhecido, deixe nulo (ou lista vazia).
- Para cada afirmação relevante (setor, funding, clientes, tecnologias, founders),
  inclua um item em `evidence` com a frase/snippet e a URL da fonte correspondente.
- `ai_technologies`: tecnologias/abordagens de IA citadas (LLMs, visão, voz, ML, modelos
  próprios, fine-tuning, APIs de terceiros, etc.).
- `website`: o site oficial da empresa, se identificável.
"""


class ExtractionResult(BaseModel):
    name: str
    website: str | None = None
    location: str | None = None
    sector: str | None = None
    one_liner: str | None = None
    description: str | None = None
    products: list[str] = Field(default_factory=list)
    founders: list[Founder] = Field(default_factory=list)
    funding: str | None = None
    employees: str | None = None
    clients: list[str] = Field(default_factory=list)
    ai_technologies: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


def _context(sources: list[RawSource]) -> str:
    blocks, total = [], 0
    for i, s in enumerate(sources, 1):
        chunk = truncate(s.text, _PER_SOURCE_CHARS)
        if total + len(chunk) > _TOTAL_CHARS:
            chunk = chunk[: max(0, _TOTAL_CHARS - total)]
        if not chunk:
            break
        blocks.append(f"=== FONTE {i}: {s.url} ===\n{chunk}")
        total += len(chunk)
    return "\n\n".join(blocks)


def _extract_one(name: str, sources: list[RawSource]) -> StartupProfile:
    urls = [s.url for s in sources]
    user = f"Startup: {name}\n\nPáginas coletadas:\n\n{_context(sources)}"
    res = complete_structured(ExtractionResult, _SYSTEM, user, fast=True)
    # Backfill source URLs onto evidence items that came back without one.
    for ev in res.evidence:
        if not ev.source_url and urls:
            ev.source_url = urls[0]
    return StartupProfile(
        name=res.name or name,
        website=res.website,
        location=res.location,
        sector=res.sector,
        one_liner=res.one_liner,
        description=res.description,
        products=res.products,
        founders=res.founders,
        funding=res.funding,
        employees=res.employees,
        clients=res.clients,
        ai_technologies=res.ai_technologies,
        evidence=res.evidence,
        sources=urls,
    )


def node(state: RadarState, config=None) -> dict:
    raw: list[RawSource] = state.get("raw_sources", [])
    groups: dict[str, list[RawSource]] = {}
    for rs in raw:
        groups.setdefault(rs.startup_hint or "desconhecida", []).append(rs)

    emit(config, "extractor", f"Estruturando dados de {len(groups)} startups…")
    profiles: list[StartupProfile] = []
    errors: list[str] = []
    for name, sources in groups.items():
        try:
            profiles.append(_extract_one(name, sources))
            emit(config, "extractor", f"Estruturado: {name}")
        except Exception as err:
            errors.append(f"extract failed for {name}: {err}")
            logger.warning("extraction failed for %s: %s", name, err)

    emit(config, "extractor", f"{len(profiles)} perfis estruturados", status="done",
         data={"count": len(profiles)})
    return {
        "startups": profiles,
        "errors": errors,
        "progress": [{"stage": "extractor", "status": "done",
                      "message": f"{len(profiles)} perfis estruturados"}],
    }
