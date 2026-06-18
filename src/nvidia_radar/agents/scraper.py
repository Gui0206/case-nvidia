"""Scraper Agent — discovers candidates and collects public information.

Round 0  : harvest candidate names (planner + search snippets) and fetch base
           pages for each target startup.
Round >=1: deepening pass — fetch extra pages only for startups the Evidence
           Validator flagged as needing more evidence (drives the retry loop).
"""
from __future__ import annotations

import logging
import unicodedata

from pydantic import BaseModel

from ..config import get_settings
from ..llm import complete_structured
from ..scraping import fetch_url, web_search
from ..state import RadarState, emit

logger = logging.getLogger("nvidia_radar.scraper")

_SOCIAL = ("facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com")


class _Names(BaseModel):
    names: list[str]


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _rank_urls(name: str, results) -> list[str]:
    """Official-looking domains first; one page per domain for variety."""
    token = _slug(name)
    seen_domains: set[str] = set()
    scored: list[tuple[int, str]] = []
    for r in results:
        url = r.url
        if not url.startswith("http"):
            continue
        domain = url.split("/")[2].lower() if len(url.split("/")) > 2 else url
        base_domain = domain.replace("www.", "")
        if base_domain in seen_domains:
            continue
        seen_domains.add(base_domain)
        score = 0
        if token and token[:8] in _slug(base_domain):
            score += 10  # likely the official site
        if any(soc in base_domain for soc in _SOCIAL):
            score -= 5
        if base_domain.endswith(".com.br") or base_domain.endswith(".com"):
            score += 1
        scored.append((score, url))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [u for _, u in scored]


def _harvest_names(snippets: list[str], query: str, n: int) -> list[str]:
    sys = (
        "Você extrai NOMES de startups brasileiras de IA a partir de trechos de busca. "
        "Retorne apenas nomes de empresas reais mencionadas, sem duplicatas, sem texto extra."
    )
    user = (
        f"Consulta: {query}\nQueremos até {n} nomes de startups.\n\n"
        "Trechos:\n" + "\n".join(f"- {s}" for s in snippets[:40])
    )
    return _safe_names(complete_structured(_Names, sys, user, fast=True).names, n)


def _safe_names(names: list[str], n: int) -> list[str]:
    out: list[str] = []
    for nm in names:
        nm = nm.strip()
        if nm and nm.lower() not in {x.lower() for x in out}:
            out.append(nm)
    return out[:n]


def _collect_for(name: str, extra_queries: list[str], max_pages: int) -> tuple[list, list[str]]:
    """Search + fetch pages for one startup. Returns (raw_sources, urls)."""
    queries = [f"{name} startup inteligência artificial Brasil"] + extra_queries
    results = []
    for q in queries:
        results.extend(web_search(q, max_results=6))
    raw, urls = [], []
    for url in _rank_urls(name, results):
        if len(raw) >= max_pages:
            break
        rs = fetch_url(url, startup_hint=name)
        if rs and rs.text:
            raw.append(rs)
            urls.append(url)
    return raw, urls


def node(state: RadarState, config=None) -> dict:
    s = get_settings()
    plan = state["plan"]
    max_startups = state.get("max_startups", s.default_max_startups)
    rounds_done = state.get("scrape_round", 0)
    existing_raw = list(state.get("raw_sources", []))
    errors: list[str] = []

    # ---------- Deepening pass (retry loop) ----------
    if rounds_done >= 1:
        flagged = [p for p in state.get("startups", []) if p.needs_more_evidence]
        emit(config, "scraper", f"Reforço de evidências para {len(flagged)} startup(s)…")
        for p in flagged:
            extra = [f"{p.name} founders", f"{p.name} rodada investimento funding",
                     f"{p.name} tecnologia modelo IA"]
            raw, urls = _collect_for(p.name, extra, s.scrape_max_pages_per_startup)
            existing_raw.extend(raw)
        return {
            "raw_sources": existing_raw,
            "scrape_round": rounds_done + 1,
            "progress": [{"stage": "scraper", "status": "done",
                          "message": "Reforço de coleta concluído"}],
        }

    # ---------- Round 0: discovery ----------
    candidates = _safe_names(list(plan.candidate_startups), max_startups * 3)
    emit(config, "scraper", "Descobrindo candidatos em fontes públicas…")
    snippets: list[str] = []
    for q in plan.queries[:4]:
        for r in web_search(q, max_results=6):
            if r.title or r.snippet:
                snippets.append(f"{r.title} — {r.snippet}")
    if snippets and len(candidates) < max_startups + 2:
        try:
            for nm in _harvest_names(snippets, state["query"], max_startups + 3):
                if nm.lower() not in {c.lower() for c in candidates}:
                    candidates.append(nm)
        except Exception as err:
            errors.append(f"name harvest failed: {err}")

    targets = candidates[:max_startups]
    emit(config, "scraper", f"Coletando dados de {len(targets)} startups: {', '.join(targets)}",
         data={"targets": targets})

    raw_sources = list(existing_raw)
    for name in targets:
        emit(config, "scraper", f"Coletando: {name}")
        try:
            raw, urls = _collect_for(name, [], s.scrape_max_pages_per_startup)
            if raw:
                raw_sources.extend(raw)
            else:
                errors.append(f"no pages fetched for {name}")
        except Exception as err:
            errors.append(f"scrape failed for {name}: {err}")

    techs = {r.startup_hint for r in raw_sources if r.startup_hint}
    emit(config, "scraper", f"{len(raw_sources)} páginas coletadas de {len(techs)} startups",
         status="done", data={"pages": len(raw_sources)})
    return {
        "raw_sources": raw_sources,
        "scrape_round": 1,
        "errors": errors,
        "progress": [{"stage": "scraper", "status": "done",
                      "message": f"{len(raw_sources)} páginas de {len(techs)} startups"}],
    }
