"""Conectores de sinais primários (descoberta antecipada). Graciosos: sem token/httpx -> []."""
from __future__ import annotations

from ..config import get_settings


def _httpx():
    try:
        import httpx
        return httpx
    except Exception:
        return None


def github_signals(org_or_query: str, limit: int = 5) -> list[dict]:
    s = get_settings(); httpx = _httpx()
    if not httpx:
        return []
    try:
        headers = {"Authorization": f"Bearer {s.github_token}"} if s.github_token else {}
        r = httpx.get("https://api.github.com/search/repositories",
                      params={"q": f"{org_or_query} language:python", "sort": "updated", "per_page": limit},
                      headers=headers, timeout=20)
        r.raise_for_status()
        return [{"type": "github", "name": it["full_name"], "url": it["html_url"],
                 "signal": "ml_repos_github", "desc": it.get("description")}
                for it in r.json().get("items", [])]
    except Exception:
        return []


def huggingface_signals(org: str, limit: int = 5) -> list[dict]:
    httpx = _httpx()
    if not httpx:
        return []
    try:
        r = httpx.get("https://huggingface.co/api/models",
                      params={"author": org, "limit": limit}, timeout=20)
        r.raise_for_status()
        return [{"type": "huggingface", "name": m.get("id"),
                 "url": f"https://huggingface.co/{m.get('id')}",
                 "signal": "publishes_hf_models"} for m in r.json()]
    except Exception:
        return []


def arxiv_signals(query: str, limit: int = 5) -> list[dict]:
    httpx = _httpx()
    if not httpx:
        return []
    try:
        r = httpx.get("http://export.arxiv.org/api/query",
                      params={"search_query": f"all:{query}", "max_results": limit}, timeout=20)
        r.raise_for_status()
        import re
        titles = re.findall(r"<title>(.*?)</title>", r.text)
        return [{"type": "arxiv", "name": t, "signal": "publishes_papers"} for t in titles[1:limit+1]]
    except Exception:
        return []


def discover(query: str) -> dict:
    """Agrega sinais primários. Retorna vazio graciosamente se nada configurado."""
    return {"github": github_signals(query), "huggingface": huggingface_signals(query),
            "arxiv": arxiv_signals(query)}
