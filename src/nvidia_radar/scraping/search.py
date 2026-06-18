"""Web search for startup discovery.

Prefers Tavily when ``TAVILY_API_KEY`` is set (higher quality, includes content);
otherwise falls back to free DuckDuckGo search via the ``ddgs`` package. Both
paths return a normalised ``list[SearchResult]``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_settings

logger = logging.getLogger("nvidia_radar.search")


@dataclass
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""

    def as_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "snippet": self.snippet}


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=8), reraise=True)
def _tavily(query: str, max_results: int) -> list[SearchResult]:
    s = get_settings()
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": s.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        },
        timeout=20,
    )
    resp.raise_for_status()
    out: list[SearchResult] = []
    for r in resp.json().get("results", []):
        out.append(
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", "")[:600],
            )
        )
    return out


def _ddg(query: str, max_results: int) -> list[SearchResult]:
    # Imported lazily so the package only loads when actually needed.
    from ddgs import DDGS

    out: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="br-pt", safesearch="off", max_results=max_results):
                url = r.get("href") or r.get("url") or r.get("link") or ""
                if not url:
                    continue
                out.append(
                    SearchResult(
                        url=url,
                        title=r.get("title", ""),
                        snippet=(r.get("body") or r.get("snippet") or "")[:600],
                    )
                )
    except Exception as err:  # DDG is rate-limit prone; degrade quietly
        logger.warning("DuckDuckGo search failed for %r: %s", query, err)
    return out


def web_search(query: str, max_results: int = 8) -> list[SearchResult]:
    """Run a single web search and return normalised results."""
    s = get_settings()
    if s.has_tavily:
        try:
            return _tavily(query, max_results)
        except Exception as err:
            logger.warning("Tavily failed (%s); falling back to DuckDuckGo", err)
    return _ddg(query, max_results)
