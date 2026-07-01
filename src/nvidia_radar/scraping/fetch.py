"""Coleta de páginas públicas — opcional e gracioso. trafilatura/Firecrawl/httpx se presentes."""
from __future__ import annotations


def fetch_url(url: str) -> str:
    try:
        import httpx, trafilatura
        html = httpx.get(url, timeout=20, follow_redirects=True).text
        return trafilatura.extract(html) or ""
    except Exception as e:
        raise RuntimeError(f"scraping indisponível ({e}); instale .[scrape]")
