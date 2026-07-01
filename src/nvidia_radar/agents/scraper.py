"""Scraper Agent — coleta pública (Firecrawl/Playwright/trafilatura). Opcional; ver scraping/."""
try:
    from ..scraping.fetch import fetch_url  # noqa: F401
except Exception:  # scraping é opcional
    def fetch_url(url: str):  # type: ignore
        raise RuntimeError("camada de scraping não instalada (pip install .[scrape])")
