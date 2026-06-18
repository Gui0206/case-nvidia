"""Fetch a URL and extract clean main text.

Strategy chain (first that yields good text wins):
  1. Firecrawl       — if ``FIRECRAWL_API_KEY`` is set (clean markdown, JS-aware)
  2. httpx + trafilatura  — fast static extraction (default path)
  3. BeautifulSoup   — visible-text fallback when trafilatura returns little
  4. Playwright      — if installed, for JS-heavy pages that returned no text

Every step is wrapped so a single bad URL never breaks the pipeline.
"""
from __future__ import annotations

import logging

import httpx
import trafilatura
from bs4 import BeautifulSoup

from ..config import get_settings
from ..models import RawSource

logger = logging.getLogger("nvidia_radar.fetch")

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "(compatible; NVIDIA-Startup-Radar/1.0; +public-research)"
)
_MAX_CHARS = 18000


def _title_from_html(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        pass
    return None


def _bs4_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg", "form"]):
            tag.decompose()
        return " ".join(soup.get_text(" ").split())
    except Exception:
        return ""


def _firecrawl(url: str, api_key: str) -> RawSource | None:
    try:
        from firecrawl import FirecrawlApp  # type: ignore

        app = FirecrawlApp(api_key=api_key)
        res = app.scrape_url(url, params={"formats": ["markdown"]})
        data = res.get("data", res) if isinstance(res, dict) else {}
        text = (data.get("markdown") or data.get("content") or "")[:_MAX_CHARS]
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        if text.strip():
            return RawSource(url=url, title=meta.get("title"), text=text, fetched_via="firecrawl")
    except Exception as err:
        logger.debug("Firecrawl failed for %s: %s", url, err)
    return None


def _playwright(url: str, timeout: float) -> RawSource | None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=_UA)
            page.goto(url, timeout=int(timeout * 1000), wait_until="domcontentloaded")
            html = page.content()
            browser.close()
        text = (trafilatura.extract(html, url=url, favor_recall=True) or _bs4_text(html))[:_MAX_CHARS]
        if text.strip():
            return RawSource(url=url, title=_title_from_html(html), text=text, fetched_via="playwright")
    except Exception as err:
        logger.debug("Playwright failed for %s: %s", url, err)
    return None


def fetch_url(url: str, startup_hint: str | None = None) -> RawSource | None:
    """Fetch one URL and return a RawSource, or ``None`` if nothing usable."""
    s = get_settings()

    if s.has_firecrawl:
        rs = _firecrawl(url, s.firecrawl_api_key)
        if rs:
            rs.startup_hint = startup_hint
            return rs

    html = ""
    try:
        with httpx.Client(timeout=s.scrape_timeout, follow_redirects=True,
                          headers={"User-Agent": _UA}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as err:
        logger.debug("httpx GET failed for %s: %s", url, err)

    if html:
        text = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=True, favor_recall=True
        ) or ""
        if len(text) < 250:  # trafilatura found little -> try raw visible text
            alt = _bs4_text(html)
            if len(alt) > len(text):
                text = alt
        text = text[:_MAX_CHARS]
        if len(text) >= 120:
            return RawSource(
                url=url, title=_title_from_html(html), text=text,
                startup_hint=startup_hint, fetched_via="trafilatura",
            )

    # Last resort: JS-heavy page -> Playwright if it happens to be installed.
    rs = _playwright(url, s.scrape_timeout)
    if rs:
        rs.startup_hint = startup_hint
    return rs
