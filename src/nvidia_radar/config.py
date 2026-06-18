"""Centralised configuration.

Everything is driven by environment variables (see ``.env.example``). Only
``OPENROUTER_API_KEY`` is strictly required; every other integration degrades
gracefully when its key is absent.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository layout ---------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parents[1]
DATA_DIR = ROOT_DIR / "data"
KB_DIR = DATA_DIR / "nvidia_kb"
QDRANT_DIR = DATA_DIR / "qdrant"
BM25_DIR = DATA_DIR / "bm25"
RUNS_DIR = DATA_DIR / "runs"
FRONTEND_DIR = ROOT_DIR / "frontend"


class Settings(BaseSettings):
    """Runtime settings, loaded from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM via OpenRouter (required) ----
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_url: str = "https://github.com/Gui0206/case_nvidia"
    openrouter_app_title: str = "NVIDIA Startup AI Radar"
    radar_llm_model: str = "anthropic/claude-sonnet-4.6"
    radar_llm_fast_model: str = "anthropic/claude-haiku-4.5"
    radar_llm_temperature: float = 0.1

    # ---- Web search / scraping ----
    tavily_api_key: str = ""
    firecrawl_api_key: str = ""
    scrape_timeout: float = 20.0
    scrape_max_pages_per_startup: int = 4

    # ---- RAG ----
    radar_embed_model: str = "BAAI/bge-small-en-v1.5"
    radar_rerank_local_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cohere_api_key: str = ""
    radar_cohere_rerank_model: str = "rerank-v3.5"
    radar_rag_top_k: int = 5
    radar_rag_candidates: int = 20
    qdrant_collection: str = "nvidia_kb"

    # ---- Structured store ----
    database_url: str = ""  # empty -> local SQLite at data/radar.db

    # ---- Pipeline defaults ----
    default_max_startups: int = 3

    # ---- Convenience flags ----
    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_firecrawl(self) -> bool:
        return bool(self.firecrawl_api_key)

    @property
    def has_cohere(self) -> bool:
        return bool(self.cohere_api_key)

    @property
    def sqlite_path(self) -> Path:
        return DATA_DIR / "radar.db"

    def provider_status(self) -> dict[str, str]:
        """Human-readable summary of which providers are active (for the UI)."""
        return {
            "llm": f"OpenRouter · {self.radar_llm_model}" if self.has_llm else "missing OPENROUTER_API_KEY",
            "search": "Tavily" if self.has_tavily else "DuckDuckGo (free)",
            "extraction": "Firecrawl" if self.has_firecrawl else "trafilatura + BeautifulSoup",
            "reranker": f"Cohere · {self.radar_cohere_rerank_model}" if self.has_cohere
            else f"local cross-encoder · {self.radar_rerank_local_model}",
            "vector_store": f"Qdrant (embedded) · {self.radar_embed_model}",
            "structured_store": "PostgreSQL" if self.database_url else "SQLite",
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_dirs() -> None:
    """Create the local data directories used by the pipeline."""
    for d in (DATA_DIR, KB_DIR, QDRANT_DIR, BM25_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
