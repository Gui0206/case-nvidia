"""Configuração central com degradação graciosa.

Nada aqui é obrigatório para o núcleo. `Settings` apenas informa quais camadas estão
habilitadas, lendo variáveis de ambiente (e um .env, se presente).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
GOLD_PATH = DATA / "gold" / "gold_set.json"
KB_DIR = DATA / "nvidia_kb"
WEB_DIR = Path(__file__).resolve().parent / "web"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass
class Settings:
    openrouter_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    nvidia_key: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    cohere_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    hf_token: str = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", ""))
    firecrawl_key: str = field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY", ""))
    tavily_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("RADAR_LLM_MODEL", "anthropic/claude-sonnet-4.5"))
    llm_fast_model: str = field(default_factory=lambda: os.getenv("RADAR_LLM_FAST_MODEL", "anthropic/claude-haiku-4.5"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))

    @property
    def has_llm(self) -> bool:
        return bool(self.openrouter_key or self.nvidia_key)

    def status(self) -> dict:
        return {
            "llm": "ready" if self.has_llm else "OFF (núcleo roda mesmo assim)",
            "reranker": "cohere" if self.cohere_key else "local/heurístico",
            "github_signals": "on" if self.github_token else "off",
            "hf_signals": "on" if self.hf_token else "off",
            "scraping": "firecrawl" if self.firecrawl_key else "básico",
            "db": "postgres" if self.database_url else "sqlite (dev)",
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
