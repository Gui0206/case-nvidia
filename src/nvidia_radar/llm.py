"""Wrapper de LLM com degradação graciosa (via OpenRouter/NVIDIA). Opcional.

Se não houver chave ou httpx, retorna None — e todo o pipeline segue em modo determinístico.
"""
from __future__ import annotations

import json
import logging

from .config import get_settings

log = logging.getLogger("nvidia_radar.llm")


def available() -> bool:
    return get_settings().has_llm


def complete(system: str, user: str, fast: bool = False) -> str | None:
    s = get_settings()
    if not s.has_llm:
        return None
    try:
        import httpx  # opcional
    except Exception:
        log.warning("httpx não instalado — LLM desligado (pip install .[llm])")
        return None
    model = s.llm_fast_model if fast else s.llm_model
    try:
        if s.openrouter_key:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {s.openrouter_key}"}
        else:
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {s.nvidia_key}"}
        payload = {"model": model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}]}
        r = httpx.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning("LLM falhou (%s) — seguindo determinístico", e)
        return None
