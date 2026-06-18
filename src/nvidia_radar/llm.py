"""OpenRouter-backed LLM helpers.

All language-model traffic goes through OpenRouter via its OpenAI-compatible API
(``langchain_openai.ChatOpenAI`` pointed at the OpenRouter base URL). Two model
tiers are exposed: a *reasoning* model (planning, classification, recommendation,
briefing) and a *fast* model (high-volume extraction / validation).

``complete_structured`` is provider-agnostic: it asks for raw JSON and validates
against a Pydantic model, repairing on failure. This is more robust across
OpenRouter-proxied providers than relying on a single tool-calling dialect.
"""
from __future__ import annotations

import json
import logging
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .config import get_settings

logger = logging.getLogger("nvidia_radar.llm")

T = TypeVar("T", bound=BaseModel)


def get_llm(fast: bool = False, temperature: float | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI client wired to OpenRouter."""
    s = get_settings()
    if not s.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        model=s.radar_llm_fast_model if fast else s.radar_llm_model,
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        temperature=s.radar_llm_temperature if temperature is None else temperature,
        timeout=120,
        max_retries=2,
        default_headers={
            "HTTP-Referer": s.openrouter_app_url,
            "X-Title": s.openrouter_app_title,
        },
    )


def complete_text(system: str, user: str, fast: bool = False,
                  temperature: float | None = None) -> str:
    """Plain text completion (used for the markdown briefing)."""
    llm = get_llm(fast=fast, temperature=temperature)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    text = text.strip()
    # strip ```json ... ``` fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the outermost {...} span
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model response")


def complete_structured(
    model_cls: type[T],
    system: str,
    user: str,
    fast: bool = False,
    retries: int = 3,
) -> T:
    """Complete and validate against a Pydantic model, repairing invalid JSON."""
    llm = get_llm(fast=fast)
    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    sys = (
        f"{system}\n\n"
        "Responda APENAS com um objeto JSON válido que satisfaça EXATAMENTE este "
        "JSON Schema. Não inclua texto, comentários ou blocos de código markdown.\n"
        f"JSON Schema:\n{schema}"
    )
    messages = [SystemMessage(content=sys), HumanMessage(content=user)]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = llm.invoke(messages)
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _extract_json(text)
            return model_cls.model_validate(data)
        except (ValidationError, ValueError, json.JSONDecodeError) as err:
            last_err = err
            logger.warning("structured completion attempt %d failed: %s", attempt + 1, err)
            messages.append(AIMessage(content=text if "text" in dir() else ""))
            messages.append(
                HumanMessage(
                    content=(
                        f"O JSON anterior era inválido: {err}. "
                        "Corrija e responda APENAS com JSON válido conforme o schema."
                    )
                )
            )
    raise RuntimeError(f"Failed to obtain valid structured output: {last_err}")
