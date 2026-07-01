"""Startup Classifier — AIMS (6 dimensões) + LDR × Resgatabilidade + AI-washing.

Determinístico e rastreável. O número final NÃO é chutado por LLM: vem das regras em
scoring/ (features.py, aims.py, leverage.py). Um LLM pode, opcionalmente, ajustar nuance
(±0.1/dim) via llm.py — mas o núcleo funciona sem chave.
"""
from ..scoring.engine import diagnose  # noqa: F401
from ..scoring.aims import score_aims   # noqa: F401
from ..scoring.leverage import score_leverage  # noqa: F401
