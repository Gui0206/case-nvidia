"""Extractor Agent — transforma conteúdo público em sinais estruturados (SignalSet).

Offline: os sinais vêm do gold set (via graph.node_extractor). Com LLM+scraping (llm.py +
scraping/), extrairia sinais de páginas reais aplicando o prompt do Kit (§6.1). Esta camada
é o ponto de extensão para produção; o núcleo de scoring não depende dela.
"""
from ..graph import node_extractor  # noqa: F401  (nó real do pipeline)
