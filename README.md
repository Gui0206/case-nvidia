# NVIDIA Startup AI Radar

Plataforma **multi-agente** que descobre startups brasileiras com potencial **AI-native**,
coleta dados públicos, **pontua a maturidade de IA**, diagnostica os gaps de stack e
**recomenda tecnologias NVIDIA** — gerando um briefing executivo para o time de
**Startups & VCs / NVIDIA Inception** no Brasil.

> Pergunta norteadora: *como a NVIDIA pode identificar, atrair e nutrir startups brasileiras
> AI-native num cenário em que os grandes labs ameaçam startups que dependem só de wrappers de LLM?*

---

## ✨ O que o sistema faz

1. **Descobre** startups a partir de uma consulta em linguagem natural (busca web pública).
2. **Coleta** informações públicas (site, blog, notícias) com rastreabilidade de fontes.
3. **Estrutura** os dados (empresa, setor, founders, funding, clientes, tecnologias de IA).
4. **Classifica** a empresa em `AI-native` / `AI-enabled` / `non-AI` com um **score de maturidade
   AI-native de 6 dimensões** e um **Lab Displacement Risk** (risco de virar irrelevante frente
   aos grandes labs) — *o diferencial do projeto*.
5. **Valida** se as afirmações têm evidência suficiente (com loop de reforço de coleta).
6. **Consulta** uma base de conhecimento NVIDIA via **RAG híbrido (vetorial + BM25) com reranking**.
7. **Recomenda** as tecnologias NVIDIA certas para cada gap, com justificativa técnica/negócio,
   prioridade, complexidade, próxima ação e **citações**.
8. **Gera** briefings executivos por startup + um panorama de portfólio priorizado.

---

## 🏗️ Arquitetura (LangGraph)

```
Consulta do usuário
   └─ Search Planner ─ Scraper ─[tem fontes?]─ Extractor ─ Classifier ─ Evidence Validator
                          ▲                                                      │
                          └──────────── reforço de coleta (retry, máx. 2) ◄──────┤ (evidência fraca?)
                                                                                 │
                          NVIDIA RAG (híbrido + rerank) ─ Recommendation ─ Briefing ─► Dashboard
```

Estado tipado, transições condicionais, **loop de retry com limite**, checkpointer
(`MemorySaver`) e emissão de progresso ao vivo — tudo em [`graph.py`](src/nvidia_radar/graph.py).

Os 8 agentes vivem em [`src/nvidia_radar/agents/`](src/nvidia_radar/agents/).

---

## 🚀 Setup

Requer Python 3.10+ (testado em 3.12 via [`uv`](https://github.com/astral-sh/uv)).
**Só `OPENROUTER_API_KEY` é obrigatório** — tudo o mais degrada graciosamente.

```bash
# 1. ambiente + dependências
uv venv --python 3.12 .venv
uv pip install --python .venv -e .

# 2. configuração
cp .env.example .env        # edite e coloque sua OPENROUTER_API_KEY

# 3. construa a base de conhecimento NVIDIA (vetorial + BM25)
.venv/bin/radar build-kb

# 4. suba o dashboard
.venv/bin/radar serve       # http://127.0.0.1:8000
```

### CLI

```bash
radar status                                   # mostra provedores ativos e tamanho da base
radar build-kb                                 # ingere data/nvidia_kb/ no Qdrant + BM25
radar run "fintechs de IA para crédito" -n 3   # pipeline completo no terminal
radar ask "reduzir latência de inferência"     # consulta direta ao RAG (híbrido + rerank)
radar serve                                     # API + dashboard
```

---

## 🔌 Provedores e LLMs (via OpenRouter)

| Função | Padrão | Configurável por |
|---|---|---|
| LLM (raciocínio) | `anthropic/claude-sonnet-4.6` | `RADAR_LLM_MODEL` |
| LLM (rápido/volume) | `anthropic/claude-haiku-4.5` | `RADAR_LLM_FAST_MODEL` |
| Busca web | DuckDuckGo (grátis) → **Tavily** se `TAVILY_API_KEY` | `.env` |
| Extração de páginas | trafilatura + BeautifulSoup → **Firecrawl** se `FIRECRAWL_API_KEY`; **Playwright** se instalado | `.env` |
| Embeddings | `BAAI/bge-small-en-v1.5` (fastembed, local, ONNX) | `RADAR_EMBED_MODEL` |
| Vector DB | Qdrant (embedded) | — |
| Lexical | BM25 (`rank-bm25`) | — |
| Reranking | cross-encoder local (ONNX) → **Cohere Rerank** se `COHERE_API_KEY` | `.env` |
| Banco estruturado | SQLite → **PostgreSQL** se `DATABASE_URL` | `.env` |

Todas as chamadas de LLM passam pelo **OpenRouter** ([`llm.py`](src/nvidia_radar/llm.py)).

---

## 🎯 Diferencial — *AI-Native Maturity Radar* + *Lab Displacement Risk*

Em vez de só rotular, o sistema pontua **por que** uma startup é (ou não) AI-native, numa
rubrica transparente e ancorada em evidências, com 6 dimensões (0–100):

- **Dados proprietários** · **Engenharia de modelos** (treina/fine-tuna/post-traina?) ·
  **Otimização de inferência** · **Profundidade de workflow** · **IA no produto** · **Defensibilidade**

E quantifica o **risco de deslocamento pelos grandes labs** (o "wrapper risk"), com vetores
de ameaça e moats. Isso responde diretamente à pergunta *"por que nem toda startup é AI-native?"*
e transforma o radar em uma ferramenta de **priorização de quem o Inception deve nutrir primeiro**.
Visualizado no dashboard como gráfico radar + gauge de risco + `inception_fit`.

---

## 📦 Mapa dos entregáveis

| Entregável | Onde |
|---|---|
| 1. Pipeline de scraping | [`scraping/`](src/nvidia_radar/scraping/) + [`agents/scraper.py`](src/nvidia_radar/agents/scraper.py) |
| 2. Sistema multiagente (LangGraph) | [`graph.py`](src/nvidia_radar/graph.py) + [`agents/`](src/nvidia_radar/agents/) |
| 3. RAG NVIDIA com reranking | [`rag/`](src/nvidia_radar/rag/) + [`data/nvidia_kb/`](data/nvidia_kb/) |
| 4. Motor de recomendação | [`agents/recommender.py`](src/nvidia_radar/agents/recommender.py) |
| 5. Interface web | [`api/app.py`](src/nvidia_radar/api/app.py) + [`frontend/index.html`](frontend/index.html) |
| 6. Diferencial | Maturity Radar + Lab Displacement Risk ([`agents/classifier.py`](src/nvidia_radar/agents/classifier.py)) |

---

## 🗂️ Estrutura

```
src/nvidia_radar/
  config.py  llm.py  models.py  state.py  graph.py  cli.py
  agents/     8 agentes (planner, scraper, extractor, classifier,
              evidence_validator, rag_agent, recommender, briefing)
  scraping/   search.py (Tavily/DDG) · fetch.py (trafilatura/bs4/Firecrawl/Playwright)
  rag/        embeddings · vectorstore (Qdrant) · bm25 · reranker · pipeline · ingest
  db/         store.py (SQLite/Postgres)
  api/        app.py (FastAPI)
data/nvidia_kb/   19 documentos da base de conhecimento NVIDIA
frontend/         dashboard single-page (Chart.js)
```

---

## 🔒 Notas

- O scraping coleta **apenas informação pública** e mantém as URLs de origem como evidência.
- A base NVIDIA é curada em `data/nvidia_kb/` com a fonte oficial citada em cada documento.
- Sem `OPENROUTER_API_KEY` o pipeline não roda; o resto (RAG, dashboard) sobe e avisa o que falta.

## 📄 Licença
Projeto acadêmico (case NVIDIA). Uso educacional.
