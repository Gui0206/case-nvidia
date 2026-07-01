# NVIDIA Startup AI Radar

Plataforma multi-agente que encontra a startup brasileira onde **a NVIDIA muda o desfecho** — não a mais AI-native (essa é a que menos precisa da NVIDIA), mas a que está no **Ponto de Alavancagem**: ameaçada pelos grandes labs, resgatável pela stack NVIDIA, e prestes a consumir compute.

> **Pergunta norteadora:** como a NVIDIA identifica, atrai e nutre startups AI-native num cenário em que os grandes labs ameaçam quem depende só de wrappers de LLM?
>
> **Resposta deste projeto:** um índice único e auditável — o **IPI (Índice de Prioridade Inception)** — que funde 4 lentes (Realness · Alavancagem · Urgência de compute · Descoberta antecipada) e ordena a fila de quem o gerente deve chamar nesta semana.

Leia o racional completo em [`docs/CONTEXTO_E_DECISOES.md`](docs/CONTEXTO_E_DECISOES.md).

---

## O diferencial em 30 segundos

- **AIMS** — maturidade AI-native em 6 dimensões, **determinística e rastreável** (o número não é chutado por LLM; vem de regras sobre features, cada uma ligada a evidência com URL).
- **Alavancagem = LDR × Resgatabilidade** — a inversão central: risco de displacement *multiplicado* por "a NVIDIA consegue reverter?". LDR alto só é bom se for resgatável.
- **CDS (Compute Demand Score)** — estimativa **direcional com confiança e premissas visíveis** de quanto/quando a startup vai consumir GPU. Sem falsa precisão.
- **Detector de AI-washing** — separa quem *diz* usar IA de quem *faz*.
- **IPI** — funde tudo; recalibrável pelo feedback do gerente (flywheel).

Resultado no gold set de 27 startups BR reais: **Spearman ρ = 0.85**, **acurácia de tier 93%**, **AI-washing precision/recall 1.00**.

---

## Rodar (o núcleo não precisa de NENHUMA chave de API)

```bash
pip install -e .                      # sem dependências obrigatórias
radar status                          # mostra camadas ativas
radar eval                            # avalia o gold set (ranking + correlação)
radar calibrate                       # calibra limiares de tier pelo gold (flywheel)
radar brief "Magie"                   # briefing executivo de uma startup
radar score "Tractian"                # diagnóstico completo em JSON
radar ask "reduzir latência de LLM"   # consulta o RAG da base NVIDIA
radar map                             # gera o dado do Mapa de Alavancagem
radar serve                           # dashboard em http://127.0.0.1:8000
```

Sem `pip install`: `PYTHONPATH=src python -m nvidia_radar.eval.gold_eval`.

### Camadas opcionais (degradação graciosa)
Configure `.env` (veja `.env.example`) para habilitar:
- **LLM** (`OPENROUTER_API_KEY` ou `NVIDIA_API_KEY`) — enriquece extração e nuance do classificador.
- **RAG avançado** (`COHERE_API_KEY`) — reranking Cohere sobre o BM25 local.
- **Descoberta** (`GITHUB_TOKEN`, `HUGGINGFACE_TOKEN`) — sinais primários (o "achar cedo").
- **Scraping** (`.[scrape]`, `FIRECRAWL_API_KEY`) — coleta pública real.
- **Grafo** (`.[graph]`) — orquestração via LangGraph (senão roda o pipeline sequencial equivalente).

Nada disso é necessário para demonstrar o diferencial: o núcleo roda 100% offline.

---

## Arquitetura

```
Descoberta (sinais primários: GitHub/HF/arXiv/vagas)  ─┐
                                                        ▼
  Search Planner → Scraper → Extractor → [SignalSet + evidências]
                                              │
        ┌──────────── DIAGNÓSTICO (scoring/) ─┴───────────────┐
        │  AIMS (Realness) ∥ LDR×Resgate (Alavancagem) ∥ CDS  │
        │        ∥ AI-washing ───────► compute IPI            │
        └──────────────────────────┬──────────────────────────┘
                                    ▼
   NVIDIA RAG (BM25 + rerank) → Recommendation → Wedge → Briefing
                                    ▼
   Mapa de Alavancagem (resgatabilidade × urgência) + fila por IPI
                                    ▼
   Feedback do gerente ──► flywheel (recalibra pesos e limiares)
```

## Mapa dos entregáveis do case

| Entregável | Onde |
|---|---|
| 1. Pipeline de scraping / descoberta | `signals/connectors.py`, `scraping/fetch.py`, `agents/scraper.py` |
| 2. Sistema multiagente (LangGraph) | `graph.py`, `agents/`, `state.py` |
| 3. RAG NVIDIA com reranking | `rag/pipeline.py`, `data/nvidia_kb/` |
| 4. Motor de recomendação | `agents/recommender.py` |
| 5. Interface web | `api/app.py`, `web/index.html` (Mapa de Alavancagem) |
| **6. Diferencial** | **`scoring/` (AIMS + LDR×RES + CDS + IPI) + AI-washing + `eval/` (gold + calibração)** |

## Estrutura

```
src/nvidia_radar/
  config.py  models.py  state.py  graph.py  llm.py  cli.py
  scoring/    features.py aims.py leverage.py cds.py ipi.py engine.py confidence.py
  agents/     recommender.py briefing.py wedge.py  (+ nós: extractor/classifier/rag/...)
  rag/        pipeline.py           (BM25 puro + rerank opcional)
  signals/    connectors.py         (GitHub/HF/arXiv, graciosos)
  eval/       gold_eval.py calibrate.py
  api/        app.py                (dashboard stdlib, zero-dep)
  web/        index.html            (Mapa de Alavancagem)
data/
  gold/       gold_set.json build_gold.py    (27 startups BR rotuladas)
  nvidia_kb/  *.md                            (base de conhecimento NVIDIA)
tests/        test_core.py
```

## Testes

```bash
pip install pytest && PYTHONPATH=src pytest -q
```

Projeto acadêmico (case NVIDIA / Inteli). Uso educacional. Dados coletados de fontes públicas com rastreabilidade de URL.
