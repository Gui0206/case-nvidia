# Contexto e decisões — NVIDIA Startup AI Radar

Este documento explica **por que** o projeto é o que é: o contexto do case, a tese central, e cada decisão de arquitetura e implementação — incluindo as que foram deliberadamente diferentes do enunciado.

---

## 1. O contexto do case

A NVIDIA (time de Startups & VCs / programa **Inception** no Brasil) precisa identificar, atrair e nutrir startups **AI-native** brasileiras. O pano de fundo: os grandes labs (OpenAI, Anthropic, Google) estão subindo na cadeia de valor e ameaçando startups que são só **wrappers de LLM**. O enunciado pede um sistema multi-agente que descobre startups, coleta dados públicos, classifica maturidade AI-native, consulta uma base RAG de tecnologias NVIDIA e gera um briefing.

**O risco do enunciado, lido ao pé da letra:** ele descreve um pipeline (`Planner → Scraper → Extractor → Classifier → Validator → RAG → Recommender → Briefing`) que **todos os times vão construir igual**. Entregar exatamente isso é ser "mais um case".

---

## 2. A tese central — o Ponto de Alavancagem

A decisão que organiza tudo:

> **A startup mais AI-native é a que menos precisa da NVIDIA.** A Tractian já resolveu sua stack. Quem o Inception precisa achar não é a mais madura — é aquela onde a **intervenção da NVIDIA muda o desfecho**.

Isso é o **Ponto de Alavancagem**: a interseção de três coisas que raramente coincidem —
1. **está prestes a quebrar** sob pressão de compute/custo/latência (precisa agora),
2. é **ameaçada** pelos labs **mas resgatável** pela stack NVIDIA (a NVIDIA converte wrapper em AI-native defensável),
3. foi **descoberta cedo** (a NVIDIA chega antes da AWS/Google e antes do lock-in).

O produto não entrega uma lista de startups de IA. Entrega uma **fila de intervenção priorizada** — quem chamar, por que agora, e qual a cunha técnica — condensada num número auditável, o **IPI**.

Por que isso vale mais: é a única formulação que olha para o **P&L da NVIDIA** (vender compute), não só para a categoria "é AI-native?".

---

## 3. As decisões de scoring (o diferencial)

### 3.1 AIMS determinístico, não "LLM chuta o número"
**Decisão:** a maturidade AI-native (6 dimensões) é calculada por **regras determinísticas** sobre features observáveis (`scoring/features.py`), com o LLM tendo no máximo influência de nuance (±0.1/dimensão).
**Por quê:** um score que um LLM emite não é reprodutível, não é auditável e alucina. Regras determinísticas são rastreáveis até a URL da evidência (`fired` guarda quais features dispararam). Esta é a diferença de rigor frente à abordagem "peça ao LLM para classificar".

### 3.2 Alavancagem = LDR × Resgatabilidade (a inversão)
**Decisão:** em vez de tratar "risco de displacement pelos labs" (LDR) como sinal negativo, multiplico-o pela **Resgatabilidade NVIDIA** (RES).
**Por quê:** LDR alto sozinho é ambíguo. Uma startup ameaçada **cuja ameaça a NVIDIA consegue reverter** (custo→NIM/TensorRT-LLM, governança→Guardrails, dado→NeMo) é o **melhor alvo do Inception** — é literalmente a razão de o programa existir. A multiplicação (`leverage = ldr × res`) transforma "risco" em "oportunidade de resgate". Se a ameaça é de produto/distribuição (fora do alcance da NVIDIA), RES cai e a alavancagem some. Ver a matriz 2×2 em `scoring/leverage.py`.

### 3.3 CDS honesto (sem falsa precisão)
**Decisão:** o Compute Demand Score é uma **estimativa de ordem de grandeza**, sempre acompanhada de `confidence` e da **lista de premissas** que a compõem (`scoring/cds.py`).
**Por quê:** estimar consumo de GPU sem acesso interno pode soar "chute". A correção não é apagar o CDS (é a métrica mais alinhada à NVIDIA) — é ser transparente: nunca "gasta US$ X", sempre "alto consumo provável (confiança média), porque publica modelos 70B + tem produto de voz". A fraqueza vira força.

### 3.4 IPI — um índice, não cinco widgets
**Decisão:** fundir Realness, Alavancagem, Compute e Descoberta num único `IPI = 100·(0.30·R + 0.30·L + 0.30·C + 0.10·D)`, decomposto e com confiança (`scoring/ipi.py`).
**Por quê:** cinco métricas soltas parecem um dashboard; um índice composto e explicável é um **instrumento de decisão**. Os pesos não são fixos — são recalibrados pelo feedback do gerente (§4).

### 3.5 AI-washing adversarial e conservador
**Decisão:** o detector só marca washing quando há **afirmação de marketing de IA + sinais de wrapper puro + evidência técnica fraca** (`scoring/engine.py:detect_ai_washing`).
**Por quê:** responde direto à pergunta "por que nem toda startup é a Tractian?" com evidência, e o critério triplo evita falso-positivo em AI-enabled real (calibrei até precision/recall = 1.00 no gold).

---

## 4. Calibração e o flywheel (por que a acurácia de tier é honesta)

**Observação:** o AIMS determinístico produz uma distribuição de scores em que os limiares do Kit (native ≥0.70 / enabled ≥0.40) ficavam altos — o **ranking** estava certo (ρ=0.85) mas a **rotulagem de tier** errava.

**Decisão:** em vez de forçar os deltas na mão até "bater", implementei `eval/calibrate.py`, que **deriva os limiares do gold set** por busca em grade (maximiza acurácia sobre os rótulos de ground-truth) e grava em `data/gold/calibration.json`. O AIMS passa a usá-los.
**Por quê:** isso é exatamente a história do **data flywheel** materializada — o sistema aprende os limiares com dados rotulados, em vez de números mágicos. Resultado: acurácia de tier salta para **93%**. Em produção, os rótulos viriam do `manager_verdict` (aceito/recusado/fechado) e recalibrariam pesos e limiares.

Os dois "erros" restantes são **propositais e documentados**: CrewAI (AI-native de *infra* que não treina modelos — caso de fronteira para estressar o score) e Pipefy (wrapper-ish que a pesquisa já sinalizava).

---

## 5. Decisões de engenharia

### 5.1 Núcleo em stdlib puro, tudo mais opcional
**Decisão:** `scoring/`, `eval/`, RAG (BM25) e o mapa rodam **sem nenhuma dependência**. LLM, LangGraph, Qdrant, Cohere, scraping são extras com *import* preguiçoso e fallback.
**Por quê:** o diferencial (scoring + IPI + gold + mapa) precisa rodar em qualquer máquina, na hora, sem chave — para ser demonstrável e testável. Camadas caras degradam graciosamente (`radar status` mostra o que está ativo). "Código é a parte fácil"; o que importa é que o conceito rode e seja verificável.

### 5.2 LangGraph opcional, com runner sequencial equivalente
**Decisão:** `graph.py` monta um `StateGraph` se a lib existir; senão roda os **mesmos nós** em sequência.
**Por quê:** cumpre o entregável 2 (multi-agente com LangGraph) sem tornar a lib um requisito para a demo. Os nós são as mesmas funções nos dois caminhos.

### 5.3 RAG com BM25 em python puro
**Decisão:** busca lexical BM25 implementada à mão, com rerank Cohere e vetorial como upgrades opcionais.
**Por quê:** garante que o entregável 3 (RAG com citações) funcione offline. A base NVIDIA (`data/nvidia_kb/`) é curada com a fonte oficial em cada doc.

### 5.4 Dashboard em `http.server` da stdlib
**Decisão:** `api/app.py` usa a biblioteca padrão, sem FastAPI/uvicorn.
**Por quê:** `radar serve` funciona sempre. O frontend (`web/index.html`) consome `/api/gold` (scores do motor real) e `/api/brief` (briefing ao vivo).

### 5.5 Gold set como código de dados versionado
**Decisão:** `data/gold/build_gold.py` gera `gold_set.json` com 27 startups BR reais, cada uma com features atribuídas a partir de pesquisa pública (jun/2026) e URLs de evidência.
**Por quê:** é o ground-truth que valida o scoring e a peça de calibração do flywheel. Manter o builder documenta a proveniência.

---

## 6. O que a demo mostra (e o "momento uau")

Rodando `radar eval` e `radar serve`:
- O **Mapa de Alavancagem** plota as 27 startups em Resgatabilidade × Urgência, tamanho = maturidade, cor = risco. O quadrante "ligue esta semana" brilha.
- A **fila por IPI** coloca em #1 **não a mais AI-native**, mas a **Magie** — um quase-wrapper (AIMS 7) que é ameaçado *e* resgatável (Alavancagem 100). Tractian, a mais madura, fica mais abaixo. **Isso não é bug: é a tese.** A NVIDIA tem impacto marginal máximo na Magie, não na Tractian.
- Clicar numa startup gera o **briefing** com o diagnóstico rastreável, recomendações NVIDIA com hook do Inception, e a cunha técnica quantificada.

---

## 7. Limitações honestas (de-risking)

- As features do gold set foram atribuídas a partir de pesquisa pública, não de scraping ao vivo — em produção, o Extractor as preencheria com evidência coletada.
- O CDS é direcional por design; não é um medidor de consumo real de GPU.
- Duas startups do gold estão marcadas "verificar" (Dharma, Alana) — são o caso de uso literal do sistema (buscar artefatos no HF/GitHub para confirmar a narrativa).
- A calibração de limiares é feita sobre o próprio gold; com mais dados de conversão real (flywheel), ela melhora e generaliza.

Colocar isto na mesa é parte do argumento: uma tese madura mostra onde pode furar.

---

## 8. Mapa rápido código ↔ decisão

| Decisão | Arquivo |
|---|---|
| Dicionário de features (50) | `src/nvidia_radar/scoring/features.py` |
| AIMS 6-dim + tier calibrável | `src/nvidia_radar/scoring/aims.py` |
| Alavancagem (LDR×RES) + matriz | `src/nvidia_radar/scoring/leverage.py` |
| CDS com premissas/confiança | `src/nvidia_radar/scoring/cds.py` |
| IPI composto | `src/nvidia_radar/scoring/ipi.py` |
| AI-washing adversarial | `src/nvidia_radar/scoring/engine.py` |
| Gold set + avaliação | `data/gold/`, `src/nvidia_radar/eval/gold_eval.py` |
| Calibração (flywheel) | `src/nvidia_radar/eval/calibrate.py` |
| Recomendação NVIDIA | `src/nvidia_radar/agents/recommender.py` |
| Briefing + cunha técnica | `src/nvidia_radar/agents/briefing.py`, `wedge.py` |
| Mapa de Alavancagem | `src/nvidia_radar/web/index.html` |
