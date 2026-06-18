---
technology: NVIDIA NeMo
category: Treinamento e Customização de Modelos
aliases: NeMo, NeMo framework, NeMo Retriever, NeMo Curator, NeMo Evaluator
source: https://www.nvidia.com/en-us/ai-data-science/products/nemo/
use_cases: fine-tuning, post-training, RLHF, avaliação de modelos, curadoria de dados, RAG, modelos próprios
---

# NVIDIA NeMo

## O que é
NeMo é a plataforma end-to-end da NVIDIA para **construir, customizar e avaliar modelos
generativos** próprios. Cobre todo o ciclo de vida do modelo:

- **NeMo Curator** — curadoria e limpeza de dados em escala (deduplicação, filtragem, qualidade).
- **NeMo (treino/customização)** — pré-treino, **fine-tuning**, **PEFT/LoRA**, **SFT** e
  **post-training / RLHF** para alinhar modelos a um domínio.
- **NeMo Evaluator** — avaliação sistemática de qualidade, com benchmarks e métricas.
- **NeMo Retriever** — embeddings e reranking de produção para pipelines de RAG (servidos como NIM).
- **NeMo Guardrails** — controle de comportamento (ver doc própria).

## Por que importa para startups AI-native
É o caminho para a startup deixar de ser um "wrapper" e passar a ter **modelos proprietários
ajustados aos seus dados**, criando defensibilidade real frente aos grandes labs. Pós-treino
forte é justamente um dos sinais de uma empresa que **produz IA**, e não apenas consome.

## Quando recomendar (sinais na startup)
- Possui **dados proprietários** e quer fine-tunar/post-trainar um modelo de domínio.
- Precisa **avaliar qualidade** de modelos de forma rigorosa (evals) antes de produção.
- Constrói **RAG** e precisa de embeddings/reranking de alto desempenho (NeMo Retriever).
- Quer reduzir dependência de modelos fechados treinando/customizando modelos abertos.

## Integra com
NIM (deploy do modelo customizado), Triton, TensorRT-LLM, NeMo Guardrails, DGX Cloud, AI Enterprise.
