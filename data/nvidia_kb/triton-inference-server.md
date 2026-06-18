---
technology: NVIDIA Triton Inference Server
category: Inferência e Deploy
aliases: Triton, Triton Inference Server, model serving
source: https://developer.nvidia.com/triton-inference-server
use_cases: serving de modelos em produção, dynamic batching, multi-framework, ensembles, alto throughput
---

# NVIDIA Triton Inference Server

## O que é
Triton é um servidor de inferência **open-source** para colocar modelos em produção de forma
padronizada. É **multi-framework** (TensorRT, PyTorch, ONNX Runtime, TensorFlow, Python
backend) e expõe endpoints HTTP/gRPC.

Recursos-chave:
- **Dynamic batching** — agrupa requisições para maximizar uso de GPU e throughput.
- **Concurrent model execution** — vários modelos/instâncias na mesma GPU.
- **Model ensembles** — encadeia pré-processamento, modelo e pós-processamento.
- **Métricas Prometheus** e model management para observabilidade e escala.

## Por que importa para startups AI-native
É a base de um serving **escalável e observável**. Quando a startup sai do protótipo e precisa
servir muitos modelos com eficiência de custo e SLAs de latência, o Triton organiza isso. É o
motor de serving por trás do **NIM**.

## Quando recomendar (sinais na startup)
- Sofre com **latência** ou **baixa utilização de GPU** em inferência.
- Serve **múltiplos modelos** (visão + NLP + ranking) e quer consolidar a infraestrutura.
- Precisa de **observabilidade** (métricas, versionamento de modelos) em produção.
- Quer **batching** e concorrência sem reescrever o serving do zero.

## Integra com
TensorRT-LLM, NIM, NeMo, RAPIDS, NVIDIA AI Enterprise, Kubernetes.
