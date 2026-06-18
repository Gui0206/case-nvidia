---
technology: TensorRT-LLM
category: Otimização de Inferência
aliases: TensorRT-LLM, TensorRT, quantização, otimização de LLM
source: https://github.com/NVIDIA/TensorRT-LLM
use_cases: reduzir latência de LLM, baixar custo por token, quantização FP8/INT4, in-flight batching
---

# TensorRT-LLM

## O que é
TensorRT-LLM é uma biblioteca **open-source** para **otimizar a inferência de LLMs** em GPUs
NVIDIA. Compila e acelera o modelo aplicando técnicas de ponta:

- **Kernel fusion** e kernels otimizados específicos de atenção.
- **Quantização** (FP8, INT8, INT4/AWQ) para reduzir memória e custo mantendo qualidade.
- **In-flight (continuous) batching** — maximiza throughput em cargas concorrentes.
- **Paged KV cache** e **tensor / pipeline parallelism** para modelos grandes.

## Por que importa para startups AI-native
Endereça diretamente **custo por token** e **latência** — dois dos maiores gargalos de quem
escala IA generativa. É a camada de "engenharia eficiente" que diferencia uma empresa que
realmente otimiza a stack de uma que só consome API. Alimenta o **NIM** por baixo.

## Quando recomendar (sinais na startup)
- **Latência de inferência** alta ou custo de GPU/token insustentável ao escalar.
- Roda modelos abertos (Llama, Mistral, Qwen) e quer extrair o máximo de cada GPU.
- Precisa de **throughput** alto para muitos usuários simultâneos.
- Quer **quantizar** modelos sem perder qualidade perceptível.

## Integra com
Triton, NIM, NeMo, NVIDIA AI Enterprise.
