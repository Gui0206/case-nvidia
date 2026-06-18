---
technology: CUDA
category: Plataforma de Computação Paralela
aliases: CUDA, CUDA Toolkit, CUDA-X, programação em GPU
source: https://developer.nvidia.com/cuda-toolkit
use_cases: programação paralela em GPU, kernels customizados, base de toda a stack acelerada
---

# CUDA

## O que é
CUDA é a **plataforma de computação paralela e o modelo de programação** da NVIDIA que permite
usar a GPU para computação de propósito geral. É a **fundação** sobre a qual todo o restante da
stack é construído — TensorRT-LLM, RAPIDS, Triton, NeMo e os frameworks de deep learning
(PyTorch, TensorFlow) dependem das bibliotecas **CUDA-X** (cuDNN, cuBLAS, NCCL, etc.).

## Por que importa para startups AI-native
Entender CUDA e as bibliotecas CUDA-X é o que separa quem **apenas chama frameworks** de quem
consegue **otimizar de verdade** kernels críticos, extrair o máximo do hardware e reduzir custo.
É a camada de engenharia profunda.

## Quando recomendar (sinais na startup)
- Tem **gargalos de performance** que exigem otimização abaixo do nível do framework.
- Desenvolve operadores/kernels customizados ou workloads HPC/numéricos.
- Quer formar time técnico em GPU (via DLI / Inception).

## Integra com
Base de TensorRT-LLM, RAPIDS, Triton, NeMo, cuDF, cuML e todo o stack NVIDIA.
