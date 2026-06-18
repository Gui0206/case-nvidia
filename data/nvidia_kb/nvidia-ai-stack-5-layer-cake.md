---
technology: NVIDIA AI Stack (5-layer cake)
category: Estratégia e Arquitetura
aliases: 5-layer cake, full-stack AI, camadas de IA, onde a startup está na stack
source: https://blogs.nvidia.com/blog/ai-5-layer-cake/
use_cases: posicionar a startup na stack, decidir onde otimizar, roadmap técnico
---

# NVIDIA AI Stack — o "bolo de 5 camadas"

## A ideia
A NVIDIA oferece uma plataforma **full-stack** para IA. Pensar em camadas ajuda a diagnosticar
**onde a startup está** e **para onde pode subir** para ganhar eficiência e defensibilidade,
em vez de depender só da camada mais alta (API de terceiros).

## As camadas (de baixo para cima)
1. **Infraestrutura acelerada** — GPUs, sistemas DGX, networking; via DGX Cloud / nuvens.
2. **CUDA e bibliotecas CUDA-X** — a base de software (cuDNN, cuBLAS, NCCL) e **CUDA**.
3. **Frameworks e modelos** — treino e customização com **NeMo**, RAPIDS para dados,
   frameworks de DL; é onde nascem **modelos proprietários**.
4. **Microsserviços de inferência (NIM)** — servir modelos otimizados com **Triton** e
   **TensorRT-LLM**, com APIs padrão.
5. **Aplicações e agentes** — produtos, copilotos e **agentes** (com **NeMo Guardrails** para
   governança) que entregam o resultado ao cliente.

## Como usar no diagnóstico
- Startup que só opera na **camada 5** (consome API) tende a ter **alto risco de wrapper**.
- Recomendar descer estrategicamente: otimizar inferência (camada 4), customizar modelos com
  dados próprios (camada 3) e, quando fizer sentido, controlar infraestrutura (camadas 1-2).
- Cada descida aumenta **controle de custo/latência** e **defensibilidade**.

## Integra com
Mapeia para todo o portfólio: Inception, NIM, NeMo, Triton, TensorRT-LLM, RAPIDS, AI Enterprise.
