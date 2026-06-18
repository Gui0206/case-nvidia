---
technology: NeMo Guardrails
category: Governança e Segurança de IA
aliases: NeMo Guardrails, guardrails, Colang, controle de agentes
source: https://github.com/NVIDIA/NeMo-Guardrails
use_cases: governança de agentes, segurança de chatbots, rails de tópico, prevenção de alucinação, compliance
---

# NeMo Guardrails

## O que é
NeMo Guardrails é um **toolkit open-source** para adicionar **guardrails programáveis** a
aplicações e agentes baseados em LLM. Permite definir, com a linguagem **Colang**, regras
declarativas que controlam o comportamento do sistema em tempo de execução.

Tipos de rails:
- **Topical rails** — mantêm a conversa dentro do escopo permitido.
- **Safety / moderation rails** — bloqueiam conteúdo tóxico, jailbreaks e respostas inadequadas.
- **Security rails** — mitigam prompt injection e uso indevido de ferramentas.
- **Fact-checking / hallucination rails** — reduzem respostas sem fundamento.

## Por que importa para startups AI-native
Agentes em produção precisam de **governança**: previsibilidade, conformidade e segurança.
Guardrails transformam um protótipo de agente em algo auditável e seguro para clientes
enterprise — frequentemente um pré-requisito de venda em setores regulados.

## Quando recomendar (sinais na startup)
- Opera **agentes ou chatbots** em produção e precisa de previsibilidade/compliance.
- Atua em setores **regulados** (saúde, financeiro, jurídico).
- Já teve (ou teme) incidentes de alucinação, vazamento ou respostas fora de escopo.
- Precisa demonstrar **governança de IA** para fechar contratos enterprise.

## Integra com
NeMo, NIM, qualquer LLM (inclusive via API), frameworks de agentes (LangGraph/LangChain).
