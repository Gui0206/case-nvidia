"""Motor de recomendação: gaps do diagnóstico -> tecnologias NVIDIA (grounded no mapa do Kit).

Determinístico e citável. Se o RAG estiver disponível, anexa trechos/citações da base NVIDIA.
"""
from __future__ import annotations

from ..models import Diagnosis, SignalSet

# Mapa gap -> recomendação NVIDIA (do Kit de Implementação, §12.1)
# cada regra: (condição sobre sig/diagnóstico) -> dict de recomendação
RULES = [
    dict(id="api_dependency", tech="NVIDIA NIM + TensorRT-LLM + Dynamo-Triton",
         when=lambda s, d: (s.has("uses_only_external_api") or s.has("external_api_plus_growth"))
                            and d.compute.pain >= 0.3,
         tech_just="Depende de API externa para LLM em produto core; NIM + TensorRT-LLM cortam custo/latência com serving próprio.",
         biz_just="Reduz custo por token e dependência de fornecedor; melhora margem à medida que o volume cresce.",
         priority="alta", complexity="média",
         next_action="Oferecer endpoint NIM gratuito (Inception) para PoC de latência/custo.",
         hook="Endpoints NIM grátis + até US$100k em créditos DGX Cloud (Inception)."),
    dict(id="latency", tech="TensorRT-LLM + Dynamo-Triton",
         when=lambda s, d: s.has("latency_sensitive_product") or s.has("realtime_critical"),
         tech_just="Produto sensível a latência; TensorRT-LLM + serving desagregado (Dynamo) reduzem p95.",
         biz_just="Latência menor melhora conversão/experiência em tempo real.",
         priority="alta", complexity="média",
         next_action="Benchmark de latência antes/depois em GPU L40S/H100.",
         hook="Suporte técnico Inception + créditos para benchmark."),
    dict(id="voice", tech="NVIDIA Riva + NIM",
         when=lambda s, d: s.modality == "voice",
         tech_just="Pipeline de voz (ASR/TTS) próprio com Riva, servido via NIM.",
         biz_just="Qualidade e custo de voz melhores que APIs genéricas; suporte a português.",
         priority="alta", complexity="média",
         next_action="PoC Riva ASR/TTS em português com dado do cliente.",
         hook="Blueprints de voz NVIDIA + créditos Inception."),
    dict(id="data_scale", tech="NVIDIA RAPIDS (cuDF/cuML)",
         when=lambda s, d: s.has("pain_is_scale_data") or (s.modality == "data" and s.has("data_scale_signal")),
         tech_just="Grandes volumes tabulares; RAPIDS acelera ETL/ML em GPU.",
         biz_just="Pipelines de dados mais rápidos e baratos que CPU.",
         priority="média", complexity="baixa",
         next_action="Prova de aceleração de um pipeline crítico com cuDF.",
         hook="Treinamento técnico RAPIDS via Inception."),
    dict(id="robotics", tech="NVIDIA Isaac + Omniverse",
         when=lambda s, d: s.modality == "robotics",
         tech_just="Simulação, percepção e autonomia com Isaac/Omniverse.",
         biz_just="Acelera desenvolvimento e validação de sistemas autônomos.",
         priority="alta", complexity="alta",
         next_action="Workshop Isaac Sim para o caso de uso do cliente.",
         hook="Acesso a Isaac + créditos de compute Inception."),
    dict(id="health", tech="NVIDIA Clara / MONAI + NeMo Guardrails",
         when=lambda s, d: d.vertical == "saúde",
         tech_just="Imagem/dados clínicos com Clara/MONAI; governança com Guardrails.",
         biz_just="Conformidade e qualidade em saúde regulada.",
         priority="média", complexity="alta",
         next_action="Avaliar Clara para o pipeline clínico específico.",
         hook="AI Enterprise + suporte Inception para saúde."),
    dict(id="governance", tech="NeMo Guardrails + avaliação NeMo",
         when=lambda s, d: s.has("pain_is_governance") or d.vertical == "cyber",
         tech_just="Governança/segurança de agentes com NeMo Guardrails.",
         biz_just="Reduz risco e habilita adoção enterprise.",
         priority="média", complexity="média",
         next_action="Integrar Guardrails no fluxo de agentes atual.",
         hook="NeMo microservices via Inception."),
    dict(id="own_model", tech="NVIDIA NeMo (+ NeMo microservices)",
         when=lambda s, d: s.has("could_own_model_with_help") and d.aims.dimensions["proprietary_data"].score >= 0.4,
         tech_just="Tem dado proprietário e quer treinar/customizar; NeMo dá o caminho (treino, PEFT, data flywheel).",
         biz_just="Converte dado proprietário em modelo defensável — reduz risco de displacement pelos labs.",
         priority="alta", complexity="alta",
         next_action="Roadmap de fine-tuning com NeMo sobre o dado proprietário.",
         hook="Créditos DGX Cloud + NeMo via Inception."),
    dict(id="scale_prod", tech="NVIDIA AI Enterprise + DGX Cloud",
         when=lambda s, d: d.compute.score >= 0.6,
         tech_just="Alta demanda de compute; AI Enterprise + DGX Cloud para produção escalável.",
         biz_just="Infra pronta para produção sem montar data center próprio.",
         priority="alta", complexity="média",
         next_action="Dimensionar DGX Cloud para a carga projetada.",
         hook="Até US$100k em créditos DGX Cloud + 30% de desconto (Inception)."),
]


def recommend(diag: Diagnosis, sig: SignalSet, rag=None) -> list[dict]:
    out = []
    for rule in RULES:
        try:
            if rule["when"](sig, diag):
                rec = {k: v for k, v in rule.items() if k != "when"}
                rec["evidence_used"] = [e.source_url for e in sig.evidence][:3]
                if rag is not None:
                    hits = rag.search(rule["tech"], top_k=1)
                    if hits:
                        rec["citations"] = [{"source": hits[0].get("source", ""),
                                             "snippet": hits[0].get("text", "")[:180]}]
                out.append(rec)
        except Exception:
            continue
    # ordena por prioridade e resgatabilidade
    order = {"alta": 0, "média": 1, "baixa": 2}
    out.sort(key=lambda r: order.get(r["priority"], 1))
    return out[:5]
