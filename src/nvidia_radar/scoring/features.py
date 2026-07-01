"""Dicionário de features do AIMS + rubricas de LDR/RES/CDS.

Cada dimensão é uma lista de (feature, delta). O score da dimensão é
clamp(base + soma dos deltas das features ativas, 0, 1). Determinístico e rastreável:
`fired` guarda quais features dispararam, e cada feature mapeia para evidências no pipeline real.

Esta é a implementação direta do "Dicionário de Features" do Kit de Implementação.
"""
from __future__ import annotations

# ─────────────────────────── AIMS (maturidade AI-native) ───────────────────────────
# dim: (peso_na_maturidade, base, [(feature, delta), ...])
AIMS_DIMENSIONS: dict[str, tuple[float, float, list[tuple[str, float]]]] = {
    "model_engineering": (0.25, 0.0, [
        ("publishes_hf_models", 0.45),
        ("trains_or_finetunes", 0.30),
        ("publishes_papers", 0.20),
        ("publishes_datasets", 0.15),
        ("ml_repos_github", 0.15),
        ("uses_only_external_api", -0.30),
        ("marketing_only_ai_claim", -0.15),
    ]),
    "proprietary_data": (0.20, 0.0, [
        ("unique_data_source", 0.40),
        ("data_scale_signal", 0.25),
        ("data_flywheel_language", 0.20),
        ("regulated_proprietary_data", 0.15),
        ("no_data_moat", -0.25),
    ]),
    "workflow_depth": (0.15, 0.0, [
        ("sells_outcome_not_tool", 0.40),
        ("end_to_end_automation", 0.30),
        ("deep_vertical_workflow", 0.20),
        ("thin_ui_over_llm", -0.30),
    ]),
    "inference_optimization": (0.15, 0.0, [
        ("own_serving_infra", 0.35),
        ("latency_cost_engineering", 0.30),
        ("gpu_infra_signal", 0.25),
        ("mlops_jobs", 0.20),
        ("pure_api_no_optimization", -0.25),
    ]),
    "ai_in_product": (0.15, 0.0, [
        ("ai_is_core_value", 0.45),
        ("ai_is_primary_feature", 0.25),
        ("ai_is_peripheral", -0.30),
    ]),
    "defensibility": (0.10, 0.0, [
        ("founder_ai_pedigree", 0.30),
        ("academic_research_root", 0.25),
        ("technical_moat", 0.25),
        ("easily_replicable_by_labs", -0.35),
    ]),
}

# ─────────────────────────── Lab Displacement Risk (LDR) ───────────────────────────
LDR_BASE = 0.5
LDR_FEATURES: list[tuple[str, float]] = [
    ("thin_llm_wrapper", 0.40),
    ("no_proprietary_data", 0.25),
    ("generic_use_case", 0.20),
    ("commodity_modality", 0.15),
    ("has_strong_data_moat", -0.35),
    ("deep_vertical_lock", -0.25),
]

# ─────────────────────────── Resgatabilidade NVIDIA (RES) ───────────────────────────
RES_BASE = 0.5
RES_FEATURES: list[tuple[str, float]] = [
    ("pain_is_cost_latency", 0.35),
    ("pain_is_governance", 0.20),
    ("pain_is_scale_data", 0.20),
    ("open_migratable_stack", 0.25),
    ("could_own_model_with_help", 0.20),
    ("pain_is_product_distribution", -0.40),
    ("irreversible_vendor_lock", -0.25),
]

# ─────────────────────────── Compute Demand Score (CDS) ───────────────────────────
CDS_MAGNITUDE: list[tuple[str, float]] = [
    ("trains_models", 0.30),
    ("heavy_modality", 0.25),
    ("large_model_class", 0.20),
    ("high_inference_volume", 0.20),
    ("realtime_critical", 0.15),
]
CDS_PAIN: list[tuple[str, float]] = [
    ("external_api_plus_growth", 0.35),
    ("cost_reduction_jobs", 0.25),
    ("public_cost_complaints", 0.20),
    ("latency_sensitive_product", 0.20),
]

# modalidades que implicam compute pesado (deriva heavy_modality automaticamente)
HEAVY_MODALITIES = {"voice", "vision", "robotics", "multimodal"}
LARGE_MODEL_CLASSES = {"70b", "405b+"}


def all_feature_names() -> set[str]:
    names: set[str] = set()
    for _, _, feats in AIMS_DIMENSIONS.values():
        names |= {f for f, _ in feats}
    for lst in (LDR_FEATURES, RES_FEATURES, CDS_MAGNITUDE, CDS_PAIN):
        names |= {f for f, _ in lst}
    return names
