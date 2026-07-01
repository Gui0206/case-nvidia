"""Testes do núcleo determinístico — rodam sem nenhuma dependência externa."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nvidia_radar.models import SignalSet
from nvidia_radar.scoring.engine import diagnose
from nvidia_radar.scoring.features import all_feature_names, AIMS_DIMENSIONS
from nvidia_radar.eval.gold_eval import evaluate


def test_aims_weights_sum_to_one():
    assert abs(sum(w for w, _, _ in AIMS_DIMENSIONS.values()) - 1.0) < 1e-9


def test_feature_count():
    assert len(all_feature_names()) >= 45


def test_strong_native_scores_high():
    sig = SignalSet.from_names(
        ["publishes_hf_models", "trains_or_finetunes", "publishes_papers", "ml_repos_github",
         "unique_data_source", "sells_outcome_not_tool", "own_serving_infra", "ai_is_core_value",
         "founder_ai_pedigree", "academic_research_root", "technical_moat", "has_strong_data_moat"],
        modality="text", model_class="70b")
    d = diagnose("StrongNative", sig)
    assert d.aims.tier == "AI-native"
    assert d.aims.overall > 0.4


def test_wrapper_flagged_as_washing():
    sig = SignalSet.from_names(
        ["marketing_only_ai_claim", "ai_is_primary_feature", "uses_only_external_api",
         "thin_ui_over_llm", "thin_llm_wrapper", "no_proprietary_data", "pure_api_no_optimization"])
    d = diagnose("Wrapper", sig)
    assert d.ai_washing is True
    assert d.aims.overall < 0.3


def test_leverage_multiplication():
    # ameaçada (LDR alto) + resgatável (RES alto) => alavancagem alta
    sig = SignalSet.from_names(
        ["thin_llm_wrapper", "no_proprietary_data", "generic_use_case",
         "pain_is_cost_latency", "open_migratable_stack", "could_own_model_with_help",
         "external_api_plus_growth"], model_class="70b")
    d = diagnose("Leverageable", sig)
    assert d.leverage.ldr > 0.6
    assert d.leverage.res > 0.6
    assert d.leverage.leverage > 0.4


def test_ipi_bounds():
    sig = SignalSet.from_names(["trains_or_finetunes"])
    d = diagnose("X", sig)
    assert 0 <= d.ipi.ipi <= 100


def test_gold_eval_quality():
    r = evaluate()
    assert r["spearman"] >= 0.80          # ranking discrimina
    assert r["washing_recall"] >= 0.99    # pega todos os wrappers rotulados
    assert r["tier_accuracy"] >= 0.85     # com calibração
