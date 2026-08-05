from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.flux_cache_offline_gate import (
    build_frozen_gate,
    evaluate_development,
    load_methodology,
    one_sided_binomial_upper_bound,
    replay_profile_selection,
    select_oil_threshold,
)


ROOT = Path(__file__).resolve().parents[3]
METHODOLOGY_PATH = (
    ROOT / "benchmark/flux_cache/offline-gate-methodology-v1.json"
)
TARGET = "adaptive-stage-oil-p30-e1p40-w6-i8-k12-o1-index"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development(*, pooled_auc: float = 0.9, target_auc: float = 0.9):
    target_scores = [0.9, 0.8] + [0.01 * index for index in range(1, 31)]
    labels = [1, 1] + [0] * 30
    rows = [
        {
            "candidate_id": TARGET,
            "prompt_index": index,
            "failed": bool(label),
            "oof_probabilities": {"combined": score},
        }
        for index, (label, score) in enumerate(zip(labels, target_scores))
    ]
    return {
        "selection": {"selected_model": "combined"},
        "failure_count": 5,
        "failure_count_by_candidate": {TARGET: 2, "enrichment": 3},
        "models": {
            "combined": {
                "feature_names": ["risk"],
                "metrics": {
                    "roc_auc": pooled_auc,
                    "average_precision": 0.8,
                },
                "by_candidate": {
                    TARGET: {
                        "roc_auc": target_auc,
                        "average_precision": 0.8,
                        "sample_count": 32,
                        "failure_count": 2,
                    }
                },
                "final_fit": {
                    "standardization_mean": [0.1],
                    "standardization_scale": [0.2],
                    "coefficients": [1.5],
                    "intercept": -0.4,
                    "probability": "sigmoid(...)"
                },
            }
        },
        "rows": rows,
    }


def test_methodology_matches_frozen_feature_implementation():
    methodology = load_methodology(METHODOLOGY_PATH)

    assert methodology["methodology_id"] == "flux-cache-offline-gate-v1"
    assert methodology["oil_threshold"]["minimum_independent_group_count"] == 29


def test_zero_failure_exact_upper_bound_requires_29_samples_for_ten_percent():
    upper_28 = one_sided_binomial_upper_bound(0, 28, confidence=0.95)
    upper_29 = one_sided_binomial_upper_bound(0, 29, confidence=0.95)

    assert upper_28 > 0.1
    assert upper_29 <= 0.1
    assert upper_29 == pytest.approx(1.0 - math.pow(0.05, 1.0 / 29.0))


def test_oil_threshold_uses_largest_qualified_low_risk_cutoff():
    labels = [1, 1] + [0] * 30
    scores = [0.9, 0.8] + [0.01 * index for index in range(1, 31)]

    result = select_oil_threshold(
        labels,
        scores,
        list(range(len(labels))),
        brake_threshold=0.8,
        confidence=0.95,
        maximum_failure_rate_upper_bound=0.1,
        minimum_independent_group_count=29,
        require_zero_observed_failures=True,
    )

    assert result["qualified"]
    assert result["selected"]["threshold"] == pytest.approx(0.30)
    assert result["selected"]["sample_count"] == 30
    assert result["selected"]["independent_group_count"] == 30
    assert result["selected"]["failure_count"] == 0


def test_oil_threshold_does_not_count_repeated_seeds_as_independent():
    labels = [1, 1] + [0] * 30
    scores = [0.9, 0.8] + [0.01 * index for index in range(1, 31)]
    repeated_prompt_groups = [0, 1] + [index // 2 + 2 for index in range(30)]

    result = select_oil_threshold(
        labels,
        scores,
        repeated_prompt_groups,
        brake_threshold=0.8,
        confidence=0.95,
        maximum_failure_rate_upper_bound=0.1,
        minimum_independent_group_count=29,
        require_zero_observed_failures=True,
    )

    assert not result["qualified"]
    assert result["best_available"]["independent_group_count"] == 15


def test_development_exports_only_after_all_checks_and_preregistration():
    methodology = load_methodology(METHODOLOGY_PATH)
    decision = evaluate_development(
        _development(),
        methodology,
        target_candidate_id=TARGET,
        methodology_preregistered=True,
    )

    assert decision["status"] == "freeze_for_prospective_confirmation"
    assert decision["export_frozen_gate"]
    assert decision["failed_checks"] == []


def test_low_oof_auc_rejects_feature_family_without_export():
    methodology = load_methodology(METHODOLOGY_PATH)
    development = _development(pooled_auc=0.65, target_auc=0.56)

    decision = evaluate_development(
        development,
        methodology,
        target_candidate_id=TARGET,
        methodology_preregistered=True,
    )

    assert decision["status"] == "reject_feature_family"
    assert not decision["export_frozen_gate"]
    assert "minimum_pooled_oof_roc_auc" in decision["failed_checks"]
    assert "minimum_target_oof_roc_auc" in decision["failed_checks"]
    with pytest.raises(ValueError, match="does not permit"):
        build_frozen_gate(development, decision, {})


def test_positive_replay_cannot_export_without_methodology_preregistration():
    methodology = load_methodology(METHODOLOGY_PATH)
    decision = evaluate_development(
        _development(),
        methodology,
        target_candidate_id=TARGET,
        methodology_preregistered=False,
    )

    assert decision["status"] == "replay_only_methodology_not_preregistered"
    assert not decision["export_frozen_gate"]
    assert decision["checks"]["methodology_preregistered"] is False


def test_stage_replay_selects_brake_profile_and_preserves_controller_thresholds():
    methodology = load_methodology(METHODOLOGY_PATH)
    stage_result = json.loads(
        (ROOT / "benchmark/flux_cache/stage-gate-study-result.json").read_text(
            encoding="utf-8"
        )
    )

    replay = replay_profile_selection(
        stage_result,
        methodology,
        {
            "stage_acceleration": ROOT
            / "benchmark/flux_cache/adaptive-stage-oil-p30-e1p40-k12-candidate.json",
            "brake_only": ROOT / "benchmark/flux_cache/adaptive-brake-candidate.json",
        },
    )

    assert replay["status"] == "legacy_profile_selected_not_methodology_preregistered"
    assert replay["selected_profile"]["profile_name"] == "brake_only"
    assert replay["selected_profile"]["policy"]["allow_acceleration"] is False
    assert replay["selected_profile"]["policy"]["tighten_error"] == pytest.approx(1.19)
    assert replay["selected_profile"]["policy"]["recovery_error"] == pytest.approx(1.5)
    assert not replay["export_serving_profile"]


def test_checked_in_engineering_closure_binds_both_decisions():
    closure = json.loads(
        (ROOT / "benchmark/flux_cache/cache-system-engineering-closure.json").read_text(
            encoding="utf-8"
        )
    )
    methodology_path = ROOT / closure["methodology"]["path"]
    profile_path = ROOT / closure["offline_profile_selection"]["artifact"]
    online_path = ROOT / closure["online_semantic_gate"]["artifact"]
    candidate_path = ROOT / closure["offline_profile_selection"]["selected_candidate"]

    assert _sha256(methodology_path) == closure["methodology"]["sha256"]
    assert _sha256(profile_path) == closure["offline_profile_selection"]["artifact_sha256"]
    assert _sha256(online_path) == closure["online_semantic_gate"]["artifact_sha256"]
    assert (
        _sha256(candidate_path)
        == closure["offline_profile_selection"]["selected_candidate_sha256"]
    )
    assert closure["effective_decision"]["enable_acceleration"] is False
    assert closure["effective_decision"]["serving_qualified"] is False
