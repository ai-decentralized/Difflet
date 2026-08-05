from __future__ import annotations

import pytest

from scripts.evaluate_flux_cache_online_signal import (
    extract_features,
    extract_output_dynamics_features,
    summarize_features,
)


def test_extract_features_uses_positive_regional_second_difference():
    records = [
        [0, [1.0] * 16],
        [1, [2.0] * 16],
        [2, [3.0] * 15 + [5.0]],
        [3, [4.0] * 16],
    ]

    features = extract_features(records, first_eligible_step=2, last_eligible_step=3)

    assert features["max_region_acceleration"] == pytest.approx(2.0)
    assert features["max_top2_region_acceleration"] == pytest.approx(1.0)
    assert features["peak_acceleration_level_cv_percent"] == pytest.approx(
        15.491933, rel=1e-6
    )


def test_extract_features_rejects_incomplete_spatial_map():
    records = [[0, [1.0] * 16], [1, [2.0] * 16], [2, [3.0] * 15]]

    with pytest.raises(ValueError, match="4x4 maps"):
        extract_features(records, first_eligible_step=2, last_eligible_step=2)


def test_summarize_features_reports_failure_rank_without_fitting_threshold():
    rows = [
        {"features": {"risk": 0.1}},
        {"features": {"risk": 0.8}},
        {"features": {"risk": 0.4}},
    ]

    summary = summarize_features(rows, [2])["risk"]

    assert summary["auc"] == pytest.approx(0.5)
    assert summary["failure_descending_rank"] == 2
    assert summary["passing_requests_at_or_above_failure"] == 1


def test_extract_output_dynamics_uses_only_cached_eligible_steps():
    records = [
        {
            "step_index": 5,
            "used_cache_prediction": True,
            "relative_l1": [9.0] * 16,
            "velocity_turn": [9.0] * 16,
            "acceleration_ratio": [9.0] * 16,
        },
        {
            "step_index": 6,
            "used_cache_prediction": False,
            "relative_l1": [8.0] * 16,
            "velocity_turn": [8.0] * 16,
            "acceleration_ratio": [8.0] * 16,
        },
        {
            "step_index": 7,
            "used_cache_prediction": True,
            "relative_l1": [1.0] * 15 + [3.0],
            "velocity_turn": [0.25] * 16,
            "acceleration_ratio": [0.5] * 16,
        },
    ]

    features = extract_output_dynamics_features(
        records,
        first_eligible_step=6,
        last_eligible_step=8,
    )

    assert features["output_max_region_relative_l1"] == pytest.approx(3.0)
    assert features["output_max_top2_region_relative_l1"] == pytest.approx(2.0)
    assert features["output_max_region_velocity_turn"] == pytest.approx(0.25)


def test_extract_output_dynamics_allows_a_pre_skip_window():
    records = [
        {
            "step_index": step,
            "used_cache_prediction": False,
            "relative_l1": [1.0] * 16,
            "velocity_turn": [1.0] * 16,
            "acceleration_ratio": [1.0] * 16,
        }
        for step in range(6)
    ]

    features = extract_output_dynamics_features(
        records,
        first_eligible_step=2,
        last_eligible_step=5,
    )

    assert set(features.values()) == {0.0}
    assert len(features) == 6
