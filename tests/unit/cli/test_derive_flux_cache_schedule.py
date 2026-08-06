from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.derive_flux_cache_schedule import (
    _derive_hardware_budget,
    _fit_affine_step_cost,
    _materialized_path_segments,
    _nearest_rank,
    _optimize_mask,
    _relative_prediction_error,
    _scheduler_sigmas,
)


def test_hardware_budget_derives_one_static_anchor_count_from_speed_target():
    fit = _fit_affine_step_cost([(12, 117.0), (13, 125.0)])

    budget = _derive_hardware_budget(
        num_steps=50,
        warmup_steps=6,
        cooldown_steps=1,
        baseline_latency_s=464.0,
        target_speedup=3.2,
        intercept_s=fit["intercept_s"],
        incremental_real_step_s=fit["incremental_real_step_s"],
        dynamic_step_reserve=2,
    )

    assert fit["intercept_s"] == pytest.approx(21.0)
    assert fit["incremental_real_step_s"] == pytest.approx(8.0)
    assert budget["total_real_step_budget"] == 15
    assert budget["static_anchor_budget"] == 13
    assert budget["dynamic_step_reserve"] == 2
    assert budget["predicted_reserved_speedup"] >= 3.2

    faster_budget = _derive_hardware_budget(
        num_steps=50,
        warmup_steps=6,
        cooldown_steps=1,
        baseline_latency_s=464.0,
        target_speedup=3.3,
        intercept_s=fit["intercept_s"],
        incremental_real_step_s=fit["incremental_real_step_s"],
        dynamic_step_reserve=2,
    )

    assert faster_budget["total_real_step_budget"] == 14
    assert faster_budget["static_anchor_budget"] == 12


def test_hardware_budget_rejects_target_that_cannot_cover_required_anchors():
    with pytest.raises(ValueError, match="too few static anchors"):
        _derive_hardware_budget(
            num_steps=50,
            warmup_steps=6,
            cooldown_steps=1,
            baseline_latency_s=100.0,
            target_speedup=10.0,
            intercept_s=5.0,
            incremental_real_step_s=2.0,
            dynamic_step_reserve=2,
        )


def test_affine_step_cost_requires_distinct_hardware_points():
    with pytest.raises(ValueError, match="at least two real-step counts"):
        _fit_affine_step_cost([(12, 117.0), (12, 118.0)])


def test_nearest_rank_is_deterministic_at_tail_quantiles():
    values = [0.4, 0.1, 0.3, 0.2]

    assert _nearest_rank(values, 0.5) == pytest.approx(0.2)
    assert _nearest_rank(values, 0.95) == pytest.approx(0.4)
    assert _nearest_rank(values, 0.99) == pytest.approx(0.4)


def test_relative_prediction_error_matches_direct_vector_math():
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [4.0, 1.0],
        ]
    )
    gram = vectors @ vectors.T
    predicted = vectors[1] + (vectors[1] - vectors[0])
    expected = np.linalg.norm(predicted - vectors[2]) / np.linalg.norm(vectors[2])

    assert _relative_prediction_error(gram, 1, 2, 3, norm_floor=1e-8) == pytest.approx(expected)


def test_optimizer_obeys_budget_gap_caps_final_anchor_and_lexical_tie_break():
    costs = {}
    for a in range(1, 9):
        for b in range(a + 1, 10):
            for c in range(b + 1, min(11, b + 3) + 1):
                costs[(a, b, c)] = 0.0

    anchors, objective = _optimize_mask(
        costs,
        num_steps=12,
        warmup_steps=3,
        anchor_budget=6,
        phase_boundary=6,
        middle_gap_cap=3,
        tail_gap_cap=3,
    )

    assert anchors == (0, 1, 2, 5, 8, 11)
    assert objective == 0.0
    assert len(anchors) == 6
    assert all(right - left <= 3 for left, right in zip(anchors, anchors[1:]))


def test_materialized_segments_exclude_unscored_warmup_transitions():
    anchors = (0, 1, 2, 5, 8, 11)
    costs = {(1, 2, 5): 0.1, (2, 5, 8): 0.2, (5, 8, 11): 0.3}

    rows = _materialized_path_segments(anchors, costs, warmup_steps=3)

    assert [(row["previous_anchor"], row["anchor"], row["next_anchor"]) for row in rows] == [
        (1, 2, 5),
        (2, 5, 8),
        (5, 8, 11),
    ]


def test_scheduler_sigmas_reproduce_registered_dynamic_shift():
    generation = {
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "num_steps": 50,
        "height": 1024,
        "width": 1024,
        "scheduler_config": {
            "base_image_seq_len": 256,
            "max_image_seq_len": 4096,
            "base_shift": 0.5,
            "max_shift": 1.15,
            "use_dynamic_shifting": True,
            "time_shift_type": "exponential",
            "invert_sigmas": False,
            "shift_terminal": None,
            "use_beta_sigmas": False,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        },
    }

    sigmas = _scheduler_sigmas(generation)

    assert len(sigmas) == 51
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == 0.0
    assert all(math.isfinite(value) for value in sigmas)
    assert all(left > right for left, right in zip(sigmas, sigmas[1:]))
