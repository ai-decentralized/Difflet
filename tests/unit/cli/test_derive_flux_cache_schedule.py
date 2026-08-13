from __future__ import annotations

import json
import math

import numpy as np
import pytest

from difflet.offline.cache_profile import derivation as schedule_derivation
from difflet.offline.cache_profile.schedule import (
    materialized_path_segments as _materialized_path_segments,
    nearest_rank as _nearest_rank,
    optimize_budget_frontier as _optimize_budget_frontier,
    optimize_mask as _optimize_mask,
    relative_prediction_error as _relative_prediction_error,
    scheduler_sigmas as _scheduler_sigmas,
)


def test_registration_uses_label_free_trajectories_without_a_speed_budget(
    tmp_path,
    monkeypatch,
):
    trajectories = []
    for index in range(48):
        path = tmp_path / f"trajectory-{index:03d}.pt"
        path.write_bytes(f"trajectory-{index}".encode())
        trajectories.append(
            {
                "sample_id": f"p{index:03d}-s0",
                "path": path.name,
                "file_sha256": schedule_derivation.sha256_file(path),
            }
        )
    trajectory_input_path = tmp_path / "trajectory-input-v1.json"
    trajectory_input_path.write_text(
        json.dumps(
            {
                "schema": "difflet-flux-cache-trajectory-input-v1",
                "protocol": {
                    "sha256": "a" * 64,
                    "generation": {"num_steps": 50, "dtype": "bfloat16"},
                },
                "prompt_count": 48,
                "trajectory_count": 48,
                "trajectories": trajectories,
                "semantic_labels_collected": False,
            }
        ),
        encoding="utf-8",
    )
    quality_contract_path = tmp_path / "quality-contract.json"
    quality_contract_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(schedule_derivation, "ROOT", tmp_path)
    monkeypatch.setattr(
        schedule_derivation.provenance,
        "implementation_bundle",
        lambda *_args, **_kwargs: {"files": []},
    )

    registration = schedule_derivation._registration_payload(
        trajectory_input_path,
        quality_contract_path,
        study_id="quality-led-frontier",
        created_at="2026-08-12T00:00:00Z",
    )

    assert registration["source"]["trajectory_input_schema"] == (
        "difflet-flux-cache-trajectory-input-v1"
    )
    assert registration["source"]["trajectory_count"] == 48
    assert registration["optimizer"]["anchor_budget_source"] == (
        "literature_informed_empirical_search_floor"
    )
    assert registration["optimizer"]["budget_selection"] == (
        "ascending_first_closed_loop_quality_pass"
    )
    assert registration["optimizer"]["warmup_steps"] == 3
    assert registration["optimizer"]["cooldown_steps"] == 0
    assert registration["optimizer"]["search_floor_formula"] == (
        "ceil(0.20 * total_steps)"
    )
    assert registration["optimizer"]["search_floor_ratio"] == 0.20
    assert registration["optimizer"]["search_floor_anchor_budget"] == 10
    assert registration["optimizer"]["search_floor_role"] == (
        "candidate_domain_floor_not_quality_guarantee"
    )
    assert registration["optimizer"]["predictor_history_floor"] == 2
    assert registration["optimizer"]["trajectory_observation_floor"] == 3
    assert registration["optimizer"]["closed_loop_mechanism_floor"] == 4
    assert registration["quality_contract_ref"] == {
        "path": "quality-contract.json",
        "file_sha256": schedule_derivation.sha256_file(quality_contract_path),
    }
    assert "hardware_budget" not in registration
    assert "anchor_budgets" not in registration["optimizer"]


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
    for a in range(1, 10):
        for b in range(a + 1, 11):
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


def test_budget_frontier_contains_every_structurally_feasible_budget():
    costs = {}
    for a in range(1, 10):
        for b in range(a + 1, 11):
            for c in range(b + 1, min(11, b + 3) + 1):
                costs[(a, b, c)] = 0.0

    frontier = _optimize_budget_frontier(
        costs,
        num_steps=12,
        warmup_steps=3,
        minimum_anchor_budget=6,
        phase_boundary=6,
        middle_gap_cap=3,
        tail_gap_cap=3,
    )

    budgets = [budget for budget, _, _ in frontier]
    assert budgets == list(range(6, 13))
    assert all(len(anchors) == budget for budget, anchors, _ in frontier)


def test_flux_frontier_really_starts_at_twenty_percent_floor():
    num_steps = 50
    costs = {}
    for a in range(1, num_steps - 2):
        for b in range(a + 1, min(num_steps - 1, a + 10) + 1):
            cap = 6 if b < 21 else 10
            for c in range(b + 1, min(num_steps - 1, b + cap) + 1):
                costs[(a, b, c)] = 0.0

    frontier = _optimize_budget_frontier(
        costs,
        num_steps=num_steps,
        warmup_steps=3,
        minimum_anchor_budget=schedule_derivation.empirical_search_floor(num_steps),
        phase_boundary=21,
        middle_gap_cap=6,
        tail_gap_cap=10,
    )

    assert frontier[0][0] == 10
    assert len(frontier[0][1]) == 10
    assert frontier[0][1][:3] == (0, 1, 2)
    assert frontier[0][1][-1] == 49


@pytest.mark.parametrize(
    ("num_steps", "expected"),
    ((50, 10), (49, 10), (51, 11)),
)
def test_empirical_search_floor_is_ceiling_twenty_percent(num_steps, expected):
    assert schedule_derivation.empirical_search_floor(num_steps) == expected


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
