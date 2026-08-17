from __future__ import annotations

import numpy as np
import pytest

from difflet.offline.cache_profile.composition import (
    analyze_mask,
    envelope_table,
    generate_stratified_masks,
)


def test_analyze_mask_matches_direct_open_loop_vector_composition():
    velocities = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [4.0, 1.0],
            [7.0, 2.0],
            [11.0, 3.0],
        ],
        dtype=np.float64,
    )
    gram = velocities @ velocities.T
    deltas = (0.0, -0.10, -0.15, -0.20, -0.25, -0.30)

    result = analyze_mask(
        gram,
        deltas,
        (0, 1, 2, 5),
        warmup_steps=3,
        norm_floor=1e-8,
    )

    predicted_3 = velocities[1] + (velocities[1] - velocities[0]) * 1.0
    predicted_4 = velocities[1] + (velocities[1] - velocities[0]) * 2.0
    error = deltas[3] * (predicted_3 - velocities[2])
    error += deltas[4] * (predicted_4 - velocities[3])
    displacement = sum(
        (deltas[step] * velocities[step - 1] for step in range(3, 6)),
        start=np.zeros(2),
    )
    expected_harm = np.linalg.norm(error) / np.linalg.norm(displacement)
    expected_z = np.linalg.norm(
        velocities[1] + (velocities[1] - velocities[0]) * 3.0 - velocities[4]
    ) / np.linalg.norm(velocities[4])

    assert result["segment_count"] == 1
    assert result["skipped_steps"] == 2
    assert result["max_endpoint_anchor_z"] == pytest.approx(expected_z)
    assert result["end_to_end_path_relative_l2"] == pytest.approx(expected_harm)


def test_stratified_masks_are_reproducible_unique_and_obey_caps():
    kwargs = {
        "num_steps": 20,
        "warmup_steps": 4,
        "budgets": range(8, 11),
        "masks_per_budget": 5,
        "phase_boundary": 10,
        "middle_gap_cap": 4,
        "tail_gap_cap": 6,
        "seed": 42,
    }

    first = generate_stratified_masks(**kwargs)
    second = generate_stratified_masks(**kwargs)

    assert first == second
    assert len(first) == len(set(first)) == 15
    for anchors in first:
        assert anchors[:4] == (0, 1, 2, 3)
        assert anchors[-1] == 19
        assert len(anchors) in (8, 9, 10)
        for left, right in zip(anchors[3:], anchors[4:]):
            cap = 4 if left < 10 else 6
            assert right - left <= cap


def test_envelope_table_detects_holdout_composition_violation_by_prompt():
    development = [
        {"sample_id": "d0", "z": 0.1, "harm": 0.2},
        {"sample_id": "d1", "z": 0.2, "harm": 0.3},
        {"sample_id": "d2", "z": 0.3, "harm": 0.4},
        {"sample_id": "d3", "z": 0.4, "harm": 0.5},
    ]
    holdout = [
        {"sample_id": "h0", "z": 0.15, "harm": 0.25},
        {"sample_id": "h1", "z": 0.18, "harm": 0.35},
        {"sample_id": "h1", "z": 0.50, "harm": 0.90},
    ]

    row = envelope_table(
        development,
        holdout,
        signal="z",
        harm="harm",
        quantiles=(0.5,),
    )[0]

    assert row["envelope"] == pytest.approx(0.2)
    assert row["development_harm_maximum_bound"] == pytest.approx(0.3)
    assert row["holdout_accepted_count"] == 2
    assert row["holdout_violation_count"] == 1
    assert row["holdout_violating_prompt_count"] == 1
    assert row["holdout_prompt_familywise_coverage"] == pytest.approx(0.5)
