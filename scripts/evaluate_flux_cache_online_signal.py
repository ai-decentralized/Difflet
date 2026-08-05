"""Historical feature extraction for the rejected FLUX online-signal audits.

The runtime probe and decision-veto path have been removed.  These pure helpers
remain only so frozen audit artifacts and the legacy collector's optional
artifact decoder stay readable; they are not a serving control surface.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence


def _positive_acceleration(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [max(float(now) - 2.0 * float(prior) + float(older), 0.0)
         for now, prior, older in zip(values[index], values[index - 1], values[index - 2])]
        for index in range(2, len(values))
    ]


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values)


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = _mean(values)
    if mean <= 1e-12:
        return 0.0
    return statistics.pstdev(float(value) for value in values) / mean


def extract_features(
    records: Sequence[Sequence[Any]],
    *,
    first_eligible_step: int,
    last_eligible_step: int,
) -> dict[str, float]:
    """Extract fixed request-level features from step-indexed 4x4 maps."""

    steps = [int(record[0]) for record in records]
    values = [[float(value) for value in record[1]] for record in records]
    if steps != list(range(len(steps))):
        raise ValueError("spatial probe records must cover consecutive steps from zero")
    if len(values) < 3 or any(len(row) != 16 for row in values):
        raise ValueError("spatial probe records must contain at least three 4x4 maps")
    if first_eligible_step < 2 or last_eligible_step < first_eligible_step:
        raise ValueError("eligible signal window is invalid")

    eligible_levels = [
        row for step, row in zip(steps, values)
        if first_eligible_step <= step <= last_eligible_step
    ]
    accelerations = _positive_acceleration(values)
    eligible_acceleration_records = [
        (step, row) for step, row in zip(steps[2:], accelerations)
        if first_eligible_step <= step <= last_eligible_step
    ]
    if not eligible_levels or not eligible_acceleration_records:
        raise ValueError("eligible signal window contains no probe records")
    eligible_accelerations = [row for _, row in eligible_acceleration_records]

    sorted_levels = [sorted(row, reverse=True) for row in eligible_levels]
    sorted_accelerations = [sorted(row, reverse=True) for row in eligible_accelerations]
    peak_acceleration_step, _ = max(
        eligible_acceleration_records,
        key=lambda record: max(record[1]),
    )
    peak_level_concentration = 100.0 * _coefficient_of_variation(
        values[peak_acceleration_step]
    )
    return {
        "max_region_level": max(row[0] for row in sorted_levels),
        "max_top2_region_level": max(_mean(row[:2]) for row in sorted_levels),
        "max_level_cv": max(_coefficient_of_variation(row) for row in eligible_levels),
        "max_region_acceleration": max(row[0] for row in sorted_accelerations),
        "max_top2_region_acceleration": max(
            _mean(row[:2]) for row in sorted_accelerations
        ),
        "max_acceleration_range": max(
            max(row) - min(row) for row in eligible_accelerations
        ),
        "max_acceleration_cv": max(
            _coefficient_of_variation(row) for row in eligible_accelerations
        ),
        # A peak can be a global schedule transition or a spatially concentrated
        # event.  Measure the level-map CV at the strongest acceleration step;
        # using the sparse positive-acceleration map itself would saturate when
        # only one region is positive.
        "peak_acceleration_level_cv_percent": peak_level_concentration,
    }


def extract_output_dynamics_features(
    records: Sequence[dict[str, Any]],
    *,
    first_eligible_step: int,
    last_eligible_step: int,
) -> dict[str, float]:
    """Extract fixed request features from cheap prediction-output dynamics."""

    eligible = [
        record
        for record in records
        if bool(record["used_cache_prediction"])
        and first_eligible_step <= int(record["step_index"]) <= last_eligible_step
    ]
    if not eligible:
        # A pre-skip warmup window intentionally has no predicted Transformer
        # outputs.  Preserve the fixed feature schema while representing that
        # absence explicitly; modulation features remain fully observed.
        return {
            f"output_max_{region}_{name}": 0.0
            for name in ("relative_l1", "velocity_turn", "acceleration_ratio")
            for region in ("region", "top2_region")
        }
    for record in eligible:
        for name in ("relative_l1", "velocity_turn", "acceleration_ratio"):
            if len(record[name]) != 16:
                raise ValueError("output-dynamics records require 16 regions per metric")

    result: dict[str, float] = {}
    for name in ("relative_l1", "velocity_turn", "acceleration_ratio"):
        ordered = [
            sorted((float(value) for value in record[name]), reverse=True)
            for record in eligible
        ]
        result[f"output_max_region_{name}"] = max(row[0] for row in ordered)
        result[f"output_max_top2_region_{name}"] = max(
            _mean(row[:2]) for row in ordered
        )
    return result


def _single_positive_auc(scores: Sequence[float], failure_index: int) -> float:
    failure = float(scores[failure_index])
    passes = [float(value) for index, value in enumerate(scores) if index != failure_index]
    wins = sum(failure > value for value in passes)
    ties = sum(failure == value for value in passes)
    return (wins + 0.5 * ties) / len(passes)


def summarize_features(
    rows: Sequence[dict[str, Any]], failure_indices: Sequence[int]
) -> dict[str, Any]:
    """Summarize one-positive opened audits without fitting a threshold."""

    if len(failure_indices) != 1:
        return {"note": "AUC summary currently requires exactly one known failure."}
    failure_index = int(failure_indices[0])
    names = tuple(rows[0]["features"])
    summary: dict[str, Any] = {}
    for name in names:
        scores = [float(row["features"][name]) for row in rows]
        failure_score = scores[failure_index]
        passes_at_or_above = sum(
            score >= failure_score for index, score in enumerate(scores)
            if index != failure_index
        )
        pass_scores = [score for index, score in enumerate(scores) if index != failure_index]
        summary[name] = {
            "auc": _single_positive_auc(scores, failure_index),
            "failure_descending_rank": 1 + sum(score > failure_score for score in pass_scores),
            "passing_requests_at_or_above_failure": passes_at_or_above,
            "failure_score": failure_score,
            "passing_median": statistics.median(pass_scores),
            "passing_maximum": max(pass_scores),
        }
    return summary
