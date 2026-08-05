from __future__ import annotations

import pytest

from scripts.analyze_flux_cache_online_gate import (
    grouped_oof_logistic,
    threshold_at_full_recall,
)


def test_full_recall_threshold_uses_lowest_positive_score():
    result = threshold_at_full_recall(
        [0, 1, 0, 1, 0],
        [0.1, 0.8, 0.6, 0.4, 0.2],
    )

    assert result["threshold"] == pytest.approx(0.4)
    assert result["failure_recall"] == pytest.approx(1.0)
    assert result["passing_false_brake_count"] == 1
    assert result["passing_false_brake_rate"] == pytest.approx(1 / 3)


def test_grouped_oof_logistic_holds_out_each_prompt_once():
    rows = []
    for prompt_index in range(6):
        failed = prompt_index in {1, 4}
        for candidate_index in range(2):
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "candidate_id": f"candidate-{candidate_index}",
                    "failed": failed and candidate_index == 1,
                    "features": {"risk": float(prompt_index + candidate_index)},
                }
            )

    result = grouped_oof_logistic(rows, ("risk",))

    assert result["fold_count"] == 6
    assert len(result["oof_probabilities"]) == len(rows)
    assert all(0.0 <= value <= 1.0 for value in result["oof_probabilities"])
    assert result["metrics"]["roc_auc"] is not None
