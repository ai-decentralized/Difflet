from __future__ import annotations

import math

import pytest

from scripts.analyze_flux_cache_semantic_boundary import correlation_report, summarize


def _row(candidate: str, signal: float, target: float):
    return {"candidate_id": candidate, "signal": signal, "target": target}


def test_semantic_boundary_summary_reports_plain_statistics():
    assert summarize([1.0, 2.0, 7.0]) == {
        "minimum": 1.0,
        "median": 2.0,
        "mean": 10.0 / 3.0,
        "maximum": 7.0,
    }


@pytest.mark.parametrize("values", [[], [math.nan], [math.inf]])
def test_semantic_boundary_summary_rejects_invalid_samples(values):
    with pytest.raises(ValueError, match="non-empty and finite"):
        summarize(values)


def test_semantic_boundary_reports_pooled_and_fixed_candidate_correlations():
    rows = [
        _row("a", 1.0, 1.0),
        _row("a", 2.0, 2.0),
        _row("a", 3.0, 3.0),
        _row("b", 10.0, 3.0),
        _row("b", 11.0, 2.0),
        _row("b", 12.0, 1.0),
    ]

    report = correlation_report(rows, signal="signal", target="target")

    assert report["pooled_spearman"] is not None
    assert report["candidate_stratified_spearman"] == pytest.approx(0.0)
    assert [row["spearman"] for row in report["per_candidate"]] == [1.0, -1.0]
