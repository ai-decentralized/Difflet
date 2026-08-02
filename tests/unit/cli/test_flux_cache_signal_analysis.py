from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_flux_cache_signals import (
    candidate_stratified_spearman,
    load_signal_study_protocol,
    spearman,
)


def test_candidate_stratified_spearman_removes_candidate_level_confounding():
    rows = [
        {"candidate_id": "a", "signal": 1.0, "lpips": 10.0},
        {"candidate_id": "a", "signal": 2.0, "lpips": 11.0},
        {"candidate_id": "b", "signal": 10.0, "lpips": 1.0},
        {"candidate_id": "b", "signal": 11.0, "lpips": 2.0},
    ]

    pooled = spearman(
        [row["signal"] for row in rows],
        [row["lpips"] for row in rows],
    )
    stratified = candidate_stratified_spearman(
        rows,
        signal="signal",
        target="lpips",
    )

    assert pooled is not None and pooled < 0.0
    assert stratified == pytest.approx(1.0)


def test_spearman_uses_average_ranks_and_rejects_nonfinite_values():
    assert spearman([1.0, 1.0, 2.0], [2.0, 2.0, 4.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0], [2.0, 3.0]) is None
    with pytest.raises(ValueError, match="finite"):
        spearman([1.0, float("nan")], [1.0, 2.0])


def test_signal_study_protocol_is_content_addressed(tmp_path):
    source = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "signal-study-protocol.json"
    )
    document = load_signal_study_protocol(source)
    assert document["primary_test"]["minimum_samples_per_candidate"] == 16

    document["primary_test"]["minimum_stratified_spearman"] = 0.1
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        load_signal_study_protocol(tampered)


def test_signal_confirmation_protocol_promotes_the_preregistered_mean_signal():
    source = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "signal-confirmation-protocol.json"
    )

    document = load_signal_study_protocol(source)

    assert document["primary_test"]["signal"] == "mean-anchor-estimate-relative-error"
    assert "maximum-anchor-estimate-relative-error" in document["exploratory_signals"]
