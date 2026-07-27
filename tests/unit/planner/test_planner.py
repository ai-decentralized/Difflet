"""End-to-end planning: ranking, objectives, evidence, and cache awareness."""
from __future__ import annotations

import json

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.planner.hardware import HardwareProfile
from difflet.planner.measurements import Measurement, MeasurementStore
from difflet.planner.planner import OBJECTIVES, plan

FLUX = "black-forest-labs/FLUX.1-dev"
WAN = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

TRN2 = HardwareProfile(
    instance_type="trn2.3xlarge",
    platform_target="trn2",
    num_devices=1,
    cores_per_device=4,
    hbm_bytes_per_device=103079215104,
    lnc=2,
    allocated_cores=4,
    source="neuron-ls",
)


def _store(*rows: tuple[str, float]) -> MeasurementStore:
    return MeasurementStore(
        tuple(
            Measurement(
                instance_type="trn2.3xlarge", model="flux", model_id=FLUX, label=label,
                height=1024, width=1024, num_frames=None, steps=28,
                step_latency_seconds=seconds, e2e_warm_seconds=None,
                compile_seconds=None, source="test",
            )
            for label, seconds in rows
        )
    )


def _plan(**kwargs):
    kwargs.setdefault("model_type", "flux")
    kwargs.setdefault("steps", 28)
    kwargs.setdefault("hardware", TRN2)
    kwargs.setdefault("store", _store(("tp4", 0.2654)))
    kwargs.setdefault("cache_dir", "/nonexistent-cache")
    return plan(kwargs.pop("model_id", FLUX), **kwargs)


# ------------------------------------------------------------------ structure


def test_plan_ranks_every_feasible_candidate():
    result = _plan()
    assert result.ranked
    assert {entry.label for entry in result.ranked} == set(result.feasibility.labels())


def test_ranking_is_sorted_by_score():
    scores = [entry.score for entry in _plan().ranked]
    assert scores == sorted(scores, reverse=True)


def test_best_candidate_scores_one():
    assert _plan().best.score == pytest.approx(1.0)


def test_rejects_unknown_objective():
    with pytest.raises(ValueError, match="unknown objective"):
        _plan(objective="cheapest")


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_objective_produces_a_ranking(objective):
    assert _plan(objective=objective).ranked


# ----------------------------------------------------------------- objectives


def test_latency_objective_prefers_the_fastest_single_request():
    result = _plan(objective="latency")
    assert result.best.request_seconds == min(e.request_seconds for e in result.ranked)


def test_throughput_objective_prefers_the_most_requests_per_second():
    result = _plan(objective="throughput")
    assert result.best.throughput == max(e.throughput for e in result.ranked)


def test_throughput_and_latency_disagree_on_this_host():
    """dp buys throughput at the cost of latency, so the winners differ."""

    assert _plan(objective="latency").best.label != _plan(objective="throughput").best.label


def test_dp4_wins_on_throughput():
    assert _plan(objective="throughput").best.parallel.dp_degree == 4


def test_balanced_sits_between_the_two():
    latency_best = _plan(objective="latency").best
    balanced_best = _plan(objective="balanced").best
    throughput_best = _plan(objective="throughput").best
    assert balanced_best.request_seconds <= throughput_best.request_seconds
    assert balanced_best.throughput >= latency_best.throughput


# ------------------------------------------------------------------- evidence


def test_a_measured_config_reports_measured():
    result = _plan()
    tp4 = next(entry for entry in result.ranked if entry.label == "tp4")
    assert tp4.prediction.evidence == "measured"
    assert tp4.step_seconds == pytest.approx(0.2654)


def test_unmeasured_configs_report_predicted():
    result = _plan()
    others = [entry for entry in result.ranked if entry.label != "tp4"]
    assert others
    assert all(entry.prediction.evidence == "predicted" for entry in others)


def test_no_measurements_flags_the_whole_plan_uncalibrated():
    result = _plan(store=MeasurementStore(()))
    assert result.calibration.kind == "uncalibrated"
    assert all(e.prediction.evidence == "predicted-uncalibrated" for e in result.ranked)


def test_two_measurements_fit_the_bandwidth():
    result = _plan(store=_store(("tp4", 0.2654), ("tp2cp2", 0.2100)))
    assert result.calibration.kind == "measured-fit"
    assert not result.calibration.bandwidth_is_assumed


def test_measurements_for_another_shape_do_not_leak_in():
    """A step latency at 1024x1024 says nothing about 512x512."""

    result = _plan(height=512, width=512)
    assert result.calibration.kind == "uncalibrated"


def test_measurements_for_another_host_do_not_leak_in():
    other = HardwareProfile(
        instance_type="trn2.48xlarge", platform_target="trn2", num_devices=1,
        cores_per_device=4, hbm_bytes_per_device=103079215104, lnc=2,
        allocated_cores=4, source="neuron-ls",
    )
    assert _plan(hardware=other).calibration.kind == "uncalibrated"


def test_evidence_summary_counts_both_kinds():
    summary = _plan().evidence_summary
    assert summary.startswith("1 measured,")


# --------------------------------------------------------------- cache aware


def test_cache_awareness_marks_compiled_configs(tmp_path):
    entry = tmp_path / "flux" / "deadbeef"
    entry.mkdir(parents=True)
    (entry / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "cache_inputs": {"parallel": {"tp_degree": 2, "cp_degree": 2}},
            }
        ),
        encoding="utf-8",
    )
    result = _plan(cache_dir=str(tmp_path))
    cached = {entry.label for entry in result.ranked if entry.cached}
    assert cached == {"tp2cp2"}


def test_a_partial_cache_entry_does_not_count_as_compiled(tmp_path):
    """A compile that died before writing its manifest left artifacts, not a cache hit."""

    (tmp_path / "flux" / "deadbeef" / "transformer").mkdir(parents=True)
    result = _plan(cache_dir=str(tmp_path))
    assert not any(entry.cached for entry in result.ranked)


def test_corrupt_manifest_is_treated_as_a_miss(tmp_path):
    entry = tmp_path / "flux" / "deadbeef"
    entry.mkdir(parents=True)
    (entry / "manifest.json").write_text("{not json", encoding="utf-8")
    assert not any(e.cached for e in _plan(cache_dir=str(tmp_path)).ranked)


def test_cache_labels_survive_the_additive_key_elisions(tmp_path):
    """to_cache_dict omits cp_mode at gather_kv, sp when off and dp at 1."""

    entry = tmp_path / "flux" / "abc123"
    entry.mkdir(parents=True)
    (entry / "manifest.json").write_text(
        json.dumps({"schema_version": 4, "cache_inputs": {"parallel": {"tp_degree": 4}}}),
        encoding="utf-8",
    )
    result = _plan(cache_dir=str(tmp_path))
    assert {e.label for e in result.ranked if e.cached} == {"tp4"}


# ------------------------------------------------------------ memory advisory


def test_weight_estimate_flags_configs_that_replicate_too_much():
    result = _plan()
    over = {entry.label for entry in result.ranked if entry.weights_over_budget}
    # Four copies of Flux's 33.7 GB do not fit one 96 GB device.
    assert "tp1cp4" in over
    assert "tp4" not in over


def test_weight_advisory_does_not_remove_candidates():
    """Advisory means advisory: the residency model is not validated yet."""

    result = _plan()
    assert any(entry.weights_over_budget for entry in result.ranked)
    assert "tp1cp4" in result.feasibility.labels()


# ------------------------------------------------------------------- serving


def test_serving_mode_drops_dp_and_cfg_candidates():
    result = _plan(model_id=WAN, model_type="wan", steps=20, serving=True,
                   store=MeasurementStore(()))
    labels = {entry.label for entry in result.ranked}
    assert not any(label.startswith("dp") for label in labels)
    assert not any("cfg" in label for label in labels)


# -------------------------------------------------------------- shape effects


def test_larger_shapes_move_more_bytes():
    small = _plan(height=512, width=512)
    large = _plan(height=1024, width=1024)
    small_cp = next(e for e in small.ranked if e.label == "tp2cp2")
    large_cp = next(e for e in large.ranked if e.label == "tp2cp2")
    assert large_cp.prediction.comm.total > small_cp.prediction.comm.total
