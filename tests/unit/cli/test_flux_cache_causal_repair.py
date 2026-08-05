from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.flux_cache_causal_repair import (
    CausalRepairLayout,
    ReplayRepairHook,
    ShadowCaptureHook,
    load_shadow_trace,
    repair_gain,
    splice_true_region,
)


def _call_hook(hook, *, latents, predicted, actual, step=2, skipped=True):
    calls = {"actual": 0}

    def compute_actual():
        calls["actual"] += 1
        return actual

    output = hook(
        step_index=step,
        timestep=torch.tensor(1.0),
        latents=latents,
        predicted=predicted,
        used_cache_prediction=skipped,
        compute_actual=compute_actual,
    )
    return output, calls


def test_splice_true_region_repairs_exactly_one_grid_quadrant():
    predicted = torch.zeros(1, 16, 2)
    actual = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
    layout = CausalRepairLayout(4, 4, region_rows=2, region_columns=2)

    repaired = splice_true_region(
        predicted,
        actual,
        layout=layout,
        region_index=1,
    )

    repaired_grid = repaired.reshape(1, 4, 4, 2)
    actual_grid = actual.reshape(1, 4, 4, 2)
    assert torch.equal(repaired_grid[:, :2, 2:], actual_grid[:, :2, 2:])
    assert torch.count_nonzero(repaired_grid[:, :2, :2]) == 0
    assert torch.count_nonzero(repaired_grid[:, 2:, :]) == 0
    assert torch.count_nonzero(predicted) == 0


def test_shadow_capture_is_read_only_and_replay_applies_full_truth(tmp_path):
    latents = torch.arange(16, dtype=torch.bfloat16).reshape(1, 8, 2)
    predicted = torch.ones(1, 16, 2, dtype=torch.bfloat16)
    actual = torch.full_like(predicted, 3)
    capture = ShadowCaptureHook(tmp_path / "trace")

    output, calls = _call_hook(
        capture,
        latents=latents,
        predicted=predicted,
        actual=actual,
    )
    assert output is predicted
    assert calls["actual"] == 1

    trace_path = capture.write_manifest(
        identity={"sample_id": "p011-s0", "candidate_id": "adaptive-oil"}
    )
    trace = load_shadow_trace(trace_path)
    assert trace["offline_teacher_only"] is True
    assert [row["step_index"] for row in trace["steps"]] == [2]

    replay = ReplayRepairHook(trace_path, target_step=2)
    repaired, replay_calls = _call_hook(
        replay,
        latents=latents.clone(),
        predicted=predicted.clone(),
        actual=torch.full_like(actual, 99),
    )
    replay.validate_complete()
    assert replay_calls["actual"] == 0
    assert torch.equal(repaired, actual)


def test_shadow_capture_ignores_real_anchor_without_teacher_call(tmp_path):
    capture = ShadowCaptureHook(tmp_path / "trace")
    tensor = torch.ones(1, 16, 2)

    output, calls = _call_hook(
        capture,
        latents=tensor,
        predicted=tensor,
        actual=tensor * 2,
        skipped=False,
    )

    assert output is tensor
    assert calls["actual"] == 0
    assert capture.records == ()


def test_region_replay_rejects_pre_intervention_drift(tmp_path):
    latents = torch.ones(1, 8, 2)
    predicted = torch.ones(1, 16, 2)
    actual = torch.full_like(predicted, 2)
    capture = ShadowCaptureHook(tmp_path / "trace")
    _call_hook(capture, latents=latents, predicted=predicted, actual=actual)
    trace_path = capture.write_manifest(identity={"sample_id": "p011-s0"})
    replay = ReplayRepairHook(
        trace_path,
        target_step=2,
        layout=CausalRepairLayout(4, 4, region_rows=2, region_columns=2),
        region_index=0,
    )

    with pytest.raises(RuntimeError, match="pre-step latent"):
        _call_hook(
            replay,
            latents=latents + 1,
            predicted=predicted,
            actual=actual,
        )


def test_trace_digest_and_artifact_hash_fail_closed(tmp_path):
    tensor = torch.ones(1, 16, 2)
    capture = ShadowCaptureHook(tmp_path / "trace")
    _call_hook(capture, latents=tensor, predicted=tensor, actual=tensor * 2)
    trace_path = capture.write_manifest(identity={"sample_id": "p011-s0"})

    document = json.loads(trace_path.read_text())
    document["identity"]["sample_id"] = "tampered"
    trace_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="digest"):
        load_shadow_trace(trace_path)


def test_repair_gain_is_signed_vqa_improvement():
    assert repair_gain(cached_vqa=0.25, repaired_vqa=0.6) == pytest.approx(0.35)
    with pytest.raises(ValueError, match="finite"):
        repair_gain(cached_vqa=float("nan"), repaired_vqa=0.6)


def test_registered_pilot_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    protocol = json.loads(
        (root / "benchmark/flux_cache/spatial-causal-repair-pilot.json").read_text()
    )
    digest = protocol.pop("sha256")

    from scripts.flux_cache_causal_repair import _canonical_sha256

    assert digest == _canonical_sha256(protocol)


def test_registered_pilot_amendment_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    amendment = json.loads(
        (
            root
            / "benchmark/flux_cache/spatial-causal-repair-pilot-amendment.json"
        ).read_text()
    )
    digest = amendment.pop("sha256")

    from scripts.flux_cache_causal_repair import _canonical_sha256

    assert digest == _canonical_sha256(amendment)


def test_registered_pilot_result_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    result = json.loads(
        (
            root / "benchmark/flux_cache/spatial-causal-repair-pilot-result.json"
        ).read_text()
    )
    digest = result.pop("sha256")

    from scripts.flux_cache_causal_repair import _canonical_sha256

    assert digest == _canonical_sha256(result)
