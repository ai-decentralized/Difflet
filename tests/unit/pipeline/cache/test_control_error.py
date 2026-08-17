from __future__ import annotations

import pytest
import torch

from difflet.pipeline.cache import (
    AnchorErrorMeasurement,
    CacheAnchor,
    CacheHistory,
    CacheRunner,
    CacheStepContext,
    StaticPlusBrakeConfig,
    StaticPlusBrakePolicy,
    TaylorSeerPredictor,
    measure_anchor_error,
)


def _context(step_index: int, num_steps: int = 4) -> CacheStepContext:
    return CacheStepContext(step_index=step_index, num_steps=num_steps)


def _history() -> CacheHistory:
    history = CacheHistory(2)
    history.push(CacheAnchor(_context(0), torch.tensor([1.0, 2.0])))
    history.push(CacheAnchor(_context(1), torch.tensor([2.0, 4.0])))
    return history


def test_anchor_error_matches_float32_l2_formula():
    measurement = measure_anchor_error(
        context=_context(2),
        output=torch.tensor([4.0, 7.0], dtype=torch.bfloat16),
        predictor=TaylorSeerPredictor(order=1),
        history=_history(),
    )

    expected = torch.linalg.vector_norm(torch.tensor([1.0, 1.0])).item()
    expected /= torch.linalg.vector_norm(torch.tensor([4.0, 7.0])).item()
    assert measurement.estimate_status == "measured"
    assert measurement.estimate_relative_error == pytest.approx(expected)
    assert measurement.numerically_valid is True


def test_anchor_error_reports_unready_and_invalid_outputs():
    predictor = TaylorSeerPredictor(order=1)
    unready = measure_anchor_error(
        context=_context(0),
        output=torch.tensor([1.0]),
        predictor=predictor,
        history=CacheHistory(2),
    )
    invalid = measure_anchor_error(
        context=_context(0),
        output=torch.tensor([float("nan")]),
        predictor=predictor,
        history=CacheHistory(2),
    )

    assert unready == AnchorErrorMeasurement(0, 4, "history_not_ready", None, True)
    assert invalid == AnchorErrorMeasurement(0, 4, "invalid_actual_output", None, False)


def test_runner_feeds_control_error_to_bounded_brake():
    policy = StaticPlusBrakePolicy(
        StaticPlusBrakeConfig(
            anchor_mask=(True, True, True, False, False, True),
            plastic_window_start=2,
            plastic_window_end=4,
            dynamic_budget=1,
            tighten_error=0.1,
            recovery_error=1.0,
            recovery_steps=1,
            disable_after_recoveries=2,
        )
    )
    runner = CacheRunner(policy, TaylorSeerPredictor(order=1))
    outputs = (
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([5.0]),
    )
    for step_index, output in enumerate(outputs):
        context = CacheStepContext(step_index=step_index, num_steps=6)
        assert runner.decide(context).should_skip is False
        runner.record_anchor(context, output)

    assert policy.stats()["static_brake_last_anchor_error"] == pytest.approx(0.4)
    assert runner.decide(CacheStepContext(step_index=3, num_steps=6)).should_skip is False


def test_runner_trace_binds_endpoint_z_to_exact_scheduler_weighted_segment():
    policy = StaticPlusBrakePolicy(
        StaticPlusBrakeConfig(
            anchor_mask=(True, True, False, False, True),
            plastic_window_start=2,
            plastic_window_end=3,
            dynamic_budget=1,
            tighten_error=10.0,
            recovery_error=20.0,
            recovery_steps=1,
            disable_after_recoveries=2,
        )
    )
    runner = CacheRunner(policy, TaylorSeerPredictor(order=1))
    contexts = tuple(
        CacheStepContext(
            step_index=index,
            num_steps=5,
            timestep=float(100 - index),
            sigma=float(5 - index) / 10.0,
        )
        for index in range(5)
    )
    for index, output in ((0, torch.tensor([1.0])), (1, torch.tensor([2.0]))):
        assert runner.decide(contexts[index]).should_skip is False
        runner.record_anchor(contexts[index], output)
    for index in (2, 3):
        assert runner.decide(contexts[index]).should_skip is True
        runner.predict(contexts[index])
    assert runner.decide(contexts[4]).should_skip is False
    runner.record_anchor(contexts[4], torch.tensor([8.0]))

    trace = runner.anchor_error_trace()
    entry = trace["entries"][-1]
    assert trace["path_semantics"] == "logical_post_restore_path"
    assert entry["previous_anchor_step_index"] == 1
    assert entry["anchor_step_index"] == 4
    assert entry["anchor_gap"] == 3
    assert entry["estimate_step_indices"] == [2, 3]
    assert entry["estimate_step_count"] == 2
    assert entry["scheduler_signed_delta_sigma"] == pytest.approx(-0.2)
    assert entry["scheduler_abs_delta_sigma"] == pytest.approx(0.2)
    assert entry["endpoint_z"] == pytest.approx(0.375)


def test_anchor_error_rejects_incoherent_records():
    with pytest.raises(ValueError, match="numerically valid"):
        AnchorErrorMeasurement(0, 1, "measured", 0.5, False)
