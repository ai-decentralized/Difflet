from __future__ import annotations

import pytest
import torch

from difflet.pipeline.cache import (
    CacheRunner,
    CacheStepContext,
    CadencePolicy,
    ExplicitMaskPolicy,
    LegacyResidualPredictor,
    PeriodicAnchorPolicy,
    QualityRecoveryConfig,
    QualityRecoveryGuard,
    RecoveryDecision,
    TaylorSeerPredictor,
)


def _ctx(index: int, steps: int, *, barrier: bool = False) -> CacheStepContext:
    return CacheStepContext(
        step_index=index,
        num_steps=steps,
        is_barrier=barrier,
    )


def test_predictions_never_enter_true_anchor_history():
    runner = CacheRunner(
        ExplicitMaskPolicy([True, True, False, False, True]),
        TaylorSeerPredictor(order=1),
    )
    values = []
    for index in range(5):
        context = _ctx(index, 5)
        if runner.should_skip(context):
            output = runner.predict()
        else:
            output = torch.tensor([float(index * index)])
            runner.record_anchor(context, output)
        values.append(float(output.item()))

    assert values == [0.0, 1.0, 2.0, 3.0, 16.0]
    assert [anchor.step_index for anchor in runner.history] == [1, 4]
    assert runner.stats()["skipped_steps"] == 2


def test_readiness_rejection_forces_real_anchors():
    runner = CacheRunner(CadencePolicy(cadence=1), TaylorSeerPredictor(order=1))
    decisions = []
    for index in range(3):
        context = _ctx(index, 3)
        decision = runner.decide(context)
        decisions.append(decision.reason)
        if decision.should_skip:
            runner.predict()
        else:
            runner.record_anchor(context, torch.tensor([float(index)]))
    assert decisions == ["history_not_ready", "history_not_ready", "policy_skip"]
    assert runner.stats()["readiness_rejections"] == 2


def test_predictor_consecutive_limit_vetoes_dynamic_second_skip():
    class AlwaysSkipPolicy:
        def should_skip(self, context, history, observation):
            return True

        def reset(self):
            return None

    runner = CacheRunner(AlwaysSkipPolicy(), LegacyResidualPredictor())
    reasons = []
    for index in range(4):
        context = _ctx(index, 4)
        decision = runner.decide(context)
        reasons.append(decision.reason)
        if decision.should_skip:
            runner.predict()
        else:
            runner.record_anchor(context, torch.tensor([float(index)]))
    assert reasons == [
        "history_not_ready",
        "history_not_ready",
        "policy_skip",
        "consecutive_skip_veto",
    ]
    assert runner.stats()["consecutive_skip_vetoes"] == 1


def test_barrier_invalidates_history_and_rebuilds_readiness():
    runner = CacheRunner(
        ExplicitMaskPolicy([True, True, False, True, True, False]),
        TaylorSeerPredictor(order=1),
    )
    reasons = []
    for index in range(6):
        context = _ctx(index, 6, barrier=index == 3)
        decision = runner.decide(context)
        reasons.append(decision.reason)
        if decision.should_skip:
            runner.predict()
        else:
            runner.record_anchor(context, torch.tensor([float(index)]))
    assert reasons == [
        "policy_compute",
        "policy_compute",
        "policy_skip",
        "barrier",
        "policy_compute",
        "policy_skip",
    ]
    assert runner.stats()["barrier_resets"] == 1


def test_quality_recovery_request_forces_fresh_anchor_window_and_history_reset():
    runner = CacheRunner(
        CadencePolicy(cadence=1),
        TaylorSeerPredictor(order=1),
        recovery=QualityRecoveryGuard(QualityRecoveryConfig(recovery_steps=2)),
    )
    for index in range(2):
        context = _ctx(index, 5)
        assert not runner.should_skip(context)
        runner.record_anchor(context, torch.tensor([float(index)]))

    runner.request_quality_recovery("online drift", steps=2, reset_history=True)
    reasons = []
    for index in range(2, 5):
        context = _ctx(index, 5)
        decision = runner.decide(context)
        reasons.append(decision.reason)
        if decision.should_skip:
            runner.predict()
        else:
            runner.record_anchor(context, torch.tensor([float(index)]))

    assert reasons == ["recovery_requested", "recovery_requested", "policy_skip"]
    stats = runner.stats()
    assert stats["recovery_triggers"] == 1
    assert stats["recovery_history_resets"] == 1
    assert stats["recovery_forced_steps"] == 2


def test_quality_consecutive_bound_can_safely_wrap_legacy_cadence_one():
    recovery = QualityRecoveryGuard(
        QualityRecoveryConfig(max_consecutive_predictions=1)
    )
    runner = CacheRunner(
        CadencePolicy(cadence=1),
        LegacyResidualPredictor(),
        recovery=recovery,
    )
    reasons = []
    for index in range(5):
        context = _ctx(index, 5)
        decision = runner.decide(context)
        reasons.append(decision.reason)
        if decision.should_skip:
            runner.predict()
        else:
            runner.record_anchor(context, torch.tensor([float(index)]))
    assert reasons == [
        "history_not_ready",
        "history_not_ready",
        "policy_skip",
        "recovery_consecutive_limit",
        "policy_skip",
    ]


def test_runner_rejects_bypassing_or_abandoning_a_decision():
    runner = CacheRunner(CadencePolicy(cadence=1), TaylorSeerPredictor(order=1))
    with pytest.raises(RuntimeError, match="accepted skip decision"):
        runner.predict(_ctx(0, 3))

    context = _ctx(0, 3)
    runner.decide(context)
    with pytest.raises(RuntimeError, match="has not been completed"):
        runner.decide(_ctx(1, 3))
    with pytest.raises(RuntimeError, match="does not match"):
        runner.record_anchor(_ctx(1, 3), torch.tensor([1.0]))


def test_runner_lifts_policy_windows_into_default_recovery_guard():
    runner = CacheRunner(
        PeriodicAnchorPolicy(
            anchor_interval=3,
            anchor_phase=1,
            warmup_steps=2,
            cooldown_steps=1,
            require_final_anchor=True,
        ),
        TaylorSeerPredictor(order=1),
    )
    reasons = []
    for index in range(6):
        context = _ctx(index, 6)
        decision = runner.decide(context)
        reasons.append(decision.reason)
        if decision.should_skip:
            runner.predict()
        else:
            runner.record_anchor(context, torch.tensor([float(index)]))
    assert reasons[:2] == ["recovery_warmup", "recovery_warmup"]
    assert reasons[-1] == "recovery_cooldown"


def test_recovery_decision_rejects_missing_or_unknown_force_reason():
    with pytest.raises(ValueError, match="requires a recovery reason"):
        RecoveryDecision(force_compute=True, reason="none")
    with pytest.raises(ValueError, match="unsupported recovery reason"):
        RecoveryDecision(force_compute=True, reason="custom")


def test_runner_rejects_malformed_custom_recovery_result():
    class MalformedRecovery:
        def before_step(self, context, history, observation):
            return object()

        def observe_anchor(self, context, output, history, observation):
            return None

        def observe_prediction(self, context, output, history, observation):
            return None

        def reset(self):
            return None

    runner = CacheRunner(
        ExplicitMaskPolicy([True]),
        TaylorSeerPredictor(order=1),
        recovery=MalformedRecovery(),
    )
    with pytest.raises(TypeError, match="must return a RecoveryDecision"):
        runner.decide(_ctx(0, 1))


def test_overlapping_recovery_windows_are_safe_full_compute():
    config = QualityRecoveryConfig(warmup_steps=3, cooldown_steps=3)
    assert config.apply_to_anchor_mask((False,) * 5) == (True,) * 5

    runner = CacheRunner(
        CadencePolicy(cadence=2, warmup_steps=3, cooldown_steps=3),
        LegacyResidualPredictor(),
    )
    reasons = []
    for index in range(5):
        context = _ctx(index, 5)
        decision = runner.decide(context)
        reasons.append(decision.reason)
        assert decision.should_compute
        runner.record_anchor(context, torch.tensor([float(index)]))
    assert reasons == [
        "recovery_warmup",
        "recovery_warmup",
        "recovery_warmup",
        "recovery_cooldown",
        "recovery_cooldown",
    ]


def test_barrier_clears_stale_pending_recovery_but_preserves_metrics():
    runner = CacheRunner(
        ExplicitMaskPolicy([True, True]),
        TaylorSeerPredictor(order=1),
    )
    first = _ctx(0, 2)
    runner.decide(first)
    runner.record_anchor(first, torch.tensor([0.0]))
    runner.request_quality_recovery("pre-barrier drift", steps=2)

    barrier = _ctx(1, 2, barrier=True)
    assert runner.decide(barrier).reason == "barrier"
    runner.record_anchor(barrier, torch.tensor([1.0]))

    assert runner.stats()["recovery_triggers"] == 1
    assert runner.recovery.stats()["quality_recovery_pending_steps"] == 0
    assert "quality_recovery_triggers" not in runner.recovery.stats()
