from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from difflet.pipeline.cache import (
    AdaptiveAnchorConfig,
    AdaptiveAnchorPolicy,
    AnchorMeasurement,
    CacheAnchor,
    CacheHistory,
    CacheStepContext,
    CadencePolicy,
    ExplicitMaskPolicy,
    LegacyResidualPredictor,
    PeriodicAnchorPolicy,
    RuntimeObservation,
    TaylorSeerPredictor,
    TeaCachePolicy,
)


def _context(index: int, steps: int = 10, **kwargs) -> CacheStepContext:
    return CacheStepContext(step_index=index, num_steps=steps, **kwargs)


def _history(points, *, coord: str = "index") -> CacheHistory:
    history = CacheHistory(len(points))
    for index, coordinate, value in points:
        kwargs = {} if coord == "index" else {coord: coordinate}
        history.push(CacheAnchor(_context(index, **kwargs), torch.tensor([value])))
    return history


def _adaptive_config(**overrides) -> AdaptiveAnchorConfig:
    values = {
        "initial_anchor_interval": 8,
        "minimum_anchor_interval": 4,
        "maximum_anchor_interval": 10,
        "warmup_steps": 6,
        "cooldown_steps": 1,
        "anchor_phase": 1,
        "tighten_error": 0.5,
        "recovery_error": 1.0,
        "acceleration_error": 0.25,
        "recovery_steps": 2,
        "disable_after_recoveries": 2,
        "stable_anchors_for_acceleration": 2,
        "allow_acceleration": False,
        "require_final_anchor": True,
    }
    values.update(overrides)
    return AdaptiveAnchorConfig(**values)


def _anchor_measurement(
    step_index: int,
    *,
    error: float | None,
    num_steps: int = 20,
    status: str = "measured",
) -> AnchorMeasurement:
    return AnchorMeasurement(
        step_index=step_index,
        num_steps=num_steps,
        timestep=None,
        sigma=None,
        decision_reason="policy_compute",
        history_size=2 if status == "measured" else 0,
        estimated_steps_since_anchor=0,
        anchor_step_gap=1 if step_index else None,
        output_shape=(1,),
        output_dtype="float32",
        output_norm=1.0,
        relative_output_change=0.0,
        relative_output_curvature=0.0,
        estimate_status=status,
        estimate_relative_error=error,
        estimate_seconds=0.0 if status == "measured" else None,
        measurement_seconds=0.0,
        numerically_valid=status in {"measured", "history_not_ready"},
    )


def test_periodic_anchor_policy_matches_documented_formula():
    policy = PeriodicAnchorPolicy(
        anchor_interval=4,
        anchor_phase=1,
        warmup_steps=3,
        cooldown_steps=1,
    )
    assert policy.materialize_anchor_mask(10) == (
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        True,
        True,
    )


def test_adaptive_anchor_policy_brakes_without_changing_the_first_safe_gap():
    policy = AdaptiveAnchorPolicy(_adaptive_config())
    history = CacheHistory(2)
    observation = RuntimeObservation()
    decisions = []
    for step_index in range(12):
        context = _context(step_index, steps=20)
        skip = policy.should_skip(context, history, observation)
        decisions.append(skip)
        if not skip:
            if step_index < 2:
                measurement = _anchor_measurement(
                    step_index,
                    error=None,
                    status="history_not_ready",
                )
            else:
                measurement = _anchor_measurement(
                    step_index,
                    error=0.6 if step_index == 2 else 0.3,
                )
            policy.observe_anchor_measurement(measurement)

    assert decisions == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        True,
        False,
    ]
    assert policy.current_anchor_interval == 4
    assert policy.stats()["adaptive_tightenings"] == 1
    assert policy.stats()["adaptive_accelerations"] == 0


def test_adaptive_anchor_policy_recovers_then_disables_after_repeated_large_errors():
    policy = AdaptiveAnchorPolicy(_adaptive_config())

    policy.observe_anchor_measurement(_anchor_measurement(2, error=1.1))
    assert policy.state == "recovery"
    policy.observe_anchor_measurement(_anchor_measurement(3, error=0.1))
    policy.observe_anchor_measurement(_anchor_measurement(4, error=0.1))
    assert policy.state == "active"

    policy.observe_anchor_measurement(_anchor_measurement(5, error=1.1))
    assert policy.state == "disabled"
    assert policy.should_skip(
        _context(8, steps=20),
        CacheHistory(2),
        RuntimeObservation(),
    ) is False
    assert policy.stats()["adaptive_recoveries"] == 2
    assert policy.stats()["adaptive_disabled"] == 1


def test_adaptive_anchor_policy_oil_branch_is_explicitly_gated():
    disabled = AdaptiveAnchorPolicy(_adaptive_config(allow_acceleration=False))
    enabled = AdaptiveAnchorPolicy(_adaptive_config(allow_acceleration=True))
    for policy in (disabled, enabled):
        policy.observe_anchor_measurement(_anchor_measurement(2, error=0.1))
        policy.observe_anchor_measurement(_anchor_measurement(3, error=0.1))

    assert disabled.current_anchor_interval == 8
    assert disabled.stats()["adaptive_accelerations"] == 0
    assert enabled.current_anchor_interval == 9
    assert enabled.stats()["adaptive_accelerations"] == 1


def test_adaptive_anchor_config_rejects_overlapping_thresholds():
    with pytest.raises(ValueError, match="acceleration < tighten < recovery"):
        _adaptive_config(tighten_error=1.0, recovery_error=0.5)


def test_cadence_and_explicit_mask_are_schedule_only():
    cadence = CadencePolicy(cadence=3, warmup_steps=2, cooldown_steps=1)
    assert cadence.materialize_anchor_mask(8) == (
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    )
    explicit = ExplicitMaskPolicy([True, True, False, True])
    assert explicit.materialize_anchor_mask(4) == (True, True, False, True)


def test_taylorseer_order_one_uses_real_anchor_coordinates():
    history = _history([(0, 10.0, 2.0), (2, 6.0, 10.0)], coord="timestep")
    predictor = TaylorSeerPredictor(order=1, coord="timestep")
    predicted = predictor.predict(_context(3, timestep=4.0), history)
    assert predicted.item() == pytest.approx(14.0)


def test_taylorseer_order_two_newton_extrapolation_is_exact_for_quadratic():
    history = _history(
        [(0, 0.0, 1.0), (1, 1.0, 4.0), (2, 3.0, 16.0)],
        coord="sigma",
    )
    predictor = TaylorSeerPredictor(order=2, coord="sigma")
    predicted = predictor.predict(_context(3, sigma=4.0), history)
    assert predicted.item() == pytest.approx(25.0)


def test_legacy_residual_scales_nonuniform_coordinate_distance():
    history = _history([(0, 0.0, 3.0), (2, 4.0, 11.0)], coord="sigma")
    predicted = LegacyResidualPredictor(coord="sigma").predict(
        _context(3, sigma=5.0), history
    )
    assert predicted.item() == pytest.approx(13.0)


def test_teacache_policy_is_independent_of_predictor():
    calibration = SimpleNamespace(
        cadence=2,
        online_delta_alpha=0.0,
        warmup_steps=2,
        cooldown_steps=1,
        skip_run_length=1,
        accumulate=False,
        threshold=0.1,
        predict_delta=lambda value: value,
    )
    policy = TeaCachePolicy(calibration)
    history = CacheHistory(2)
    observation = RuntimeObservation()
    decisions = [
        policy.should_skip(_context(index, steps=7), history, observation)
        for index in range(7)
    ]
    assert decisions == [False, False, False, True, False, True, False]
