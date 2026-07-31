from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from difflet.pipeline.cache import (
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
