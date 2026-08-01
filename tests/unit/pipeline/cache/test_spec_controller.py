from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from difflet.pipeline.cache import (
    CACHE_MASK_SCHEMA,
    CACHE_PLAN_SCHEMA,
    CacheRunner,
    CacheSession,
    CacheSpecError,
    LegacyResidualPredictor,
    PeriodicAnchorPolicy,
    QualityRecoveryConfig,
    QualityRecoveryGuard,
    ResolvedCacheSession,
    TaylorSeerPredictor,
    TeaCacheControllerAdapter,
    TeaCachePolicy,
    load_cache_mask,
    load_cache_plan,
    resolve_cache_config,
    resolve_cache_plan,
    validate_schedule_safety,
)
from difflet.pipeline.teacache import TeaCacheCalibration


def _plan_dict(num_steps: int = 10):
    policy = PeriodicAnchorPolicy(
        anchor_interval=3,
        anchor_phase=1,
        warmup_steps=3,
        cooldown_steps=1,
        require_final_anchor=True,
    )
    return {
        "schema": CACHE_PLAN_SCHEMA,
        "compatibility": {
            "model": "flux",
            "shape_label": "1024x1024",
            "num_steps": num_steps,
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        },
        "policy": {
            "type": "periodic_anchor",
            "anchor_interval": 3,
            "anchor_phase": 1,
            "warmup_steps": 3,
            "cooldown_steps": 1,
            "require_final_anchor": True,
        },
        "predictor": {
            "type": "taylorseer",
            "order": 1,
            "coord": "index",
        },
        "frozen_mask": list(policy.materialize_anchor_mask(num_steps)),
    }


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.update(extra=True), "unknown fields"),
        (lambda data: data["policy"].pop("anchor_phase"), "missing required"),
        (lambda data: data["predictor"].update(order=True), "positive integer"),
        (lambda data: data.update(schema="wrong"), "schema must be"),
    ],
)
def test_plan_parser_is_fail_closed(mutation, message):
    data = _plan_dict()
    mutation(data)
    with pytest.raises(CacheSpecError, match=message):
        load_cache_plan(data)


def test_mask_schema_is_strict_but_allows_description():
    mask = load_cache_mask(
        {
            "schema": CACHE_MASK_SCHEMA,
            "num_steps": 3,
            "anchor_mask": [True, False, True],
            "description": "external SCM schedule",
        }
    )
    assert mask.anchor_mask == (True, False, True)
    with pytest.raises(CacheSpecError, match="JSON booleans"):
        load_cache_mask(
            {
                "schema": CACHE_MASK_SCHEMA,
                "num_steps": 3,
                "anchor_mask": [1, 0, 1],
            }
        )


def test_plan_resolution_checks_identity_and_policy_mask_drift():
    plan = load_cache_plan(_plan_dict())
    with pytest.raises(CacheSpecError, match="compatibility does not match"):
        resolve_cache_plan(
            plan,
            model="flux",
            shape_label="768x768",
            num_steps=10,
            scheduler_class="FlowMatchEulerDiscreteScheduler",
        )

    drifted = _plan_dict()
    drifted["frozen_mask"][5] = not drifted["frozen_mask"][5]
    with pytest.raises(CacheSpecError, match="semantics drifted"):
        resolve_cache_plan(
            load_cache_plan(drifted),
            model="flux",
            shape_label="1024x1024",
            num_steps=10,
            scheduler_class="FlowMatchEulerDiscreteScheduler",
        )


def test_schedule_safety_restarts_history_after_barrier():
    predictor = TaylorSeerPredictor(order=1)
    with pytest.raises(CacheSpecError, match="after barrier step 3"):
        validate_schedule_safety(
            [True, True, False, True, False],
            predictor,
            barrier_steps=[3],
        )
    with pytest.raises(CacheSpecError, match="must be a real anchor"):
        validate_schedule_safety(
            [True, True, False, False, True],
            predictor,
            barrier_steps=[3],
        )


def test_independent_recovery_materializes_effective_mask():
    mask = load_cache_mask(
        {
            "schema": CACHE_MASK_SCHEMA,
            "num_steps": 5,
            "anchor_mask": [True, True, False, False, False],
        }
    )
    resolved = resolve_cache_config(
        num_steps=5,
        mask=mask,
        predictor=LegacyResidualPredictor(),
        recovery=QualityRecoveryConfig(
            max_consecutive_predictions=1,
            require_final_anchor=True,
        ),
    )
    assert resolved.anchor_mask == (True, True, False, True, True)
    resolved.build_runner()


def test_resolved_session_binds_schedule_coordinates_and_clears_request_state():
    data = _plan_dict()
    data["predictor"]["coord"] = "timestep"
    plan = load_cache_plan(data)
    resolved = resolve_cache_plan(
        plan,
        model="flux",
        shape_label="1024x1024",
        num_steps=10,
        scheduler_class="FlowMatchEulerDiscreteScheduler",
    )
    session = ResolvedCacheSession(resolved)
    session.clear_request_state()
    session.bind_schedule_coordinates(
        [10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
    )
    for index in range(10):
        decision = session.decide_step(index)
        if decision.should_skip:
            session.estimate_output(index)
        else:
            session.record_anchor(index, torch.tensor([float(index)]))

    stats = session.statistics()
    assert stats["full_steps"] == stats["planned_anchor_steps"]
    assert stats["skipped_steps"] == stats["planned_skip_steps"]
    session.clear_request_state()
    assert session.statistics()["full_steps"] == 0


def test_cache_session_composes_policy_predictor_and_recovery():
    calibration = TeaCacheCalibration(
        model="flux",
        shape_label="1024x1024",
        num_steps=6,
        poly_coef=(0.0, 1.0),
        threshold=0.0,
        warmup_steps=0,
        cooldown_steps=0,
        cadence=1,
    )
    recovery = QualityRecoveryGuard(
        QualityRecoveryConfig(max_consecutive_predictions=1)
    )
    session = CacheSession(
        CacheRunner(
            TeaCachePolicy(calibration),
            LegacyResidualPredictor(),
            recovery=recovery,
        ),
        num_steps=6,
        configuration_source="teacache_cadence",
    )

    assert session.policy_requires_signal() is False
    session.bind_schedule_coordinates(
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0],
    )
    for index in range(6):
        decision = session.decide_step(index)
        if decision.should_skip:
            session.estimate_output(index)
        else:
            session.record_anchor(index, torch.tensor([float(index)]))

    stats = session.statistics()
    assert stats["source"] == "teacache_cadence"
    assert stats["full_steps"] == 4
    assert stats["skipped_steps"] == 2
    assert stats["readiness_rejections"] == 2
    assert stats["recovery_forced_steps"] == 2
    assert "planned_anchor_steps" not in stats
    assert stats["quality_recovery_pending_steps"] == 0


def test_teacache_adapter_translates_legacy_loop_calls_only():
    resolved = resolve_cache_config(
        num_steps=4,
        mask=(True, True, False, True),
        predictor=TaylorSeerPredictor(order=1),
        require_final_anchor=True,
    )
    session = ResolvedCacheSession(resolved)
    adapter = TeaCacheControllerAdapter(session)

    adapter.bind_schedule([4.0, 3.0, 2.0, 1.0])
    outputs = []
    for index in range(4):
        if adapter.should_skip(index):
            output = adapter.skip_noise_pred()
        else:
            output = torch.tensor([float(index * index)])
            adapter.record_full_step(output)
        outputs.append(float(output.item()))

    assert outputs == [0.0, 1.0, 2.0, 9.0]
    assert adapter.session is session
    assert adapter.stats()["full_steps"] == 3
    assert adapter.stats()["skipped_steps"] == 1


def test_cache_session_requires_explicit_matching_step_completion():
    resolved = resolve_cache_config(
        num_steps=2,
        mask=(True, True),
        predictor=TaylorSeerPredictor(order=1),
    )
    session = ResolvedCacheSession(resolved)

    with pytest.raises(ValueError, match="step_index must be an integer"):
        session.decide_step(True)

    decision = session.decide_step(0)
    assert decision.should_compute
    with pytest.raises(RuntimeError, match="not step 1"):
        session.record_anchor(1, torch.tensor([0.0]))
    with pytest.raises(RuntimeError, match="cannot change"):
        session.bind_schedule_coordinates([2.0, 1.0])

    session.record_anchor(0, torch.tensor([0.0]))
    assert session.active_step_index is None


def test_plan_roundtrip_dict_is_stable():
    original = _plan_dict()
    plan = load_cache_plan(deepcopy(original))
    assert plan.to_dict() == original


def test_plan_v1_keeps_strict_non_overlapping_protection_windows():
    data = _plan_dict(num_steps=5)
    data["policy"]["warmup_steps"] = 3
    data["policy"]["cooldown_steps"] = 2
    data["frozen_mask"] = [True] * 5
    with pytest.raises(CacheSpecError, match=r"warmup_steps \+ cooldown_steps"):
        resolve_cache_plan(
            load_cache_plan(data),
            model="flux",
            shape_label="1024x1024",
            num_steps=5,
            scheduler_class="FlowMatchEulerDiscreteScheduler",
        )
