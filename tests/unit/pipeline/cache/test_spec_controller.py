from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from difflet.pipeline.cache import (
    CACHE_MASK_SCHEMA,
    CACHE_PLAN_SCHEMA,
    CachePlanController,
    CacheRunner,
    CacheRuntimeController,
    CacheSpecError,
    LegacyResidualPredictor,
    PeriodicAnchorPolicy,
    QualityRecoveryConfig,
    QualityRecoveryGuard,
    TaylorSeerPredictor,
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


def test_controller_binds_scheduler_coordinates_and_resets_per_request():
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
    controller = CachePlanController(resolved)
    controller.reset()
    controller.bind_schedule(
        [10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
    )
    for index in range(10):
        if controller.should_skip(index):
            controller.skip_noise_pred()
        else:
            controller.record_full_step(torch.tensor([float(index)]))

    stats = controller.stats()
    assert stats["full_steps"] == stats["planned_anchor_steps"]
    assert stats["skipped_steps"] == stats["planned_skip_steps"]
    controller.reset()
    assert controller.stats()["full_steps"] == 0


def test_dynamic_controller_composes_policy_predictor_and_recovery():
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
    controller = CacheRuntimeController(
        CacheRunner(
            TeaCachePolicy(calibration),
            LegacyResidualPredictor(),
            recovery=recovery,
        ),
        num_steps=6,
        source="teacache_cadence",
    )

    assert controller.needs_signal() is False
    controller.bind_schedule(
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0],
    )
    for index in range(6):
        if controller.should_skip(index):
            controller.skip_noise_pred()
        else:
            controller.record_full_step(torch.tensor([float(index)]))

    stats = controller.stats()
    assert stats["source"] == "teacache_cadence"
    assert stats["full_steps"] == 4
    assert stats["skipped_steps"] == 2
    assert stats["readiness_rejections"] == 2
    assert stats["recovery_forced_steps"] == 2
    assert "planned_anchor_steps" not in stats
    assert stats["quality_recovery_pending_steps"] == 0


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
