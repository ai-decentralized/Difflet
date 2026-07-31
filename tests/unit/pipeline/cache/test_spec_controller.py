from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from difflet.pipeline.cache import (
    CACHE_MASK_SCHEMA,
    CACHE_PLAN_SCHEMA,
    CachePlanController,
    CacheSpecError,
    LegacyResidualPredictor,
    PeriodicAnchorPolicy,
    QualityRecoveryConfig,
    TaylorSeerPredictor,
    load_cache_mask,
    load_cache_plan,
    resolve_cache_config,
    resolve_cache_plan,
    validate_schedule_safety,
)


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


def test_plan_roundtrip_dict_is_stable():
    original = _plan_dict()
    plan = load_cache_plan(deepcopy(original))
    assert plan.to_dict() == original
