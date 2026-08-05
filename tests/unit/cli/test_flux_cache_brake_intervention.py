from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from difflet.pipeline.cache import AdaptiveAnchorConfig
from scripts.flux_cache_brake_intervention import (
    BrakeInterventionPolicy,
    TargetAnchorFingerprintHook,
    canonical_sha256,
    evaluate_pilot_decision,
    select_target_anchor,
)


class _Measurement:
    def __init__(self, *, step: int, error: float, num_steps: int = 50) -> None:
        self.step_index = step
        self.num_steps = num_steps
        self.estimate_status = "measured"
        self.numerically_valid = True
        self.estimate_relative_error = error
        self.relative_output_change = 0.2
        self.relative_output_curvature = 0.3
        self.anchor_step_gap = 8

    def to_dict(self):
        return {
            "step_index": self.step_index,
            "num_steps": self.num_steps,
            "estimate_status": self.estimate_status,
            "numerically_valid": self.numerically_valid,
            "estimate_relative_error": self.estimate_relative_error,
            "relative_output_change": self.relative_output_change,
            "relative_output_curvature": self.relative_output_curvature,
            "anchor_step_gap": self.anchor_step_gap,
        }


def _config() -> AdaptiveAnchorConfig:
    return AdaptiveAnchorConfig(
        initial_anchor_interval=8,
        minimum_anchor_interval=4,
        maximum_anchor_interval=12,
        warmup_steps=6,
        cooldown_steps=1,
        anchor_phase=1,
        tighten_error=1.4,
        recovery_error=1.75,
        acceleration_error=0.7,
        recovery_steps=2,
        disable_after_recoveries=2,
        stable_anchors_for_acceleration=2,
        acceleration_start_progress=0.3,
        allow_acceleration=True,
        require_final_anchor=True,
    )


@pytest.mark.parametrize(
    ("action", "state", "interval", "next_anchor", "recovery_remaining"),
    (
        ("continue", "active", 8, 23, 0),
        ("brake", "active", 4, 19, 0),
        ("recovery", "recovery", 4, None, 2),
        ("terminal", "disabled", 4, None, 0),
    ),
)
def test_intervention_overrides_only_the_target_anchor_action(
    action, state, interval, next_anchor, recovery_remaining
):
    policy = BrakeInterventionPolicy(_config(), action=action, target_step=15)

    policy.observe_anchor_measurement(_Measurement(step=15, error=2.0))
    policy.validate_complete()

    assert policy.state == state
    assert policy.current_anchor_interval == interval
    assert policy.target_event["post_next_anchor_step"] == next_anchor
    assert policy.target_event["post_recovery_steps_remaining"] == recovery_remaining
    assert policy.target_event["estimate_relative_error"] == 2.0


def test_non_target_anchor_keeps_the_production_policy_semantics():
    policy = BrakeInterventionPolicy(_config(), action="continue", target_step=23)

    policy.observe_anchor_measurement(_Measurement(step=15, error=2.0))

    assert policy.state == "recovery"
    assert policy.current_anchor_interval == 4
    assert policy.target_event is None
    with pytest.raises(RuntimeError, match="never applied"):
        policy.validate_complete()


def test_target_selector_uses_only_active_valid_anchors_and_breaks_ties_early():
    events = [
        {
            "step_index": 7,
            "pre_state": "active",
            "estimate_status": "measured",
            "numerically_valid": True,
            "estimate_relative_error": 0.5,
        },
        {
            "step_index": 15,
            "pre_state": "recovery",
            "estimate_status": "measured",
            "numerically_valid": True,
            "estimate_relative_error": 0.6,
        },
        {
            "step_index": 23,
            "pre_state": "active",
            "estimate_status": "measured",
            "numerically_valid": True,
            "estimate_relative_error": 0.7,
        },
        {
            "step_index": 31,
            "pre_state": "active",
            "estimate_status": "measured",
            "numerically_valid": True,
            "estimate_relative_error": 0.8,
        },
        {
            "step_index": 47,
            "pre_state": "active",
            "estimate_status": "measured",
            "numerically_valid": True,
            "estimate_relative_error": 0.9,
        },
    ]

    selected = select_target_anchor(
        events,
        target_progress=27 / 49,
        num_steps=50,
        minimum_remaining_steps=6,
    )

    # 23 and 31 are equally distant; the deterministic tie-break is earlier.
    assert selected["step_index"] == 23


def test_target_fingerprint_requires_a_real_anchor_and_is_repeatable():
    latents = torch.arange(8, dtype=torch.bfloat16).reshape(1, 4, 2)
    output = torch.ones(1, 4, 2, dtype=torch.bfloat16)
    hook = TargetAnchorFingerprintHook(3)

    returned = hook(
        step_index=3,
        timestep=torch.tensor(0.5),
        latents=latents,
        predicted=output,
        used_cache_prediction=False,
        compute_actual=lambda: output,
    )

    assert returned is output
    assert hook.validate_complete()["step_index"] == 3
    with pytest.raises(RuntimeError, match="more than once"):
        hook(
            step_index=3,
            timestep=torch.tensor(0.5),
            latents=latents,
            predicted=output,
            used_cache_prediction=False,
            compute_actual=lambda: output,
        )


def test_registered_brake_intervention_protocol_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    protocol = json.loads((root / "benchmark/flux_cache/brake-intervention-pilot.json").read_text())
    digest = protocol.pop("sha256")

    assert digest == canonical_sha256(protocol)


def test_registered_brake_intervention_result_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    result = json.loads(
        (root / "benchmark/flux_cache/brake-intervention-pilot-result.json").read_text()
    )
    digest = result.pop("sha256")

    assert digest == canonical_sha256(result)


def test_pilot_decision_is_negative_when_no_known_failure_is_rescued():
    protocol = {
        "quality_contract": {
            "image_reward_max_harm": 0.78,
            "vqa_score_max_harm": 0.25,
        },
        "decision_rule": {
            "positive_pilot": "positive",
            "negative_pilot": "negative",
            "next_if_positive": "confirm",
            "next_if_negative": "change predictor",
        },
    }
    rows = [
        {
            "sample_id": "failure",
            "phase": "early",
            "action": "brake",
            "prior_label": "target-candidate-vqa-failure",
            "quality": {
                "outcome": "both_fail",
                "continue_harm": {"image_reward": 0.1, "vqa_score": 0.4},
            },
        },
        {
            "sample_id": "control",
            "phase": "early",
            "action": "brake",
            "prior_label": "target-candidate-pass-control",
            "quality": {
                "outcome": "both_pass",
                "continue_harm": {"image_reward": 0.0, "vqa_score": 0.0},
            },
        },
    ]

    decision = evaluate_pilot_decision(
        protocol=protocol,
        rows=rows,
        correlations={"brake": {"anchor_error": {"vqa_score": 0.9}}},
    )

    assert decision["status"] == "negative_pilot"
    assert decision["known_failure_rescued_target_count"] == 0
    assert decision["introduced_control_failure_target_count"] == 0
