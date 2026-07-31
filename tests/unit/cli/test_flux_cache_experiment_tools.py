from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from difflet.pipeline.cache import load_cache_plan, resolve_cache_plan
from scripts.calibrate_flux_cache_plan import (
    NO_ACCEPTABLE_CANDIDATE,
    NoAcceptableCandidateError,
    calibrate,
)
from scripts.collect_flux_cache_ab import (
    FOREGROUND_ACK,
    CandidateArm,
    _build_baseline_controller,
    build_candidate_arms,
    build_manifests,
    collect,
)
from scripts.evaluate_cache_quality import (
    QUALITY_CURVE_SCHEMA,
    _validate_lpips_images,
    default_thresholds,
    evaluate,
    metric_config,
    psnr_db,
    ssim,
    tensor_cosine,
    validate_metric_config,
)


def test_default_flux_sweep_has_twelve_unique_safe_arms():
    arms = build_candidate_arms()

    assert len(arms) == 12
    assert len({arm.candidate_id for arm in arms}) == 12
    assert {(arm.warmup_steps, arm.anchor_interval, arm.order) for arm in arms} == {
        (warmup, interval, order)
        for warmup in (10, 12, 14)
        for interval in (4, 5)
        for order in (1, 2)
    }
    for arm in arms:
        controller = arm.build_controller(50)
        assert controller.num_steps == 50
        assert controller.stats()["planned_skip_steps"] > 0


def test_measured_50_step_schedule_matches_frozen_mask_without_vetoes():
    arm = CandidateArm(
        warmup_steps=14,
        anchor_interval=4,
        order=1,
        anchor_phase=1,
        cooldown_steps=1,
    )
    controller = arm.build_controller(50)
    actual_anchors = []
    for step in range(50):
        should_skip = controller.should_skip(step)
        actual_anchors.append(not should_skip)
        if should_skip:
            controller.skip_noise_pred()
        else:
            controller.record_full_step(torch.tensor([float(step)]))

    assert tuple(actual_anchors) == controller.config.anchor_mask
    assert controller.stats()["readiness_rejections"] == 0
    assert controller.stats()["consecutive_skip_vetoes"] == 0
    assert controller.stats()["full_steps"] == controller.stats()["planned_anchor_steps"]
    assert controller.stats()["skipped_steps"] == controller.stats()["planned_skip_steps"]


def test_baseline_controller_forces_every_step_through_the_same_runner_loop():
    controller = _build_baseline_controller(6)
    for step in range(6):
        assert controller.should_skip(step) is False
        controller.record_full_step(torch.tensor([float(step)]))

    assert controller.stats()["full_steps"] == 6
    assert controller.stats()["skipped_steps"] == 0
    assert controller.stats()["planned_anchor_steps"] == 6


def test_collector_requires_explicit_hardware_ack():
    with pytest.raises(RuntimeError, match="--allow-hardware"):
        collect(
            SimpleNamespace(
                allow_hardware=False,
                foreground_ack=FOREGROUND_ACK,
            )
        )


def _save_artifacts(
    root: Path,
    *,
    label: str,
    sample_id: str,
    trajectory: torch.Tensor,
    image_value: int,
) -> dict[str, str]:
    directory = root / "artifacts" / label
    directory.mkdir(parents=True, exist_ok=True)
    trajectory_path = directory / f"{sample_id}.trajectory.pt"
    final_path = directory / f"{sample_id}.final-latent.pt"
    image_path = directory / f"{sample_id}.png"
    torch.save(trajectory, trajectory_path)
    torch.save(trajectory[-1], final_path)
    Image.new("RGB", (16, 16), color=(image_value,) * 3).save(image_path)
    return {
        "trajectory": trajectory_path.relative_to(root).as_posix(),
        "final_latent": final_path.relative_to(root).as_posix(),
        "image": image_path.relative_to(root).as_posix(),
    }


def _experiment_manifests(tmp_path: Path):
    samples = (
        {
            "sample_id": "p000-s0",
            "prompt_index": 0,
            "prompt": "first",
            "seed": 0,
        },
        {
            "sample_id": "p000-s1",
            "prompt_index": 0,
            "prompt": "first",
            "seed": 1,
        },
        {
            "sample_id": "p001-s0",
            "prompt_index": 1,
            "prompt": "second",
            "seed": 0,
        },
        {
            "sample_id": "p001-s1",
            "prompt_index": 1,
            "prompt": "second",
            "seed": 1,
        },
    )
    arm = CandidateArm(
        warmup_steps=2,
        anchor_interval=2,
        order=1,
        anchor_phase=0,
        cooldown_steps=1,
    )
    baseline_runs = []
    candidate_rows = []
    base = torch.arange(1, 25, dtype=torch.float32).reshape(6, 1, 4)
    for index, sample in enumerate(samples):
        baseline_artifacts = _save_artifacts(
            tmp_path,
            label="baseline",
            sample_id=sample["sample_id"],
            trajectory=base + index,
            image_value=128,
        )
        candidate_artifacts = _save_artifacts(
            tmp_path,
            label=arm.candidate_id,
            sample_id=sample["sample_id"],
            trajectory=(base + index) * (1.0 + 0.001 * (index + 1)),
            image_value=130 + index * 2,
        )
        baseline_runs.append(
            {
                "sample_id": sample["sample_id"],
                "elapsed_s": 2.0,
                "artifacts": baseline_artifacts,
            }
        )
        candidate_rows.append(
            {
                "sample_id": sample["sample_id"],
                "elapsed_s": 1.0,
                "artifacts": candidate_artifacts,
                "runner_stats": {
                    "full_steps": 5,
                    "skipped_steps": 1,
                    "consecutive_skip_vetoes": 0,
                },
            }
        )
    identity = {
        "model": "flux",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "shape_label": "16x16",
        "num_steps": 6,
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "guidance_scale": 3.5,
        "prompt_count": 2,
        "seed_count": 2,
        "sample_count": 4,
    }
    quality, speedup = build_manifests(
        identity=identity,
        samples=samples,
        arms=(arm,),
        baseline_runs=baseline_runs,
        candidate_runs={arm.candidate_id: candidate_rows},
        started_at="2026-07-31T00:00:00Z",
        completed_at="2026-07-31T00:01:00Z",
    )
    quality_path = tmp_path / "quality-input-v2.json"
    speedup_path = tmp_path / "speedup-candidates-v1.json"
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    speedup_path.write_text(json.dumps(speedup), encoding="utf-8")
    return quality_path, speedup_path, arm


def test_quality_evaluator_computes_per_sample_and_worst_metrics(tmp_path):
    quality_path, speedup_path, arm = _experiment_manifests(tmp_path)

    result = evaluate(
        quality_path,
        speedup_path,
        thresholds=default_thresholds(
            min_speedup=1.5,
            min_trajectory_cosine=0.99,
            min_final_latent_cosine=0.99,
            min_psnr_db=30.0,
            max_lpips=0.1,
        ),
        lpips_function=lambda left, right: float((left.float() - right.float()).abs().mean()),
    )

    assert result["schema"] == QUALITY_CURVE_SCHEMA
    assert result["passing_candidate_ids"] == [arm.candidate_id]
    candidate = result["candidates"][0]
    assert candidate["hardware_measured"] is True
    assert candidate["measured_speedup"] == pytest.approx(2.0)
    assert candidate["passes_gate"] is True
    assert candidate["failed_gates"] == []
    assert len(candidate["per_sample"]) == 4
    assert candidate["worst_sample"]["lpips"] == pytest.approx(8 / 255)
    assert candidate["worst_sample"]["psnr_db"] < candidate["per_sample"][0]["psnr_db"]
    assert candidate["runner_stats"]["consecutive_skip_vetoes"] == 0


def test_offline_artifacts_close_into_a_production_cache_plan(tmp_path):
    quality_path, speedup_path, arm = _experiment_manifests(tmp_path)
    curve = evaluate(
        quality_path,
        speedup_path,
        thresholds=default_thresholds(
            min_speedup=1.5,
            min_trajectory_cosine=0.99,
            min_final_latent_cosine=0.99,
            min_psnr_db=30.0,
            max_lpips=0.1,
        ),
        lpips_function=lambda left, right: float((left.float() - right.float()).abs().mean()),
    )

    plan, selected = calibrate(curve)

    assert selected["candidate_id"] == arm.candidate_id
    assert plan.compatibility.num_steps == 6
    assert plan.policy == arm.policy_spec()
    assert plan.predictor == arm.predictor_spec()
    assert len(plan.frozen_mask) == 6


def test_quality_evaluator_rejects_identity_mismatch(tmp_path):
    quality_path, speedup_path, _ = _experiment_manifests(tmp_path)
    speedup = json.loads(speedup_path.read_text(encoding="utf-8"))
    speedup["num_steps"] = 4
    speedup_path.write_text(json.dumps(speedup), encoding="utf-8")

    with pytest.raises(ValueError, match="identities do not match"):
        evaluate(
            quality_path,
            speedup_path,
            thresholds=default_thresholds(),
            lpips_function=lambda left, right: 0.0,
        )


def test_quality_evaluator_rejects_unreproducible_speedup(tmp_path):
    quality_path, speedup_path, _ = _experiment_manifests(tmp_path)
    speedup = json.loads(speedup_path.read_text(encoding="utf-8"))
    speedup["candidates"][0]["measured_speedup"] = 9.0
    speedup_path.write_text(json.dumps(speedup), encoding="utf-8")

    with pytest.raises(ValueError, match="measured_speedup disagrees"):
        evaluate(
            quality_path,
            speedup_path,
            thresholds=default_thresholds(),
            lpips_function=lambda left, right: 0.0,
        )


def test_metric_primitives_have_expected_identity_behavior():
    image = torch.full((3, 16, 16), 0.5)
    tensor = torch.arange(8, dtype=torch.float32)

    assert tensor_cosine(tensor, tensor) == pytest.approx(1.0)
    assert psnr_db(image, image) == pytest.approx(120.0)
    assert ssim(image, image) == pytest.approx(1.0)


def test_protocol_metric_config_requires_exact_lpips_model_state():
    base = metric_config("alex")
    with pytest.raises(ValueError, match="model-state provenance"):
        validate_metric_config(base, require_lpips_provenance=True)

    provenance = {
        "implementation": "lpips.LPIPS",
        "package_version": "0.1.4",
        "calibration_version": "0.1",
        "net": "alex",
        "model_state_sha256": "a" * 64,
    }
    exact = metric_config("alex", provenance)

    assert validate_metric_config(exact, require_lpips_provenance=True) == exact


def test_lpips_input_validation_rejects_tiny_or_mismatched_images():
    with pytest.raises(ValueError, match="at least 64x64"):
        _validate_lpips_images(torch.zeros(3, 16, 16), torch.zeros(3, 16, 16))
    with pytest.raises(ValueError, match="identically shaped"):
        _validate_lpips_images(torch.zeros(3, 64, 64), torch.zeros(3, 64, 65))
    _validate_lpips_images(torch.zeros(3, 64, 64), torch.zeros(3, 64, 64))


def _quality_curve(candidates):
    return {
        "schema": QUALITY_CURVE_SCHEMA,
        "model": "flux",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "shape_label": "1024x1024",
        "num_steps": 50,
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "guidance_scale": 3.5,
        "prompt_count": 2,
        "seed_count": 2,
        "sample_count": 4,
        "hardware_measured": True,
        "aggregation": "worst-sample",
        "metric_config": metric_config("alex"),
        "thresholds": {
            "min_speedup": 1.5,
            "min_trajectory_cosine": 0.9999,
            "min_final_latent_cosine": 0.9995,
            "min_psnr_db": 30.0,
            "max_lpips": 0.1,
        },
        "candidates": candidates,
        "passing_candidate_ids": [
            candidate["candidate_id"] for candidate in candidates if candidate["passes_gate"]
        ],
    }


def _passing_candidate(
    candidate_id: str,
    *,
    lpips: float,
    psnr: float,
    speedup: float,
    vetoes: int = 0,
):
    sample_metrics = {
        "trajectory_cosine": 0.99995,
        "final_latent_cosine": 0.9997,
        "psnr_db": psnr,
        "ssim": 0.99,
        "lpips": lpips,
    }
    return {
        "candidate_id": candidate_id,
        "policy": {
            "type": "periodic_anchor",
            "anchor_interval": 4,
            "anchor_phase": 1,
            "warmup_steps": 14,
            "cooldown_steps": 1,
            "require_final_anchor": True,
        },
        "predictor": {
            "type": "taylorseer",
            "order": 1,
            "coord": "index",
        },
        "hardware_measured": True,
        "measured_speedup": speedup,
        "runner_stats": {"consecutive_skip_vetoes": vetoes},
        "per_sample": [
            {
                "sample_id": f"p{index // 2:03d}-s{index % 2}",
                "prompt_index": index // 2,
                "seed": index % 2,
                **sample_metrics,
            }
            for index in range(4)
        ],
        "worst_sample": sample_metrics,
        "passes_gate": True,
        "failed_gates": [],
    }


def test_calibrator_ranks_quality_first_and_emits_resolvable_plan():
    faster = _passing_candidate(
        "faster",
        lpips=0.04,
        psnr=38.0,
        speedup=2.2,
    )
    better_lpips = _passing_candidate(
        "better-lpips",
        lpips=0.03,
        psnr=34.0,
        speedup=1.6,
    )

    plan, selected = calibrate(_quality_curve([faster, better_lpips]))

    assert selected["candidate_id"] == "better-lpips"
    roundtripped = load_cache_plan(plan.to_dict())
    resolved = resolve_cache_plan(
        roundtripped,
        model="flux",
        shape_label="1024x1024",
        num_steps=50,
        scheduler_class="FlowMatchEulerDiscreteScheduler",
    )
    assert resolved.plan is not None
    assert resolved.planned_skip_steps > 0


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (
            {"lpips": 0.03, "psnr": 36.0, "speedup": 2.2},
            {"lpips": 0.03, "psnr": 38.0, "speedup": 1.6},
            "right",
        ),
        (
            {"lpips": 0.03, "psnr": 38.0, "speedup": 1.6},
            {"lpips": 0.03, "psnr": 38.0, "speedup": 2.2},
            "right",
        ),
    ],
)
def test_calibrator_uses_psnr_then_speedup_as_tiebreakers(left, right, expected):
    candidates = [
        _passing_candidate("left", **left),
        _passing_candidate("right", **right),
    ]

    _, selected = calibrate(_quality_curve(candidates))

    assert selected["candidate_id"] == expected


def test_calibrator_rejects_vetoed_candidates_without_relaxing_gates():
    candidate = _passing_candidate(
        "vetoed",
        lpips=0.01,
        psnr=40.0,
        speedup=2.5,
        vetoes=1,
    )

    with pytest.raises(NoAcceptableCandidateError, match=NO_ACCEPTABLE_CANDIDATE):
        calibrate(_quality_curve([candidate]))


def test_calibrator_detects_tampered_gate_boolean():
    candidate = _passing_candidate(
        "tampered",
        lpips=0.2,
        psnr=40.0,
        speedup=2.0,
    )
    candidate["passes_gate"] = True
    candidate["failed_gates"] = []

    with pytest.raises(ValueError, match="passes_gate disagrees"):
        calibrate(_quality_curve([candidate]))
