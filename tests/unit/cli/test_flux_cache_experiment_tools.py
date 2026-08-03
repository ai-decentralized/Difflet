from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from difflet.pipeline.cache import (
    InMemoryMeasurementSink,
    build_policy,
    load_cache_measurements,
    load_cache_plan,
    resolve_cache_plan,
)
from scripts.calibrate_flux_cache_plan import (
    NO_ACCEPTABLE_CANDIDATE,
    NoAcceptableCandidateError,
    calibrate,
)
from scripts.collect_flux_cache_ab import (
    FOREGROUND_ACK,
    CandidateArm,
    _build_baseline_adapter,
    _run_sample,
    build_candidate_arms,
    build_manifests,
    collect,
    load_candidate_ladder,
    select_candidate_arms,
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
        adapter = arm.build_pipeline_adapter(50)
        assert adapter.num_steps == 50
        assert adapter.stats()["planned_skip_steps"] > 0


def test_boundary_pilot_ladder_has_four_explicit_paired_arms():
    ladder_path = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "boundary-pilot-candidates.json"
    )

    ladder = load_candidate_ladder(ladder_path)
    arms = ladder.arms

    assert [
        (arm.warmup_steps, arm.anchor_interval, arm.order, arm.coord)
        for arm in arms
    ] == [
        (14, 4, 1, "index"),
        (10, 5, 1, "index"),
        (6, 8, 1, "index"),
        (4, 12, 1, "index"),
    ]
    assert len({arm.candidate_id for arm in arms}) == 4
    assert ladder.labels == (
        "conservative",
        "current-fast",
        "aggressive",
        "deliberate-boundary",
    )
    assert ladder.descriptor()["candidate_ids_in_order"] == [
        arm.candidate_id for arm in arms
    ]
    assert [
        arm.build_pipeline_adapter(50).stats()["planned_skip_steps"] for arm in arms
    ] == [26, 31, 37, 41]


def test_candidate_ladder_cli_selection_does_not_form_cartesian_product():
    ladder_path = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "boundary-pilot-candidates.json"
    )

    arms, descriptor = select_candidate_arms(
        SimpleNamespace(
            candidate_ladder=str(ladder_path),
            warmup_steps=None,
            anchor_intervals=None,
            orders=None,
            coord=None,
        )
    )

    assert len(arms) == 4
    assert descriptor["kind"] == "explicit_ladder"
    assert descriptor["content_sha256"] == (
        "814e04ca2de7d1855a9dc1134cbe01ff2e70e10dbbd09fd18878901b87c4f8f1"
    )


def test_candidate_ladder_rejects_sweep_overrides():
    with pytest.raises(ValueError, match="cannot be combined"):
        select_candidate_arms(
            SimpleNamespace(
                candidate_ladder="unused.json",
                warmup_steps=[14],
                anchor_intervals=None,
                orders=None,
                coord=None,
            )
        )


def test_candidate_ladder_rejects_rehashed_field_drift(tmp_path):
    source_path = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "boundary-pilot-candidates.json"
    )
    document = json.loads(source_path.read_text(encoding="utf-8"))
    document["candidates"][0]["warmup_steps"] = 13
    tampered_path = tmp_path / "candidate-ladder.json"
    tampered_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 does not match"):
        load_candidate_ladder(tampered_path)


def test_measured_50_step_schedule_matches_frozen_mask_without_vetoes():
    arm = CandidateArm(
        warmup_steps=14,
        anchor_interval=4,
        order=1,
        anchor_phase=1,
        cooldown_steps=1,
    )
    adapter = arm.build_pipeline_adapter(50)
    actual_anchors = []
    for step in range(50):
        should_skip = adapter.should_skip(step)
        actual_anchors.append(not should_skip)
        if should_skip:
            adapter.skip_noise_pred()
        else:
            adapter.record_full_step(torch.tensor([float(step)]))

    assert tuple(actual_anchors) == adapter.session.config.anchor_mask
    assert adapter.stats()["readiness_rejections"] == 0
    assert adapter.stats()["consecutive_skip_vetoes"] == 0
    assert adapter.stats()["full_steps"] == adapter.stats()["planned_anchor_steps"]
    assert adapter.stats()["skipped_steps"] == adapter.stats()["planned_skip_steps"]


def test_baseline_adapter_forces_every_step_through_the_same_runner_loop():
    adapter = _build_baseline_adapter(6)
    for step in range(6):
        assert adapter.should_skip(step) is False
        adapter.record_full_step(torch.tensor([float(step)]))

    assert adapter.stats()["full_steps"] == 6
    assert adapter.stats()["skipped_steps"] == 0
    assert adapter.stats()["planned_anchor_steps"] == 6


def test_collector_writes_and_references_one_measurement_report_per_sample(tmp_path):
    sink = InMemoryMeasurementSink()
    adapter = _build_baseline_adapter(3, measurement_sink=sink)
    flux_pipeline = SimpleNamespace(
        teacache_controller=adapter,
        _tc_last_trajectory=[],
    )

    def fake_pipe(**kwargs):
        del kwargs
        adapter.reset()
        flux_pipeline._tc_last_trajectory = []
        latent = torch.tensor([1.0])
        for step_index in range(3):
            assert adapter.should_skip(step_index) is False
            adapter.record_full_step(torch.tensor([float(step_index)]))
            previous = latent
            latent = latent + 1.0
            adapter.record_latent_update(step_index, previous, latent)
            flux_pipeline._tc_last_trajectory.append(latent.detach().cpu())
        return SimpleNamespace(images=[Image.new("RGB", (16, 16), color="white")])

    run = _run_sample(
        fake_pipe,
        flux_pipeline,
        sample={"sample_id": "p000-s0", "prompt": "test", "seed": 0},
        num_steps=3,
        height=16,
        width=16,
        guidance_scale=3.5,
        artifact_dir=tmp_path / "artifacts" / "baseline",
        output_root=tmp_path,
        measurement_sink=sink,
        configuration_source=adapter.source,
    )

    measurement_relative = run["artifacts"]["cache_measurements"]
    measurement_path = tmp_path / measurement_relative
    assert run["artifacts"]["cache_measurements_sha256"] == hashlib.sha256(
        measurement_path.read_bytes()
    ).hexdigest()
    report = load_cache_measurements(measurement_path)
    assert [record.step_index for record in report.latent_updates] == [0, 1, 2]
    assert all(not record.used_estimate for record in report.latent_updates)
    assert len(report.anchor_measurements) == 3


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


def _save_measurement_artifact(
    root: Path,
    *,
    label: str,
    sample_id: str,
    adapter_builder,
) -> dict[str, str]:
    sink = InMemoryMeasurementSink()
    adapter = adapter_builder(sink)
    for step_index in range(adapter.num_steps):
        if adapter.should_skip(step_index):
            adapter.skip_noise_pred()
        else:
            adapter.record_full_step(torch.tensor([float(step_index)]))
        before = torch.tensor([float(step_index + 1)])
        adapter.record_latent_update(step_index, before, before + 0.5)
    report = sink.build_report(
        num_steps=adapter.num_steps,
        configuration_source=adapter.source,
    )
    path = root / "artifacts" / label / f"{sample_id}.cache-measurements.json"
    report.write_json(path)
    return {
        "cache_measurements": path.relative_to(root).as_posix(),
        "cache_measurements_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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
        baseline_artifacts.update(
            _save_measurement_artifact(
                tmp_path,
                label="baseline",
                sample_id=sample["sample_id"],
                adapter_builder=lambda sink: _build_baseline_adapter(
                    6,
                    measurement_sink=sink,
                ),
            )
        )
        candidate_artifacts = _save_artifacts(
            tmp_path,
            label=arm.candidate_id,
            sample_id=sample["sample_id"],
            trajectory=(base + index) * (1.0 + 0.001 * (index + 1)),
            image_value=130 + index * 2,
        )
        candidate_artifacts.update(
            _save_measurement_artifact(
                tmp_path,
                label=arm.candidate_id,
                sample_id=sample["sample_id"],
                adapter_builder=lambda sink: arm.build_pipeline_adapter(
                    6,
                    measurement_sink=sink,
                ),
            )
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


def test_quality_evaluator_rejects_replaced_cache_measurements(tmp_path):
    quality_path, speedup_path, _ = _experiment_manifests(tmp_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    relative = quality["comparisons"][0]["candidate"]["cache_measurements"]
    measurement_path = tmp_path / relative
    measurement_path.write_text(
        measurement_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
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


def test_calibrator_freezes_recovery_overlay_when_cooldown_is_zero():
    candidate = _passing_candidate(
        "final-anchor-overlay",
        lpips=0.03,
        psnr=38.0,
        speedup=2.0,
    )
    candidate["policy"].update(
        {
            "anchor_interval": 3,
            "anchor_phase": 1,
            "warmup_steps": 3,
            "cooldown_steps": 0,
        }
    )
    curve = _quality_curve([candidate])
    curve["num_steps"] = 10

    plan, _ = calibrate(curve)

    bare_mask = build_policy(candidate["policy"]).materialize_anchor_mask(10)
    assert bare_mask[-1] is False
    assert plan.frozen_mask[-1] is True
    resolved = resolve_cache_plan(
        plan,
        model="flux",
        shape_label="1024x1024",
        num_steps=10,
        scheduler_class="FlowMatchEulerDiscreteScheduler",
    )
    assert resolved.anchor_mask == plan.frozen_mask
    assert resolved.anchor_mask[-1] is True


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
