from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.multires_quality_contract as multires
from scripts.flux_cache_protocol import (
    DEFAULT_PROMPT_SUITE_PATH,
    EXPERIMENT_PROTOCOL_SCHEMA,
    canonical_sha256,
    load_prompt_suite,
)


ROOT = Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _registration() -> dict:
    selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "metric_margin_calibration")
    prompt_binding = {
        "path": "benchmark/flux_cache/prompt-suite-v1.json",
        "file_sha256": _file_sha256(DEFAULT_PROMPT_SUITE_PATH),
        "split": "metric_margin_calibration",
        "split_sha256": selection.descriptor["sha256"],
        "prompt_count": 16,
        "seeds": [0, 1, 2],
        "sample_count": 48,
    }
    payload = {
        "schema": multires.PROTOCOL_SCHEMA,
        "schema_revision": 1,
        "study_id": "unit-test-multires",
        "created_at": "2026-08-04T00:00:00Z",
        "status": "registered-not-collected",
        "controlled_generation": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "model_revision": "b" * 40,
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "scheduler_config_sha256": canonical_sha256({"shift": 3.0}),
            "num_steps": 50,
            "guidance_scale": 3.5,
            "dtype": "bfloat16",
            "tp_degree": 4,
        },
        "resolution_buckets": [
            {
                "bucket_id": "square-1024",
                "height": 1024,
                "width": 1024,
                "prompt_suite": dict(prompt_binding),
            },
            {
                "bucket_id": "landscape-1344x768",
                "height": 768,
                "width": 1344,
                "prompt_suite": dict(prompt_binding),
            },
        ],
        "margin_calibration": {
            "method": multires.CALIBRATION_METHOD,
            "quantile": 0.95,
            "candidate_images_excluded_from_margin_estimation": True,
            "pooling_across_resolutions_forbidden": True,
        },
        "automatic_damage_rule": multires.AUTOMATIC_DAMAGE_RULE,
        "semantic_metrics": {"image_reward": {}, "vqa_score": {}},
        "source_registration": {
            "python_source_sha256": "a" * 64,
            "dirty_worktree_policy": "exact-python-source-sha256-required",
            "implementation_files": [
                {
                    "path": "scripts/multires_quality_contract.py",
                    "file_sha256": _file_sha256(ROOT / "scripts/multires_quality_contract.py"),
                }
            ],
        },
        "parameters_frozen_before_collection": True,
        "limitations": ["unit test"],
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _experiment(registration: dict, *, height: int, width: int) -> dict:
    bucket = registration["resolution_buckets"][0]
    selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "metric_margin_calibration")
    payload = {
        "schema": EXPERIMENT_PROTOCOL_SCHEMA,
        "source": {
            "git_commit": "a" * 40,
            "git_branch": "feature/cache-system",
            "git_dirty": True,
        },
        "model": {
            "model_id": registration["controlled_generation"]["model_id"],
            "requested_revision": registration["controlled_generation"]["model_revision"],
            "resolved_revision": registration["controlled_generation"]["model_revision"],
        },
        "compile": {
            "cache_key": "unit-test-key",
            "cache_inputs": {
                "model_id": registration["controlled_generation"]["model_id"],
                "revision": registration["controlled_generation"]["model_revision"],
                "dtype": "bfloat16",
                "shape": {"height": height, "width": width},
                "parallel": {"tp_degree": 4},
            },
            "manifest_schema_version": 4,
        },
        "runtime": {
            "python": "3.12.3",
            "platform": "Linux",
            "packages": {"torch": "2.9.1"},
        },
        "hardware": {"product_name": "trn2", "backend": "neuron", "tp_degree": 4},
        "generation": {
            "height": height,
            "width": width,
            "num_steps": 50,
            "guidance_scale": 3.5,
            "dtype": "bfloat16",
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "scheduler_config": {"shift": 3.0},
        },
        "rng": {
            "generator": "torch.Generator(cpu)",
            "seed_reset_per_sample": True,
            "seeds": bucket["prompt_suite"]["seeds"],
        },
        "cache_semantics": {
            "prediction_target": "transformer_noise_prediction",
            "anchor_history": "real-compute-only",
            "predictor_math": "newton-divided-differences",
            "coordinate": "index",
        },
        "timing": {
            "clock": "time.perf_counter",
            "boundary": "DiffletPipeline.__call__",
            "pipeline_warmup_enabled": True,
            "execution_order": "baseline-only",
            "sample_order": "prompt-major-seed-minor",
            "completion_barrier": "decoded-image-materialized-before-return",
            "includes": ["denoise-loop"],
            "excludes": ["artifact-save"],
        },
        "prompt_selection": selection.descriptor,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def test_protocol_is_digest_bound_and_keeps_resolution_buckets_separate(tmp_path):
    registration = _registration()
    path = tmp_path / "registration.json"
    _write_json(path, registration)

    loaded = multires.load_protocol(path)
    assert [row["bucket_id"] for row in loaded["resolution_buckets"]] == [
        "square-1024",
        "landscape-1344x768",
    ]
    tampered = json.loads(json.dumps(registration))
    tampered["resolution_buckets"][1]["width"] = 1312
    _write_json(path, tampered)
    with pytest.raises(ValueError, match="sha256 does not match"):
        multires.load_protocol(path)


def test_observed_generation_rejects_shape_drift():
    registration = _registration()
    bucket = registration["resolution_buckets"][0]
    multires.validate_observed_generation(
        _experiment(registration, height=1024, width=1024),
        registration,
        bucket,
    )

    with pytest.raises(ValueError, match="shape differs"):
        multires.validate_observed_generation(
            _experiment(registration, height=768, width=1344),
            registration,
            bucket,
        )


def test_calibration_computes_independent_nearest_rank_margins(tmp_path, monkeypatch):
    registration = _registration()
    registration_path = tmp_path / "registration.json"
    _write_json(registration_path, registration)
    reports: dict[str, Path] = {}
    evidence: dict[str, tuple[dict, Path, dict]] = {}
    metric_config = {
        "image_reward": {"model": "ImageReward-v1.0"},
        "vqa_score": {"model": "clip-flant5-xl"},
    }
    scales = {
        "square-1024": (0.1, 0.01),
        "landscape-1344x768": (0.4, 0.05),
    }
    selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "metric_margin_calibration")
    for bucket in registration["resolution_buckets"]:
        bucket_id = bucket["bucket_id"]
        report_path = tmp_path / f"{bucket_id}.report.json"
        manifest_path = tmp_path / f"{bucket_id}.manifest.json"
        _write_json(report_path, {})
        _write_json(manifest_path, {})
        image_scale, vqa_scale = scales[bucket_id]
        images = [
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "seed": seed,
                "scores": {
                    "image_reward": seed * image_scale,
                    "vqa_score": seed * vqa_scale,
                },
            }
            for prompt_index, prompt in enumerate(selection.prompts)
            for seed in (0, 1, 2)
        ]
        report = {"metrics": metric_config, "images": images}
        reports[bucket_id] = report_path
        evidence[bucket_id] = (report, manifest_path, {})

    monkeypatch.setattr(
        multires,
        "_load_report_evidence",
        lambda report_path, protocol, bucket: evidence[bucket["bucket_id"]],
    )
    contract = multires.calibrate_contract(registration_path, reports)
    by_id = {row["bucket_id"]: row for row in contract["resolution_contracts"]}

    assert by_id["square-1024"]["margins"] == pytest.approx(
        {"image_reward": 0.2, "vqa_score": 0.02}
    )
    assert by_id["landscape-1344x768"]["margins"] == pytest.approx(
        {"image_reward": 0.8, "vqa_score": 0.1}
    )
    assert all(
        summary["pair_count"] == 48
        for row in by_id.values()
        for summary in row["calibration_summary"].values()
    )
