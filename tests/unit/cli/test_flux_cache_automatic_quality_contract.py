from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.automatic_quality_contract import (
    CONTRACT_SCHEMA,
    SCHEMA_REVISION,
    calibrate_contract,
    clopper_pearson_upper,
    evaluate_contract,
    load_protocol,
    metric_identity,
)
from scripts.flux_cache_protocol import (
    DEFAULT_PROMPT_SUITE_PATH,
    canonical_sha256,
    load_prompt_suite,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "benchmark" / "flux_cache" / "automatic-quality-contract-protocol.json"


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_config() -> dict:
    return {
        "image_reward": {"model": "ImageReward-v1.0", "load_seconds": 1.0},
        "vqa_score": {"model": "clip-flant5-xl", "load_seconds": 2.0},
    }


def _semantic_report(
    manifest_path: Path,
    *,
    split: str,
    images: list[dict],
    comparisons: list[dict],
) -> dict:
    return {
        "schema": "difflet-cache-semantic-scores",
        "schema_revision": 1,
        "complete": True,
        "started_at": "2026-08-03T00:00:00Z",
        "completed_at": "2026-08-03T00:01:00Z",
        "sources": [{"split": split, "path": str(manifest_path), "sha256": _sha256(manifest_path)}],
        "metrics": _metric_config(),
        "runtime": {},
        "images": images,
        "comparisons": comparisons,
        "summary": [],
    }


def _experiment_manifest(selection: object, seeds: list[int]) -> dict:
    return {
        "candidates": [],
        "protocol": {
            "prompt_selection": selection.descriptor,
            "rng": {"seeds": seeds},
            "generation": {
                "num_steps": 50,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
                "dtype": "bfloat16",
                "scheduler_class": "FlowMatchEulerDiscreteScheduler",
                "scheduler_config": {
                    "_class_name": "FlowMatchEulerDiscreteScheduler",
                    "_diffusers_version": "0.30.0.dev0",
                    "_use_default_values": [
                        "invert_sigmas",
                        "shift_terminal",
                        "stochastic_sampling",
                        "time_shift_type",
                        "use_beta_sigmas",
                        "use_exponential_sigmas",
                        "use_karras_sigmas",
                    ],
                    "base_image_seq_len": 256,
                    "base_shift": 0.5,
                    "invert_sigmas": False,
                    "max_image_seq_len": 4096,
                    "max_shift": 1.15,
                    "num_train_timesteps": 1000,
                    "shift": 3.0,
                    "shift_terminal": None,
                    "stochastic_sampling": False,
                    "time_shift_type": "exponential",
                    "use_beta_sigmas": False,
                    "use_dynamic_shifting": True,
                    "use_exponential_sigmas": False,
                    "use_karras_sigmas": False,
                },
            },
            "model": {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "resolved_revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
            },
            "compile": {"cache_inputs": {"parallel": {"tp_degree": 4}}},
            "source": {"git_dirty": False},
        },
    }


def test_automatic_quality_protocol_and_prompt_splits_are_frozen_and_disjoint():
    protocol = load_protocol(PROTOCOL_PATH)
    margin = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "metric_margin_calibration")
    holdout = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "static_profile_holdout")
    previous = set()
    for split in ("legacy_parity", "calibration", "holdout", "boundary_pilot"):
        previous.update(load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, split).prompts)

    assert len(margin.prompts) == 16
    assert margin.descriptor["sha256"] == (
        "36ce6b7881991d8dc7a27fb5b9a6bc70e66e210fb6cbcf94882359bf8db1c3fa"
    )
    assert len(holdout.prompts) == 32
    assert holdout.descriptor["sha256"] == (
        "2c52ee4b0f5ff469c40db4b1b89ea259f6d5da66dcbeee2ebdd8d09a9655239a"
    )
    assert previous.isdisjoint(margin.prompts)
    assert previous.isdisjoint(holdout.prompts)
    assert set(margin.prompts).isdisjoint(holdout.prompts)
    assert protocol["margin_calibration"]["split_sha256"] == margin.descriptor["sha256"]
    assert protocol["holdout"]["split_sha256"] == holdout.descriptor["sha256"]


def test_contract_uses_only_baseline_seed_variation_and_evaluates_strict_margins(tmp_path):
    selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "metric_margin_calibration")
    manifest_path = tmp_path / "margin-quality.json"
    _write_json(
        manifest_path,
        _experiment_manifest(selection, [0, 1, 2]),
    )
    images = []
    for prompt_index, prompt in enumerate(selection.prompts):
        for seed in (0, 1, 2):
            images.append(
                {
                    "role": "baseline",
                    "candidate_id": None,
                    "split": "metric_margin_calibration",
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "seed": seed,
                    "scores": {
                        "image_reward": seed * 0.1,
                        "vqa_score": seed * 0.01,
                    },
                }
            )
    report_path = tmp_path / "margin-semantic.json"
    _write_json(
        report_path,
        _semantic_report(
            manifest_path,
            split="metric_margin_calibration",
            images=images,
            comparisons=[],
        ),
    )

    contract = calibrate_contract(PROTOCOL_PATH, report_path)

    assert contract["margin_calibration"]["candidate_images_used"] is False
    assert contract["margin_calibration"]["summaries"]["image_reward"]["pair_count"] == 48
    assert contract["margins"]["image_reward"] == pytest.approx(0.2)
    assert contract["margins"]["vqa_score"] == pytest.approx(0.02)
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)

    boundary_selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "boundary_pilot")
    evaluation_manifest_path = tmp_path / "boundary-quality.json"
    _write_json(
        evaluation_manifest_path,
        _experiment_manifest(boundary_selection, [0]),
    )
    comparisons = [
        {
            "candidate_id": "candidate",
            "sample_id": "pass",
            "prompt_index": 0,
            "seed": 0,
            "prompt": boundary_selection.prompts[0],
            "candidate_minus_baseline": {"image_reward": -0.2, "vqa_score": 0.0},
        },
        {
            "candidate_id": "candidate",
            "sample_id": "fail",
            "prompt_index": 1,
            "seed": 0,
            "prompt": boundary_selection.prompts[1],
            "candidate_minus_baseline": {"image_reward": -0.201, "vqa_score": 0.0},
        },
    ]
    evaluation_report_path = tmp_path / "boundary-semantic.json"
    _write_json(
        evaluation_report_path,
        _semantic_report(
            evaluation_manifest_path,
            split="boundary_pilot",
            images=[],
            comparisons=comparisons,
        ),
    )

    evaluation = evaluate_contract(contract_path, evaluation_report_path)

    assert [row["passes"] for row in evaluation["rows"]] == [True, False]
    assert evaluation["candidate_summaries"][0]["failure_count"] == 1
    assert evaluation["registered_holdout"] is False


def test_zero_of_32_has_below_ten_percent_one_sided_upper_bound():
    upper = clopper_pearson_upper(0, 32, 0.95)

    assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / 32.0))
    assert upper < 0.1
    assert clopper_pearson_upper(1, 32, 0.95) > 0.1


def test_metric_identity_ignores_transient_cache_bookkeeping():
    left = _metric_config()
    right = _metric_config()
    left["image_reward"]["checkpoint_files"] = [
        {"path": "/cache/model.bin", "bytes": 10, "sha256": "a" * 64},
        {"path": "/cache/model.bin.metadata", "bytes": 20, "sha256": "b" * 64},
    ]
    right["image_reward"]["checkpoint_files"] = [
        {"path": "/cache/model.bin", "bytes": 10, "sha256": "a" * 64},
        {"path": "/cache/model.bin.metadata", "bytes": 21, "sha256": "c" * 64},
    ]

    assert metric_identity(left)["sha256"] == metric_identity(right)["sha256"]


def test_registered_holdout_can_examine_static_and_brake_candidates_together(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    selection = load_prompt_suite(DEFAULT_PROMPT_SUITE_PATH, "static_profile_holdout")
    manifest = _experiment_manifest(selection, [0])
    static = protocol["static_candidate"]
    static_id = static["candidate_id"]
    brake_id = "adaptive-brake"
    manifest["candidates"] = [
        {
            "candidate_id": static_id,
            "policy": {
                "type": "periodic_anchor",
                "warmup_steps": 6,
                "anchor_interval": 8,
                "anchor_phase": 1,
                "cooldown_steps": 1,
                "require_final_anchor": True,
            },
            "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
        },
        {
            "candidate_id": brake_id,
            "policy": {"type": "adaptive_anchor"},
            "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
        },
    ]
    manifest_path = tmp_path / "holdout-quality.json"
    _write_json(manifest_path, manifest)
    comparisons = []
    for candidate_id in (static_id, brake_id):
        for prompt_index, prompt in enumerate(selection.prompts):
            comparisons.append(
                {
                    "candidate_id": candidate_id,
                    "sample_id": f"p{prompt_index:03d}-s0",
                    "prompt_index": prompt_index,
                    "seed": 0,
                    "prompt": prompt,
                    "candidate_minus_baseline": {
                        "image_reward": 0.0,
                        "vqa_score": 0.0,
                    },
                }
            )
    report_path = tmp_path / "holdout-semantic.json"
    _write_json(
        report_path,
        _semantic_report(
            manifest_path,
            split="static_profile_holdout",
            images=[],
            comparisons=comparisons,
        ),
    )
    payload = {
        "schema": CONTRACT_SCHEMA,
        "schema_revision": SCHEMA_REVISION,
        "controlled_generation": protocol["controlled_generation"],
        "static_candidate": static,
        "metric_identity": metric_identity(_metric_config()),
        "margins": {"image_reward": 0.1, "vqa_score": 0.1},
        "holdout": protocol["holdout"],
    }
    contract = {**payload, "sha256": canonical_sha256(payload)}
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)

    evaluation = evaluate_contract(contract_path, report_path)

    assert evaluation["registered_holdout"] is True
    assert {row["candidate_id"] for row in evaluation["candidate_summaries"]} == {
        static_id,
        brake_id,
    }
    assert all(row["passes_statistical_gate"] for row in evaluation["candidate_summaries"])


def test_protocol_digest_rejects_rehashed_field_drift(tmp_path):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["margin_calibration"]["quantile"] = 0.9
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    protocol["sha256"] = canonical_sha256(payload)
    tampered = tmp_path / "protocol.json"
    _write_json(tampered, protocol)

    loaded = load_protocol(tampered)
    assert loaded["margin_calibration"]["quantile"] == 0.9

    loaded["sha256"] = "0" * 64
    _write_json(tampered, loaded)
    with pytest.raises(ValueError, match="sha256 does not match"):
        load_protocol(tampered)
