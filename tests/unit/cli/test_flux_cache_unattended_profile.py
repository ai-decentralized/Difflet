from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.collect_flux_cache_authorized as authorized_collector
import scripts.flux_cache_natural_range_gate as natural_gate
from scripts.flux_cache_execution_policy import (
    POLICY_SCHEMA,
    POLICY_SCHEMA_REVISION,
    ExecutionRequest,
    authorize_execution,
    load_execution_policy,
    write_authorization_record,
)
from scripts.flux_cache_protocol import canonical_sha256

MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _write_candidate(root: Path) -> Path:
    horizon = root / "horizon.json"
    contract = root / "contract.json"
    _write_json(horizon, {})
    _write_json(contract, {})
    payload = {
        "schema": "difflet-flux-cache-phased-candidate",
        "schema_revision": 1,
        "candidate_id": "unit-static-plus-brake",
        "policy": {
            "type": "phased_static_plus_brake",
            "num_steps": 50,
            "static_anchor_steps": [
                0,
                1,
                2,
                3,
                4,
                5,
                9,
                13,
                17,
                21,
                25,
                29,
                33,
                37,
                41,
                45,
                49,
            ],
            "warmup_steps": 6,
            "cooldown_steps": 1,
            "require_final_anchor": True,
            "dynamic_budget": 2,
            "invalid_measurement_fail_closed": True,
            "plastic_window": [6, 37],
            "tighten_error": 1.19,
            "recovery_error": 1.5,
            "recovery_steps": 2,
            "disable_after_recoveries": 2,
            "tighten_rule": "bisect_next_static_gap",
            "allow_acceleration": False,
        },
        "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
        "horizon_ref": {
            "path": str(horizon),
            "sha256": hashlib.sha256(horizon.read_bytes()).hexdigest(),
        },
        "quality_contract_ref": {
            "path": str(contract),
            "sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        },
    }
    candidate = {**payload, "sha256": canonical_sha256(payload)}
    path = root / "candidate.json"
    _write_json(path, candidate)
    return path


def _execution_policy(output_root: Path) -> dict:
    payload = {
        "schema": POLICY_SCHEMA,
        "schema_revision": POLICY_SCHEMA_REVISION,
        "policy_id": "unit-test-profile-run",
        "created_at": "2026-08-06T00:00:00Z",
        "status": "authorized",
        "authorization": {
            "mode": "one_time_scoped",
            "substage_reauthorization_required": False,
        },
        "scope": {
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            },
            "hardware": {
                "backend": "trainium",
                "product_name": "trn2.3xlarge",
                "tp_degree": 4,
            },
            "generation": {
                "num_steps": 50,
                "dtype": "bfloat16",
                "guidance_scale": 3.5,
                "resolutions": [{"height": 1024, "width": 1024}],
            },
            "allowed_stages": ["candidate_screen", "confirmation"],
            "stage_request_limits": {
                "candidate_screen": 160,
                "confirmation": 64,
            },
            "output_root": str(output_root.resolve()),
        },
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _execution_request(output_root: Path, **overrides) -> ExecutionRequest:
    values = {
        "stage": "candidate_screen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "trainium",
        "product_name": "trn2.3xlarge",
        "tp_degree": 4,
        "num_steps": 50,
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "dtype": "bfloat16",
        "request_count": 160,
        "output_directory": output_root / "candidate-screen",
    }
    values.update(overrides)
    return ExecutionRequest(**values)


def test_one_time_policy_authorizes_multiple_in_scope_stages_without_reack(tmp_path):
    output_root = tmp_path / "artifacts"
    policy_path = tmp_path / "execution-policy.json"
    _write_json(policy_path, _execution_policy(output_root))

    screen = authorize_execution(policy_path, _execution_request(output_root))
    confirmation = authorize_execution(
        policy_path,
        _execution_request(
            output_root,
            stage="confirmation",
            request_count=64,
            output_directory=output_root / "confirmation",
        ),
    )

    assert screen["substage_reauthorization_required"] is False
    assert confirmation["substage_reauthorization_required"] is False
    record_path = write_authorization_record(output_root / "candidate-screen", screen)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["stage"] == "candidate_screen"
    assert record["policy"]["content_sha256"] == load_execution_policy(policy_path)["sha256"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stage": "trajectory_collection"}, "stage"),
        ({"height": 768, "width": 1344}, "generation settings"),
        ({"request_count": 161}, "request count"),
        ({"model_revision": "different"}, "model"),
        ({"output_directory": Path("/tmp/outside")}, "output directory"),
    ],
)
def test_execution_policy_rejects_every_out_of_scope_dimension(
    tmp_path,
    overrides,
    message,
):
    output_root = tmp_path / "artifacts"
    policy_path = tmp_path / "execution-policy.json"
    _write_json(policy_path, _execution_policy(output_root))

    with pytest.raises(ValueError, match=message):
        authorize_execution(policy_path, _execution_request(output_root, **overrides))


def test_execution_policy_rejects_tampering(tmp_path):
    output_root = tmp_path / "artifacts"
    policy_path = tmp_path / "execution-policy.json"
    policy = _execution_policy(output_root)
    policy["scope"]["stage_request_limits"]["candidate_screen"] = 999
    _write_json(policy_path, policy)

    with pytest.raises(ValueError, match="sha256 does not match"):
        load_execution_policy(policy_path)


def _quality_contract(metric_config: dict) -> dict:
    payload = {
        "schema": natural_gate.CONTRACT_SCHEMA,
        "schema_revision": natural_gate.SCHEMA_REVISION,
        "protocol": {},
        "controlled_generation": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "scheduler_config_sha256": "a" * 64,
            "num_steps": 50,
            "guidance_scale": 3.5,
            "dtype": "bfloat16",
            "tp_degree": 4,
        },
        "margin_calibration": {},
        "automatic_damage_rule": {},
        "metric_identity": {
            "config": metric_config,
            "sha256": canonical_sha256(metric_config),
        },
        "resolution_contracts": [
            {
                "bucket_id": "square-1024",
                "height": 1024,
                "width": 1024,
                "margins": {"image_reward": 0.8, "vqa_score": 0.2},
                "calibration_summary": {
                    "image_reward": {"maximum": 1.0},
                    "vqa_score": {"maximum": 0.25},
                },
                "evidence": {},
            }
        ],
        "limitations": [],
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def test_natural_range_gate_automatically_rejects_without_review_state(
    tmp_path,
    monkeypatch,
):
    metric_config = {
        "image_reward": {
            "implementation": "ImageReward.score",
            "package": "image-reward",
            "package_version": "1.5",
            "model": "ImageReward-v1.0",
            "dtype": "float32",
            "preprocessing": "official",
            "checkpoint_files": [{"path": "/weights/ir", "sha256": "1" * 64}],
        },
        "vqa_score": {
            "implementation": "VQAScore",
            "package": "t2v-metrics",
            "package_version": "1.2",
            "model": "clip-flant5-xl",
            "dtype": "bfloat16",
            "question_template": "question",
            "answer_template": "Yes",
            "repositories": [],
            "checkpoint_files": [{"path": "/weights/vqa", "sha256": "2" * 64}],
        },
    }
    contract_path = tmp_path / "contract.json"
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(contract_path, _quality_contract(metric_config))
    _write_json(report_path, {})
    report = {
        "metrics": metric_config,
        "comparisons": [
            {
                "candidate_id": "safe",
                "sample_id": "p000-s0",
                "prompt_index": 0,
                "seed": 0,
                "prompt": "safe",
                "candidate_minus_baseline": {
                    "image_reward": -1.0,
                    "vqa_score": -0.25,
                },
            },
            {
                "candidate_id": "unsafe",
                "sample_id": "p001-s0",
                "prompt_index": 1,
                "seed": 0,
                "prompt": "unsafe",
                "candidate_minus_baseline": {
                    "image_reward": -1.000001,
                    "vqa_score": 0.0,
                },
            },
        ],
    }
    manifest = {
        "comparisons": [
            {
                key: row[key]
                for key in (
                    "candidate_id",
                    "sample_id",
                    "prompt_index",
                    "seed",
                    "prompt",
                )
            }
            for row in report["comparisons"]
        ]
    }
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(natural_gate, "load_semantic_report", lambda _path: report)
    monkeypatch.setattr(
        natural_gate,
        "semantic_source",
        lambda _report: (manifest, {"split": "test", "sha256": "b" * 64}, manifest_path),
    )
    monkeypatch.setattr(natural_gate, "validate_generation_identity", lambda *args: None)

    result = natural_gate.evaluate_natural_range(
        contract_path,
        report_path,
        bucket_id="square-1024",
    )
    summaries = {row["candidate_id"]: row for row in result["candidate_summaries"]}

    assert summaries["safe"]["decision"] == "automatic_pass"
    assert summaries["unsafe"]["decision"] == "automatic_reject"
    assert result["review_policy"] == {
        "within_natural_range_action": "automatic_pass",
        "outside_natural_range_action": "automatic_reject",
        "human_review_required": False,
        "manual_override_permitted": False,
    }
    serialized = json.dumps(result)
    assert "ambiguous" not in serialized
    assert "needs_review" not in serialized


def test_natural_range_gate_rejects_incomplete_scoring_coverage(tmp_path, monkeypatch):
    metric_config = {
        "image_reward": {
            "implementation": "ImageReward.score",
            "package": "image-reward",
            "package_version": "1.5",
            "model": "ImageReward-v1.0",
            "dtype": "float32",
            "preprocessing": "official",
            "checkpoint_files": [{"sha256": "1" * 64}],
        },
        "vqa_score": {
            "implementation": "VQAScore",
            "package": "t2v-metrics",
            "package_version": "1.2",
            "model": "clip-flant5-xl",
            "dtype": "bfloat16",
            "question_template": "question",
            "answer_template": "Yes",
            "repositories": [],
            "checkpoint_files": [{"sha256": "2" * 64}],
        },
    }
    contract_path = tmp_path / "contract.json"
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(contract_path, _quality_contract(metric_config))
    _write_json(report_path, {})
    _write_json(manifest_path, {})
    report = {"metrics": metric_config, "comparisons": []}
    manifest = {
        "comparisons": [
            {
                "candidate_id": "candidate",
                "sample_id": "p000-s0",
                "prompt_index": 0,
                "seed": 0,
                "prompt": "prompt",
            }
        ]
    }
    monkeypatch.setattr(natural_gate, "load_semantic_report", lambda _path: report)
    monkeypatch.setattr(
        natural_gate,
        "semantic_source",
        lambda _report: (manifest, {"split": "test", "sha256": "b" * 64}, manifest_path),
    )
    monkeypatch.setattr(natural_gate, "validate_generation_identity", lambda *args: None)

    with pytest.raises(ValueError, match="no cache comparisons"):
        natural_gate.evaluate_natural_range(
            contract_path,
            report_path,
            bucket_id="square-1024",
        )


def test_natural_range_cli_returns_nonzero_when_any_candidate_is_rejected(
    tmp_path,
    monkeypatch,
):
    result = {
        "candidate_summaries": [
            {
                "candidate_id": "unsafe",
                "passes_zero_failure_gate": False,
            }
        ]
    }
    monkeypatch.setattr(natural_gate, "evaluate_natural_range", lambda *args, **kwargs: result)

    exit_code = natural_gate.main(
        [
            "--contract",
            str(tmp_path / "contract.json"),
            "--semantic-report",
            str(tmp_path / "report.json"),
            "--bucket-id",
            "square-1024",
            "--out",
            str(tmp_path / "result.json"),
        ]
    )

    assert exit_code == 1


def test_scoped_wrapper_runs_one_frozen_confirmation_candidate(tmp_path, monkeypatch):
    output_root = tmp_path / "artifacts"
    output_directory = output_root / "confirmation"
    policy_path = tmp_path / "execution-policy.json"
    _write_json(policy_path, _execution_policy(output_root))
    candidate_path = _write_candidate(tmp_path / "candidate")
    prompt_suite = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "qualified-profile-confirmation-prompt-suite-20260808.json"
    )
    observed = {}

    def fake_collect(args, arm):
        output_directory.mkdir(parents=True)
        observed["candidate_id"] = arm.candidate_id
        return output_directory / "quality.json", output_directory / "speed.json"

    monkeypatch.setattr(
        authorized_collector.confirmation_collector,
        "collect_confirmation",
        fake_collect,
    )
    wrapper_args = type(
        "Args",
        (),
        {
            "execution_stage": "confirmation",
            "execution_policy": str(policy_path),
            "hardware_backend": "trainium",
            "hardware_product": "trn2.3xlarge",
            "phased_candidate": (str(candidate_path),),
        },
    )()
    authorized_collector._collect_ab(
        wrapper_args,
        (
            "--out-dir",
            str(output_directory),
            "--model-revision",
            MODEL_REVISION,
            "--prompt-suite",
            str(prompt_suite),
            "--prompt-split",
            "qualified_profile_confirmation",
            "--seed",
            "0",
        ),
    )

    record = json.loads(
        (output_directory / "execution-authorization.json").read_text(encoding="utf-8")
    )
    assert record["stage"] == "confirmation"
    assert record["request_count"] == 64
    assert observed["candidate_id"] == "unit-static-plus-brake"


def test_scoped_wrapper_rejects_user_supplied_legacy_ack(tmp_path):
    wrapper_args = type(
        "Args",
        (),
        {
            "execution_stage": "confirmation",
            "execution_policy": str(tmp_path / "policy.json"),
            "hardware_backend": "trainium",
            "hardware_product": "trn2.3xlarge",
            "phased_candidate": None,
        },
    )()
    with pytest.raises(ValueError, match="owns hardware acknowledgement"):
        authorized_collector._collect_ab(wrapper_args, ("--allow-hardware",))
