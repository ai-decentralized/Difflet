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
        "candidate_id": "unit-static-frontier-a17",
        "policy": {
            "type": "phased_static",
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
            "dynamic_budget": 0,
            "invalid_measurement_fail_closed": True,
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
        "observed_seed_variation_calibration": {
            "method": natural_gate.CALIBRATION_METHOD,
            "decision_statistic": "maximum_observed",
            "candidate_images_excluded_from_margin_estimation": True,
            "pooling_across_resolutions_forbidden": True,
        },
        "automatic_damage_rule": natural_gate.AUTOMATIC_DAMAGE_RULE,
        "metric_identity": {
            "config": metric_config,
            "sha256": canonical_sha256(metric_config),
        },
        "resolution_contracts": [
            {
                "bucket_id": "square-1024",
                "height": 1024,
                "width": 1024,
                "observed_seed_variation_envelope": {
                    "image_reward": 1.0,
                    "vqa_score": 0.25,
                },
                "calibration_diagnostics": {
                    "image_reward": {"maximum_observed": 1.0},
                    "vqa_score": {"maximum_observed": 0.25},
                },
                "evidence": {},
            }
        ],
        "limitations": [],
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def test_natural_range_contract_rejects_legacy_quantile_contract(tmp_path):
    contract = _quality_contract({"image_reward": {}, "vqa_score": {}})
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    payload.pop("observed_seed_variation_calibration")
    payload["margin_calibration"] = {
        "method": "absolute-baseline-seed-pair-difference-nearest-rank",
        "quantile": 0.95,
    }
    path = tmp_path / "legacy-contract.json"
    _write_json(path, {**payload, "sha256": canonical_sha256(payload)})

    with pytest.raises(ValueError, match="observed-maximum calibration rule"):
        natural_gate.load_natural_range_contract(path)


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

    assert result["max_allowed_failures"] == 0
    assert summaries["safe"]["decision"] == "automatic_pass"
    assert summaries["safe"]["passes_failure_budget_gate"] is True
    assert summaries["unsafe"]["decision"] == "automatic_reject"
    assert summaries["unsafe"]["passes_failure_budget_gate"] is False
    assert summaries["unsafe"]["max_allowed_failures"] == 0
    assert result["review_policy"] == {
        "within_natural_range_action": "automatic_pass",
        "outside_natural_range_action": "automatic_reject",
        "human_review_required": False,
        "manual_override_permitted": False,
    }
    serialized = json.dumps(result)
    assert "ambiguous" not in serialized
    assert "needs_review" not in serialized

    budgeted = natural_gate.evaluate_natural_range(
        contract_path,
        report_path,
        bucket_id="square-1024",
        max_allowed_failures=1,
    )
    budgeted_summaries = {
        row["candidate_id"]: row for row in budgeted["candidate_summaries"]
    }

    assert budgeted["max_allowed_failures"] == 1
    assert budgeted_summaries["unsafe"]["failure_count"] == 1
    assert budgeted_summaries["unsafe"]["decision"] == "automatic_pass"
    assert budgeted_summaries["unsafe"]["passes_failure_budget_gate"] is True


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
                "passes_failure_budget_gate": False,
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
    assert observed["candidate_id"] == "unit-static-frontier-a17"


def test_scoped_wrapper_authorizes_only_candidate_requests_when_reusing_baseline(
    tmp_path,
    monkeypatch,
):
    output_root = tmp_path / "artifacts"
    output_directory = output_root / "confirmation-rung-2"
    policy_path = tmp_path / "execution-policy.json"
    _write_json(policy_path, _execution_policy(output_root))
    candidate_path = _write_candidate(tmp_path / "candidate")
    baseline_quality = tmp_path / "baseline-quality.json"
    baseline_speed = tmp_path / "baseline-speed.json"
    _write_json(baseline_quality, {})
    _write_json(baseline_speed, {})
    prompt_suite = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "qualified-profile-confirmation-prompt-suite-20260808.json"
    )

    def fake_collect(args, _arm):
        output_directory.mkdir(parents=True)
        assert args.baseline_quality_manifest == str(baseline_quality)
        assert args.baseline_speed_manifest == str(baseline_speed)
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
            "--baseline-quality-manifest",
            str(baseline_quality),
            "--baseline-speed-manifest",
            str(baseline_speed),
        ),
    )

    record = json.loads(
        (output_directory / "execution-authorization.json").read_text(encoding="utf-8")
    )
    assert record["request_count"] == 32


def test_scoped_wrapper_runs_baseline_only_trajectory_calibration(tmp_path, monkeypatch):
    output_root = tmp_path / "artifacts"
    output_directory = output_root / "calibration"
    policy_path = tmp_path / "execution-policy.json"
    policy = _execution_policy(output_root)
    policy["scope"]["allowed_stages"].append("trajectory_collection")
    policy["scope"]["stage_request_limits"]["trajectory_collection"] = 192
    policy["sha256"] = canonical_sha256(
        {key: value for key, value in policy.items() if key != "sha256"}
    )
    _write_json(policy_path, policy)

    prompt_suite = (
        Path(__file__).resolve().parents[3]
        / "benchmark"
        / "flux_cache"
        / "phase-schedule-development-prompt-suite.json"
    )
    observed = {}

    def fake_collect(args):
        output_directory.mkdir(parents=True)
        observed["called"] = True
        return output_directory / "trajectory-input-v1.json"

    monkeypatch.setattr(
        authorized_collector.confirmation_collector,
        "collect_calibration",
        fake_collect,
    )
    wrapper_args = type(
        "Args",
        (),
        {
            "execution_stage": "trajectory_collection",
            "execution_policy": str(policy_path),
            "hardware_backend": "trainium",
            "hardware_product": "trn2.3xlarge",
            "phased_candidate": None,
        },
    )()

    authorized_collector._collect_calibration(
        wrapper_args,
        (
            "--out-dir",
            str(output_directory),
            "--model-revision",
            MODEL_REVISION,
            "--prompt-suite",
            str(prompt_suite),
            "--prompt-split",
            "phase_schedule_horizon_development",
            "--seed",
            "2",
        ),
    )

    record = json.loads(
        (output_directory / "execution-authorization.json").read_text(encoding="utf-8")
    )
    assert record["stage"] == "trajectory_collection"
    assert record["request_count"] == 48
    assert observed["called"] is True


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
