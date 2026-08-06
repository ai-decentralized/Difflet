from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_flux_cache_profile as profile_builder
from scripts.flux_cache_protocol import canonical_sha256


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _hashed(payload: dict) -> dict:
    return {**payload, "sha256": canonical_sha256(payload)}


def _registration() -> dict:
    return {
        "controlled_generation": {
            "num_steps": 50,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "dtype": "bfloat16",
        },
        "hardware_budget": {"target_speedup": 3.2},
        "optimizer": {
            "warmup_steps": 6,
            "cooldown_steps": 1,
            "require_final_anchor": True,
        },
        "bounded_brake": {
            "dynamic_budget": 2,
            "plastic_window": [6, 29],
            "recovery_steps": 2,
            "disable_after_recoveries": 2,
            "tighten_rule": "bisect_next_static_gap",
        },
    }


def _candidate(candidate_id: str = "target-static-brake-s3p2-a13-b2-o1-index") -> dict:
    payload = {
        "schema": profile_builder.PHASED_CANDIDATE_SCHEMA,
        "schema_revision": profile_builder.PHASED_CANDIDATE_SCHEMA_REVISION,
        "candidate_id": candidate_id,
        "policy": {"type": "phased_static_plus_brake"},
        "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
        "horizon_ref": {"path": "/tmp/horizon", "sha256": "a" * 64},
        "quality_contract_ref": {"path": "/tmp/contract", "sha256": "b" * 64},
    }
    return _hashed(payload)


def test_materialize_combined_candidate_writes_one_runtime_loadable_profile(tmp_path):
    derivation_path = tmp_path / "derivation.json"
    contract_path = tmp_path / "contract.json"
    candidate_path = tmp_path / "candidate.json"
    derivation_payload = {
        "schedules": [
            {
                "anchor_budget": 13,
                "target_speedup": 3.2,
                "static_anchor_steps": [0, 1, 2, 3, 4, 5, 9, 15, 21, 29, 38, 45, 49],
                "brake_thresholds": {
                    "tighten_error": 1.18,
                    "recovery_error": 1.45,
                },
            }
        ]
    }
    _write_json(derivation_path, _hashed(derivation_payload))
    _write_json(contract_path, {})

    document = profile_builder._materialize_combined_candidate(
        _registration(),
        derivation_path,
        contract_path,
        candidate_path,
        reference_root=tmp_path,
    )

    loaded = profile_builder.load_phased_candidate(candidate_path)
    assert loaded.candidate_id == "target-static-brake-s3p2-a13-b2-o1-index"
    assert document["policy"]["static_anchor_steps"] == [
        0,
        1,
        2,
        3,
        4,
        5,
        9,
        15,
        21,
        29,
        38,
        45,
        49,
    ]
    assert document["policy"]["allow_acceleration"] is False
    assert document["horizon_ref"]["path"] == "derivation.json"
    assert document["quality_contract_ref"]["path"] == "contract.json"


def test_confirmation_manifest_validation_requires_exact_candidate(tmp_path):
    candidate = _candidate()
    definition = {
        "candidate_id": candidate["candidate_id"],
        "policy": candidate["policy"],
        "predictor": candidate["predictor"],
    }
    quality_path = tmp_path / "quality.json"
    speed_path = tmp_path / "speed.json"
    _write_json(quality_path, {"candidates": [definition]})
    _write_json(
        speed_path,
        {
            "hardware_measured": True,
            "candidates": [{**definition, "measured_speedup": 3.25}],
        },
    )

    assert (
        profile_builder._validate_confirmation_manifests(
            quality_path,
            speed_path,
            candidate,
        )
        == 3.25
    )

    wrong = {**definition, "candidate_id": "different", "measured_speedup": 3.25}
    _write_json(speed_path, {"hardware_measured": True, "candidates": [wrong]})
    with pytest.raises(ValueError, match="different candidate"):
        profile_builder._validate_confirmation_manifests(
            quality_path,
            speed_path,
            candidate,
        )


def _prompt_suite(path: Path) -> None:
    _write_json(
        path,
        {
            "schema": "difflet-flux-cache-prompt-suite-v1",
            "suite_id": "test-confirmation",
            "source": {"origin": "unit test"},
            "splits": {
                "confirmation": [
                    {
                        "prompt_id": "p1",
                        "category": "test",
                        "text": "one test prompt",
                    }
                ]
            },
        },
    )


def _orchestration_fixture(tmp_path, monkeypatch, *, quality_passed, measured_speedup):
    output_root = tmp_path / "output"
    registration_path = tmp_path / "registration.json"
    contract_path = tmp_path / "contract.json"
    policy_path = tmp_path / "policy.json"
    prompt_path = tmp_path / "prompts.json"
    spec_path = tmp_path / "spec.json"
    for path in (registration_path, policy_path, spec_path):
        _write_json(path, {})
    _write_json(
        contract_path,
        {
            "controlled_generation": {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "model_revision": "3" * 40,
                "tp_degree": 4,
            }
        },
    )
    _prompt_suite(prompt_path)
    spec = {
        "build_id": "test-build",
        "calibration": {"schedule_registration": {"path": str(registration_path)}},
        "quality_contract": {
            "contract": {"path": str(contract_path)},
            "bucket_id": "square-1024",
        },
        "confirmation": {
            "output_directory": str(output_root),
            "prompt_suite": {"path": str(prompt_path)},
            "prompt_split": "confirmation",
            "seeds": [0],
            "minimum_speedup": 3.2,
        },
        "execution_policy": {"path": str(policy_path)},
        "runtimes": {
            "hardware_python": "/python/hardware",
            "semantic_python": "/python/semantic",
        },
        "scoring": {
            "image_reward_cache": "/cache/ir",
            "vqa_model_cache": "/cache/vqa",
            "huggingface_cache": "/cache/hf",
            "vqa_batch_size": 4,
            "cpu_threads": 12,
        },
        "sha256": "f" * 64,
    }
    registration = _registration()
    candidate = _candidate()
    monkeypatch.setattr(profile_builder, "load_build_spec", lambda _path: spec)
    monkeypatch.setattr(profile_builder, "_require_clean_worktree", lambda: None)
    monkeypatch.setattr(
        profile_builder.schedule_derivation,
        "load_registration",
        lambda _path: registration,
    )

    def fake_derive(args):
        _write_json(Path(args.out), _hashed({"schedules": [{}]}))

    def fake_materialize(
        _registration,
        _derivation,
        _contract,
        candidate_path,
        **_kwargs,
    ):
        _write_json(candidate_path, candidate)
        return candidate

    def fake_command(command):
        if any(value.endswith("collect_flux_cache_authorized.py") for value in command):
            confirmation = output_root / "confirmation"
            _write_json(confirmation / "quality-input-v2.json", {})
            _write_json(confirmation / "speedup-candidates-v1.json", {})
        elif any(value.endswith("evaluate_flux_cache_semantics.py") for value in command):
            semantic_path = Path(command[command.index("--out") + 1])
            _write_json(semantic_path, {})
        else:
            raise AssertionError(command)

    gate_payload = {
        "candidate_summaries": [
            {
                "candidate_id": candidate["candidate_id"],
                "failure_count": 0 if quality_passed else 1,
                "passes_zero_failure_gate": quality_passed,
            }
        ]
    }
    gate = _hashed(gate_payload)
    monkeypatch.setattr(profile_builder.schedule_derivation, "derive", fake_derive)
    monkeypatch.setattr(profile_builder, "_materialize_combined_candidate", fake_materialize)
    monkeypatch.setattr(profile_builder, "_run_command", fake_command)
    monkeypatch.setattr(profile_builder, "evaluate_natural_range", lambda *args, **kwargs: gate)
    monkeypatch.setattr(
        profile_builder,
        "_validate_confirmation_manifests",
        lambda *args: measured_speedup,
    )
    monkeypatch.setattr(
        profile_builder,
        "load_phased_candidate",
        lambda path: SimpleNamespace(
            candidate_id=candidate["candidate_id"],
            file_sha256=profile_builder.sha256_file(path),
            content_sha256=candidate["sha256"],
        ),
    )
    return spec_path, output_root


def test_one_command_exports_profile_only_after_quality_and_speed_pass(tmp_path, monkeypatch):
    spec_path, output_root = _orchestration_fixture(
        tmp_path,
        monkeypatch,
        quality_passed=True,
        measured_speedup=3.25,
    )

    profile = profile_builder.build_profile(spec_path)

    assert profile == output_root / "cache-profile.json"
    assert profile.is_file()
    qualification = json.loads(
        (output_root / "profile-qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "qualified"
    assert qualification["decision"]["quality_passed"] is True
    assert qualification["decision"]["speed_passed"] is True
    assert not (output_root / "rejection-report.json").exists()


@pytest.mark.parametrize(
    ("quality_passed", "measured_speedup"),
    [(False, 3.25), (True, 3.19)],
)
def test_one_command_rejects_without_exporting_profile(
    tmp_path,
    monkeypatch,
    quality_passed,
    measured_speedup,
):
    spec_path, output_root = _orchestration_fixture(
        tmp_path,
        monkeypatch,
        quality_passed=quality_passed,
        measured_speedup=measured_speedup,
    )

    with pytest.raises(profile_builder.ProfileRejected):
        profile_builder.build_profile(spec_path)

    assert not (output_root / "cache-profile.json").exists()
    rejection = json.loads((output_root / "rejection-report.json").read_text(encoding="utf-8"))
    assert rejection["status"] == "rejected"
    assert rejection["deployable_profile_written"] is False


def test_cli_uses_distinct_exit_code_for_a_quality_rejection(monkeypatch):
    monkeypatch.setattr(
        profile_builder,
        "build_profile",
        lambda _path: (_ for _ in ()).throw(profile_builder.ProfileRejected("rejected")),
    )
    assert profile_builder.main(["build", "--spec", "/tmp/spec.json"]) == 1


def test_profile_build_rejects_dirty_source_before_hardware(monkeypatch):
    monkeypatch.setattr(
        profile_builder.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M scripts/example.py\n"),
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        profile_builder._require_clean_worktree()
