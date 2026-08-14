from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.offline.cache_profile import builder as profile_builder
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
        "optimizer": {
            "warmup_steps": 6,
            "cooldown_steps": 1,
            "require_final_anchor": True,
        },
    }


def _candidate(candidate_id: str = "quality-static-a13-o1-index", budget: int = 13) -> dict:
    payload = {
        "schema": profile_builder.PHASED_CANDIDATE_SCHEMA,
        "schema_revision": profile_builder.PHASED_CANDIDATE_SCHEMA_REVISION,
        "candidate_id": candidate_id,
        "policy": {
            "type": "phased_static",
            "static_anchor_steps": list(range(budget)),
        },
        "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
        "horizon_ref": {"path": "/tmp/horizon", "sha256": "a" * 64},
        "quality_contract_ref": {"path": "/tmp/contract", "sha256": "b" * 64},
    }
    return _hashed(payload)


def test_materialize_static_frontier_writes_runtime_loadable_profiles(tmp_path):
    derivation_path = tmp_path / "derivation.json"
    contract_path = tmp_path / "contract.json"
    candidate_path = tmp_path / "candidate.json"
    derivation_payload = {
        "schedules": [
            {
                "anchor_budget": 13,
                "static_anchor_steps": [0, 1, 2, 3, 4, 5, 9, 15, 21, 29, 38, 45, 49],
            },
            {
                "anchor_budget": 14,
                "static_anchor_steps": [
                    0, 1, 2, 3, 4, 5, 9, 15, 21, 29, 35, 40, 45, 49
                ],
            },
        ]
    }
    _write_json(derivation_path, _hashed(derivation_payload))
    _write_json(contract_path, {})

    frontier = profile_builder._materialize_static_frontier(
        _registration(),
        derivation_path,
        contract_path,
        tmp_path,
        reference_root=tmp_path,
    )

    candidate_path, document = frontier[0]
    loaded = profile_builder.load_phased_candidate(candidate_path)
    assert [item[1]["candidate_id"] for item in frontier] == [
        "quality-static-a13-o1-index",
        "quality-static-a14-o1-index",
    ]
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
    assert document["policy"]["type"] == "phased_static"
    assert document["policy"]["dynamic_budget"] == 0
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


def test_quality_ladder_prefix_appends_and_rebases_prior_evidence(tmp_path):
    protocol = {"sha256": "a" * 64}
    candidates = (
        _candidate("quality-static-a11-o1-index", 11),
        _candidate("quality-static-a12-o1-index", 12),
    )
    previous_prefix = None
    for index, candidate in enumerate(candidates):
        rung = tmp_path / f"rung-{index}"
        baseline = tmp_path / "rung-0" / "artifacts" / "baseline.png"
        image = rung / "artifacts" / "candidate.png"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        image.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(b"baseline")
        image.write_bytes(candidate["candidate_id"].encode())
        quality_path = rung / "quality-input-v2.json"
        definition = {
            "candidate_id": candidate["candidate_id"],
            "policy": candidate["policy"],
            "predictor": candidate["predictor"],
        }
        _write_json(
            quality_path,
            {
                "schema": "quality-input-v2",
                "protocol": protocol,
                "hardware_measured": True,
                "started_at": "start",
                "completed_at": f"end-{index}",
                "candidates": [definition],
                "comparisons": [
                    {
                        "sample_id": "p000-s0",
                        "candidate_id": candidate["candidate_id"],
                        "baseline": {
                            "image": Path(
                                profile_builder.os.path.relpath(baseline, rung)
                            ).as_posix(),
                            "image_sha256": profile_builder.sha256_file(baseline),
                        },
                        "candidate": {
                            "image": "artifacts/candidate.png",
                            "image_sha256": profile_builder.sha256_file(image),
                        },
                    }
                ],
            },
        )
        prefix_path = rung / "quality-ladder-prefix-v2.json"
        prefix = profile_builder._write_ladder_quality_prefix(
            previous_prefix,
            quality_path,
            prefix_path,
            candidate,
        )
        previous_prefix = prefix_path

    assert [row["candidate_id"] for row in prefix["candidates"]] == [
        candidate["candidate_id"] for candidate in candidates
    ]
    assert prefix["completed_at"] == "end-1"
    for comparison in prefix["comparisons"]:
        for role in ("baseline", "candidate"):
            assert (previous_prefix.parent / comparison[role]["image"]).resolve().is_file()


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


def _orchestration_fixture(
    tmp_path,
    monkeypatch,
    *,
    quality_passed,
    measured_speedup,
    max_allowed_failures=0,
    failure_counts=None,
    evidence_role=None,
):
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
        "schema_revision": profile_builder.LEGACY_BUILD_SPEC_SCHEMA_REVISION,
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
            "max_allowed_failures": max_allowed_failures,
            "selection_rule": "ascending_first_quality_pass",
            "speed_is_selection_input": False,
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
    if evidence_role is not None:
        spec["schema_revision"] = profile_builder.BUILD_SPEC_SCHEMA_REVISION
        spec["evidence_role"] = evidence_role
    registration = _registration()
    candidates = (
        _candidate("quality-static-a11-o1-index", 11),
        _candidate("quality-static-a12-o1-index", 12),
        _candidate("quality-static-a13-o1-index", 13),
    )
    monkeypatch.setattr(profile_builder, "load_build_spec", lambda _path: spec)
    monkeypatch.setattr(
        profile_builder,
        "_require_frozen_repository_inputs",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        profile_builder.schedule_derivation,
        "load_registration",
        lambda _path: registration,
    )

    def fake_derive(args):
        _write_json(Path(args.out), _hashed({"schedules": [{}, {}]}))

    def fake_materialize(
        _registration,
        _derivation,
        _contract,
        output_directory,
        **_kwargs,
    ):
        frontier = []
        for candidate in candidates:
            candidate_path = output_directory / f".{candidate['candidate_id']}.json"
            _write_json(candidate_path, candidate)
            frontier.append((candidate_path, candidate))
        return tuple(frontier)

    hardware_commands = []

    def fake_command(command):
        if any(value.endswith("collect_flux_cache_authorized.py") for value in command):
            hardware_commands.append(command)
            confirmation = Path(command[command.index("--out-dir") + 1])
            candidate_path = Path(command[command.index("--phased-candidate") + 1])
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            definition = {
                "candidate_id": candidate["candidate_id"],
                "policy": candidate["policy"],
                "predictor": candidate["predictor"],
            }
            _write_json(
                confirmation / "quality-input-v2.json",
                {"candidates": [definition]},
            )
            index = [row["candidate_id"] for row in candidates].index(
                candidate["candidate_id"]
            )
            _write_json(
                confirmation / "speedup-candidates-v1.json",
                {
                    "hardware_measured": True,
                    "candidates": [
                        {
                            **definition,
                            "measured_speedup": measured_speedup - index * 0.1,
                        }
                    ],
                },
            )
        elif any(value.endswith("evaluate_flux_cache_semantics.py") for value in command):
            semantic_path = Path(command[command.index("--out") + 1])
            _write_json(semantic_path, {})
        else:
            raise AssertionError(command)

    quality_results = (
        (quality_passed,) * len(candidates)
        if isinstance(quality_passed, bool)
        else tuple(quality_passed)
    )
    assert len(quality_results) == len(candidates)
    if failure_counts is None:
        failure_counts = tuple(
            0 if passing else max_allowed_failures + 1 for passing in quality_results
        )
    assert len(failure_counts) == len(candidates)
    assert all(
        (count <= max_allowed_failures) == passing
        for count, passing in zip(failure_counts, quality_results)
    )

    def fake_prefix(previous_path, _rung_path, output_path, candidate):
        definitions = []
        if previous_path is not None:
            definitions.extend(
                json.loads(previous_path.read_text(encoding="utf-8"))["candidates"]
            )
        definitions.append(
            {
                "candidate_id": candidate["candidate_id"],
                "policy": candidate["policy"],
                "predictor": candidate["predictor"],
            }
        )
        document = {"candidates": definitions}
        _write_json(output_path, document)
        return document

    def fake_gate(_contract, semantic_path, **kwargs):
        rung_index = int(semantic_path.parent.name.split("-", 1)[0])
        allowed = kwargs.get("max_allowed_failures", 0)
        payload = {
            "max_allowed_failures": allowed,
            "candidate_summaries": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "failure_count": failure_counts[index],
                    "passes_failure_budget_gate": failure_counts[index] <= allowed,
                }
                for index, candidate in enumerate(candidates[: rung_index + 1])
            ],
        }
        return _hashed(payload)

    monkeypatch.setattr(profile_builder.schedule_derivation, "derive", fake_derive)
    monkeypatch.setattr(profile_builder, "_materialize_static_frontier", fake_materialize)
    monkeypatch.setattr(profile_builder, "_write_ladder_quality_prefix", fake_prefix)
    monkeypatch.setattr(profile_builder, "_run_command", fake_command)
    monkeypatch.setattr(profile_builder, "evaluate_natural_range", fake_gate)
    monkeypatch.setattr(
        profile_builder,
        "load_phased_candidate",
        lambda path: SimpleNamespace(
            candidate_id=json.loads(path.read_text(encoding="utf-8"))["candidate_id"],
            file_sha256=profile_builder.sha256_file(path),
            content_sha256=json.loads(path.read_text(encoding="utf-8"))["sha256"],
        ),
    )
    return spec_path, output_root, hardware_commands


def test_one_command_exports_minimum_quality_passed_budget(tmp_path, monkeypatch):
    spec_path, output_root, hardware_commands = _orchestration_fixture(
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
    assert qualification["decision"]["max_allowed_failures"] == 0
    assert qualification["decision"]["selected_anchor_budget"] == 11
    assert qualification["decision"]["candidate_domain"] == [11, 12, 13]
    assert qualification["decision"]["tested_anchor_budgets"] == [11]
    assert qualification["decision"]["stop_reason"] == "first_quality_pass"
    assert qualification["decision"]["speed_is_selection_input"] is False
    assert qualification["decision"]["measured_speedup"] == 3.25
    assert len(hardware_commands) == 1
    assert not (output_root / "rejection-report.json").exists()


def test_one_command_selects_first_passing_budget_after_a_failure(tmp_path, monkeypatch):
    spec_path, output_root, hardware_commands = _orchestration_fixture(
        tmp_path,
        monkeypatch,
        quality_passed=(False, True, True),
        measured_speedup=2.5,
    )

    profile_builder.build_profile(spec_path)

    qualification = json.loads(
        (output_root / "profile-qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["decision"]["selected_anchor_budget"] == 12
    assert [
        row["quality_passed"]
        for row in qualification["decision"]["tested_frontier"]
    ] == [False, True]
    assert len(hardware_commands) == 2
    assert "--baseline-quality-manifest" not in hardware_commands[0]
    assert "--baseline-quality-manifest" in hardware_commands[1]
    assert "--baseline-speed-manifest" in hardware_commands[1]


def test_hardware_ladder_smoke_never_exports_a_deployable_profile(
    tmp_path,
    monkeypatch,
):
    spec_path, output_root, hardware_commands = _orchestration_fixture(
        tmp_path,
        monkeypatch,
        quality_passed=(False, True, True),
        measured_speedup=2.5,
        evidence_role=profile_builder.HARDWARE_LADDER_SMOKE_ROLE,
    )

    report = profile_builder.build_profile(spec_path)

    assert report == output_root / "hardware-ladder-smoke-report.json"
    smoke = json.loads(report.read_text(encoding="utf-8"))
    assert smoke["status"] == "complete"
    assert smoke["evidence_role"] == profile_builder.HARDWARE_LADDER_SMOKE_ROLE
    assert smoke["decision"]["tested_anchor_budgets"] == [11, 12]
    assert smoke["decision"]["stop_reason"] == "first_quality_pass"
    assert smoke["deployable_profile_written"] is False
    assert len(hardware_commands) == 2
    assert not (output_root / "cache-profile.json").exists()
    assert not (output_root / "profile-qualification.json").exists()


def test_one_command_applies_the_registered_failure_budget(tmp_path, monkeypatch):
    spec_path, output_root, hardware_commands = _orchestration_fixture(
        tmp_path,
        monkeypatch,
        quality_passed=(False, True, True),
        measured_speedup=2.5,
        max_allowed_failures=2,
        failure_counts=(3, 2, 0),
    )

    profile_builder.build_profile(spec_path)

    qualification = json.loads(
        (output_root / "profile-qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["decision"]["selected_anchor_budget"] == 12
    assert qualification["decision"]["max_allowed_failures"] == 2
    assert qualification["decision"]["failure_count"] == 2
    frontier = qualification["decision"]["tested_frontier"]
    assert [row["quality_passed"] for row in frontier] == [False, True]
    assert [row["failure_count"] for row in frontier] == [3, 2]
    assert len(hardware_commands) == 2


def test_one_command_rejects_only_when_no_budget_passes_quality(
    tmp_path,
    monkeypatch,
):
    spec_path, output_root, hardware_commands = _orchestration_fixture(
        tmp_path,
        monkeypatch,
        quality_passed=False,
        measured_speedup=3.25,
    )

    with pytest.raises(profile_builder.ProfileRejected):
        profile_builder.build_profile(spec_path)

    assert not (output_root / "cache-profile.json").exists()
    rejection = json.loads((output_root / "rejection-report.json").read_text(encoding="utf-8"))
    assert rejection["status"] == "rejected"
    assert rejection["deployable_profile_written"] is False
    assert rejection["decision"]["stop_reason"] == "candidate_domain_exhausted"
    assert len(hardware_commands) == 3


def test_measured_speed_below_historical_target_does_not_reject(tmp_path, monkeypatch):
    spec_path, output_root, _hardware_commands = _orchestration_fixture(
        tmp_path,
        monkeypatch,
        quality_passed=True,
        measured_speedup=1.01,
    )

    profile_builder.build_profile(spec_path)

    qualification = json.loads(
        (output_root / "profile-qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "qualified"
    assert qualification["decision"]["measured_speedup"] == 1.01
    assert qualification["decision"]["speed_is_selection_input"] is False


def test_cli_uses_distinct_exit_code_for_a_quality_rejection(monkeypatch):
    monkeypatch.setattr(
        profile_builder,
        "build_profile",
        lambda _path: (_ for _ in ()).throw(profile_builder.ProfileRejected("rejected")),
    )
    assert profile_builder.main(["build", "--spec", "/tmp/spec.json"]) == 1


def test_profile_build_rejects_changed_bound_source_before_hardware(monkeypatch):
    bound = profile_builder.ROOT / "difflet" / "pipeline" / "cache" / "runner.py"
    spec = {
        "implementation": {"files": [{"path": bound.relative_to(profile_builder.ROOT).as_posix()}]},
        "calibration": {"schedule_registration": {"path": str(bound)}},
        "quality_contract": {"contract": {"path": str(bound)}},
        "confirmation": {"prompt_suite": {"path": str(bound)}},
        "execution_policy": {"path": str(bound)},
    }
    commands = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        return SimpleNamespace(
            returncode=1 if command[1:3] == ["diff", "--quiet"] else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(profile_builder.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="hash-bound implementation"):
        profile_builder._require_frozen_repository_inputs(bound, spec)
    assert all("status" not in command for command in commands)


def test_subprocess_inherits_the_selected_runtime_bin_on_path(monkeypatch):
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(profile_builder.subprocess, "run", fake_run)
    profile_builder._run_command(["/opt/example-venv/bin/python", "script.py"])

    assert observed["command"][0] == "/opt/example-venv/bin/python"
    assert observed["environment"]["PATH"].split(":")[0] == "/opt/example-venv/bin"
