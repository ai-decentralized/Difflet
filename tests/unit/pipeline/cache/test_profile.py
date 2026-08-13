from __future__ import annotations

import json
from pathlib import Path

import pytest

from difflet.pipeline.cache import (
    CacheProfileError,
    CacheSession,
    QualifiedCacheProfile,
)
from difflet.pipeline.cache.profile import (
    PHASED_CANDIDATE_SCHEMA,
    PHASED_CANDIDATE_SCHEMA_REVISION,
    PROFILE_QUALIFICATION_SCHEMA,
    PROFILE_QUALIFICATION_SCHEMA_REVISION,
    canonical_sha256,
    load_qualified_cache_profile,
    sha256_file,
)


def _write_json(path: Path, payload: dict, *, hashed: bool = True) -> dict:
    document = {**payload, "sha256": canonical_sha256(payload)} if hashed else payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _qualified_bundle(tmp_path: Path) -> tuple[Path, Path]:
    horizon_path = tmp_path / "internal" / "schedule.json"
    horizon_path.parent.mkdir(parents=True)
    horizon_path.write_text("{}\n", encoding="utf-8")
    contract_path = tmp_path / "internal" / "quality-contract.json"
    contract = _write_json(
        contract_path,
        {
            "schema": "difflet-flux-cache-multires-quality-contract",
            "schema_revision": 1,
            "controlled_generation": {
                "model_id": "black-forest-labs/FLUX.1-dev",
                "model_revision": "3" * 40,
                "tp_degree": 4,
                "num_steps": 8,
                "guidance_scale": 3.5,
                "dtype": "bfloat16",
                "scheduler_class": "FlowMatchEulerDiscreteScheduler",
                "scheduler_config_sha256": "4" * 64,
            },
            "resolution_contracts": [
                {"bucket_id": "square-64", "height": 64, "width": 64}
            ],
        },
    )
    profile_path = tmp_path / "cache-profile.json"
    candidate = _write_json(
        profile_path,
        {
            "schema": PHASED_CANDIDATE_SCHEMA,
            "schema_revision": PHASED_CANDIDATE_SCHEMA_REVISION,
            "candidate_id": "qualified-test-profile",
            "policy": {
                "type": "phased_static_plus_brake",
                "num_steps": 8,
                "static_anchor_steps": [0, 1, 2, 4, 7],
                "warmup_steps": 2,
                "cooldown_steps": 1,
                "require_final_anchor": True,
                "dynamic_budget": 1,
                "invalid_measurement_fail_closed": True,
                "plastic_window": [2, 6],
                "tighten_error": 1.1,
                "recovery_error": 1.4,
                "recovery_steps": 1,
                "disable_after_recoveries": 2,
                "tighten_rule": "bisect_next_static_gap",
                "allow_acceleration": False,
            },
            "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
            "horizon_ref": {
                "path": "internal/schedule.json",
                "sha256": sha256_file(horizon_path),
            },
            "quality_contract_ref": {
                "path": "internal/quality-contract.json",
                "sha256": sha256_file(contract_path),
            },
        },
    )
    build_spec_path = tmp_path / "build-spec.json"
    build_spec = _write_json(
        build_spec_path,
        {
            "schema": "test-build-spec",
            "quality_contract": {
                "contract": {
                    "path": str(contract_path),
                    "file_sha256": sha256_file(contract_path),
                    "content_sha256": contract["sha256"],
                },
                "bucket_id": "square-64",
            },
        },
    )
    qualification_path = tmp_path / "profile-qualification.json"
    _write_json(
        qualification_path,
        {
            "schema": PROFILE_QUALIFICATION_SCHEMA,
            "schema_revision": PROFILE_QUALIFICATION_SCHEMA_REVISION,
            "build_id": "qualified-test-build",
            "completed_at": "2026-08-08T00:00:00Z",
            "status": "qualified",
            "build_spec": {
                "path": str(build_spec_path),
                "file_sha256": sha256_file(build_spec_path),
                "content_sha256": build_spec["sha256"],
            },
            "profile": {
                "path": "/original/output/cache-profile.json",
                "file_sha256": sha256_file(profile_path),
                "content_sha256": candidate["sha256"],
                "candidate_id": candidate["candidate_id"],
            },
            "decision": {
                "quality_passed": True,
                "failure_count": 0,
                "max_allowed_failures": 0,
                "selected_anchor_budget": 5,
                "measured_speedup": 3.4,
                "selection_rule": "ascending_first_quality_pass",
                "qualification_order": "ascending_anchor_budget",
                "candidate_domain": [4, 5, 6],
                "tested_anchor_budgets": [4, 5],
                "stop_reason": "first_quality_pass",
                "speed_is_selection_input": False,
                "tested_frontier": [
                    {
                        "candidate_id": "quality-static-a4-o1-index",
                        "anchor_budget": 4,
                        "quality_passed": False,
                        "failure_count": 1,
                        "measured_speedup": 4.0,
                    },
                    {
                        "candidate_id": candidate["candidate_id"],
                        "anchor_budget": 5,
                        "quality_passed": True,
                        "failure_count": 0,
                        "measured_speedup": 3.4,
                    },
                ],
            },
            "evidence": {},
        },
    )
    return profile_path, qualification_path


def _runtime_identity() -> dict:
    return {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "model_revision": "3" * 40,
        "height": 64,
        "width": 64,
        "num_steps": 8,
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "scheduler_config_sha256": "4" * 64,
        "dtype": "bfloat16",
        "guidance_scale": 3.5,
        "tp_degree": 4,
    }


def test_load_qualified_profile_validates_evidence_and_builds_request_session(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)

    profile = load_qualified_cache_profile(profile_path, qualification_path)
    profile.validate_runtime(**_runtime_identity())
    session = profile.build_session(8)

    assert isinstance(profile, QualifiedCacheProfile)
    assert isinstance(session, CacheSession)
    assert profile.candidate_id == "qualified-test-profile"
    assert profile.measured_speedup == 3.4
    assert profile.minimum_speedup is None
    assert profile.selected_anchor_budget == 5


def test_load_qualified_profile_keeps_revision_one_compatibility(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["schema_revision"] = 1
    qualification["decision"] = {
        "quality_passed": True,
        "failure_count": 0,
        "measured_speedup": 3.4,
        "minimum_speedup": 3.2,
        "speed_passed": True,
    }
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile = load_qualified_cache_profile(profile_path, qualification_path)

    assert profile.minimum_speedup == 3.2
    assert profile.selected_anchor_budget is None


def test_load_qualified_profile_keeps_revision_two_compatibility(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["schema_revision"] = 2
    qualification["decision"] = {
        "quality_passed": True,
        "failure_count": 0,
        "selected_anchor_budget": 5,
        "measured_speedup": 3.4,
        "selection_rule": "minimum_anchor_budget_among_quality_passed_candidates",
        "speed_is_selection_input": False,
        "frontier": [
            {
                "candidate_id": "quality-static-a4-o1-index",
                "anchor_budget": 4,
                "quality_passed": False,
                "failure_count": 1,
                "measured_speedup": 4.0,
            },
            {
                "candidate_id": "qualified-test-profile",
                "anchor_budget": 5,
                "quality_passed": True,
                "failure_count": 0,
                "measured_speedup": 3.4,
            },
        ],
    }
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile = load_qualified_cache_profile(profile_path, qualification_path)

    assert profile.minimum_speedup is None
    assert profile.selected_anchor_budget == 5


def test_load_qualified_profile_keeps_revision_three_compatibility(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["schema_revision"] = 3
    del qualification["decision"]["max_allowed_failures"]
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile = load_qualified_cache_profile(profile_path, qualification_path)

    assert profile.selected_anchor_budget == 5


def test_load_qualified_profile_accepts_a_registered_failure_budget(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    decision = qualification["decision"]
    decision["max_allowed_failures"] = 2
    decision["failure_count"] = 2
    decision["tested_frontier"][0]["failure_count"] = 3
    decision["tested_frontier"][1]["failure_count"] = 2
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile = load_qualified_cache_profile(profile_path, qualification_path)

    assert profile.selected_anchor_budget == 5


def test_load_qualified_profile_rejects_failures_beyond_the_registered_budget(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    decision = qualification["decision"]
    decision["failure_count"] = 1
    decision["tested_frontier"][1]["failure_count"] = 1
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CacheProfileError, match="ladder is invalid"):
        load_qualified_cache_profile(profile_path, qualification_path)


def test_qualified_profile_rejects_runtime_identity_mismatch(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    profile = load_qualified_cache_profile(profile_path, qualification_path)
    identity = _runtime_identity()
    identity["width"] = 96

    with pytest.raises(CacheProfileError, match="does not match the runtime"):
        profile.validate_runtime(**identity)


def test_qualified_profile_rejects_profile_changed_after_confirmation(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    document["candidate_id"] = "tampered"
    profile_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(CacheProfileError, match="sha256"):
        load_qualified_cache_profile(profile_path, qualification_path)


def test_qualified_profile_rejects_failed_decision(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["decision"]["quality_passed"] = False
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CacheProfileError, match="ladder is invalid"):
        load_qualified_cache_profile(profile_path, qualification_path)


def test_qualified_profile_rejects_inconsistent_frontier_decision(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["decision"]["tested_frontier"][0]["quality_passed"] = True
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CacheProfileError, match="ladder is invalid"):
        load_qualified_cache_profile(profile_path, qualification_path)


def test_qualified_profile_rejects_a_ladder_that_skipped_a_lower_budget(tmp_path):
    profile_path, qualification_path = _qualified_bundle(tmp_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["decision"]["tested_anchor_budgets"] = [5]
    qualification["decision"]["tested_frontier"] = qualification["decision"][
        "tested_frontier"
    ][1:]
    payload = {key: value for key, value in qualification.items() if key != "sha256"}
    qualification["sha256"] = canonical_sha256(payload)
    qualification_path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CacheProfileError, match="ladder is invalid"):
        load_qualified_cache_profile(profile_path, qualification_path)


def _calibrated_candidate(tmp_path: Path, weights: dict) -> Path:
    horizon_path = tmp_path / "internal" / "schedule.json"
    horizon_path.parent.mkdir(parents=True, exist_ok=True)
    horizon_path.write_text("{}\n", encoding="utf-8")
    contract_path = tmp_path / "internal" / "quality-contract.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    profile_path = tmp_path / "calibrated-profile.json"
    _write_json(
        profile_path,
        {
            "schema": PHASED_CANDIDATE_SCHEMA,
            "schema_revision": PHASED_CANDIDATE_SCHEMA_REVISION,
            "candidate_id": "calibrated-test-profile",
            "policy": {
                "type": "phased_static",
                "num_steps": 8,
                "static_anchor_steps": [0, 1, 2, 4, 7],
                "warmup_steps": 2,
                "cooldown_steps": 1,
                "require_final_anchor": True,
                "dynamic_budget": 0,
                "invalid_measurement_fail_closed": True,
            },
            "predictor": {"type": "calibrated_linear", "coord": "index", "weights": weights},
            "horizon_ref": {
                "path": str(horizon_path),
                "sha256": sha256_file(horizon_path),
            },
            "quality_contract_ref": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
        },
    )
    return profile_path


def test_calibrated_candidate_round_trips_and_builds_session(tmp_path):
    from difflet.pipeline.cache.profile import load_phased_candidate

    weights = {
        "3": [[1, -0.5], [2, 1.5]],
        "5": [[2, -0.25], [4, 1.25]],
        "6": [[2, -0.75], [4, 1.75]],
    }
    arm = load_phased_candidate(_calibrated_candidate(tmp_path, weights))
    assert arm.weights == {
        3: ((1, -0.5), (2, 1.5)),
        5: ((2, -0.25), (4, 1.25)),
        6: ((2, -0.75), (4, 1.75)),
    }
    assert arm.predictor_spec()["type"] == "calibrated_linear"
    assert isinstance(arm.build_session(8), CacheSession)


def test_calibrated_candidate_rejects_incomplete_weight_coverage(tmp_path):
    weights = {"3": [[1, -0.5], [2, 1.5]], "5": [[2, -0.25], [4, 1.25]]}
    from difflet.pipeline.cache.profile import load_phased_candidate

    with pytest.raises(CacheProfileError, match="exactly the skipped steps"):
        load_phased_candidate(_calibrated_candidate(tmp_path, weights))
