from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.collect_flux_cache_phase_schedule_source as source_collector
import scripts.flux_cache_phase_schedule_registration as registration_module
from difflet.pipeline.cache import (
    AdaptiveAnchorConfig,
    CacheHistory,
    CacheStepContext,
    RuntimeObservation,
)
from scripts.flux_cache_phase_schedule_horizon import (
    RegisteredRepairBurstPolicy,
    RegisteredTerminalPolicy,
    _intervention_id,
    _unit_id,
    build_selection,
    evaluate_repair,
    evaluate_terminal,
)
from scripts.flux_cache_phase_schedule_registration import load_registration
from scripts.flux_cache_protocol import canonical_sha256

ROOT = Path(__file__).resolve().parents[3]
REGISTRATION_PATH = (
    ROOT / "benchmark" / "flux_cache" / "phase-schedule-horizon-registration.json"
)


def _adaptive_config() -> AdaptiveAnchorConfig:
    return AdaptiveAnchorConfig(
        initial_anchor_interval=12,
        minimum_anchor_interval=8,
        maximum_anchor_interval=16,
        warmup_steps=6,
        cooldown_steps=1,
        anchor_phase=1,
        tighten_error=3.0,
        recovery_error=4.0,
        acceleration_error=1.5,
        recovery_steps=2,
        disable_after_recoveries=2,
        stable_anchors_for_acceleration=1,
        acceleration_start_progress=0.2,
        allow_acceleration=True,
        require_final_anchor=True,
    )


def test_frozen_phase_schedule_registration_is_preserved_and_detects_runtime_drift():
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))

    assert registration["status"] == "registered_not_collected"
    assert registration["source_collection"]["candidate_request_count"] == 96
    assert registration["source_collection"]["expected_unique_semantic_images"] == 144
    assert registration["terminal_horizon"]["steps"] == [7, 13, 17, 21, 25, 29, 37]
    assert registration["repair_depth"]["consecutive_real_steps"] == [4, 8, 16]
    with pytest.raises(ValueError, match="implementation flux_application file sha256 mismatch"):
        load_registration(REGISTRATION_PATH)


def test_source_collector_uses_neutral_quality_manifest_name(monkeypatch, tmp_path):
    source_path = tmp_path / "collector-quality.json"
    speed_path = tmp_path / "speed.json"
    source_path.write_text("{}", encoding="utf-8")
    speed_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(source_collector, "_parse_args", lambda argv: object())
    monkeypatch.setattr(
        source_collector,
        "collect",
        lambda args: (source_path, speed_path),
    )

    assert source_collector.main([]) == 0
    assert not source_path.exists()
    assert (tmp_path / "quality-input.json").is_file()


def test_registration_rejects_post_registration_grid_change(tmp_path):
    document = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    for name, relative_path in registration_module.IMPLEMENTATION_PATHS.items():
        document["implementation"][name]["file_sha256"] = (
            registration_module.sha256_file(ROOT / relative_path)
        )
    document["terminal_horizon"]["steps"] = [7, 13, 21, 29, 37]
    payload = {key: value for key, value in document.items() if key != "sha256"}
    document["sha256"] = canonical_sha256(payload)
    changed = tmp_path / "changed-registration.json"
    changed.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal horizon rules"):
        load_registration(changed)


def test_registered_terminal_policy_disables_at_exact_step():
    policy = RegisteredTerminalPolicy(_adaptive_config(), terminal_step=13)
    history = CacheHistory(2)
    observation = RuntimeObservation()

    policy.should_skip(CacheStepContext(13, 50), history, observation)

    assert policy.state == "disabled"
    policy.validate_complete()
    assert policy.stats()["registered_terminal_step"] == 13


def test_registered_terminal_policy_handles_recovery_guard_overlap():
    policy = RegisteredTerminalPolicy(_adaptive_config(), terminal_step=13)

    policy.observe_anchor(CacheStepContext(13, 50), None, None, None)

    assert policy.state == "disabled"
    policy.validate_complete()


def test_registered_repair_burst_records_guard_forced_real_steps():
    policy = RegisteredRepairBurstPolicy(_adaptive_config(), start_step=13, real_steps=4)

    for step in range(13, 17):
        policy.observe_anchor(CacheStepContext(step, 50), None, None, None)

    policy.validate_complete()
    assert policy.stats()["registered_repair_applied_steps"] == [13, 14, 15, 16]


def _source_registration() -> dict:
    return {
        "study_id": "study",
        "sha256": "registration-digest",
        "quality_contract": {
            "margins": {"image_reward": 0.7824214100837708, "vqa_score": 0.25}
        },
        "source_profiles": [
            {"candidate_id": "profile-a"},
            {"candidate_id": "profile-b"},
        ],
        "source_selection": {
            "minimum_source_failures": 6,
            "maximum_source_failures": 6,
        },
    }


def _source_rows(failure_count: int) -> tuple[dict, dict]:
    semantic_rows = []
    quality_rows = []
    for index in range(12):
        sample_id = f"p{index:03d}-s2"
        failed = index < failure_count
        delta = {
            "image_reward": 0.0,
            "vqa_score": -0.5 if failed else 0.0,
        }
        common = {
            "candidate_id": "profile-a",
            "sample_id": sample_id,
            "prompt_index": index,
            "seed": 2,
            "prompt": f"prompt {index}",
        }
        semantic_rows.append({**common, "candidate_minus_baseline": delta})
        quality_rows.append(
            {
                **common,
                "baseline": {"image": f"baseline/{sample_id}.png"},
                "candidate": {
                    "image": f"candidate/{sample_id}.png",
                    "trajectory": f"candidate/{sample_id}.pt",
                },
            }
        )
    return {"comparisons": semantic_rows}, {"comparisons": quality_rows}


def test_selection_uses_six_failures_and_exact_profile_category_controls(
    monkeypatch, tmp_path
):
    registration = _source_registration()
    semantic, quality = _source_rows(6)
    registration_path = tmp_path / "registration.json"
    semantic_path = tmp_path / "semantic.json"
    quality_path = tmp_path / "quality.json"
    for path in (registration_path, semantic_path, quality_path):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon.load_registration",
        lambda path: registration,
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon._validate_source_evidence",
        lambda registered, path: (semantic, quality, quality_path),
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon._prompt_categories",
        lambda registered: {index: "counting" for index in range(12)},
    )

    selection = build_selection(registration_path, semantic_path)

    assert selection["status"] == "ready_for_terminal_horizon"
    assert len(selection["selected_failures"]) == 6
    assert len(selection["selected_controls"]) == 6
    assert {row["sample_id"] for row in selection["selected_failures"]}.isdisjoint(
        row["sample_id"] for row in selection["selected_controls"]
    )


def test_selection_stops_when_source_failure_count_is_below_six(monkeypatch, tmp_path):
    registration = _source_registration()
    semantic, quality = _source_rows(5)
    registration_path = tmp_path / "registration.json"
    semantic_path = tmp_path / "semantic.json"
    quality_path = tmp_path / "quality.json"
    for path in (registration_path, semantic_path, quality_path):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon.load_registration",
        lambda path: registration,
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon._validate_source_evidence",
        lambda registered, path: (semantic, quality, quality_path),
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon._prompt_categories",
        lambda registered: {index: "counting" for index in range(12)},
    )

    selection = build_selection(registration_path, semantic_path)

    assert selection["status"] == "insufficient_source_failures"
    assert selection["source_failure_count"] == 5
    assert selection["selected_controls"] == []


def test_terminal_analysis_applies_registered_full_and_dead_rules(monkeypatch, tmp_path):
    steps = [7, 13, 17, 21, 25, 29, 37]
    registration = {
        "study_id": "study",
        "sha256": "registration-digest",
        "quality_contract": {
            "margins": {"image_reward": 0.7824214100837708, "vqa_score": 0.25}
        },
        "terminal_horizon": {"steps": steps},
    }
    failures = [
        {
            "evaluation_id": f"profile-0::p{index:03d}-s2",
            "candidate_id": "profile-a",
        }
        for index in range(6)
    ]
    controls = [
        {
            "evaluation_id": f"profile-0::p{index + 10:03d}-s2",
            "candidate_id": "profile-a",
        }
        for index in range(6)
    ]
    selection = {
        "status": "ready_for_terminal_horizon",
        "sha256": "selection-digest",
        "selected_failures": failures,
        "selected_controls": controls,
    }
    comparisons = []
    for step in steps:
        for unit in failures:
            comparisons.append(
                {
                    "sample_id": _unit_id(unit),
                    "candidate_id": _intervention_id(
                        "terminal", unit["candidate_id"], step
                    ),
                    "candidate_minus_baseline": {
                        "image_reward": 0.0,
                        "vqa_score": -0.1 if step <= 21 else -0.5,
                    },
                }
            )
        for unit in controls:
            comparisons.append(
                {
                    "sample_id": _unit_id(unit),
                    "candidate_id": _intervention_id(
                        "terminal", unit["candidate_id"], step
                    ),
                    "candidate_minus_baseline": {
                        "image_reward": 0.0,
                        "vqa_score": 0.0,
                    },
                }
            )
    registration_path = tmp_path / "registration.json"
    selection_path = tmp_path / "selection.json"
    semantic_path = tmp_path / "semantic.json"
    for path in (registration_path, selection_path, semantic_path):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon.load_registration",
        lambda path: registration,
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon.load_selection",
        lambda path, registration_path: selection,
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon._validate_intervention_semantic",
        lambda path, registration, split: {"comparisons": comparisons},
    )

    result = evaluate_terminal(registration_path, selection_path, semantic_path)

    assert result["status"] == "usable"
    assert result["t_full_observed"] == 21
    assert result["t_dead_observed"] == 25


def test_repair_analysis_reports_registered_depth_grid(monkeypatch, tmp_path):
    registration = {
        "study_id": "study",
        "sha256": "registration-digest",
        "quality_contract": {
            "margins": {"image_reward": 0.7824214100837708, "vqa_score": 0.25}
        },
        "repair_depth": {
            "start_steps": [13, 21],
            "consecutive_real_steps": [4, 8, 16],
        },
    }
    failures = [
        {
            "evaluation_id": f"profile-0::p{index:03d}-s2",
            "candidate_id": "profile-a",
        }
        for index in range(6)
    ]
    controls = [
        {
            "evaluation_id": f"profile-0::p{index + 10:03d}-s2",
            "candidate_id": "profile-a",
        }
        for index in range(6)
    ]
    selection = {
        "status": "ready_for_terminal_horizon",
        "sha256": "selection-digest",
        "selected_failures": failures,
        "selected_controls": controls,
    }
    comparisons = []
    for start in (13, 21):
        for real_steps in (4, 8, 16):
            for unit in failures:
                comparisons.append(
                    {
                        "sample_id": _unit_id(unit),
                        "candidate_id": _intervention_id(
                            "repair", unit["candidate_id"], start, real_steps
                        ),
                        "candidate_minus_baseline": {
                            "image_reward": 0.0,
                            "vqa_score": -0.1 if (start, real_steps) == (13, 4) else -0.5,
                        },
                    }
                )
            for unit in controls:
                comparisons.append(
                    {
                        "sample_id": _unit_id(unit),
                        "candidate_id": _intervention_id(
                            "repair", unit["candidate_id"], start, real_steps
                        ),
                        "candidate_minus_baseline": {
                            "image_reward": 0.0,
                            "vqa_score": 0.0,
                        },
                    }
                )
    registration_path = tmp_path / "registration.json"
    selection_path = tmp_path / "selection.json"
    terminal_path = tmp_path / "terminal.json"
    semantic_path = tmp_path / "semantic.json"
    for path in (registration_path, selection_path, terminal_path, semantic_path):
        path.write_text("{}", encoding="utf-8")
    from scripts.flux_cache_phase_schedule_registration import sha256_file

    terminal = {
        "status": "usable",
        "sha256": "terminal-digest",
        "selection": {"file_sha256": sha256_file(selection_path)},
        "t_full_observed": 21,
        "t_dead_observed": 37,
        "terminal_curve": [],
    }
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon.load_registration",
        lambda path: registration,
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon.load_selection",
        lambda path, registration_path: selection,
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon._load_hashed",
        lambda *args, **kwargs: terminal,
    )
    monkeypatch.setattr(
        "scripts.flux_cache_phase_schedule_horizon._validate_intervention_semantic",
        lambda path, registration, split: {"comparisons": comparisons},
    )

    result = evaluate_repair(
        registration_path,
        selection_path,
        terminal_path,
        semantic_path,
    )

    assert result["status"] == "usable_for_candidate_generation"
    assert len(result["repair_depth_curve"]) == 6
    first = result["repair_depth_curve"][0]
    assert (first["start_step"], first["consecutive_real_steps"]) == (13, 4)
    assert first["D_rescue"] == 1.0
    assert first["D_introduce"] == 0.0
