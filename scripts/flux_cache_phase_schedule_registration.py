#!/usr/bin/env python3
"""Validate the frozen FLUX phase-schedule horizon registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_flux_cache_ab import load_adaptive_candidate  # noqa: E402
from scripts.flux_cache_offline_gate import load_methodology  # noqa: E402
from scripts.flux_cache_protocol import (  # noqa: E402
    canonical_sha256,
    load_prompt_suite,
)

REGISTRATION_SCHEMA = "difflet-flux-cache-phase-schedule-horizon-registration"
REGISTRATION_SCHEMA_REVISION = 1

IMPLEMENTATION_PATHS = {
    "base_collector": "scripts/collect_flux_cache_ab.py",
    "source_collector": "scripts/collect_flux_cache_phase_schedule_source.py",
    "semantic_scorer": "scripts/evaluate_flux_cache_semantics.py",
    "registration_validator": "scripts/flux_cache_phase_schedule_registration.py",
    "intervention_tool": "scripts/flux_cache_phase_schedule_horizon.py",
    "flux_application": "difflet/models/flux/application.py",
    "flux_pipeline": "difflet/models/flux/pipeline.py",
    "cache_public_api": "difflet/pipeline/cache/__init__.py",
    "cache_policy": "difflet/pipeline/cache/policies.py",
    "cache_predictor": "difflet/pipeline/cache/predictors.py",
    "cache_recovery": "difflet/pipeline/cache/recovery.py",
    "cache_runner": "difflet/pipeline/cache/runner.py",
    "cache_session": "difflet/pipeline/cache/session.py",
    "teacache_adapter": "difflet/pipeline/cache/teacache_adapter.py",
    "cache_measurements": "difflet/pipeline/cache/measurements.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _repo_artifact(relative_path: str, name: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"{name} path must be repository-relative")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError(f"{name} path is invalid: {relative_path!r}")
    return resolved


def _check_file_binding(binding: Mapping[str, Any], name: str) -> Path:
    if not {"path", "file_sha256"}.issubset(binding):
        raise ValueError(f"{name} file binding is incomplete")
    path = _repo_artifact(str(binding["path"]), name)
    observed = sha256_file(path)
    if observed != binding["file_sha256"]:
        raise ValueError(
            f"{name} file sha256 mismatch: expected {binding['file_sha256']}, "
            f"got {observed}"
        )
    return path


def _extract_prompts(path: Path) -> set[str]:
    document = _load_json(path, "prompt source")
    prompts: set[str] = set()
    rows = document.get("prompts")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, str):
                prompts.add(row)
            elif isinstance(row, dict):
                value = row.get("prompt", row.get("text"))
                if isinstance(value, str):
                    prompts.add(value)
    splits = document.get("splits")
    if isinstance(splits, dict):
        for split_rows in splits.values():
            if not isinstance(split_rows, list):
                continue
            for row in split_rows:
                if isinstance(row, str):
                    prompts.add(row)
                elif isinstance(row, dict):
                    value = row.get("prompt", row.get("text"))
                    if isinstance(value, str):
                        prompts.add(value)
    return prompts


def _strict_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_quality_contract(
    registered: Mapping[str, Any], methodology: Mapping[str, Any]
) -> None:
    source = methodology["contract"]
    expected = {
        "paired_loss": "baseline_score_minus_candidate_score",
        "comparison": "strictly_greater_than_metric_margin",
        "combination": "fail_if_either_metric_fails",
        "margins": {
            "image_reward": source["image_reward_max_harm"],
            "vqa_score": source["vqa_score_max_harm"],
        },
        "weighted_metric_average_forbidden": True,
    }
    if registered != expected:
        raise ValueError("phase-schedule quality contract differs from methodology")


def _validate_profiles(rows: Any, request_count: int) -> tuple[str, ...]:
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("phase-schedule registration requires exactly two source profiles")
    candidate_ids: list[str] = []
    for index, binding in enumerate(rows):
        name = f"source_profiles[{index}]"
        if not isinstance(binding, dict) or set(binding) != {
            "path",
            "file_sha256",
            "content_sha256",
            "candidate_id",
            "registered_request_count",
        }:
            raise ValueError(f"{name} fields do not match the protocol")
        path = _check_file_binding(binding, name)
        arm = load_adaptive_candidate(path)
        document = _load_json(path, name)
        if (
            arm.candidate_id != binding["candidate_id"]
            or document["sha256"] != binding["content_sha256"]
        ):
            raise ValueError(f"{name} candidate identity differs from the registration")
        if binding["registered_request_count"] != request_count:
            raise ValueError(f"{name} request count differs from the prompt matrix")
        candidate_ids.append(arm.candidate_id)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("source profile candidate ids must be unique")
    return tuple(candidate_ids)


def _validate_interventions(document: Mapping[str, Any]) -> None:
    selection = document["source_selection"]
    expected_selection = {
        "source_failure_metric": "vqa_score",
        "source_failure_comparison": "baseline_minus_cache_strictly_greater",
        "minimum_source_failures": 6,
        "maximum_source_failures": 6,
        "failure_order": ["sample_id", "registered_profile_order"],
        "control_pass_rule": "both_metric_harms_are_within_contract",
        "control_match_keys": ["registered_profile", "semantic_category"],
        "control_order": ["sample_id"],
        "selected_failure_sample_ids_excluded_from_controls": True,
        "insufficient_exact_controls_action": "stop_insufficient_matched_controls",
    }
    if selection != expected_selection:
        raise ValueError("source selection rules differ from the frozen protocol")

    terminal = document["terminal_horizon"]
    if terminal != {
        "steps": [7, 13, 17, 21, 25, 29, 37],
        "prefix_identity_gate": "bit_exact_latent_at_step_before_intervention",
        "action": "disable_cache_before_registered_step_and_compute_all_remaining_steps",
        "rescue_rule": "baseline_vqa_minus_intervention_vqa_is_not_greater_than_margin",
        "introduced_failure_rule": "either_metric_harm_is_strictly_greater_than_margin",
        "t_full_observed": "maximum_tested_step_where_R_equals_1_and_I_equals_0",
        "t_dead_observed": "earliest_tested_step_where_R_is_at_most_0.2_and_all_later_R_are_at_most_0.2",
    }:
        raise ValueError("terminal horizon rules differ from the frozen protocol")

    repair = document["repair_depth"]
    if repair != {
        "start_steps": [13, 21],
        "consecutive_real_steps": [4, 8, 16],
        "prefix_identity_gate": "bit_exact_latent_at_step_before_intervention",
        "action": "force_k_consecutive_real_steps_then_resume_the_registered_source_policy",
        "rescue_rule": "baseline_vqa_minus_intervention_vqa_is_not_greater_than_margin",
        "introduced_failure_rule": "either_metric_harm_is_strictly_greater_than_margin",
        "run_only_when_terminal_horizon_status_is_usable": True,
    }:
        raise ValueError("repair-depth rules differ from the frozen protocol")

    stopping = document["stopping_rules"]
    if stopping != {
        "collect_all_source_requests_before_scoring": True,
        "source_failure_count_below_6": "insufficient_source_failures",
        "t_full_observed_null": "no_observed_full_rescue_window",
        "t_dead_observed_null": "no_observed_tail_relaxation_window",
        "t_dead_not_after_t_full": "invalid_horizon_order",
        "otherwise": "run_registered_repair_depth_grid",
        "prompt_extension_permitted": False,
        "profile_substitution_permitted": False,
        "grid_retuning_permitted": False,
    }:
        raise ValueError("phase-schedule stopping rules differ from the frozen protocol")


def load_registration(path: Path) -> dict[str, Any]:
    """Load and validate every frozen input without opening experiment outcomes."""

    document = _load_json(path, "phase-schedule horizon registration")
    expected_fields = {
        "schema",
        "schema_revision",
        "study_id",
        "created_at",
        "status",
        "evidence_role",
        "methodology",
        "quality_contract",
        "controlled_generation",
        "metric_identity_source",
        "semantic_metrics",
        "prompt_suite",
        "novelty_audit",
        "source_profiles",
        "implementation",
        "source_collection",
        "registered_workflow",
        "source_selection",
        "terminal_horizon",
        "repair_depth",
        "stopping_rules",
        "registered_outputs",
        "parameters_frozen_before_collection",
        "sha256",
    }
    if set(document) != expected_fields:
        raise ValueError("phase-schedule registration fields do not match the protocol")
    if (
        document["schema"] != REGISTRATION_SCHEMA
        or document["schema_revision"] != REGISTRATION_SCHEMA_REVISION
    ):
        raise ValueError("phase-schedule registration schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != document["sha256"]:
        raise ValueError("phase-schedule registration sha256 does not match its contents")
    if document["status"] != "registered_not_collected":
        raise ValueError("phase-schedule registration is not in its frozen initial state")
    if document["parameters_frozen_before_collection"] is not True:
        raise ValueError("phase-schedule parameters were not frozen before collection")
    if document["evidence_role"] != {
        "stage": "phase_schedule_horizon_and_repair_depth_development",
        "serving_claim_permitted": False,
        "profile_selection_permitted": False,
        "endpoint_labels_opened_before_registration": False,
    }:
        raise ValueError("phase-schedule evidence role is invalid")

    methodology_binding = document["methodology"]
    methodology_path = _check_file_binding(methodology_binding, "methodology")
    methodology = load_methodology(methodology_path)
    if methodology.get("methodology_id") != methodology_binding.get("methodology_id"):
        raise ValueError("methodology id differs from the registration")
    _validate_quality_contract(document["quality_contract"], methodology)

    metric_binding = document["metric_identity_source"]
    metric_path = _check_file_binding(metric_binding, "metric identity source")
    metric_source = _load_json(metric_path, "metric identity source")
    if metric_binding.get("json_pointer") != "/semantic_metrics":
        raise ValueError("metric identity source pointer is unsupported")
    if metric_source.get("semantic_metrics") != document["semantic_metrics"]:
        raise ValueError("semantic metric identity differs from its frozen source")

    prompt_binding = document["prompt_suite"]
    prompt_path = _check_file_binding(prompt_binding, "phase-schedule prompt suite")
    prompt_selection = load_prompt_suite(prompt_path, prompt_binding["split"])
    seeds = prompt_binding["seeds"]
    if (
        prompt_selection.descriptor["sha256"] != prompt_binding["split_sha256"]
        or len(prompt_selection.prompts) != prompt_binding["prompt_count"]
        or not isinstance(seeds, list)
        or len(seeds) != len(set(seeds))
        or any(_strict_integer(seed, "prompt seed") < 0 for seed in seeds)
        or prompt_binding["sample_count"]
        != prompt_binding["prompt_count"] * len(seeds)
    ):
        raise ValueError("phase-schedule prompt/seed matrix differs from registration")
    if prompt_binding["prompt_count"] != 48 or seeds != [2]:
        raise ValueError("phase-schedule development matrix must contain 48 seed-2 prompts")

    novelty = document["novelty_audit"]
    prior: set[str] = set()
    for index, source in enumerate(novelty["prior_prompt_sources"]):
        source_path = _check_file_binding(source, f"novelty source {index}")
        prior.update(_extract_prompts(source_path))
    overlap = sorted(set(prompt_selection.prompts) & prior)
    if (
        novelty.get("method")
        != "exact prompt-text comparison against every previously registered prompt JSON"
        or novelty.get("exact_overlap_count") != len(overlap)
        or novelty.get("exact_overlap_prompts") != overlap
        or novelty.get("prior_unique_prompt_count") != len(prior)
        or overlap
    ):
        raise ValueError("phase-schedule prompt novelty audit does not reproduce")

    sample_count = prompt_binding["sample_count"]
    candidate_ids = _validate_profiles(document["source_profiles"], sample_count)
    source_collection = document["source_collection"]
    if (
        source_collection["prompt_count"] != prompt_binding["prompt_count"]
        or source_collection["sample_count"] != sample_count
        or source_collection["candidate_request_count"]
        != sample_count * len(candidate_ids)
        or source_collection["expected_unique_semantic_images"]
        != sample_count * (1 + len(candidate_ids))
        or source_collection["candidate_order"] != list(candidate_ids)
        or source_collection["pipeline_warmup_enabled"] is not True
        or source_collection["early_stopping_permitted"] is not False
        or source_collection["retuning_after_collection_starts_permitted"] is not False
    ):
        raise ValueError("source collection counts or candidate order are invalid")

    implementation = document["implementation"]
    if set(implementation) != set(IMPLEMENTATION_PATHS):
        raise ValueError("phase-schedule implementation bindings are incomplete")
    for name, binding in implementation.items():
        if binding.get("path") != IMPLEMENTATION_PATHS[name]:
            raise ValueError(f"implementation {name} path differs from the protocol")
        _check_file_binding(binding, f"implementation {name}")

    workflow = document["registered_workflow"]
    required_commands = {
        "validate_registration_argv",
        "collect_terminal_argv",
        "score_terminal_argv",
        "evaluate_terminal_argv",
        "collect_repair_argv",
        "score_repair_argv",
        "evaluate_repair_argv",
    }
    if set(workflow) != required_commands:
        raise ValueError("registered workflow commands are incomplete")
    if any(
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
        for argv in workflow.values()
    ):
        raise ValueError("registered workflow argv values must be non-empty string lists")

    controlled = document["controlled_generation"]
    if controlled != {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "model_revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "scheduler_config_sha256": "d32e7bf63b561cb6aa9101116b0bd3e4946d17af13a0dd1b674f24ca7dae509b",
        "num_steps": 50,
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "dtype": "bfloat16",
        "tp_degree": 4,
    }:
        raise ValueError("controlled generation identity differs from the registered study")

    _validate_interventions(document)
    outputs = document["registered_outputs"]
    if outputs != {
        "selection_plan": "phase-schedule-selection.json",
        "terminal_quality_input": "terminal-quality-input.json",
        "terminal_analysis": "terminal-analysis.json",
        "repair_quality_input": "repair-depth-quality-input.json",
        "horizon_result": "phase-schedule-horizon.json",
    }:
        raise ValueError("registered output names differ from the protocol")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = load_registration(Path(args.registration).expanduser().resolve())
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(
        f"[phase-schedule-registration] valid study={document['study_id']} "
        f"sha256={document['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
