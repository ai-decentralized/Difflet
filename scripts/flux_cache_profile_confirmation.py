#!/usr/bin/env python3
"""Validate and evaluate the methodology-v1 brake-only confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_quality_contract import (
    load_semantic_report,
    semantic_source,
    validate_generation_identity,
)
from scripts.collect_flux_cache_ab import load_adaptive_candidate
from scripts.flux_cache_offline_gate import (
    load_methodology,
    one_sided_binomial_upper_bound,
)
from scripts.flux_cache_protocol import canonical_sha256, load_prompt_suite


REGISTRATION_SCHEMA = "difflet-flux-cache-profile-confirmation-registration"
REGISTRATION_SCHEMA_REVISION = 1
EVALUATION_SCHEMA = "difflet-flux-cache-profile-confirmation-evaluation"
EVALUATION_SCHEMA_REVISION = 1


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
    if not {"path", "file_sha256"} <= set(binding):
        raise ValueError(f"{name} file binding is incomplete")
    path = _repo_artifact(str(binding["path"]), name)
    observed = sha256_file(path)
    if observed != binding["file_sha256"]:
        raise ValueError(
            f"{name} file sha256 mismatch: expected {binding['file_sha256']}, got {observed}"
        )
    return path


def _extract_registered_prompts(path: Path) -> set[str]:
    document = _load_json(path, "prior prompt source")
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
                if isinstance(row, dict) and isinstance(row.get("text"), str):
                    prompts.add(row["text"])
    return prompts


def _validate_metric_identity(metrics: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if set(metrics) != {"image_reward", "vqa_score"}:
        raise ValueError("semantic report must contain only ImageReward and VQAScore")
    image_reward = metrics["image_reward"]
    expected_image_reward = expected["image_reward"]
    for field in ("implementation", "package", "package_version", "model", "dtype", "preprocessing"):
        if image_reward.get(field) != expected_image_reward[field]:
            raise ValueError(f"ImageReward {field} differs from the registration")
    observed_files = {
        str(row["path"]): row["sha256"]
        for row in image_reward.get("checkpoint_files", [])
        if isinstance(row, dict) and "path" in row and "sha256" in row
    }
    if any(
        not any(path.endswith(suffix) and observed == digest for path, observed in observed_files.items())
        for suffix, digest in expected_image_reward["files_by_suffix"].items()
    ):
        raise ValueError("ImageReward checkpoint files differ from the registration")

    vqa = metrics["vqa_score"]
    expected_vqa = expected["vqa_score"]
    for field in (
        "implementation",
        "package",
        "package_version",
        "model",
        "dtype",
        "question_template",
        "answer_template",
    ):
        if vqa.get(field) != expected_vqa[field]:
            raise ValueError(f"VQAScore {field} differs from the registration")
    observed_revisions = {
        row.get("repository"): row.get("resolved_revision")
        for row in vqa.get("repositories", [])
        if isinstance(row, dict)
    }
    if observed_revisions != expected_vqa["repository_revisions"]:
        raise ValueError("VQAScore repository revisions differ from the registration")
    observed_vqa_files = {
        str(row["path"]): row["sha256"]
        for row in vqa.get("checkpoint_files", [])
        if isinstance(row, dict) and "path" in row and "sha256" in row
    }
    if any(
        not any(
            path.endswith(suffix) and observed == digest
            for path, observed in observed_vqa_files.items()
        )
        for suffix, digest in expected_vqa["files_by_suffix"].items()
    ):
        raise ValueError("VQAScore checkpoint files differ from the registration")


def load_registration(path: Path) -> dict[str, Any]:
    """Validate every frozen input without opening confirmation outcomes."""

    document = _load_json(path, "profile confirmation registration")
    expected_fields = {
        "schema",
        "schema_revision",
        "study_id",
        "created_at",
        "status",
        "evidence_role",
        "methodology",
        "selection_provenance",
        "quality_contract",
        "controlled_generation",
        "semantic_metrics",
        "prompt_suite",
        "novelty_audit",
        "candidate",
        "implementation",
        "statistical_gate",
        "collection",
        "confirmation_claim",
        "parameters_frozen_before_collection",
        "sha256",
    }
    if set(document) != expected_fields:
        raise ValueError("profile confirmation registration fields do not match the protocol")
    if (
        document["schema"] != REGISTRATION_SCHEMA
        or document["schema_revision"] != REGISTRATION_SCHEMA_REVISION
    ):
        raise ValueError("profile confirmation registration schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != document["sha256"]:
        raise ValueError("profile confirmation registration sha256 does not match its contents")
    if document["status"] != "registered_not_collected":
        raise ValueError("profile confirmation registration is no longer in its frozen initial state")
    if document["parameters_frozen_before_collection"] is not True:
        raise ValueError("profile confirmation parameters were not frozen before collection")
    if document["evidence_role"] != {
        "stage": "prospective_profile_confirmation",
        "profile_selection_or_refit_permitted": False,
        "serving_claim_permitted": False,
    }:
        raise ValueError("profile confirmation evidence role is invalid")

    methodology_binding = document["methodology"]
    methodology_path = _check_file_binding(methodology_binding, "methodology")
    methodology = load_methodology(methodology_path)
    if methodology.get("methodology_id") != methodology_binding.get("methodology_id"):
        raise ValueError("methodology id differs from the registration")
    methodology_contract = methodology["contract"]
    registered_contract = document["quality_contract"]
    if registered_contract != {
        "paired_loss": "baseline_score_minus_candidate_score",
        "comparison": "strictly_greater_than_metric_margin",
        "combination": "fail_if_either_metric_fails",
        "margins": {
            "image_reward": methodology_contract["image_reward_max_harm"],
            "vqa_score": methodology_contract["vqa_score_max_harm"],
        },
        "weighted_metric_average_forbidden": True,
    }:
        raise ValueError("quality contract differs from methodology-v1")

    selection_binding = document["selection_provenance"]
    selection_path = _check_file_binding(selection_binding, "selection provenance")
    selection = _load_json(selection_path, "selection provenance")
    if (
        selection.get("status") != selection_binding.get("status")
        or selection.get("selected_profile", {}).get("candidate_id")
        != selection_binding.get("candidate_id")
    ):
        raise ValueError("selection provenance differs from the registration")

    prompt_binding = document["prompt_suite"]
    prompt_path = _check_file_binding(prompt_binding, "confirmation prompt suite")
    prompt_selection = load_prompt_suite(prompt_path, prompt_binding["split"])
    if (
        prompt_selection.descriptor["sha256"] != prompt_binding["split_sha256"]
        or len(prompt_selection.prompts) != prompt_binding["prompt_count"]
        or prompt_binding["sample_count"]
        != prompt_binding["prompt_count"] * len(prompt_binding["seeds"])
        or len(set(prompt_binding["seeds"])) != len(prompt_binding["seeds"])
    ):
        raise ValueError("confirmation prompt/seed matrix differs from the registration")

    novelty = document["novelty_audit"]
    prior_prompts: set[str] = set()
    for binding in novelty["prior_prompt_sources"]:
        prior_prompts.update(_extract_registered_prompts(_check_file_binding(binding, "prior prompt source")))
    overlap = set(prompt_selection.prompts) & prior_prompts
    if (
        len(prior_prompts) != novelty["prior_unique_prompt_count"]
        or len(overlap) != novelty["exact_overlap_count"]
        or overlap
    ):
        raise ValueError("confirmation prompts are not novel under the registered audit")

    candidate_binding = document["candidate"]
    candidate_path = _check_file_binding(candidate_binding, "brake-only candidate")
    candidate_document = _load_json(candidate_path, "brake-only candidate")
    candidate = load_adaptive_candidate(candidate_path)
    if (
        candidate_document.get("sha256") != candidate_binding["content_sha256"]
        or candidate.candidate_id != candidate_binding["candidate_id"]
        or candidate_document.get("policy") != candidate_binding["policy"]
        or candidate_document.get("predictor") != candidate_binding["predictor"]
        or candidate_binding["policy"].get("allow_acceleration") is not False
    ):
        raise ValueError("brake-only candidate differs from the registration")
    if selection.get("selected_profile", {}).get("candidate_content_sha256") != candidate_binding[
        "content_sha256"
    ]:
        raise ValueError("selection provenance did not select the registered candidate")

    for name, binding in document["implementation"].items():
        _check_file_binding(binding, f"implementation.{name}")

    controlled = document["controlled_generation"]
    expected_controlled = {
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
    }
    if controlled != expected_controlled:
        raise ValueError("controlled generation identity is unsupported")

    gate = document["statistical_gate"]
    if gate.get("independent_unit") != "prompt_index" or gate.get("required_failures") != 0:
        raise ValueError("confirmation gate must require zero failing prompt groups")
    group_count = prompt_binding["prompt_count"]
    upper_bound = one_sided_binomial_upper_bound(
        0,
        group_count,
        confidence=float(gate["confidence"]),
    )
    if (
        int(gate["independent_group_count"]) != group_count
        or group_count < int(methodology["profile_selection"]["minimum_independent_group_count"])
        or float(gate["confidence"]) != float(methodology["profile_selection"]["confidence"])
        or float(gate["maximum_failure_rate_upper_bound"])
        != float(methodology["profile_selection"]["maximum_failure_rate_upper_bound"])
        or not math.isclose(upper_bound, float(gate["required_upper_bound"]), abs_tol=1e-15)
        or upper_bound > float(gate["maximum_failure_rate_upper_bound"])
        or gate.get("stopping_rule") != "collect_and_score_all_registered_samples"
    ):
        raise ValueError("confirmation statistical gate differs from methodology-v1")
    return document


def evaluate_confirmation(
    registration_path: Path,
    semantic_report_path: Path,
) -> dict[str, Any]:
    registration = load_registration(registration_path)
    report = load_semantic_report(semantic_report_path)
    _validate_metric_identity(report["metrics"], registration["semantic_metrics"])
    manifest, observed_selection, manifest_path = semantic_source(report)
    prompt_binding = registration["prompt_suite"]
    if (
        observed_selection.get("split") != prompt_binding["split"]
        or observed_selection.get("sha256") != prompt_binding["split_sha256"]
        or list(manifest["protocol"]["rng"]["seeds"]) != prompt_binding["seeds"]
    ):
        raise ValueError("confirmation evidence uses a different prompt or seed set")
    validate_generation_identity(manifest, registration["controlled_generation"])
    candidate_binding = registration["candidate"]
    candidate_path = _repo_artifact(candidate_binding["path"], "brake-only candidate")
    candidate = load_adaptive_candidate(candidate_path)
    if manifest.get("candidates") != [
        {
            "candidate_id": candidate_binding["candidate_id"],
            "policy": candidate.policy_spec(),
            "predictor": candidate.predictor_spec(),
        }
    ]:
        raise ValueError("confirmation evidence does not contain only the frozen candidate")

    comparisons = report["comparisons"]
    if len(comparisons) != prompt_binding["sample_count"]:
        raise ValueError("confirmation semantic report has the wrong sample count")
    margins = registration["quality_contract"]["margins"]
    rows: list[dict[str, Any]] = []
    observed_prompt_indices: set[int] = set()
    for comparison in comparisons:
        if comparison["candidate_id"] != candidate_binding["candidate_id"]:
            raise ValueError("confirmation report contains an unregistered candidate")
        prompt_index = int(comparison["prompt_index"])
        if prompt_index in observed_prompt_indices:
            raise ValueError("confirmation report repeats a prompt group")
        observed_prompt_indices.add(prompt_index)
        delta = comparison["candidate_minus_baseline"]
        harms = {metric: -float(delta[metric]) for metric in margins}
        metric_failures = {
            metric: harms[metric] > float(margin)
            for metric, margin in margins.items()
        }
        rows.append(
            {
                "prompt_index": prompt_index,
                "sample_id": comparison["sample_id"],
                "seed": int(comparison["seed"]),
                "harms": harms,
                "metric_failures": metric_failures,
                "failed": any(metric_failures.values()),
            }
        )
    if observed_prompt_indices != set(range(prompt_binding["prompt_count"])):
        raise ValueError("confirmation report does not cover the registered prompt groups")

    failure_count = sum(row["failed"] for row in rows)
    gate = registration["statistical_gate"]
    upper_bound = one_sided_binomial_upper_bound(
        failure_count,
        prompt_binding["prompt_count"],
        confidence=float(gate["confidence"]),
    )
    passed = (
        failure_count == gate["required_failures"]
        and upper_bound <= float(gate["maximum_failure_rate_upper_bound"])
    )
    payload = {
        "schema": EVALUATION_SCHEMA,
        "schema_revision": EVALUATION_SCHEMA_REVISION,
        "registration": {
            "path": str(registration_path),
            "file_sha256": sha256_file(registration_path),
            "content_sha256": registration["sha256"],
        },
        "semantic_report": {
            "path": str(semantic_report_path),
            "file_sha256": sha256_file(semantic_report_path),
        },
        "quality_manifest": {
            "path": str(manifest_path),
            "file_sha256": sha256_file(manifest_path),
        },
        "candidate_id": candidate_binding["candidate_id"],
        "independent_group_count": prompt_binding["prompt_count"],
        "failure_count": failure_count,
        "failure_rate_upper_bound": upper_bound,
        "passes_registered_confirmation": passed,
        "serving_qualified": False,
        "next_stage": (
            "prospective_interventional_quality_and_speed_holdout"
            if passed
            else "reject_or_redesign_profile_without_retuning_this_holdout"
        ),
        "rows": sorted(rows, key=lambda row: row["prompt_index"]),
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate frozen preregistration")
    validate.add_argument("--registration", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate", help="score completed confirmation evidence")
    evaluate.add_argument("--registration", required=True, type=Path)
    evaluate.add_argument("--semantic-report", required=True, type=Path)
    evaluate.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        registration_path = args.registration.expanduser().resolve()
        if args.command == "validate":
            registration = load_registration(registration_path)
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "study_id": registration["study_id"],
                        "candidate_id": registration["candidate"]["candidate_id"],
                        "prompt_count": registration["prompt_suite"]["prompt_count"],
                        "required_failures": registration["statistical_gate"]["required_failures"],
                        "required_upper_bound": registration["statistical_gate"]["required_upper_bound"],
                        "registration_sha256": registration["sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        out = args.out.expanduser().resolve()
        if out.exists():
            raise ValueError(f"output already exists: {out}")
        evaluation = evaluate_confirmation(
            registration_path,
            args.semantic_report.expanduser().resolve(),
        )
        _write_json(out, evaluation)
        print(f"[profile-confirmation] {evaluation['passes_registered_confirmation']=} -> {out}")
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
