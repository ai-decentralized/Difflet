#!/usr/bin/env python3
"""Register, validate, and evaluate derived FLUX cache schedule screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_quality_contract import (  # noqa: E402
    load_semantic_report,
    semantic_source,
    validate_generation_identity,
)
from scripts.collect_flux_cache_ab import load_adaptive_candidate  # noqa: E402
from scripts.flux_cache_offline_gate import (  # noqa: E402
    load_methodology,
    one_sided_binomial_upper_bound,
)
from scripts.flux_cache_phased_candidate import (  # noqa: E402
    load_phased_candidate,
)
from scripts.flux_cache_profile_confirmation import (  # noqa: E402
    _extract_registered_prompts,
    _validate_metric_identity,
)
from scripts.flux_cache_protocol import (  # noqa: E402
    canonical_sha256,
    load_prompt_suite,
)

REGISTRATION_SCHEMA = "difflet-flux-cache-derived-schedule-screen-registration"
REGISTRATION_SCHEMA_REVISION = 1
EVALUATION_SCHEMA = "difflet-flux-cache-derived-schedule-screen-evaluation"
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


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _repo_path(value: str, name: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{name} must be repository-relative")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError(f"{name} path is invalid: {value}")
    return resolved


def _binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "file_sha256": sha256_file(resolved),
    }


def _check_binding(binding: Mapping[str, Any], name: str) -> Path:
    path = _repo_path(str(binding.get("path")), name)
    if sha256_file(path) != binding.get("file_sha256"):
        raise ValueError(f"{name} file hash differs")
    return path


def _quality_contract(methodology: Mapping[str, Any]) -> dict[str, Any]:
    contract = methodology["contract"]
    return {
        "paired_loss": "baseline_score_minus_candidate_score",
        "comparison": "strictly_greater_than_metric_margin",
        "combination": "fail_if_either_metric_fails",
        "margins": {
            "image_reward": contract["image_reward_max_harm"],
            "vqa_score": contract["vqa_score_max_harm"],
        },
        "weighted_metric_average_forbidden": True,
    }


def _candidate_bindings(candidate_set_path: Path, legacy_path: Path) -> list[dict[str, Any]]:
    legacy = load_adaptive_candidate(legacy_path)
    rows: list[dict[str, Any]] = [
        {
            **_binding(legacy_path),
            "candidate_id": legacy.candidate_id,
            "family": "legacy_comparator",
            "anchor_budget": None,
            "content_sha256": _load_json(legacy_path, "legacy candidate")["sha256"],
        }
    ]
    candidate_set = _load_json(candidate_set_path, "derived candidate set")
    payload = {key: value for key, value in candidate_set.items() if key != "sha256"}
    if canonical_sha256(payload) != candidate_set.get("sha256"):
        raise ValueError("derived candidate set content hash differs")
    for source in candidate_set.get("candidates", []):
        path = _repo_path(source["path"], "derived candidate")
        arm = load_phased_candidate(path)
        if (
            arm.file_sha256 != source["file_sha256"]
            or arm.content_sha256 != source["content_sha256"]
            or arm.candidate_id != source["candidate_id"]
        ):
            raise ValueError("derived candidate identity differs from its set")
        rows.append(
            {
                **_binding(path),
                "candidate_id": arm.candidate_id,
                "family": source["family"],
                "anchor_budget": int(source["anchor_budget"]),
                "content_sha256": arm.content_sha256,
            }
        )
    if len(rows) != 5:
        raise ValueError("screen requires one legacy and four derived candidates")
    return rows


def register_screen(args: argparse.Namespace) -> None:
    methodology_path = _repo_path(args.methodology, "methodology")
    methodology = load_methodology(methodology_path)
    prompt_path = _repo_path(args.prompt_suite, "prompt suite")
    selection = load_prompt_suite(prompt_path, args.prompt_split)
    if len(selection.prompts) != 32:
        raise ValueError("development screen requires exactly 32 prompts")
    categories: dict[str, int] = {}
    for row in selection.descriptor["prompts"]:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    if len(categories) != 8 or set(categories.values()) != {4}:
        raise ValueError("development screen requires eight categories of four prompts")

    prior_bindings = []
    prior_prompts: set[str] = set()
    for value in args.prior_prompt_source:
        path = _repo_path(value, "prior prompt source")
        prior_bindings.append(_binding(path))
        prior_prompts.update(_extract_registered_prompts(path))
    overlap = sorted(set(selection.prompts) & prior_prompts)
    if overlap:
        raise ValueError(f"development prompts overlap prior prompts: {overlap}")

    metric_source_path = _repo_path(args.metric_source, "metric identity source")
    metric_source = _load_json(metric_source_path, "metric identity source")
    semantic_metrics = metric_source.get("semantic_metrics")
    if not isinstance(semantic_metrics, dict):
        raise ValueError("metric identity source has no semantic metrics")

    candidate_set_path = _repo_path(args.candidate_set, "candidate set")
    legacy_path = _repo_path(args.legacy_candidate, "legacy candidate")
    candidates = _candidate_bindings(candidate_set_path, legacy_path)
    controlled = {
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
    output_directory = str(Path(args.output_directory).expanduser().resolve())
    evaluator_path = Path(__file__).resolve()
    collector_path = ROOT / "scripts" / "collect_flux_cache_schedule_screen.py"
    base_collector_path = ROOT / "scripts" / "collect_flux_cache_ab.py"
    phased_loader_path = ROOT / "scripts" / "flux_cache_phased_candidate.py"
    semantic_scorer_path = ROOT / "scripts" / "evaluate_flux_cache_semantics.py"
    payload = {
        "schema": REGISTRATION_SCHEMA,
        "schema_revision": REGISTRATION_SCHEMA_REVISION,
        "study_id": args.study_id,
        "created_at": args.created_at,
        "status": "registered_not_collected",
        "evidence_role": {
            "stage": "derived_schedule_development_screen",
            "profile_selection_permitted": True,
            "serving_claim_permitted": False,
        },
        "methodology": {
            **_binding(methodology_path),
            "methodology_id": methodology["methodology_id"],
        },
        "candidate_set": {
            **_binding(candidate_set_path),
            "content_sha256": _load_json(candidate_set_path, "candidate set")["sha256"],
        },
        "quality_contract": _quality_contract(methodology),
        "controlled_generation": controlled,
        "semantic_metrics": semantic_metrics,
        "metric_identity_source": _binding(metric_source_path),
        "prompt_suite": {
            **_binding(prompt_path),
            "split": args.prompt_split,
            "split_sha256": selection.descriptor["sha256"],
            "prompt_count": 32,
            "seeds": [3],
            "sample_count": 32,
            "category_counts": categories,
        },
        "novelty_audit": {
            "method": "exact_prompt_text_comparison",
            "prior_unique_prompt_count": len(prior_prompts),
            "exact_overlap_count": 0,
            "prior_prompt_sources": prior_bindings,
        },
        "candidates": candidates,
        "implementation": {
            "screen_evaluator": _binding(evaluator_path),
            "screen_collector": _binding(collector_path),
            "base_collector": _binding(base_collector_path),
            "phased_candidate_loader": _binding(phased_loader_path),
            "semantic_scorer": _binding(semantic_scorer_path),
        },
        "quality_gate": {
            "independent_unit": "prompt_index",
            "independent_group_count": 32,
            "confidence": 0.95,
            "maximum_failure_rate_upper_bound": 0.1,
            "required_failures": 0,
            "required_upper_bound": one_sided_binomial_upper_bound(0, 32, confidence=0.95),
        },
        "speed_gate": {
            "comparator_candidate_id": candidates[0]["candidate_id"],
            "relative_speed_estimand": "sum_legacy_wall_time_divided_by_sum_candidate_wall_time",
            "minimum_relative_speed": 1.05,
            "paired_bootstrap_repetitions": 10000,
            "paired_bootstrap_seed": 20260805,
            "paired_bootstrap_lower_quantile": 0.025,
            "minimum_bootstrap_lower_bound": 1.0,
        },
        "selection_rule": {
            "eligible_families": ["static", "combined"],
            "quality_before_speed": True,
            "within_family": "minimum_total_wall_time_then_candidate_id",
            "maximum_representatives_per_family": 1,
            "no_eligible_representatives_action": "stop_schedule_iteration",
        },
        "collection": {
            "output_directory": output_directory,
            "candidate_order": [row["candidate_id"] for row in candidates],
            "expected_unique_semantic_images": 192,
            "pipeline_warmup_enabled": True,
            "early_stopping_permitted": False,
            "retuning_after_collection_starts_permitted": False,
            "collector_argv": [
                "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python",
                "scripts/collect_flux_cache_schedule_screen.py",
                "--registration",
                "benchmark/flux_cache/derived-schedule-development-screen-registration.json",
                "--allow-hardware",
                "--foreground-ack",
                "I am running the registered derived schedule screen",
            ],
            "semantic_scorer_argv": [
                "/home/ubuntu/.venvs/difflet-cache-eval/bin/python",
                "scripts/evaluate_flux_cache_semantics.py",
                "--quality-input",
                f"{output_directory}/quality-input-v2.json",
                "--out",
                f"{output_directory}/semantic-scores.json",
                "--metrics",
                "image_reward",
                "vqa_score",
                "--expected-images",
                "192",
            ],
            "evaluator_argv": [
                "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python",
                "scripts/flux_cache_schedule_screen.py",
                "evaluate",
                "--registration",
                "benchmark/flux_cache/derived-schedule-development-screen-registration.json",
                "--semantic-report",
                f"{output_directory}/semantic-scores.json",
                "--speed-manifest",
                f"{output_directory}/speedup-candidates-v1.json",
                "--out",
                f"{output_directory}/screen-evaluation.json",
            ],
        },
        "parameters_frozen_before_collection": True,
    }
    _write_json(
        Path(args.out).expanduser().resolve(), {**payload, "sha256": canonical_sha256(payload)}
    )


def load_registration(path: Path) -> dict[str, Any]:
    document = _load_json(path, "derived schedule screen registration")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if (
        document.get("schema") != REGISTRATION_SCHEMA
        or document.get("schema_revision") != REGISTRATION_SCHEMA_REVISION
        or canonical_sha256(payload) != document.get("sha256")
    ):
        raise ValueError("derived schedule screen registration identity is invalid")
    if document.get("status") != "registered_not_collected":
        raise ValueError("derived schedule screen registration status is invalid")
    if document.get("parameters_frozen_before_collection") is not True:
        raise ValueError("screen parameters were not frozen before collection")
    methodology_path = _check_binding(document["methodology"], "methodology")
    methodology = load_methodology(methodology_path)
    if document["quality_contract"] != _quality_contract(methodology):
        raise ValueError("screen quality contract differs from methodology")
    _check_binding(document["candidate_set"], "candidate set")
    for binding in document["implementation"].values():
        _check_binding(binding, "implementation")
    metric_source = _check_binding(document["metric_identity_source"], "metric source")
    if (
        _load_json(metric_source, "metric source").get("semantic_metrics")
        != document["semantic_metrics"]
    ):
        raise ValueError("screen metric identity differs from its source")
    prompt = document["prompt_suite"]
    prompt_path = _check_binding(prompt, "prompt suite")
    selection = load_prompt_suite(prompt_path, prompt["split"])
    if selection.descriptor["sha256"] != prompt["split_sha256"] or len(selection.prompts) != 32:
        raise ValueError("screen prompt suite differs from registration")
    prior: set[str] = set()
    for binding in document["novelty_audit"]["prior_prompt_sources"]:
        prior.update(_extract_registered_prompts(_check_binding(binding, "prior prompt source")))
    if (
        len(prior) != document["novelty_audit"]["prior_unique_prompt_count"]
        or set(selection.prompts) & prior
    ):
        raise ValueError("screen prompt novelty audit differs")
    candidates = document["candidates"]
    if len(candidates) != 5 or candidates[0]["family"] != "legacy_comparator":
        raise ValueError("screen candidate matrix is invalid")
    observed_ids = []
    for index, binding in enumerate(candidates):
        candidate_path = _check_binding(binding, f"candidate {index}")
        arm = (
            load_adaptive_candidate(candidate_path)
            if index == 0
            else load_phased_candidate(candidate_path)
        )
        if (
            arm.candidate_id != binding["candidate_id"]
            or arm.content_sha256 != binding["content_sha256"]
        ):
            raise ValueError("screen candidate identity differs")
        observed_ids.append(arm.candidate_id)
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError("screen candidate ids are not unique")
    return document


def load_registered_arms(registration: Mapping[str, Any]) -> tuple[Any, ...]:
    arms = []
    for index, binding in enumerate(registration["candidates"]):
        path = _repo_path(binding["path"], f"candidate {index}")
        arms.append(load_adaptive_candidate(path) if index == 0 else load_phased_candidate(path))
    return tuple(arms)


def _paired_bootstrap_lower(
    comparator: Sequence[float],
    candidate: Sequence[float],
    *,
    repetitions: int,
    seed: int,
    quantile: float,
) -> float:
    if len(comparator) != len(candidate) or not comparator:
        raise ValueError("paired bootstrap inputs must be equal and non-empty")
    left = np.asarray(comparator, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("paired bootstrap durations must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), size=(repetitions, len(left)))
    ratios = left[indices].sum(axis=1) / right[indices].sum(axis=1)
    return float(np.quantile(ratios, quantile, method="linear"))


def evaluate_screen(
    registration_path: Path,
    semantic_report_path: Path,
    speed_manifest_path: Path,
) -> dict[str, Any]:
    registration = load_registration(registration_path)
    report = load_semantic_report(semantic_report_path)
    _validate_metric_identity(report["metrics"], registration["semantic_metrics"])
    manifest, observed_selection, manifest_path = semantic_source(report)
    prompt = registration["prompt_suite"]
    if (
        observed_selection.get("split") != prompt["split"]
        or observed_selection.get("sha256") != prompt["split_sha256"]
        or list(manifest["protocol"]["rng"]["seeds"]) != prompt["seeds"]
    ):
        raise ValueError("screen semantic evidence uses a different prompt matrix")
    validate_generation_identity(manifest, registration["controlled_generation"])
    arms = load_registered_arms(registration)
    expected_candidates = [
        {
            "candidate_id": arm.candidate_id,
            "policy": arm.policy_spec(),
            "predictor": arm.predictor_spec(),
        }
        for arm in arms
    ]
    if manifest.get("candidates") != expected_candidates:
        raise ValueError("screen quality manifest candidate order differs")

    margins = registration["quality_contract"]["margins"]
    registered_ids = [row["candidate_id"] for row in registration["candidates"]]
    row_map: dict[tuple[str, int], dict[str, Any]] = {}
    for comparison in report["comparisons"]:
        candidate_id = comparison["candidate_id"]
        prompt_index = int(comparison["prompt_index"])
        key = (candidate_id, prompt_index)
        if candidate_id not in registered_ids or key in row_map:
            raise ValueError("screen semantic report has an invalid comparison matrix")
        harms = {
            metric: -float(comparison["candidate_minus_baseline"][metric]) for metric in margins
        }
        failures = {metric: harms[metric] > float(margin) for metric, margin in margins.items()}
        row_map[key] = {
            "sample_id": comparison["sample_id"],
            "prompt_index": prompt_index,
            "seed": int(comparison["seed"]),
            "harms": harms,
            "metric_failures": failures,
            "failed": any(failures.values()),
        }
    expected_keys = {
        (candidate_id, index) for candidate_id in registered_ids for index in range(32)
    }
    if set(row_map) != expected_keys:
        raise ValueError("screen semantic report does not cover every registered comparison")

    speed = _load_json(speed_manifest_path, "screen speed manifest")
    if (
        speed.get("candidates") is None
        or speed.get("protocol", {}).get("sha256") != manifest["protocol"]["sha256"]
    ):
        raise ValueError("screen speed and quality manifests use different protocols")
    speed_rows = {row["candidate_id"]: row for row in speed["candidates"]}
    if set(speed_rows) != set(registered_ids):
        raise ValueError("screen speed manifest candidate matrix differs")
    comparator_id = registration["speed_gate"]["comparator_candidate_id"]
    comparator_samples = sorted(
        speed_rows[comparator_id]["samples"], key=lambda row: row["sample_id"]
    )
    comparator_durations = [float(row["elapsed_s"]) for row in comparator_samples]

    summaries = []
    for binding in registration["candidates"]:
        candidate_id = binding["candidate_id"]
        failures = sum(row_map[(candidate_id, index)]["failed"] for index in range(32))
        upper = one_sided_binomial_upper_bound(failures, 32, confidence=0.95)
        quality_passed = (
            failures == 0
            and upper <= registration["quality_gate"]["maximum_failure_rate_upper_bound"]
        )
        candidate_speed = speed_rows[candidate_id]
        candidate_samples = sorted(candidate_speed["samples"], key=lambda row: row["sample_id"])
        if [row["sample_id"] for row in candidate_samples] != [
            row["sample_id"] for row in comparator_samples
        ]:
            raise ValueError("screen speed samples are not paired")
        candidate_durations = [float(row["elapsed_s"]) for row in candidate_samples]
        relative = float(speed_rows[comparator_id]["total_s"]) / float(candidate_speed["total_s"])
        if candidate_id == comparator_id:
            lower = 1.0
            speed_passed = False
        else:
            gate = registration["speed_gate"]
            lower = _paired_bootstrap_lower(
                comparator_durations,
                candidate_durations,
                repetitions=int(gate["paired_bootstrap_repetitions"]),
                seed=int(gate["paired_bootstrap_seed"]),
                quantile=float(gate["paired_bootstrap_lower_quantile"]),
            )
            speed_passed = relative >= float(gate["minimum_relative_speed"]) and lower > float(
                gate["minimum_bootstrap_lower_bound"]
            )
        summaries.append(
            {
                **binding,
                "failure_count": failures,
                "failure_rate_upper_bound": upper,
                "quality_passed": quality_passed,
                "total_s": float(candidate_speed["total_s"]),
                "measured_speedup_vs_full_dit": float(candidate_speed["measured_speedup"]),
                "relative_speed_vs_legacy": relative,
                "paired_bootstrap_lower_bound": lower,
                "speed_passed": speed_passed,
                "eligible": binding["family"] in {"static", "combined"}
                and quality_passed
                and speed_passed,
                "rows": [row_map[(candidate_id, index)] for index in range(32)],
            }
        )

    representatives = []
    for family in registration["selection_rule"]["eligible_families"]:
        eligible = [row for row in summaries if row["family"] == family and row["eligible"]]
        if eligible:
            selected = min(eligible, key=lambda row: (row["total_s"], row["candidate_id"]))
            representatives.append(
                {
                    key: selected[key]
                    for key in (
                        "candidate_id",
                        "family",
                        "anchor_budget",
                        "path",
                        "file_sha256",
                        "content_sha256",
                        "failure_count",
                        "failure_rate_upper_bound",
                        "relative_speed_vs_legacy",
                        "paired_bootstrap_lower_bound",
                        "total_s",
                    )
                }
            )
    result_payload = {
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
        "quality_manifest": {"path": str(manifest_path), "file_sha256": sha256_file(manifest_path)},
        "speed_manifest": {
            "path": str(speed_manifest_path),
            "file_sha256": sha256_file(speed_manifest_path),
        },
        "candidate_summaries": summaries,
        "selected_representatives": representatives,
        "status": (
            "eligible_representatives" if representatives else "stop_no_eligible_representatives"
        ),
        "serving_qualified": False,
    }
    return {**result_payload, "sha256": canonical_sha256(result_payload)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--study-id", required=True)
    register_parser.add_argument("--created-at", required=True)
    register_parser.add_argument("--methodology", required=True)
    register_parser.add_argument("--metric-source", required=True)
    register_parser.add_argument("--candidate-set", required=True)
    register_parser.add_argument("--legacy-candidate", required=True)
    register_parser.add_argument("--prompt-suite", required=True)
    register_parser.add_argument("--prompt-split", required=True)
    register_parser.add_argument("--prior-prompt-source", action="append", required=True)
    register_parser.add_argument("--output-directory", required=True)
    register_parser.add_argument("--out", required=True)
    register_parser.set_defaults(func=register_screen)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--registration", required=True)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--registration", required=True)
    evaluate_parser.add_argument("--semantic-report", required=True)
    evaluate_parser.add_argument("--speed-manifest", required=True)
    evaluate_parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    try:
        args = _parse_args(argv)
        if args.command == "validate":
            registration = load_registration(Path(args.registration).expanduser().resolve())
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "study_id": registration["study_id"],
                        "sha256": registration["sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "evaluate":
            out = Path(args.out).expanduser().resolve()
            if out.exists():
                raise ValueError(f"output already exists: {out}")
            result = evaluate_screen(
                Path(args.registration).expanduser().resolve(),
                Path(args.semantic_report).expanduser().resolve(),
                Path(args.speed_manifest).expanduser().resolve(),
            )
            _write_json(out, result)
            print(f"[schedule-screen] {result['status']} -> {out}", flush=True)
            return 0
        args.func(args)
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
