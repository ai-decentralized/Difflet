#!/usr/bin/env python3
"""Prospective, per-resolution FLUX semantic-quality contract calibration.

The registration freezes the generation identity, prompt/seed matrix, metric
identity, source tree, and calibration rule before any baseline images are
collected.  A calibrated contract contains one observed seed-variation
envelope for every metric in every registered AOT resolution bucket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.offline.cache_profile.quality import (
    METRICS,
    load_semantic_report,
    metric_identity,
)
from scripts.flux_cache_protocol import (
    canonical_sha256,
    load_prompt_suite,
    validate_experiment_protocol,
)


PROTOCOL_SCHEMA = "difflet-flux-cache-multires-quality-contract-protocol"
CONTRACT_SCHEMA = "difflet-flux-cache-multires-quality-contract"
BASELINE_MANIFEST_SCHEMA = "difflet-flux-baseline-calibration-manifest"
SCHEMA_REVISION = 1
CALIBRATION_METHOD = "maximum-observed-absolute-baseline-seed-pair-difference"
AUTOMATIC_DAMAGE_RULE = {
    "metrics": list(METRICS),
    "paired_loss": "baseline_score_minus_candidate_score",
    "comparison": "strictly-greater-than-observed-seed-variation-envelope",
    "combination": "fail-if-either-metric-fails",
    "weighted_average_forbidden": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_digest(document: Mapping[str, Any], name: str) -> None:
    digest = document.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name} sha256 is invalid")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != digest:
        raise ValueError(f"{name} sha256 does not match its contents")


def _repo_file(relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{name} must be a non-empty repository-relative path")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"{name} does not identify a repository file: {relative!r}")
    return path


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bucket_map(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["bucket_id"]): row for row in protocol["resolution_buckets"]}


def load_protocol(path: Path, *, verify_local_source: bool = True) -> dict[str, Any]:
    document = load_json(path, "multires quality protocol")
    expected = {
        "schema",
        "schema_revision",
        "study_id",
        "created_at",
        "status",
        "controlled_generation",
        "resolution_buckets",
        "observed_seed_variation_calibration",
        "automatic_damage_rule",
        "semantic_metrics",
        "source_registration",
        "parameters_frozen_before_collection",
        "limitations",
        "sha256",
    }
    if set(document) != expected:
        raise ValueError("multires quality protocol fields do not match the schema")
    if document["schema"] != PROTOCOL_SCHEMA or document["schema_revision"] != SCHEMA_REVISION:
        raise ValueError("multires quality protocol schema is unsupported")
    _validate_digest(document, "multires quality protocol")
    if document["status"] != "registered-not-collected":
        raise ValueError("multires protocol must be prospectively registered")
    if document["parameters_frozen_before_collection"] is not True:
        raise ValueError("multires parameters were not frozen before collection")

    controlled = document["controlled_generation"]
    controlled_fields = {
        "model_id",
        "model_revision",
        "scheduler_class",
        "scheduler_config_sha256",
        "num_steps",
        "guidance_scale",
        "dtype",
        "tp_degree",
    }
    if not isinstance(controlled, dict) or set(controlled) != controlled_fields:
        raise ValueError("controlled generation fields are invalid")
    _positive_int(controlled["num_steps"], "controlled_generation.num_steps")
    _positive_int(controlled["tp_degree"], "controlled_generation.tp_degree")
    _finite(controlled["guidance_scale"], "controlled_generation.guidance_scale")

    buckets = document["resolution_buckets"]
    if not isinstance(buckets, list) or len(buckets) < 2:
        raise ValueError("multires protocol requires at least two resolution buckets")
    bucket_ids: set[str] = set()
    shapes: set[tuple[int, int]] = set()
    for index, bucket in enumerate(buckets):
        name = f"resolution_buckets[{index}]"
        if not isinstance(bucket, dict) or set(bucket) != {
            "bucket_id",
            "height",
            "width",
            "prompt_suite",
        }:
            raise ValueError(f"{name} fields are invalid")
        bucket_id = bucket["bucket_id"]
        if not isinstance(bucket_id, str) or not bucket_id or bucket_id != bucket_id.strip():
            raise ValueError(f"{name}.bucket_id is invalid")
        height = _positive_int(bucket["height"], f"{name}.height")
        width = _positive_int(bucket["width"], f"{name}.width")
        if height % 32 or width % 32:
            raise ValueError(f"{name} dimensions must be multiples of 32")
        if bucket_id in bucket_ids or (height, width) in shapes:
            raise ValueError("resolution bucket identifiers and shapes must be unique")
        bucket_ids.add(bucket_id)
        shapes.add((height, width))

        prompt = bucket["prompt_suite"]
        if not isinstance(prompt, dict) or set(prompt) != {
            "path",
            "file_sha256",
            "split",
            "split_sha256",
            "prompt_count",
            "seeds",
            "sample_count",
        }:
            raise ValueError(f"{name}.prompt_suite fields are invalid")
        prompt_path = _repo_file(prompt["path"], f"{name}.prompt_suite.path")
        if sha256_file(prompt_path) != prompt["file_sha256"]:
            raise ValueError(f"{name} prompt-suite file hash differs from registration")
        selection = load_prompt_suite(prompt_path, prompt["split"])
        seeds = prompt["seeds"]
        if (
            selection.descriptor["sha256"] != prompt["split_sha256"]
            or len(selection.prompts)
            != _positive_int(prompt["prompt_count"], f"{name}.prompt_count")
            or not isinstance(seeds, list)
            or len(seeds) < 2
            or len(seeds) != len(set(seeds))
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
            )
            or prompt["sample_count"] != prompt["prompt_count"] * len(seeds)
        ):
            raise ValueError(f"{name} prompt/seed registration is inconsistent")

    calibration = document["observed_seed_variation_calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "method",
        "decision_statistic",
        "candidate_images_excluded_from_margin_estimation",
        "pooling_across_resolutions_forbidden",
    }:
        raise ValueError("observed seed-variation calibration fields are invalid")
    if (
        calibration["method"] != CALIBRATION_METHOD
        or calibration["decision_statistic"] != "maximum_observed"
        or calibration["candidate_images_excluded_from_margin_estimation"] is not True
        or calibration["pooling_across_resolutions_forbidden"] is not True
    ):
        raise ValueError("observed seed-variation calibration rule is unsupported")
    if document["automatic_damage_rule"] != AUTOMATIC_DAMAGE_RULE:
        raise ValueError("automatic damage rule is unsupported")
    if set(document["semantic_metrics"]) != set(METRICS):
        raise ValueError("semantic metric registration is incomplete")

    source = document["source_registration"]
    if not isinstance(source, dict) or set(source) != {
        "python_source_sha256",
        "dirty_worktree_policy",
        "implementation_files",
    }:
        raise ValueError("source registration fields are invalid")
    source_hash = source["python_source_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("registered Python source hash is invalid")
    if source["dirty_worktree_policy"] != "exact-python-source-sha256-required":
        raise ValueError("dirty worktree policy is unsupported")
    implementation = source["implementation_files"]
    if not isinstance(implementation, list) or not implementation:
        raise ValueError("implementation file registration is empty")
    seen_paths: set[str] = set()
    for binding in implementation:
        if not isinstance(binding, dict) or set(binding) != {"path", "file_sha256"}:
            raise ValueError("implementation file binding is invalid")
        if binding["path"] in seen_paths:
            raise ValueError("implementation file binding is duplicated")
        seen_paths.add(binding["path"])
        if verify_local_source:
            bound_path = _repo_file(binding["path"], "implementation file")
            if sha256_file(bound_path) != binding["file_sha256"]:
                raise ValueError(f"implementation file changed: {binding['path']}")
    if not isinstance(document["limitations"], list) or not document["limitations"]:
        raise ValueError("multires protocol must state its limitations")
    return document


def bucket_for(protocol: Mapping[str, Any], bucket_id: str) -> Mapping[str, Any]:
    try:
        return _bucket_map(protocol)[bucket_id]
    except KeyError as error:
        raise ValueError(f"unregistered resolution bucket: {bucket_id!r}") from error


def validate_observed_generation(
    experiment: Mapping[str, Any],
    protocol: Mapping[str, Any],
    bucket: Mapping[str, Any],
) -> None:
    observed_protocol = validate_experiment_protocol(dict(experiment))
    generation = observed_protocol["generation"]
    controlled = protocol["controlled_generation"]
    observed = {
        "model_id": observed_protocol["model"]["model_id"],
        "model_revision": observed_protocol["model"]["resolved_revision"],
        "scheduler_class": generation["scheduler_class"],
        "scheduler_config_sha256": canonical_sha256(generation["scheduler_config"]),
        "num_steps": generation["num_steps"],
        "guidance_scale": generation["guidance_scale"],
        "dtype": generation["dtype"],
        "tp_degree": observed_protocol["hardware"]["tp_degree"],
    }
    if observed != controlled:
        raise ValueError("baseline generation identity differs from registration")
    if (generation["height"], generation["width"]) != (
        bucket["height"],
        bucket["width"],
    ):
        raise ValueError("baseline generation shape differs from resolution bucket")
    prompt = bucket["prompt_suite"]
    selection = observed_protocol["prompt_selection"]
    if (
        selection["split"] != prompt["split"]
        or selection["sha256"] != prompt["split_sha256"]
        or len(selection["prompts"]) != prompt["prompt_count"]
        or observed_protocol["rng"]["seeds"] != prompt["seeds"]
    ):
        raise ValueError("baseline prompt/seed identity differs from registration")


def load_baseline_manifest(
    path: Path,
    protocol: Mapping[str, Any],
    bucket_id: str,
    *,
    verify_images: bool = True,
) -> dict[str, Any]:
    document = load_json(path, "baseline calibration manifest")
    if set(document) != {
        "schema",
        "schema_revision",
        "registration",
        "bucket_id",
        "python_source_sha256",
        "started_at",
        "completed_at",
        "protocol",
        "baseline_samples",
    }:
        raise ValueError("baseline calibration manifest fields are invalid")
    if (
        document["schema"] != BASELINE_MANIFEST_SCHEMA
        or document["schema_revision"] != SCHEMA_REVISION
        or document["bucket_id"] != bucket_id
    ):
        raise ValueError("baseline calibration manifest schema or bucket is invalid")
    registration = document["registration"]
    if registration.get("content_sha256") != protocol["sha256"]:
        raise ValueError("baseline manifest uses a different protocol")
    if document["python_source_sha256"] != protocol["source_registration"]["python_source_sha256"]:
        raise ValueError("baseline manifest Python source differs from registration")
    bucket = bucket_for(protocol, bucket_id)
    validate_observed_generation(document["protocol"], protocol, bucket)

    prompt_path = _repo_file(bucket["prompt_suite"]["path"], "prompt suite")
    selection = load_prompt_suite(prompt_path, bucket["prompt_suite"]["split"])
    expected = [
        (index, prompt, seed, f"p{index:03d}-s{seed}")
        for index, prompt in enumerate(selection.prompts)
        for seed in bucket["prompt_suite"]["seeds"]
    ]
    rows = document["baseline_samples"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("baseline manifest has the wrong sample count")
    for row, (index, prompt, seed, sample_id) in zip(rows, expected, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "sample_id",
            "prompt_index",
            "prompt",
            "seed",
            "elapsed_s",
            "artifacts",
        }:
            raise ValueError("baseline sample fields are invalid")
        if (
            row["sample_id"] != sample_id
            or row["prompt_index"] != index
            or row["prompt"] != prompt
            or row["seed"] != seed
            or _finite(row["elapsed_s"], "baseline elapsed_s") <= 0.0
        ):
            raise ValueError("baseline sample identity is invalid")
        artifacts = row["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != {"image", "image_sha256"}:
            raise ValueError("baseline image artifact binding is invalid")
        image_path = (path.parent / artifacts["image"]).resolve()
        if verify_images and (
            not image_path.is_file() or sha256_file(image_path) != artifacts["image_sha256"]
        ):
            raise ValueError(f"baseline image is missing or changed: {image_path}")
    return document


def _validate_metric_identity(metrics: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if set(metrics) != set(METRICS):
        raise ValueError("semantic report must contain only ImageReward and VQAScore")
    image_reward = metrics["image_reward"]
    expected_ir = expected["image_reward"]
    for field in (
        "implementation",
        "package",
        "package_version",
        "model",
        "dtype",
        "preprocessing",
    ):
        if image_reward.get(field) != expected_ir[field]:
            raise ValueError(f"ImageReward {field} differs from registration")
    observed_ir_files = {
        str(row.get("path")): row.get("sha256")
        for row in image_reward.get("checkpoint_files", [])
        if isinstance(row, dict)
    }
    for suffix, digest in expected_ir["files_by_suffix"].items():
        if not any(
            path.endswith(suffix) and value == digest for path, value in observed_ir_files.items()
        ):
            raise ValueError("ImageReward checkpoint files differ from registration")

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
            raise ValueError(f"VQAScore {field} differs from registration")
    revisions = {
        row.get("repository"): row.get("resolved_revision")
        for row in vqa.get("repositories", [])
        if isinstance(row, dict)
    }
    if revisions != expected_vqa["repository_revisions"]:
        raise ValueError("VQAScore repository revisions differ from registration")
    observed_vqa_files = {
        str(row.get("path")): row.get("sha256")
        for row in vqa.get("checkpoint_files", [])
        if isinstance(row, dict)
    }
    for suffix, digest in expected_vqa["files_by_suffix"].items():
        if not any(
            path.endswith(suffix) and value == digest for path, value in observed_vqa_files.items()
        ):
            raise ValueError("VQAScore checkpoint files differ from registration")


def _load_report_evidence(
    report_path: Path,
    protocol: Mapping[str, Any],
    bucket: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    report = load_semantic_report(report_path)
    _validate_metric_identity(report["metrics"], protocol["semantic_metrics"])
    sources = report["sources"]
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("baseline semantic report must bind one manifest")
    source = sources[0]
    if (
        source.get("bucket_id") != bucket["bucket_id"]
        or source.get("split") != bucket["prompt_suite"]["split"]
    ):
        raise ValueError("semantic report source bucket differs from registration")
    manifest_path = Path(source["path"]).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != source["sha256"]:
        raise ValueError("semantic report baseline-manifest binding is invalid")
    manifest = load_baseline_manifest(manifest_path, protocol, bucket["bucket_id"])
    if report["comparisons"] or report["summary"]:
        raise ValueError(
            "seed-variation envelope calibration must not contain candidate comparisons"
        )
    rows = report["images"]
    if len(rows) != bucket["prompt_suite"]["sample_count"]:
        raise ValueError("semantic report has the wrong baseline image count")
    manifest_rows = {row["sample_id"]: row for row in manifest["baseline_samples"]}
    seen: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        baseline = manifest_rows.get(sample_id)
        if (
            baseline is None
            or sample_id in seen
            or row.get("role") != "baseline"
            or row.get("candidate_id") is not None
            or row.get("split") != bucket["prompt_suite"]["split"]
            or row.get("prompt_index") != baseline["prompt_index"]
            or row.get("prompt") != baseline["prompt"]
            or row.get("seed") != baseline["seed"]
            or row.get("image_sha256") != baseline["artifacts"]["image_sha256"]
            or set(row.get("scores", {})) != set(METRICS)
        ):
            raise ValueError("semantic baseline image identity is invalid")
        seen.add(sample_id)
    return report, manifest_path, manifest


def calibrate_contract(
    protocol_path: Path,
    semantic_reports: Mapping[str, Path],
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    expected_ids = set(_bucket_map(protocol))
    if set(semantic_reports) != expected_ids:
        raise ValueError("semantic reports must cover every registered bucket exactly once")
    bucket_results: list[dict[str, Any]] = []
    shared_metric_identity: dict[str, Any] | None = None
    for bucket in protocol["resolution_buckets"]:
        bucket_id = bucket["bucket_id"]
        report_path = semantic_reports[bucket_id].expanduser().resolve()
        report, manifest_path, _ = _load_report_evidence(report_path, protocol, bucket)
        observed_metric_identity = metric_identity(report["metrics"])
        if shared_metric_identity is None:
            shared_metric_identity = observed_metric_identity
        elif observed_metric_identity["sha256"] != shared_metric_identity["sha256"]:
            raise ValueError("semantic metric identity differs across resolution buckets")

        grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in report["images"]:
            grouped[(int(row["prompt_index"]), str(row["prompt"]))].append(row)
        if len(grouped) != bucket["prompt_suite"]["prompt_count"]:
            raise ValueError("semantic report prompt count differs from registration")
        differences: dict[str, list[float]] = {metric: [] for metric in METRICS}
        expected_seeds = tuple(bucket["prompt_suite"]["seeds"])
        for rows in grouped.values():
            ordered = sorted(rows, key=lambda row: int(row["seed"]))
            if tuple(int(row["seed"]) for row in ordered) != expected_seeds:
                raise ValueError("each prompt must contain every registered seed")
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    for metric in METRICS:
                        differences[metric].append(
                            abs(
                                _finite(left["scores"][metric], metric)
                                - _finite(right["scores"][metric], metric)
                            )
                        )
        envelopes: dict[str, float] = {}
        summaries: dict[str, Any] = {}
        for metric, values in differences.items():
            envelope = max(values)
            envelopes[metric] = envelope
            summaries[metric] = {
                "pair_count": len(values),
                "minimum": min(values),
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "maximum_observed": envelope,
            }
        bucket_results.append(
            {
                "bucket_id": bucket_id,
                "height": bucket["height"],
                "width": bucket["width"],
                "observed_seed_variation_envelope": envelopes,
                "calibration_diagnostics": summaries,
                "evidence": {
                    "semantic_report": str(report_path),
                    "semantic_report_sha256": sha256_file(report_path),
                    "baseline_manifest": str(manifest_path),
                    "baseline_manifest_sha256": sha256_file(manifest_path),
                },
            }
        )
    payload = {
        "schema": CONTRACT_SCHEMA,
        "schema_revision": SCHEMA_REVISION,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "file_sha256": sha256_file(protocol_path),
            "content_sha256": protocol["sha256"],
            "study_id": protocol["study_id"],
        },
        "controlled_generation": protocol["controlled_generation"],
        "observed_seed_variation_calibration": protocol[
            "observed_seed_variation_calibration"
        ],
        "automatic_damage_rule": protocol["automatic_damage_rule"],
        "metric_identity": shared_metric_identity,
        "resolution_contracts": bucket_results,
        "limitations": protocol["limitations"],
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _parse_report_bindings(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        bucket_id, separator, path = value.partition("=")
        if not separator or not bucket_id or not path or bucket_id in result:
            raise ValueError("--semantic-report must be a unique BUCKET_ID=PATH binding")
        result[bucket_id] = Path(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-protocol")
    validate.add_argument("--protocol", required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--protocol", required=True)
    calibrate.add_argument("--semantic-report", action="append", required=True)
    calibrate.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol_path = Path(args.protocol).expanduser().resolve()
        if args.command == "validate-protocol":
            protocol = load_protocol(protocol_path)
            print(
                f"[multires-contract] valid {protocol['study_id']} "
                f"buckets={len(protocol['resolution_buckets'])}",
                flush=True,
            )
        else:
            reports = _parse_report_bindings(args.semantic_report)
            contract = calibrate_contract(protocol_path, reports)
            output = Path(args.out).expanduser().resolve()
            write_json(output, contract)
            print(f"[multires-contract] calibrated -> {output}", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
