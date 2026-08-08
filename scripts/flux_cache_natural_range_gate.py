#!/usr/bin/env python3
"""Apply an automatic fail-closed natural-variation gate to cache candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.offline.cache_profile.quality import (  # noqa: E402
    METRICS,
    load_semantic_report,
    metric_identity,
    semantic_source,
    sha256_file,
    validate_generation_identity,
)
from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402
from scripts.multires_quality_contract import (  # noqa: E402
    CONTRACT_SCHEMA,
    SCHEMA_REVISION,
)


EVALUATION_SCHEMA = "difflet-flux-cache-natural-range-evaluation"
EVALUATION_SCHEMA_REVISION = 1
AUTOMATIC_REVIEW_POLICY = {
    "within_natural_range_action": "automatic_pass",
    "outside_natural_range_action": "automatic_reject",
    "human_review_required": False,
    "manual_override_permitted": False,
}


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


def _validate_digest(document: Mapping[str, Any], name: str) -> None:
    digest = document.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name} sha256 is invalid")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != digest:
        raise ValueError(f"{name} sha256 does not match its contents")


def load_natural_range_contract(path: Path) -> dict[str, Any]:
    document = _load_json(path, "multires quality contract")
    if (
        document.get("schema") != CONTRACT_SCHEMA
        or document.get("schema_revision") != SCHEMA_REVISION
    ):
        raise ValueError("multires quality contract schema is unsupported")
    _validate_digest(document, "multires quality contract")
    if set(document.get("metric_identity", {}).get("config", {})) != set(METRICS):
        raise ValueError("multires quality contract metric identity is incomplete")
    buckets = document.get("resolution_contracts")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("multires quality contract has no resolution contracts")
    seen: set[str] = set()
    for bucket in buckets:
        if not isinstance(bucket, dict):
            raise ValueError("multires quality contract contains an invalid bucket")
        bucket_id = bucket.get("bucket_id")
        if not isinstance(bucket_id, str) or not bucket_id or bucket_id in seen:
            raise ValueError("multires quality contract bucket id is invalid")
        seen.add(bucket_id)
        summary = bucket.get("calibration_summary")
        if not isinstance(summary, dict) or set(summary) != set(METRICS):
            raise ValueError("multires quality contract natural ranges are incomplete")
        for metric in METRICS:
            maximum = summary[metric].get("maximum")
            if isinstance(maximum, bool) or not math.isfinite(float(maximum)):
                raise ValueError("multires quality contract natural-range maximum is invalid")
            if float(maximum) < 0.0:
                raise ValueError("multires quality contract natural-range maximum is negative")
    return document


def _bucket_for(
    contract: Mapping[str, Any],
    bucket_id: str,
) -> Mapping[str, Any]:
    matches = [
        bucket for bucket in contract["resolution_contracts"] if bucket["bucket_id"] == bucket_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown natural-range bucket {bucket_id!r}")
    return matches[0]


def _finite_score(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_comparison_coverage(
    manifest: Mapping[str, Any],
    observed: Sequence[Mapping[str, Any]],
) -> None:
    expected_rows = manifest.get("comparisons")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError("quality manifest contains no cache comparisons")
    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in expected_rows:
        if not isinstance(row, dict):
            raise ValueError("quality manifest contains an invalid comparison")
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key in expected:
            raise ValueError("quality manifest contains duplicate comparisons")
        expected[key] = row

    actual: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in observed:
        if not isinstance(row, dict):
            raise ValueError("semantic report contains an invalid comparison")
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key in actual:
            raise ValueError("semantic report contains duplicate comparisons")
        actual[key] = row
    if set(actual) != set(expected):
        raise ValueError("semantic report comparison coverage differs from the quality manifest")
    identity_fields = ("prompt_index", "seed", "prompt")
    for key, observed_row in actual.items():
        expected_row = expected[key]
        if any(observed_row.get(field) != expected_row.get(field) for field in identity_fields):
            raise ValueError(
                "semantic report comparison identity differs from the quality manifest"
            )


def _validate_metric_compatibility(
    observed_metrics: Mapping[str, Any],
    contract_identity: Mapping[str, Any],
) -> None:
    observed = metric_identity(observed_metrics)["config"]
    expected = contract_identity["config"]
    scalar_fields = {
        "image_reward": (
            "implementation",
            "package",
            "package_version",
            "model",
            "dtype",
            "preprocessing",
        ),
        "vqa_score": (
            "implementation",
            "package",
            "package_version",
            "model",
            "dtype",
            "question_template",
            "answer_template",
            "repositories",
        ),
    }
    for metric, fields in scalar_fields.items():
        for field in fields:
            if observed[metric].get(field) != expected[metric].get(field):
                raise ValueError(f"{metric} {field} differs from the natural-range contract")
        observed_digests = {
            row.get("sha256")
            for row in observed[metric].get("checkpoint_files", [])
            if isinstance(row, dict)
        }
        expected_digests = {
            row.get("sha256")
            for row in expected[metric].get("checkpoint_files", [])
            if isinstance(row, dict)
        }
        if not observed_digests or not observed_digests <= expected_digests:
            raise ValueError(f"{metric} checkpoint identity differs from the contract")


def evaluate_natural_range(
    contract_path: Path,
    semantic_report_path: Path,
    *,
    bucket_id: str,
) -> dict[str, Any]:
    contract_path = Path(contract_path).expanduser().resolve()
    semantic_report_path = Path(semantic_report_path).expanduser().resolve()
    contract = load_natural_range_contract(contract_path)
    bucket = _bucket_for(contract, bucket_id)
    report = load_semantic_report(semantic_report_path)
    manifest, selection, manifest_path = semantic_source(report)
    controlled = {
        **contract["controlled_generation"],
        "height": int(bucket["height"]),
        "width": int(bucket["width"]),
    }
    validate_generation_identity(manifest, controlled)
    _validate_metric_compatibility(report["metrics"], contract["metric_identity"])
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("semantic report contains no cache comparisons")
    _validate_comparison_coverage(manifest, comparisons)

    limits = {metric: float(bucket["calibration_summary"][metric]["maximum"]) for metric in METRICS}
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        deltas = comparison.get("candidate_minus_baseline")
        if not isinstance(deltas, dict) or set(deltas) != set(METRICS):
            raise ValueError("semantic comparison metric deltas are incomplete")
        harms = {metric: -_finite_score(deltas[metric], f"{metric} delta") for metric in METRICS}
        failed_metrics = [metric for metric in METRICS if harms[metric] > limits[metric]]
        rows.append(
            {
                "candidate_id": comparison["candidate_id"],
                "sample_id": comparison["sample_id"],
                "prompt_index": int(comparison["prompt_index"]),
                "seed": int(comparison["seed"]),
                "prompt": comparison["prompt"],
                "harms": harms,
                "failed_metrics": failed_metrics,
                "decision": "automatic_reject" if failed_metrics else "automatic_pass",
                "passes": not failed_metrics,
            }
        )

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    summaries = []
    for candidate_id in sorted(grouped):
        candidate_rows = grouped[candidate_id]
        failures = [row for row in candidate_rows if not row["passes"]]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "sample_count": len(candidate_rows),
                "failure_count": len(failures),
                "decision": "automatic_reject" if failures else "automatic_pass",
                "passes_zero_failure_gate": not failures,
            }
        )

    payload = {
        "schema": EVALUATION_SCHEMA,
        "schema_revision": EVALUATION_SCHEMA_REVISION,
        "contract": {
            "path": str(contract_path),
            "file_sha256": sha256_file(contract_path),
            "content_sha256": contract["sha256"],
        },
        "semantic_report": {
            "path": str(semantic_report_path),
            "file_sha256": sha256_file(semantic_report_path),
        },
        "quality_manifest": {
            "path": str(manifest_path),
            "file_sha256": sha256_file(manifest_path),
        },
        "bucket_id": bucket_id,
        "height": int(bucket["height"]),
        "width": int(bucket["width"]),
        "split": selection["split"],
        "split_sha256": selection["sha256"],
        "natural_range_limits": limits,
        "review_policy": AUTOMATIC_REVIEW_POLICY,
        "candidate_summaries": summaries,
        "rows": rows,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--semantic-report", required=True)
    parser.add_argument("--bucket-id", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        result = evaluate_natural_range(
            Path(args.contract),
            Path(args.semantic_report),
            bucket_id=args.bucket_id,
        )
        output = Path(args.out).expanduser().resolve()
        _write_json(output, result)
        passed = sum(row["passes_zero_failure_gate"] for row in result["candidate_summaries"])
        rejected = len(result["candidate_summaries"]) - passed
        print(
            f"[natural-range] evaluated={len(result['candidate_summaries'])} "
            f"passed={passed} rejected={rejected} "
            f"-> {output}",
            flush=True,
        )
        if rejected:
            return 1
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
