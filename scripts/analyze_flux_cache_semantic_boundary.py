#!/usr/bin/env python3
"""Relate cheap FLUX cache measurements to offline semantic quality changes.

This is an exploratory boundary analysis, not a deployment gate.  It reports
both pooled correlations and correlations after ranking requests separately
inside each fixed cache configuration.  The latter prevents an aggressive
configuration from making a weak per-request signal look useful.
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

from difflet.pipeline.cache import load_cache_measurements  # noqa: E402
from scripts.analyze_flux_cache_signals import (  # noqa: E402
    _signal_summaries,
    candidate_stratified_spearman,
    spearman,
)
from scripts.calibrate_flux_cache_plan import _load_quality_curve, _validate_curve  # noqa: E402
from scripts.evaluate_cache_quality import (  # noqa: E402
    _artifact_path,
    _load_json,
    _validate_cache_measurement_artifact,
    _validate_quality_input,
)
from scripts.evaluate_flux_cache_semantics import (  # noqa: E402
    REPORT_SCHEMA,
    REPORT_SCHEMA_REVISION,
)
from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402

BOUNDARY_ANALYSIS_SCHEMA = "difflet-flux-cache-semantic-boundary"
BOUNDARY_ANALYSIS_SCHEMA_REVISION = 1
SIGNALS = (
    "mean-anchor-estimate-relative-error",
    "maximum-anchor-estimate-relative-error",
    "maximum-anchor-output-change",
    "maximum-anchor-output-curvature",
    "maximum-latent-relative-update",
)
HARM_TARGETS = ("image_reward_harm", "vqa_harm", "lpips")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Summarize a non-empty finite sample without choosing a gate."""

    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("summary values must be non-empty and finite")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def correlation_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    target: str,
) -> dict[str, Any]:
    """Report pooled and fixed-configuration rank correlations."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    per_candidate = []
    for candidate_id in sorted(grouped):
        candidate_rows = grouped[candidate_id]
        per_candidate.append(
            {
                "candidate_id": candidate_id,
                "sample_count": len(candidate_rows),
                "spearman": spearman(
                    [float(row[signal]) for row in candidate_rows],
                    [float(row[target]) for row in candidate_rows],
                ),
            }
        )
    return {
        "signal": signal,
        "harm_target": target,
        "pooled_spearman": spearman(
            [float(row[signal]) for row in rows],
            [float(row[target]) for row in rows],
        ),
        "candidate_stratified_spearman": candidate_stratified_spearman(
            rows,
            signal=signal,
            target=target,
        ),
        "per_candidate": per_candidate,
    }


def _validate_semantic_report(
    document: Mapping[str, Any],
    *,
    quality_input_path: Path,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected_keys = {
        "schema",
        "schema_revision",
        "complete",
        "started_at",
        "completed_at",
        "sources",
        "metrics",
        "runtime",
        "images",
        "comparisons",
        "summary",
    }
    if set(document) != expected_keys:
        raise ValueError("semantic report fields do not match its protocol")
    if (
        document["schema"] != REPORT_SCHEMA
        or document["schema_revision"] != REPORT_SCHEMA_REVISION
        or document["complete"] is not True
    ):
        raise ValueError("semantic report is unsupported or incomplete")
    if set(document["metrics"]) != {"image_reward", "vqa_score"}:
        raise ValueError("semantic boundary analysis requires both semantic metrics")
    sources = document["sources"]
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("semantic report must bind exactly one quality manifest")
    source = sources[0]
    if (
        Path(source["path"]).resolve() != quality_input_path
        or source["sha256"] != _sha256_file(quality_input_path)
    ):
        raise ValueError("semantic report does not bind the requested quality manifest")
    comparisons = document["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("semantic report contains no comparisons")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in comparisons:
        key = (str(row["candidate_id"]), str(row["sample_id"]))
        if key in by_key:
            raise ValueError("semantic report contains duplicate candidate/sample rows")
        if set(row["candidate_minus_baseline"]) != {"image_reward", "vqa_score"}:
            raise ValueError("semantic comparison metric fields are incomplete")
        by_key[key] = row
    return by_key


def analyze_boundary(
    quality_input_path: Path,
    quality_curve_path: Path,
    semantic_report_path: Path,
) -> dict[str, Any]:
    """Join all boundary-pilot evidence and return a digest-bearing report."""

    quality_input = _load_json(quality_input_path, "quality input")
    identity, definitions, comparisons = _validate_quality_input(quality_input)
    quality_curve = _load_quality_curve(quality_curve_path)
    _validate_curve(quality_curve)
    if quality_curve.get("protocol") != identity.get("protocol"):
        raise ValueError("quality curve protocol does not match quality input")
    semantic_document = _load_json(semantic_report_path, "semantic report")
    semantic_by_key = _validate_semantic_report(
        semantic_document,
        quality_input_path=quality_input_path,
    )

    quality_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for candidate in quality_curve["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in definitions:
            raise ValueError("quality curve contains an unknown candidate")
        for sample in candidate["per_sample"]:
            key = (candidate_id, sample["sample_id"])
            if key in quality_by_key:
                raise ValueError("quality curve contains duplicate candidate/sample rows")
            quality_by_key[key] = sample

    rows = []
    for comparison in comparisons:
        key = (comparison["candidate_id"], comparison["sample_id"])
        quality = quality_by_key.get(key)
        semantic = semantic_by_key.get(key)
        if quality is None or semantic is None:
            raise ValueError("quality, semantic, and measurement matrices differ")
        if any(
            semantic[field] != comparison[field]
            for field in ("candidate_id", "sample_id", "prompt_index", "seed", "prompt")
        ):
            raise ValueError("semantic comparison identity does not match quality input")
        artifacts = comparison["candidate"]
        if "cache_measurements" not in artifacts:
            raise ValueError("semantic boundary analysis requires cache measurements")
        measurement_path = _artifact_path(
            quality_input_path,
            artifacts["cache_measurements"],
            f"{comparison['sample_id']} candidate cache measurements",
        )
        _validate_cache_measurement_artifact(
            measurement_path,
            expected_sha256=artifacts["cache_measurements_sha256"],
            num_steps=identity["num_steps"],
            require_all_anchors=False,
        )
        deltas = semantic["candidate_minus_baseline"]
        rows.append(
            {
                "candidate_id": comparison["candidate_id"],
                "sample_id": comparison["sample_id"],
                "prompt_index": comparison["prompt_index"],
                "seed": comparison["seed"],
                "image_reward_delta": _finite(deltas["image_reward"], "ImageReward delta"),
                "vqa_delta": _finite(deltas["vqa_score"], "VQAScore delta"),
                "image_reward_harm": -_finite(
                    deltas["image_reward"], "ImageReward delta"
                ),
                "vqa_harm": -_finite(deltas["vqa_score"], "VQAScore delta"),
                "lpips": _finite(quality["lpips"], "LPIPS"),
                **_signal_summaries(load_cache_measurements(measurement_path)),
            }
        )
    expected_keys = {(row["candidate_id"], row["sample_id"]) for row in rows}
    if expected_keys != set(quality_by_key) or expected_keys != set(semantic_by_key):
        raise ValueError("quality, semantic, and measurement matrices differ")

    candidate_summaries = []
    for candidate_id in sorted(definitions):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "sample_count": len(candidate_rows),
                "signals": {
                    signal: summarize([row[signal] for row in candidate_rows])
                    for signal in SIGNALS
                },
                "quality": {
                    target: summarize([row[target] for row in candidate_rows])
                    for target in HARM_TARGETS
                },
            }
        )
    payload = {
        "schema": BOUNDARY_ANALYSIS_SCHEMA,
        "schema_revision": BOUNDARY_ANALYSIS_SCHEMA_REVISION,
        "evidence_role": "exploratory-boundary-analysis",
        "deployment_gate": False,
        "input_sha256": {
            "quality_input": _sha256_file(quality_input_path),
            "quality_curve": _sha256_file(quality_curve_path),
            "semantic_report": _sha256_file(semantic_report_path),
        },
        "model_id": identity["model_id"],
        "prompt_split": identity["protocol"]["prompt_selection"]["split"],
        "prompt_count": identity["prompt_count"],
        "seed_count": identity["seed_count"],
        "candidate_count": len(definitions),
        "row_count": len(rows),
        "correlations": [
            correlation_report(rows, signal=signal, target=target)
            for target in HARM_TARGETS
            for signal in SIGNALS
        ],
        "candidate_summaries": candidate_summaries,
        "rows": rows,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _write_json(path: Path, document: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", required=True)
    parser.add_argument("--quality-curve", required=True)
    parser.add_argument("--semantic-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = Path(args.out).expanduser().resolve()
        result = analyze_boundary(
            Path(args.quality_input).expanduser().resolve(),
            Path(args.quality_curve).expanduser().resolve(),
            Path(args.semantic_report).expanduser().resolve(),
        )
        _write_json(output, result, force=bool(args.force))
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    mean_error = next(
        row
        for row in result["correlations"]
        if row["signal"] == "mean-anchor-estimate-relative-error"
        and row["harm_target"] == "image_reward_harm"
    )
    print(
        "[semantic-boundary] "
        f"pooled_rho={mean_error['pooled_spearman']:.6f} "
        f"stratified_rho={mean_error['candidate_stratified_spearman']:.6f} "
        f"-> {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
