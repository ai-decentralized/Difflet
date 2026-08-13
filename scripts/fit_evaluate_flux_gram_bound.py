#!/usr/bin/env python3
"""Fit and evaluate a monotone segment-conditional FLUX Gram risk bound.

The bound family is intentionally small and auditable::

    U_c(z) = lambda_c * z

For each fixed A12 segment, ``lambda_c`` is a conformal order statistic of the
development scores ``D_{c->d} / z_c``.  With 32 development prompts and alpha
0.05, the selected statistic is the maximum score and the finite-sample
marginal coverage lower bound within a fixed segment is 32/33.

This is not distribution-free pointwise conditional coverage in z. Evaluation
therefore reports z-quartile coverage as a diagnostic and never calls the
result a serving or semantic-quality guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.pipeline.cache.profile import canonical_sha256  # noqa: E402
from scripts.collect_flux_gram_transfer_rollout import _segment_rows  # noqa: E402


MODEL_SCHEMA = "difflet-flux-gram-monotone-bound"
RESULT_SCHEMA = "difflet-flux-gram-monotone-bound-holdout-result"
EXPECTED_SEGMENTS = {5: 9, 9: 15, 15: 21, 21: 31, 31: 41, 41: 49}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("quantile values must be nonempty and finite")
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.9)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _validate_trace(
    document: dict[str, Any],
    *,
    expected_samples: int,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError(f"trace must contain exactly {expected_samples} samples")
    if len({row.get("sample_id") for row in samples}) != len(samples):
        raise ValueError("trace sample identifiers must be unique")
    for sample in samples:
        stats = sample.get("runner_stats")
        if not isinstance(stats, dict):
            raise ValueError("trace sample is missing runner_stats")
        if stats.get("full_steps") != 12 or stats.get("skipped_steps") != 38:
            raise ValueError("trace is not an exact static A12 rollout")
        if sample.get("shadow_full_dit_calls") != 38:
            raise ValueError("trace does not have one shadow label per A12 skip")
    segments = _segment_rows(samples)
    if set(segments) != set(EXPECTED_SEGMENTS):
        raise ValueError("trace has the wrong fixed segment set")
    for anchor, next_anchor in EXPECTED_SEGMENTS.items():
        rows = segments[anchor]
        if len(rows) != expected_samples:
            raise ValueError(f"segment {anchor} has incomplete sample coverage")
        if {row["next_anchor_step"] for row in rows} != {next_anchor}:
            raise ValueError(f"segment {anchor} has an unexpected next anchor")
    return samples, segments


def fit_bound(args: argparse.Namespace) -> int:
    source_path = args.development_trace.expanduser().resolve()
    source = _load_json(source_path, "development trace")
    _, segments = _validate_trace(source, expected_samples=args.expected_samples)
    alpha = float(args.alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    segment_models: dict[str, Any] = {}
    for anchor, next_anchor in EXPECTED_SEGMENTS.items():
        rows = segments[anchor]
        scores = []
        for row in rows:
            z_value = float(row["anchor_relative_error"])
            damage = float(row["next_segment_integrated_ratio"])
            if not math.isfinite(z_value) or z_value <= 0.0:
                raise ValueError(f"segment {anchor} has a nonpositive anchor signal")
            if not math.isfinite(damage) or damage < 0.0:
                raise ValueError(f"segment {anchor} has invalid damage")
            scores.append(damage / z_value)
        count = len(scores)
        order_rank = math.ceil((count + 1) * (1.0 - alpha))
        if order_rank > count:
            raise ValueError(
                "development cohort is too small for the requested conformal alpha"
            )
        scale = float(sorted(scores)[order_rank - 1])
        slack = [scale / max(score, 1e-30) for score in scores]
        segment_models[str(anchor)] = {
            "anchor_step": anchor,
            "next_anchor_step": next_anchor,
            "development_sample_count": count,
            "score": "next_segment_integrated_ratio/anchor_relative_error",
            "conformal_order_rank_one_based": order_rank,
            "lambda": scale,
            "finite_sample_marginal_coverage_lower_bound": order_rank / (count + 1),
            "development_score_quantiles": _quantiles(scores),
            "development_bound_slack_quantiles": _quantiles(slack),
        }

    payload = {
        "schema": MODEL_SCHEMA,
        "schema_revision": 1,
        "model_id": "flux-a12-segment-monotone-conformal-bound-20260812",
        "serving_claim": False,
        "semantic_quality_claim": False,
        "bound_family": "U_c(z)=lambda_c*z",
        "coverage_scope": (
            "finite-sample marginal coverage within each fixed denoise segment under exchangeability; "
            "not distribution-free pointwise conditional coverage in z"
        ),
        "target_miscoverage_alpha": alpha,
        "development_source": {
            "path": str(source_path),
            "file_sha256": _sha256_file(source_path),
            "sample_count": args.expected_samples,
        },
        "candidate_id": "derived-static-a12-o1-index",
        "segments": segment_models,
        "holdout_gate": {
            "expected_sample_count": 48,
            "minimum_coverage_per_segment": 46 / 48,
            "minimum_overall_coverage": 0.95,
            "maximum_overall_median_bound_slack": 2.0,
            "maximum_overall_q90_bound_slack": 3.0,
            "maximum_per_segment_median_bound_slack": 2.5,
            "z_quartile_coverage": "diagnostic_only",
            "gate_interpretation": (
                "pass authorizes a separately frozen supported-action dynamic-policy experiment; "
                "it is not a semantic quality or serving qualification"
            ),
        },
    }
    document = {**payload, "sha256": canonical_sha256(payload)}
    _write_json(args.out.expanduser().resolve(), document)
    print(
        f"wrote {args.out}: segments={len(segment_models)} alpha={alpha} "
        f"coverage_lower={next(iter(segment_models.values()))['finite_sample_marginal_coverage_lower_bound']:.6f}"
    )
    return 0


def _quartile_rows(rows: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["anchor_relative_error"]))
    groups = np.array_split(np.arange(len(ordered)), 4)
    result = []
    for index, indices in enumerate(groups):
        selected = [ordered[int(value)] for value in indices]
        covered = sum(
            float(row["next_segment_integrated_ratio"])
            <= scale * float(row["anchor_relative_error"])
            for row in selected
        )
        result.append(
            {
                "z_quartile": index + 1,
                "sample_count": len(selected),
                "covered_count": covered,
                "coverage": covered / len(selected),
                "z_range": [
                    float(selected[0]["anchor_relative_error"]),
                    float(selected[-1]["anchor_relative_error"]),
                ],
            }
        )
    return result


def evaluate_bound(args: argparse.Namespace) -> int:
    model_path = args.bound_model.expanduser().resolve()
    trace_path = args.holdout_trace.expanduser().resolve()
    model = _load_json(model_path, "bound model")
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported bound model schema")
    payload = {key: value for key, value in model.items() if key != "sha256"}
    if model.get("sha256") != canonical_sha256(payload):
        raise ValueError("bound model content hash is invalid")
    trace = _load_json(trace_path, "holdout trace")
    samples, segments = _validate_trace(
        trace, expected_samples=int(model["holdout_gate"]["expected_sample_count"])
    )
    sample_categories = {row["sample_id"]: row.get("category") for row in samples}

    all_slack: list[float] = []
    all_covered: list[bool] = []
    category_rows: dict[str, list[bool]] = {}
    segment_results: dict[str, Any] = {}
    for anchor, next_anchor in EXPECTED_SEGMENTS.items():
        rows = segments[anchor]
        segment_model = model["segments"][str(anchor)]
        if int(segment_model["next_anchor_step"]) != next_anchor:
            raise ValueError("bound model segment identity is inconsistent")
        scale = float(segment_model["lambda"])
        outcomes = []
        slack = []
        for row in rows:
            signal = float(row["anchor_relative_error"])
            damage = float(row["next_segment_integrated_ratio"])
            upper = scale * signal
            covered = damage <= upper
            item_slack = upper / max(damage, 1e-30)
            outcomes.append(covered)
            slack.append(item_slack)
            all_covered.append(covered)
            all_slack.append(item_slack)
            category = str(sample_categories[row["sample_id"]])
            category_rows.setdefault(category, []).append(covered)
        covered_count = sum(outcomes)
        segment_results[str(anchor)] = {
            "anchor_step": anchor,
            "next_anchor_step": next_anchor,
            "lambda": scale,
            "sample_count": len(rows),
            "covered_count": covered_count,
            "miss_count": len(rows) - covered_count,
            "coverage": covered_count / len(rows),
            "coverage_wilson_95pct_ci": _wilson_interval(covered_count, len(rows)),
            "bound_slack_quantiles": _quantiles(slack),
            "z_quartile_coverage": _quartile_rows(rows, scale),
            "gate": {
                "coverage_passed": covered_count / len(rows)
                >= float(model["holdout_gate"]["minimum_coverage_per_segment"]),
                "median_slack_passed": float(np.median(slack))
                <= float(model["holdout_gate"]["maximum_per_segment_median_bound_slack"]),
            },
        }

    overall_covered = sum(all_covered)
    overall_coverage = overall_covered / len(all_covered)
    overall_slack = _quantiles(all_slack)
    segment_coverage_passed = all(
        row["gate"]["coverage_passed"] for row in segment_results.values()
    )
    segment_tightness_passed = all(
        row["gate"]["median_slack_passed"] for row in segment_results.values()
    )
    overall_coverage_passed = overall_coverage >= float(
        model["holdout_gate"]["minimum_overall_coverage"]
    )
    overall_tightness_passed = (
        overall_slack["median"]
        <= float(model["holdout_gate"]["maximum_overall_median_bound_slack"])
        and overall_slack["q90"]
        <= float(model["holdout_gate"]["maximum_overall_q90_bound_slack"])
    )
    passed = bool(
        segment_coverage_passed
        and segment_tightness_passed
        and overall_coverage_passed
        and overall_tightness_passed
    )

    category_results = {}
    for category, outcomes in sorted(category_rows.items()):
        covered = sum(outcomes)
        category_results[category] = {
            "sample_segment_count": len(outcomes),
            "covered_count": covered,
            "coverage": covered / len(outcomes),
            "coverage_wilson_95pct_ci": _wilson_interval(covered, len(outcomes)),
        }
    result_payload = {
        "schema": RESULT_SCHEMA,
        "schema_revision": 1,
        "study_id": "flux-gram-monotone-bound-holdout-20260812",
        "status": "bound_holdout_passed" if passed else "bound_holdout_rejected",
        "serving_claim": False,
        "semantic_quality_claim": False,
        "artifacts": {
            "bound_model": {
                "path": str(model_path),
                "file_sha256": _sha256_file(model_path),
                "content_sha256": model["sha256"],
            },
            "holdout_trace": {
                "path": str(trace_path),
                "file_sha256": _sha256_file(trace_path),
            },
        },
        "sample_count": len(samples),
        "segment_results": segment_results,
        "category_diagnostics": category_results,
        "overall": {
            "sample_segment_count": len(all_covered),
            "covered_count": overall_covered,
            "miss_count": len(all_covered) - overall_covered,
            "coverage": overall_coverage,
            "coverage_wilson_95pct_ci": _wilson_interval(
                overall_covered, len(all_covered)
            ),
            "bound_slack_quantiles": overall_slack,
        },
        "gate": {
            "segment_coverage_passed": segment_coverage_passed,
            "segment_tightness_passed": segment_tightness_passed,
            "overall_coverage_passed": overall_coverage_passed,
            "overall_tightness_passed": overall_tightness_passed,
            "passed": passed,
            "next_if_passed": (
                "freeze a supported-action dynamic policy and compare its anchor use and semantic "
                "quality frontier against static A12/A13 and qualified A14"
            ),
            "not_established": [
                "pointwise conditional coverage for every z",
                "coverage after policy-induced distribution shift",
                "semantic quality protection",
                "serving speedup",
            ],
        },
    }
    result = {**result_payload, "sha256": canonical_sha256(result_payload)}
    _write_json(args.out.expanduser().resolve(), result)
    print(
        f"{result['status']}: coverage={overall_coverage:.6f} "
        f"slack_median={overall_slack['median']:.4f} slack_q90={overall_slack['q90']:.4f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="fit and freeze the development bound")
    fit.add_argument("--development-trace", required=True, type=Path)
    fit.add_argument("--expected-samples", type=int, default=32)
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--out", required=True, type=Path)
    fit.set_defaults(handler=fit_bound)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a frozen bound")
    evaluate.add_argument("--bound-model", required=True, type=Path)
    evaluate.add_argument("--holdout-trace", required=True, type=Path)
    evaluate.add_argument("--out", required=True, type=Path)
    evaluate.set_defaults(handler=evaluate_bound)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
