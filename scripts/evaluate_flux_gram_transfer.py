#!/usr/bin/env python3
"""Evaluate whether clean-trajectory anchor errors transfer to skipped steps.

This is the cheap, offline Gate A for the proposed on-device Gram tap.  It is
deliberately not a cache-rollout or serving qualification: all labels come
from full-DiT trajectories.  The script also compares an FP32 Gram-quadratic
anchor error with the numerically stable direct residual norm so cancellation
in the proposed scalar primitive is visible rather than assumed away.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from difflet.offline.cache_profile.schedule import scheduler_sigmas
from difflet.pipeline.cache.profile import load_phased_candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, matching scipy.stats.rankdata(method='average')."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return float((x @ y) / denominator) if denominator > 0.0 else float("nan")


def _spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    return _pearson(_rankdata(x), _rankdata(y))


def _correlation_summary(
    left: Iterable[float],
    right: Iterable[float],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    observed = _spearman(x, y)
    bootstrap = []
    for _ in range(replicates):
        indices = rng.integers(0, len(x), len(x))
        value = _spearman(x[indices], y[indices])
        if math.isfinite(value):
            bootstrap.append(value)
    permutations = []
    for _ in range(replicates):
        permutations.append(_spearman(x, rng.permutation(y)))
    one_sided_p = (1 + sum(value >= observed for value in permutations)) / (
        replicates + 1
    )
    return {
        "sample_count": len(x),
        "spearman_rho": observed,
        "pearson_r": _pearson(x, y),
        "bootstrap_95pct_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "positive_association_permutation_p_one_sided": float(one_sided_p),
        "replicates": replicates,
    }


def _weights(a: int, b: int, target: int) -> tuple[float, float]:
    ratio = (target - b) / (b - a)
    return -ratio, 1.0 + ratio


def _gram_error(
    gram: np.ndarray,
    a: int,
    b: int,
    target: int,
) -> tuple[float, float, float]:
    wa, wb = _weights(a, b, target)
    ai, bi, ti = a - 1, b - 1, target - 1
    squared = (
        wa * wa * float(gram[ai, ai])
        + wb * wb * float(gram[bi, bi])
        + float(gram[ti, ti])
        + 2.0 * wa * wb * float(gram[ai, bi])
        - 2.0 * wa * float(gram[ai, ti])
        - 2.0 * wb * float(gram[bi, ti])
    )
    absolute = math.sqrt(max(squared, 0.0))
    reference = math.sqrt(max(float(gram[ti, ti]), 0.0))
    return absolute, absolute / max(reference, 1e-12), squared


def _direct_error(velocities: Any, a: int, b: int, target: int) -> tuple[float, float]:
    import torch

    wa, wb = _weights(a, b, target)
    actual = velocities[target - 1]
    residual = wa * velocities[a - 1] + wb * velocities[b - 1] - actual
    absolute = float(torch.linalg.vector_norm(residual.reshape(-1), ord=2))
    reference = float(torch.linalg.vector_norm(actual.reshape(-1), ord=2))
    return absolute, absolute / max(reference, 1e-12)


def _weighted_ratio(rows: list[dict[str, float]], key: str) -> float:
    numerator = sum(row["delta_sigma_abs"] * row[key] for row in rows)
    denominator = sum(row["delta_sigma_abs"] * row["reference_norm"] for row in rows)
    return numerator / max(denominator, 1e-12)


def _weighted_relative_mean(rows: list[dict[str, float]], key: str) -> float:
    numerator = sum(row["delta_sigma_abs"] * row[key] for row in rows)
    denominator = sum(row["delta_sigma_abs"] for row in rows)
    return numerator / max(denominator, 1e-12)


def _source_rows(quality_input: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for comparison in quality_input["comparisons"]:
        relative = comparison["baseline"]["trajectory"]
        unique.setdefault(
            relative,
            {
                "sample_id": comparison["sample_id"],
                "prompt_index": int(comparison["prompt_index"]),
                "prompt": comparison["prompt"],
                "path": (base_dir / relative).resolve(),
            },
        )
    return sorted(unique.values(), key=lambda row: row["prompt_index"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")

    import torch

    quality_input = json.loads(args.quality_input.read_text(encoding="utf-8"))
    candidate = load_phased_candidate(args.candidate)
    generation = quality_input["protocol"]["generation"]
    num_steps = int(generation["num_steps"])
    if candidate.policy["num_steps"] != num_steps:
        raise SystemExit("candidate and quality input disagree on num_steps")
    predictor_spec = candidate.predictor_spec()
    if predictor_spec != {"type": "taylorseer", "order": 1, "coord": "index"}:
        raise SystemExit("Gate A currently requires an order-1 index TaylorSeer candidate")

    sigmas = np.asarray(scheduler_sigmas(generation), dtype=np.float64)
    delta_sigma = np.diff(sigmas)
    anchors = list(candidate.policy["static_anchor_steps"])
    anchor_set = set(anchors)
    source_rows = _source_rows(quality_input, args.quality_input.parent)
    rng = np.random.default_rng(args.seed)

    prompt_metrics: list[dict[str, Any]] = []
    segment_values: dict[int, list[dict[str, float]]] = {}
    cancellation_rows: list[dict[str, float | int | str]] = []
    for source in source_rows:
        trajectory = torch.load(source["path"], map_location="cpu", weights_only=True)
        if not torch.is_tensor(trajectory) or trajectory.shape[0] != num_steps:
            raise ValueError(f"unexpected trajectory shape: {source['path']}")
        states = trajectory.float()
        delta = torch.tensor(delta_sigma[1:], dtype=torch.float32)
        reshape = (len(delta),) + (1,) * (states.ndim - 1)
        velocities = (states[1:] - states[:-1]) / delta.reshape(reshape)
        flat = velocities.reshape(len(delta), -1)
        # Host FP32 simulation of scalar tap arithmetic.  Device reduction
        # precision remains a separate hardware gate.
        gram = torch.mm(flat, flat.t()).double().numpy()

        previous: list[int] = []
        anchor_errors: list[dict[str, float]] = []
        future_anchor_errors: list[dict[str, float]] = []
        skip_errors: list[dict[str, float]] = []
        per_anchor: dict[int, dict[str, float]] = {}
        for step in range(num_steps):
            if step in anchor_set:
                if step >= 1 and len(previous) >= 2 and previous[-2] >= 1:
                    a, b = previous[-2:]
                    gram_abs, gram_rel, gram_squared = _gram_error(gram, a, b, step)
                    direct_abs, direct_rel = _direct_error(velocities, a, b, step)
                    row = {
                        "step": float(step),
                        "delta_sigma_abs": abs(float(delta_sigma[step])),
                        "absolute_error": gram_abs,
                        "relative_error": gram_rel,
                        "direct_absolute_error": direct_abs,
                        "direct_relative_error": direct_rel,
                        "reference_norm": direct_abs / max(direct_rel, 1e-30),
                    }
                    anchor_errors.append(row)
                    per_anchor[step] = row
                    next_anchor_index = anchors.index(step) + 1
                    if (
                        next_anchor_index < len(anchors)
                        and anchors[next_anchor_index] > step + 1
                    ):
                        future_anchor_errors.append(row)
                    cancellation_rows.append(
                        {
                            "sample_id": source["sample_id"],
                            "step": step,
                            "gram_squared_before_clamp": gram_squared,
                            "gram_relative_error": gram_rel,
                            "direct_relative_error": direct_rel,
                            "relative_discrepancy": abs(gram_rel - direct_rel)
                            / max(direct_rel, 1e-30),
                        }
                    )
                previous = (previous + [step])[-2:]
            elif len(previous) == 2 and previous[-2] >= 1:
                a, b = previous
                direct_abs, direct_rel = _direct_error(velocities, a, b, step)
                skip_errors.append(
                    {
                        "step": float(step),
                        "delta_sigma_abs": abs(float(delta_sigma[step])),
                        "absolute_error": direct_abs,
                        "relative_error": direct_rel,
                        "reference_norm": direct_abs / max(direct_rel, 1e-30),
                    }
                )

        for anchor_step, anchor_row in per_anchor.items():
            next_anchors = [value for value in anchors if value > anchor_step]
            if not next_anchors:
                continue
            next_anchor = next_anchors[0]
            next_rows = [
                row for row in skip_errors if anchor_step < int(row["step"]) < next_anchor
            ]
            if not next_rows:
                continue
            segment_values.setdefault(anchor_step, []).append(
                {
                    "anchor_relative_error": anchor_row["relative_error"],
                    "next_segment_relative_weighted_mean": _weighted_relative_mean(
                        next_rows, "relative_error"
                    ),
                    "next_segment_integrated_ratio": _weighted_ratio(
                        next_rows, "absolute_error"
                    ),
                }
            )

        prompt_metrics.append(
            {
                "sample_id": source["sample_id"],
                "prompt_index": source["prompt_index"],
                "anchor_probe_count": len(future_anchor_errors),
                "skip_count": len(skip_errors),
                "anchor_relative_weighted_mean": _weighted_relative_mean(
                    future_anchor_errors, "relative_error"
                ),
                "skip_relative_weighted_mean": _weighted_relative_mean(
                    skip_errors, "relative_error"
                ),
                "anchor_integrated_ratio": _weighted_ratio(
                    future_anchor_errors, "absolute_error"
                ),
                "skip_integrated_ratio": _weighted_ratio(skip_errors, "absolute_error"),
            }
        )

    primary = _correlation_summary(
        [row["anchor_integrated_ratio"] for row in prompt_metrics],
        [row["skip_integrated_ratio"] for row in prompt_metrics],
        rng=rng,
        replicates=args.bootstrap_replicates,
    )
    sensitivity = _correlation_summary(
        [row["anchor_relative_weighted_mean"] for row in prompt_metrics],
        [row["skip_relative_weighted_mean"] for row in prompt_metrics],
        rng=rng,
        replicates=args.bootstrap_replicates,
    )
    segment_results = {
        str(step): _correlation_summary(
            [row["anchor_relative_error"] for row in rows],
            [row["next_segment_relative_weighted_mean"] for row in rows],
            rng=rng,
            replicates=args.bootstrap_replicates,
        )
        for step, rows in sorted(segment_values.items())
    }
    positive_segments = sum(
        result["spearman_rho"] > 0.0 for result in segment_results.values()
    )
    advance = bool(
        primary["spearman_rho"] >= 0.4
        and primary["bootstrap_95pct_ci"][0] > 0.0
        and primary["positive_association_permutation_p_one_sided"] < 0.05
        and positive_segments >= math.ceil(len(segment_results) / 2)
    )

    discrepancies = np.asarray(
        [row["relative_discrepancy"] for row in cancellation_rows], dtype=np.float64
    )
    negative_squared = sum(
        float(row["gram_squared_before_clamp"]) < 0.0 for row in cancellation_rows
    )
    result = {
        "schema": "difflet-flux-gram-transfer-clean-gate-result",
        "schema_revision": 1,
        "study_id": "flux-gram-transfer-clean-gate-20260811",
        "status": "advance_to_real_cache_gate_b" if advance else "clean_gate_rejected",
        "serving_claim": False,
        "evidence_scope": "teacher_forced_full_dit_trajectories_only",
        "sources": {
            "quality_input": {
                "path": str(args.quality_input.resolve()),
                "file_sha256": _sha256_file(args.quality_input),
            },
            "candidate": {
                "path": str(args.candidate.resolve()),
                "file_sha256": _sha256_file(args.candidate),
                "candidate_id": candidate.candidate_id,
            },
        },
        "generation_identity": generation,
        "schedule": {
            "anchors": anchors,
            "counterfactual_anchor_steps_with_future_skips": sorted(segment_values),
            "predictor": predictor_spec,
        },
        "frozen_gate": {
            "primary_metric": "prompt-level scheduler-weighted integrated relative L2 scale",
            "advance_if": {
                "minimum_spearman_rho": 0.4,
                "bootstrap_95pct_lower_bound_strictly_positive": True,
                "maximum_one_sided_permutation_p": 0.05,
                "minimum_positive_fixed_segment_fraction": 0.5,
            },
            "interpretation": "advance only authorizes Gate B; it does not establish adaptive control or serving quality",
        },
        "primary_prompt_level": primary,
        "relative_mean_sensitivity": sensitivity,
        "fixed_next_segment_diagnostics": segment_results,
        "positive_fixed_segment_count": positive_segments,
        "fixed_segment_count": len(segment_results),
        "fp32_gram_numerical_audit": {
            "comparison": "Gram quadratic relative error versus direct residual relative L2",
            "measurement_count": len(cancellation_rows),
            "negative_squared_residual_count_before_clamp": negative_squared,
            "relative_discrepancy_median": float(np.median(discrepancies)),
            "relative_discrepancy_q95": float(np.quantile(discrepancies, 0.95)),
            "relative_discrepancy_max": float(np.max(discrepancies)),
            "limitation": "CPU FP32 reduction simulation; Trainium compiled reduction precision still requires measurement",
        },
        "prompt_metrics": prompt_metrics,
        "decision": {
            "advance_to_gate_b": advance,
            "gate_b_required_measurement": "real cache rollout; z_c at actual anchors versus shadow full-DiT error on each subsequent skipped step evaluated on the identical pre-step cache latent",
            "quality_budget_claim_supported": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{result['status']}: rho={primary['spearman_rho']:.4f}, "
        f"95% CI=[{primary['bootstrap_95pct_ci'][0]:.4f}, "
        f"{primary['bootstrap_95pct_ci'][1]:.4f}], "
        f"p={primary['positive_association_permutation_p_one_sided']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
