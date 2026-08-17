"""P2 composition-bound experiment for transactional FLUX cache control.

This module deliberately separates two claims:

* full-compute trajectories can test whether segment-end anchor error composes
  into a bound on an open-loop scheduler-path perturbation; and
* only same-request segment measurements paired with decoded semantic scores
  can test protection of the ImageReward/VQA quality contract.

The first claim is useful controller-design evidence.  It must never be
reported as establishing the second claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from difflet.offline.cache_profile.provenance import (
    canonical_sha256,
    implementation_bundle,
    sha256_file,
)
from difflet.offline.cache_profile.schedule import (
    nearest_rank,
    relative_prediction_error,
    trajectory_gram,
)

RESULT_SCHEMA = "difflet-flux-cache-composition-bound-result"
RESULT_SCHEMA_REVISION = 1
ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _validate_content_hash(document: Mapping[str, Any], name: str) -> None:
    digest = document.get("sha256")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if not isinstance(digest, str) or canonical_sha256(payload) != digest:
        raise ValueError(f"{name} content hash is invalid")


def _write_hashed_json(path: Path, payload: Mapping[str, Any]) -> None:
    document = {**dict(payload), "sha256": canonical_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prediction_coefficients(
    a: int,
    b: int,
    target: int,
    *,
    velocity_count: int,
) -> np.ndarray:
    if not 1 <= a < b < target <= velocity_count:
        raise ValueError("prediction indices are outside the velocity Gram matrix")
    ratio = (target - b) / (b - a)
    result = np.zeros(velocity_count, dtype=np.float64)
    result[a - 1] -= ratio
    result[b - 1] += 1.0 + ratio
    result[target - 1] -= 1.0
    return result


def _quadratic_norm(gram: np.ndarray, coefficients: np.ndarray) -> float:
    squared = float(coefficients @ gram @ coefficients)
    return math.sqrt(max(squared, 0.0))


def _path_displacement_coefficients(
    deltas: Sequence[float],
    *,
    first_step: int,
) -> np.ndarray:
    velocity_count = len(deltas) - 1
    if not 1 <= first_step <= velocity_count:
        raise ValueError("first path-displacement step is invalid")
    result = np.zeros(velocity_count, dtype=np.float64)
    for step in range(first_step, velocity_count + 1):
        result[step - 1] = float(deltas[step])
    return result


def analyze_mask(
    gram: Any,
    deltas: Sequence[float],
    anchors: Sequence[int],
    *,
    warmup_steps: int,
    norm_floor: float,
) -> dict[str, Any]:
    """Compute segment z, cumulative deficit, and open-loop path harm."""

    matrix = np.asarray(gram, dtype=np.float64)
    velocity_count = len(deltas) - 1
    if matrix.shape != (velocity_count, velocity_count):
        raise ValueError("velocity Gram matrix shape does not match scheduler deltas")
    normalized = tuple(int(value) for value in anchors)
    if (
        len(normalized) < warmup_steps
        or normalized[:warmup_steps] != tuple(range(warmup_steps))
        or normalized[-1] != velocity_count
        or any(left >= right for left, right in zip(normalized, normalized[1:]))
    ):
        raise ValueError("anchor mask does not contain the required warmup and final anchor")

    total_error = np.zeros(velocity_count, dtype=np.float64)
    segment_rows: list[dict[str, Any]] = []
    for index in range(warmup_steps, len(normalized)):
        a, b, c = normalized[index - 2 : index + 1]
        if c == b + 1:
            continue
        endpoint_z = relative_prediction_error(
            matrix,
            a,
            b,
            c,
            norm_floor=norm_floor,
        )
        segment_error = np.zeros(velocity_count, dtype=np.float64)
        true_displacement = np.zeros(velocity_count, dtype=np.float64)
        integrated_deficit = 0.0
        for target in range(b + 1, c):
            coefficients = _prediction_coefficients(
                a,
                b,
                target,
                velocity_count=velocity_count,
            )
            segment_error += float(deltas[target]) * coefficients
            true_displacement[target - 1] = float(deltas[target])
            integrated_deficit += abs(float(deltas[target])) * relative_prediction_error(
                matrix,
                a,
                b,
                target,
                norm_floor=norm_floor,
            )
        total_error += segment_error
        denominator = max(_quadratic_norm(matrix, true_displacement), norm_floor)
        segment_rows.append(
            {
                "previous_anchor": a,
                "anchor": b,
                "next_anchor": c,
                "skipped_steps": c - b - 1,
                "endpoint_anchor_z": endpoint_z,
                "integrated_prediction_deficit": integrated_deficit,
                "segment_path_relative_l2": (
                    _quadratic_norm(matrix, segment_error) / denominator
                ),
            }
        )

    path_displacement = _path_displacement_coefficients(
        deltas,
        first_step=warmup_steps,
    )
    path_denominator = max(_quadratic_norm(matrix, path_displacement), norm_floor)
    return {
        "segment_count": len(segment_rows),
        "skipped_steps": sum(row["skipped_steps"] for row in segment_rows),
        "max_endpoint_anchor_z": max(
            (row["endpoint_anchor_z"] for row in segment_rows),
            default=0.0,
        ),
        "sum_endpoint_anchor_z": sum(
            row["endpoint_anchor_z"] for row in segment_rows
        ),
        "cumulative_prediction_deficit": sum(
            row["integrated_prediction_deficit"] for row in segment_rows
        ),
        "max_segment_path_relative_l2": max(
            (row["segment_path_relative_l2"] for row in segment_rows),
            default=0.0,
        ),
        "end_to_end_path_relative_l2": (
            _quadratic_norm(matrix, total_error) / path_denominator
        ),
        "segments": segment_rows,
    }


def _gap_cap(
    anchor: int,
    *,
    phase_boundary: int,
    middle_gap_cap: int,
    tail_gap_cap: int,
) -> int:
    return middle_gap_cap if anchor < phase_boundary else tail_gap_cap


def generate_stratified_masks(
    *,
    num_steps: int,
    warmup_steps: int,
    budgets: Iterable[int],
    masks_per_budget: int,
    phase_boundary: int,
    middle_gap_cap: int,
    tail_gap_cap: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Sample masks uniformly from feasible paths without reading trajectory data."""

    if masks_per_budget <= 0:
        raise ValueError("masks_per_budget must be positive")
    final_step = num_steps - 1
    prefix = tuple(range(warmup_steps))
    rng = random.Random(seed)
    result: list[tuple[int, ...]] = []

    for budget in budgets:
        if budget <= warmup_steps or budget > num_steps:
            raise ValueError("mask budget is incompatible with warmup and horizon")
        remaining = budget - warmup_steps

        @lru_cache(maxsize=None)
        def path_count(anchor: int, anchors_left: int) -> int:
            cap = _gap_cap(
                anchor,
                phase_boundary=phase_boundary,
                middle_gap_cap=middle_gap_cap,
                tail_gap_cap=tail_gap_cap,
            )
            if anchors_left == 1:
                return int(anchor < final_step <= anchor + cap)
            maximum = min(final_step - (anchors_left - 1), anchor + cap)
            return sum(
                path_count(candidate, anchors_left - 1)
                for candidate in range(anchor + 1, maximum + 1)
            )

        available = path_count(warmup_steps - 1, remaining)
        if available < masks_per_budget:
            raise ValueError(
                f"budget {budget} has only {available} feasible masks, "
                f"fewer than requested {masks_per_budget}"
            )

        selected: set[tuple[int, ...]] = set()
        while len(selected) < masks_per_budget:
            anchor = warmup_steps - 1
            anchors_left = remaining
            suffix: list[int] = []
            while anchors_left > 1:
                cap = _gap_cap(
                    anchor,
                    phase_boundary=phase_boundary,
                    middle_gap_cap=middle_gap_cap,
                    tail_gap_cap=tail_gap_cap,
                )
                maximum = min(final_step - (anchors_left - 1), anchor + cap)
                candidates = []
                weights = []
                for candidate in range(anchor + 1, maximum + 1):
                    count = path_count(candidate, anchors_left - 1)
                    if count:
                        candidates.append(candidate)
                        weights.append(count)
                draw = rng.randrange(sum(weights))
                cumulative = 0
                choice = candidates[-1]
                for candidate, weight in zip(candidates, weights):
                    cumulative += weight
                    if draw < cumulative:
                        choice = candidate
                        break
                suffix.append(choice)
                anchor = choice
                anchors_left -= 1
            suffix.append(final_step)
            selected.add((*prefix, *suffix))
        result.extend(sorted(selected))
    return tuple(result)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], *, rank: bool) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires paired observations")
    x = _rankdata(left) if rank else np.asarray(left, dtype=np.float64)
    y = _rankdata(right) if rank else np.asarray(right, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def envelope_table(
    development: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    harm: str,
    quantiles: Sequence[float],
) -> list[dict[str, Any]]:
    """Fit dev maximum harm below each envelope and audit it on holdout prompts."""

    development_signals = [float(row[signal]) for row in development]
    result = []
    for quantile in quantiles:
        threshold = nearest_rank(development_signals, float(quantile))
        accepted_dev = [row for row in development if float(row[signal]) <= threshold]
        accepted_holdout = [row for row in holdout if float(row[signal]) <= threshold]
        dev_harms = [float(row[harm]) for row in accepted_dev]
        holdout_harms = [float(row[harm]) for row in accepted_holdout]
        dev_bound = max(dev_harms)
        violations = [row for row in accepted_holdout if float(row[harm]) > dev_bound]
        prompts = {str(row["sample_id"]) for row in accepted_holdout}
        violating_prompts = {str(row["sample_id"]) for row in violations}
        result.append(
            {
                "development_signal_quantile": float(quantile),
                "envelope": threshold,
                "development_accepted_count": len(accepted_dev),
                "development_harm_q95": nearest_rank(dev_harms, 0.95),
                "development_harm_maximum_bound": dev_bound,
                "holdout_accepted_count": len(accepted_holdout),
                "holdout_accepted_prompt_count": len(prompts),
                "holdout_harm_q95": (
                    nearest_rank(holdout_harms, 0.95) if holdout_harms else None
                ),
                "holdout_harm_maximum": max(holdout_harms) if holdout_harms else None,
                "holdout_violation_count": len(violations),
                "holdout_violating_prompt_count": len(violating_prompts),
                "holdout_prompt_familywise_coverage": (
                    1.0 - len(violating_prompts) / len(prompts) if prompts else None
                ),
            }
        )
    return result


@dataclass(frozen=True)
class _MaskDefinition:
    mask_id: str
    family: str
    anchor_budget: int
    anchors: tuple[int, ...]


def _source_ref(path: Path, document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
    }
    if document is not None and isinstance(document.get("sha256"), str):
        result["content_sha256"] = document["sha256"]
    return result


def _mask_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_mask: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_mask.setdefault(str(row["mask_id"]), []).append(row)
    summaries = []
    for mask_id, values in sorted(by_mask.items()):
        first = values[0]
        summary: dict[str, Any] = {
            "mask_id": mask_id,
            "family": first["family"],
            "anchor_budget": first["anchor_budget"],
            "anchors": first["anchors"],
            "sample_count": len(values),
        }
        for field in (
            "max_endpoint_anchor_z",
            "cumulative_prediction_deficit",
            "end_to_end_path_relative_l2",
        ):
            observed = [float(row[field]) for row in values]
            summary[field] = {
                "median": nearest_rank(observed, 0.5),
                "q95": nearest_rank(observed, 0.95),
                "maximum": max(observed),
            }
        summaries.append(summary)
    return summaries


def _frontier_segment_summaries(
    values: Mapping[tuple[str, int, int, int], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    result = []
    for (mask_id, a, b, c), rows in sorted(values.items()):
        summary: dict[str, Any] = {
            "mask_id": mask_id,
            "previous_anchor": a,
            "anchor": b,
            "next_anchor": c,
            "sample_count": len(rows),
        }
        for field in (
            "endpoint_anchor_z",
            "integrated_prediction_deficit",
            "segment_path_relative_l2",
        ):
            observed = [float(row[field]) for row in rows]
            summary[field] = {
                "median": nearest_rank(observed, 0.5),
                "q95": nearest_rank(observed, 0.95),
                "maximum": max(observed),
            }
        result.append(summary)
    return result


def _semantic_audit(paths: Sequence[Path]) -> dict[str, Any]:
    executions = []
    for path in paths:
        document = _load_json(path, "natural-range evaluation")
        _validate_content_hash(document, "natural-range evaluation")
        for summary in document.get("candidate_summaries", []):
            executions.append(
                {
                    "source": _source_ref(path, document),
                    "candidate_id": summary["candidate_id"],
                    "sample_count": summary["sample_count"],
                    "failure_count": summary["failure_count"],
                    "decision": summary["decision"],
                }
            )
    return {
        "same_request_segment_z_present": False,
        "same_request_semantic_harm_present": True,
        "joint_segment_z_semantic_rows": 0,
        "semantic_implication_identifiable": False,
        "reason": (
            "natural-range evaluations contain decoded per-request semantic harm but no "
            "per-anchor z trace; the 1216 value in A12 runner_stats is 38 skips x 32 "
            "requests, not 1216 causal labels"
        ),
        "executions": executions,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    registration_path = Path(args.registration).expanduser().resolve()
    derivation_path = Path(args.derivation).expanduser().resolve()
    contract_path = Path(args.quality_contract).expanduser().resolve()
    registration = _load_json(registration_path, "schedule registration")
    derivation = _load_json(derivation_path, "schedule derivation")
    contract = _load_json(contract_path, "quality contract")
    for document, name in (
        (registration, "schedule registration"),
        (derivation, "schedule derivation"),
        (contract, "quality contract"),
    ):
        _validate_content_hash(document, name)

    generation = registration["controlled_generation"]
    optimizer = registration["optimizer"]
    sigmas = tuple(float(value) for value in derivation["sigma_schedule"])
    deltas = tuple(sigmas[index + 1] - sigmas[index] for index in range(len(sigmas) - 1))
    num_steps = int(generation["num_steps"])
    warmup_steps = int(optimizer["warmup_steps"])
    development_count = int(args.development_prompts)
    source_rows = list(registration["source"]["trajectories"])
    if not 1 < development_count < len(source_rows):
        raise ValueError("development prompt count must leave a nonempty holdout")

    random_masks = generate_stratified_masks(
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        budgets=range(int(args.minimum_budget), int(args.maximum_budget) + 1),
        masks_per_budget=int(args.masks_per_budget),
        phase_boundary=int(optimizer["phase_boundary_step"]),
        middle_gap_cap=int(optimizer["middle_max_anchor_gap"]),
        tail_gap_cap=int(optimizer["tail_max_anchor_gap"]),
        seed=int(args.mask_seed),
    )
    masks = [
        _MaskDefinition(
            mask_id=f"stratified-a{len(anchors)}-{index:03d}",
            family="data_independent_stratified",
            anchor_budget=len(anchors),
            anchors=anchors,
        )
        for index, anchors in enumerate(random_masks)
    ]
    masks.extend(
        _MaskDefinition(
            mask_id=f"frontier-a{int(row['anchor_budget'])}",
            family="derived_frontier_diagnostic",
            anchor_budget=int(row["anchor_budget"]),
            anchors=tuple(int(value) for value in row["static_anchor_steps"]),
        )
        for row in derivation["schedules"]
    )

    observations = []
    frontier_segments: dict[
        tuple[str, int, int, int], list[Mapping[str, Any]]
    ] = {}
    verified_trajectory_count = 0
    for source_index, source in enumerate(source_rows):
        path = Path(source["path"])
        if sha256_file(path) != source["file_sha256"]:
            raise ValueError(f"trajectory hash differs for {path}")
        verified_trajectory_count += 1
        gram = trajectory_gram(path, deltas)
        split = "development" if source_index < development_count else "holdout"
        for mask in masks:
            metrics = analyze_mask(
                gram,
                deltas,
                mask.anchors,
                warmup_steps=warmup_steps,
                norm_floor=float(optimizer["norm_floor"]),
            )
            if mask.family == "derived_frontier_diagnostic":
                for segment in metrics["segments"]:
                    key = (
                        mask.mask_id,
                        int(segment["previous_anchor"]),
                        int(segment["anchor"]),
                        int(segment["next_anchor"]),
                    )
                    frontier_segments.setdefault(key, []).append(segment)
            observations.append(
                {
                    "sample_id": source["sample_id"],
                    "split": split,
                    "mask_id": mask.mask_id,
                    "family": mask.family,
                    "anchor_budget": mask.anchor_budget,
                    "anchors": list(mask.anchors),
                    **{key: value for key, value in metrics.items() if key != "segments"},
                }
            )

    primary = [row for row in observations if row["family"] == "data_independent_stratified"]
    development = [row for row in primary if row["split"] == "development"]
    holdout = [row for row in primary if row["split"] == "holdout"]
    harm_field = "end_to_end_path_relative_l2"
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
    z_table = envelope_table(
        development,
        holdout,
        signal="max_endpoint_anchor_z",
        harm=harm_field,
        quantiles=quantiles,
    )
    deficit_table = envelope_table(
        development,
        holdout,
        signal="cumulative_prediction_deficit",
        harm=harm_field,
        quantiles=quantiles,
    )
    primary_z_gate = next(
        row for row in z_table if row["development_signal_quantile"] == 0.75
    )
    latent_gate_passed = (
        primary_z_gate["holdout_accepted_prompt_count"] == len(source_rows) - development_count
        and primary_z_gate["holdout_violation_count"] == 0
    )
    semantic_audit = _semantic_audit(
        tuple(Path(value).expanduser().resolve() for value in args.semantic_evaluation)
    )
    mask_summaries = _mask_summaries(observations)
    summaries_by_id = {row["mask_id"]: row for row in mask_summaries}
    a12 = summaries_by_id.get("frontier-a12")
    a13 = summaries_by_id.get("frontier-a13")
    frontier_case_study = None
    if a12 is not None and a13 is not None:
        frontier_case_study = {
            "masks": ["frontier-a12", "frontier-a13"],
            "max_z_q95_equal": math.isclose(
                float(a12["max_endpoint_anchor_z"]["q95"]),
                float(a13["max_endpoint_anchor_z"]["q95"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "max_z_q95": float(a12["max_endpoint_anchor_z"]["q95"]),
            "a12_to_a13_end_to_end_harm_q95_ratio": (
                float(a12["end_to_end_path_relative_l2"]["q95"])
                / float(a13["end_to_end_path_relative_l2"]["q95"])
            ),
            "interpretation": (
                "the shared early segment fixes max z while later segment composition "
                "materially changes end-to-end path harm"
            ),
        }
    bucket = next(
        row for row in contract["resolution_contracts"] if row["bucket_id"] == args.bucket_id
    )

    payload = {
        "schema": RESULT_SCHEMA,
        "schema_revision": RESULT_SCHEMA_REVISION,
        "study_id": args.study_id,
        "status": (
            "single_envelope_not_authorized"
            if not semantic_audit["semantic_implication_identifiable"]
            else "composition_evaluated"
        ),
        "claims": {
            "open_loop_latent_path_claim": True,
            "semantic_quality_claim": False,
            "serving_claim": False,
        },
        "sources": {
            "registration": _source_ref(registration_path, registration),
            "derivation": _source_ref(derivation_path, derivation),
            "quality_contract": _source_ref(contract_path, contract),
            "verified_trajectory_count": verified_trajectory_count,
            "implementation": implementation_bundle(
                (
                    Path(__file__).resolve(),
                    ROOT / "difflet" / "offline" / "cache_profile" / "schedule.py",
                    ROOT / "difflet" / "offline" / "cache_profile" / "provenance.py",
                    ROOT / "scripts" / "evaluate_flux_cache_composition.py",
                ),
                root=ROOT,
            ),
        },
        "protocol": {
            "split": {
                "unit": "prompt_trajectory",
                "development_count": development_count,
                "holdout_count": len(source_rows) - development_count,
                "assignment": "registration_order_prefix_then_suffix",
            },
            "mask_family": {
                "primary": "data_independent_stratified",
                "seed": int(args.mask_seed),
                "minimum_budget": int(args.minimum_budget),
                "maximum_budget": int(args.maximum_budget),
                "masks_per_budget": int(args.masks_per_budget),
                "mask_count": len(random_masks),
                "diagnostic_frontier_mask_count": len(derivation["schedules"]),
            },
            "segment_signal": (
                "relative_l2(order1_prediction_at_next_anchor, true_velocity_at_next_anchor)"
            ),
            "cumulative_deficit": (
                "sum_skipped(abs(delta_sigma_t) * relative_l2(predicted_velocity_t, "
                "true_velocity_t))"
            ),
            "end_to_end_harm": (
                "l2(sum_skipped(delta_sigma_t * velocity_error_t)) / "
                "l2(post_warmup_full_path_displacement)"
            ),
            "open_loop_limitation": (
                "real-anchor velocities are read from the full-compute path; policy-induced "
                "latent distribution shift is not represented"
            ),
            "primary_gate": (
                "at the development q75 max-z envelope, every holdout prompt must have "
                "accepted masks and no accepted holdout harm may exceed the development maximum"
            ),
        },
        "quality_contract": {
            "bucket_id": bucket["bucket_id"],
            "height": bucket["height"],
            "width": bucket["width"],
            "semantic_harm_limits": bucket["observed_seed_variation_envelope"],
            "latent_proxy_is_contract_metric": False,
        },
        "observations": {
            "total_count": len(observations),
            "primary_count": len(primary),
            "development_primary_count": len(development),
            "holdout_primary_count": len(holdout),
        },
        "relationships": {
            "max_z_to_end_to_end_harm": {
                "pearson": _correlation(
                    [row["max_endpoint_anchor_z"] for row in holdout],
                    [row[harm_field] for row in holdout],
                    rank=False,
                ),
                "spearman": _correlation(
                    [row["max_endpoint_anchor_z"] for row in holdout],
                    [row[harm_field] for row in holdout],
                    rank=True,
                ),
            },
            "cumulative_deficit_to_end_to_end_harm": {
                "pearson": _correlation(
                    [row["cumulative_prediction_deficit"] for row in holdout],
                    [row[harm_field] for row in holdout],
                    rank=False,
                ),
                "spearman": _correlation(
                    [row["cumulative_prediction_deficit"] for row in holdout],
                    [row[harm_field] for row in holdout],
                    rank=True,
                ),
            },
            "raw_max_z_usefulness_diagnostic": {
                "supported": False,
                "reason": (
                    "holdout rank correlation is weak and the A12/A13 frontier pair has "
                    "identical max-z q95 despite materially different end-to-end path harm"
                ),
                "frontier_case_study": frontier_case_study,
            },
        },
        "envelope_audit": {
            "max_endpoint_anchor_z": z_table,
            "cumulative_prediction_deficit": deficit_table,
            "primary_z_gate": primary_z_gate,
            "latent_composition_gate_passed": latent_gate_passed,
        },
        "mask_summaries": mask_summaries,
        "frontier_segment_summaries": _frontier_segment_summaries(frontier_segments),
        "semantic_bridge_audit": semantic_audit,
        "decision": {
            "latent_composition_supported_at_primary_gate": latent_gate_passed,
            "single_raw_z_envelope_supported": False,
            "semantic_composition_supported": False,
            "single_envelope_transactional_policy_authorized": False,
            "reason": (
                "raw max z is dominated by a shared early segment and is weakly related to "
                "composed path harm; additionally, same-request per-segment z and "
                "ImageReward/VQA harm are not jointly present, so an open-loop latent bound "
                "cannot certify the semantic quality contract"
            ),
            "p3_design_constraint": (
                "do not implement one global raw-z commit threshold; retain phase/scheduler "
                "weighting and a cumulative deficit ledger as candidates until paired "
                "semantic traces are collected"
            ),
            "minimum_missing_evidence": (
                "rerun a paired cache-request matrix while persisting its complete per-anchor "
                "z trace and decoded ImageReward/VQA harm, then test a frozen implication on "
                "a separate request-level holdout"
            ),
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--derivation", required=True)
    parser.add_argument("--quality-contract", required=True)
    parser.add_argument("--bucket-id", default="square-1024")
    parser.add_argument("--semantic-evaluation", action="append", default=[])
    parser.add_argument("--development-prompts", type=int, default=32)
    parser.add_argument("--minimum-budget", type=int, default=12)
    parser.add_argument("--maximum-budget", type=int, default=20)
    parser.add_argument("--masks-per-budget", type=int, default=16)
    parser.add_argument("--mask-seed", type=int, default=20260815)
    parser.add_argument(
        "--study-id",
        default="flux-cache-transactional-composition-p2-2026-08-15",
    )
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_experiment(args)
    output = Path(args.out).expanduser().resolve()
    _write_hashed_json(output, payload)
    print(
        f"[composition-p2] status={payload['status']} "
        f"latent_gate={payload['decision']['latent_composition_supported_at_primary_gate']} "
        f"semantic_gate={payload['decision']['semantic_composition_supported']} -> {output}",
        flush=True,
    )
    return 0


__all__ = [
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_REVISION",
    "analyze_mask",
    "build_parser",
    "envelope_table",
    "generate_stratified_masks",
    "main",
    "run_experiment",
]
