"""P2b paired study: per-segment cache risk versus decoded semantic harm.

The P2 composition experiment ruled out a single raw-z commit rule using
full-compute trajectories alone.  It could not test the ImageReward/VQAScore
contract itself, because no historical artifact carried per-anchor endpoint
errors and decoded semantic harm for the *same* request.

This module consumes exactly that evidence once it is collected: one
natural-range evaluation supplies per-request semantic harm, and the quality
manifest it is hash-bound to supplies the per-request anchor-error trace.  The
study reports how the fixed-schedule segment signals relate to contract
failures.

It deliberately selects nothing.  Fitting the per-segment thresholds ``tau_s``
and the cumulative budget ``C`` is a separate registered step; this module only
reports whether the observed number of contract positives authorizes that step.
A diagnostic computed here is development evidence, never a serving claim.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from difflet.offline.cache_profile.composition import (
    _correlation,
    _load_json,
    _source_ref,
    _validate_content_hash,
    _write_hashed_json,
)
from difflet.offline.cache_profile.provenance import implementation_bundle, sha256_file
from difflet.offline.cache_profile.schedule import nearest_rank

RESULT_SCHEMA = "difflet-flux-cache-segment-risk-study-result"
RESULT_SCHEMA_REVISION = 1
ROOT = Path(__file__).resolve().parents[3]

EVALUATION_SCHEMA = "difflet-flux-cache-natural-range-evaluation"
QUALITY_INPUT_SCHEMA = "difflet-flux-cache-quality-input-v1"
METRICS = ("image_reward", "vqa_score")
REQUEST_SIGNALS = (
    "max_endpoint_z",
    "max_segment_risk",
    "cumulative_risk_ledger",
)
HARM_FIELDS = (
    "image_reward_harm",
    "vqa_score_harm",
    "contract_utilization",
)
DEFAULT_MINIMUM_POSITIVES = 6


@dataclass(frozen=True)
class SegmentRisk:
    """One cache segment of one request, as seen by an online controller."""

    previous_anchor_step_index: int
    anchor_step_index: int
    policy_region: str
    estimate_status: str
    numerically_valid: bool
    endpoint_z: float | None
    scheduler_exposure: float | None

    @property
    def segment_id(self) -> str:
        return f"{self.previous_anchor_step_index}->{self.anchor_step_index}"

    @property
    def skipped_step_count(self) -> int:
        return self.anchor_step_index - self.previous_anchor_step_index - 1

    @property
    def risk(self) -> float | None:
        """Return the candidate online score ``r_s = W_s z_s``.

        This is a proposal under test, not a proven bound on the skipped-step
        errors of the segment.
        """

        if self.endpoint_z is None or self.scheduler_exposure is None:
            return None
        return float(self.scheduler_exposure * self.endpoint_z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "previous_anchor_step_index": self.previous_anchor_step_index,
            "anchor_step_index": self.anchor_step_index,
            "skipped_step_count": self.skipped_step_count,
            "policy_region": self.policy_region,
            "estimate_status": self.estimate_status,
            "numerically_valid": self.numerically_valid,
            "endpoint_z": self.endpoint_z,
            "scheduler_exposure": self.scheduler_exposure,
            "risk": self.risk,
        }


def _finite_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def segment_risks(trace: Mapping[str, Any]) -> tuple[SegmentRisk, ...]:
    """Convert one bound anchor-error trace into its cache-segment observations.

    The first real anchor of a request opens the first segment but closes none,
    so it produces no observation.  Consecutive real anchors do produce an
    observation with zero scheduler exposure: no step was skipped, so the
    segment carries no cache risk even when the predictor happens to disagree
    with the measured output.
    """

    if not isinstance(trace, Mapping):
        raise ValueError("anchor-error trace must be a mapping")
    entries = trace.get("entries")
    if not isinstance(entries, list):
        raise ValueError("anchor-error trace entries must be a list")
    if trace.get("physical_rollback_attempts_included") is not False:
        raise ValueError("segment risk requires a logical post-restore trace")

    risks: list[SegmentRisk] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("anchor-error trace entry must be a mapping")
        previous = entry.get("previous_anchor_step_index")
        anchor = entry.get("anchor_step_index")
        region = entry.get("policy_region")
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            raise ValueError("anchor-error trace entry has no anchor step")
        if not isinstance(region, str) or not region:
            raise ValueError("anchor-error trace entry is not phase-bound")
        if previous is None:
            continue
        if isinstance(previous, bool) or not isinstance(previous, int) or previous >= anchor:
            raise ValueError("anchor-error trace entry segment coordinates are invalid")
        status = entry.get("estimate_status")
        if not isinstance(status, str) or not status:
            raise ValueError("anchor-error trace entry has no estimate status")
        valid = entry.get("numerically_valid")
        if type(valid) is not bool:
            raise ValueError("anchor-error trace entry validity must be a boolean")
        endpoint_z = _finite_or_none(entry.get("endpoint_z"), "endpoint z")
        if endpoint_z is not None and endpoint_z < 0.0:
            raise ValueError("endpoint z must be nonnegative")
        exposure = _finite_or_none(
            entry.get("scheduler_abs_delta_sigma"),
            "scheduler exposure",
        )
        if exposure is not None and exposure < 0.0:
            raise ValueError("scheduler exposure must be nonnegative")
        if status != "measured":
            endpoint_z = None
        risks.append(
            SegmentRisk(
                previous_anchor_step_index=previous,
                anchor_step_index=anchor,
                policy_region=region,
                estimate_status=status,
                numerically_valid=valid,
                endpoint_z=endpoint_z,
                scheduler_exposure=exposure,
            )
        )
    return tuple(risks)


def request_signals(risks: Sequence[SegmentRisk]) -> dict[str, Any]:
    """Summarize one request the way a bounded online ledger would see it."""

    scored = [risk for risk in risks if risk.risk is not None]
    unmeasured = [risk for risk in risks if risk.risk is None]
    return {
        "segment_count": len(risks),
        "scored_segment_count": len(scored),
        "unmeasured_segment_count": len(unmeasured),
        "unmeasured_segment_ids": [risk.segment_id for risk in unmeasured],
        "fail_closed_by_numerics": bool(unmeasured),
        "max_endpoint_z": (
            max(float(risk.endpoint_z) for risk in scored) if scored else None
        ),
        "max_segment_risk": (
            max(float(risk.risk) for risk in scored) if scored else None
        ),
        "cumulative_risk_ledger": (
            float(sum(float(risk.risk) for risk in scored)) if scored else None
        ),
        "total_scheduler_exposure": float(
            sum(float(risk.scheduler_exposure or 0.0) for risk in risks)
        ),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "q95": None, "maximum": None, "minimum": None}
    observed = [float(value) for value in values]
    return {
        "count": len(observed),
        "minimum": min(observed),
        "median": nearest_rank(observed, 0.5),
        "q95": nearest_rank(observed, 0.95),
        "maximum": max(observed),
    }


def separation_diagnostic(
    values: Sequence[float | None],
    failed: Sequence[bool],
) -> dict[str, Any]:
    """Report whether any single threshold on one signal separates failures.

    A separable signal is a necessary, not sufficient, condition for a
    threshold rule: separation on development data says nothing about a
    holdout, and this diagnostic never proposes the threshold it measures.
    """

    if len(values) != len(failed):
        raise ValueError("separation diagnostic requires paired observations")
    failure_values = [
        float(value) for value, flag in zip(values, failed, strict=True) if flag and value is not None
    ]
    pass_values = [
        float(value)
        for value, flag in zip(values, failed, strict=True)
        if not flag and value is not None
    ]
    unscored = sum(1 for value in values if value is None)
    result: dict[str, Any] = {
        "failure_count": len(failure_values),
        "pass_count": len(pass_values),
        "unscored_count": unscored,
        "failure_minimum": min(failure_values) if failure_values else None,
        "pass_maximum": max(pass_values) if pass_values else None,
        "separable": None,
        "passes_below_failure_minimum_fraction": None,
    }
    if failure_values and pass_values:
        failure_minimum = min(failure_values)
        result["separable"] = failure_minimum > max(pass_values)
        below = sum(1 for value in pass_values if value < failure_minimum)
        result["passes_below_failure_minimum_fraction"] = below / len(pass_values)
    return result


def signal_harm_correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Correlate each online signal with each decoded harm on scored requests."""

    result: list[dict[str, Any]] = []
    for signal in REQUEST_SIGNALS:
        for harm in HARM_FIELDS:
            paired = [
                (float(row[signal]), float(row[harm]))
                for row in rows
                if row.get(signal) is not None
            ]
            entry: dict[str, Any] = {
                "signal": signal,
                "harm": harm,
                "paired_count": len(paired),
                "pearson": None,
                "spearman": None,
            }
            if len(paired) >= 2:
                left = [value for value, _ in paired]
                right = [value for _, value in paired]
                entry["pearson"] = _correlation(left, right, rank=False)
                entry["spearman"] = _correlation(left, right, rank=True)
            result.append(entry)
    return result


def segment_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe every fixed segment separately for contract passes and failures."""

    by_segment: dict[str, list[tuple[Mapping[str, Any], bool]]] = {}
    order: dict[str, tuple[int, int]] = {}
    for row in rows:
        failed = bool(row["contract_failed"])
        for segment in row["segments"]:
            segment_id = str(segment["segment_id"])
            by_segment.setdefault(segment_id, []).append((segment, failed))
            order[segment_id] = (
                int(segment["previous_anchor_step_index"]),
                int(segment["anchor_step_index"]),
            )

    summaries: list[dict[str, Any]] = []
    for segment_id in sorted(by_segment, key=lambda key: order[key]):
        observations = by_segment[segment_id]
        first = observations[0][0]
        summary: dict[str, Any] = {
            "segment_id": segment_id,
            "previous_anchor_step_index": order[segment_id][0],
            "anchor_step_index": order[segment_id][1],
            "skipped_step_count": int(first["skipped_step_count"]),
            "policy_region": str(first["policy_region"]),
            "request_count": len(observations),
            "unmeasured_count": sum(
                1 for segment, _ in observations if segment["risk"] is None
            ),
        }
        for field in ("endpoint_z", "scheduler_exposure", "risk"):
            summary[field] = {
                "all": _distribution(
                    [segment[field] for segment, _ in observations if segment[field] is not None]
                ),
                "contract_pass": _distribution(
                    [
                        segment[field]
                        for segment, failed in observations
                        if not failed and segment[field] is not None
                    ]
                ),
                "contract_failure": _distribution(
                    [
                        segment[field]
                        for segment, failed in observations
                        if failed and segment[field] is not None
                    ]
                ),
            }
        summary["separation"] = separation_diagnostic(
            [segment["risk"] for segment, _ in observations],
            [failed for _, failed in observations],
        )
        summaries.append(summary)
    return summaries


def _evaluation_rows(evaluation: Mapping[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    rows = evaluation.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("natural-range evaluation has no rows")
    limits = evaluation.get("natural_range_limits")
    if not isinstance(limits, Mapping) or set(limits) != set(METRICS):
        raise ValueError("natural-range evaluation has no complete contract limits")
    if any(
        isinstance(limits[metric], bool)
        or not isinstance(limits[metric], (int, float))
        or not math.isfinite(float(limits[metric]))
        or float(limits[metric]) <= 0.0
        for metric in METRICS
    ):
        raise ValueError("natural-range contract limits must be finite and positive")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("candidate_id")) != candidate_id:
            continue
        harms = row.get("harms")
        if not isinstance(harms, Mapping) or set(harms) != set(METRICS):
            raise ValueError("natural-range evaluation row has incomplete harms")
        failed_metrics = row.get("failed_metrics")
        if not isinstance(failed_metrics, list):
            raise ValueError("natural-range evaluation row has no failed-metric list")
        utilization = max(
            float(harms[metric]) / float(limits[metric])
            for metric in METRICS
        )
        selected.append(
            {
                "sample_id": str(row["sample_id"]),
                "prompt_index": int(row["prompt_index"]),
                "seed": int(row["seed"]),
                "image_reward_harm": float(harms["image_reward"]),
                "vqa_score_harm": float(harms["vqa_score"]),
                "contract_utilization": utilization,
                "failed_metrics": sorted(str(metric) for metric in failed_metrics),
                "contract_failed": bool(failed_metrics),
            }
        )
    if not selected:
        raise ValueError(f"natural-range evaluation has no rows for {candidate_id}")
    return sorted(selected, key=lambda row: row["sample_id"])


def _manifest_traces(
    manifest: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Mapping[str, Any]]:
    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("quality manifest has no comparisons")
    traces: dict[str, Mapping[str, Any]] = {}
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise ValueError("quality manifest comparison is malformed")
        if str(comparison.get("candidate_id")) != candidate_id:
            continue
        trace = comparison.get("anchor_error_trace")
        if trace is None:
            continue
        if not isinstance(trace, Mapping):
            raise ValueError("quality manifest anchor-error trace is malformed")
        binding = trace.get("candidate_binding")
        if (
            not isinstance(binding, Mapping)
            or str(binding.get("candidate_id")) != candidate_id
        ):
            raise ValueError("anchor-error trace is bound to a different candidate")
        sample_id = str(comparison.get("sample_id"))
        if sample_id in traces:
            raise ValueError(f"quality manifest repeats sample {sample_id}")
        traces[sample_id] = trace
    return traces


def build_study(
    evaluation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    candidate_id: str,
    split_role: str,
    minimum_positives: int,
) -> dict[str, Any]:
    """Join semantic harm with same-request segment risk and gate calibration."""

    if split_role not in {"development", "holdout"}:
        raise ValueError("split role must be development or holdout")
    if isinstance(minimum_positives, bool) or not isinstance(minimum_positives, int):
        raise ValueError("minimum positives must be an integer")
    if minimum_positives < 1:
        raise ValueError("minimum positives must be positive")

    rows = _evaluation_rows(evaluation, candidate_id)
    traces = _manifest_traces(manifest, candidate_id)
    failures = [row for row in rows if row["contract_failed"]]
    common = {
        "candidate_id": candidate_id,
        "split_role": split_role,
        "minimum_quality_positives": minimum_positives,
        "request_count": len(rows),
        "contract_failure_count": len(failures),
        "traced_request_count": len(traces),
        "threshold_calibration_authorized": False,
    }

    missing = [row["sample_id"] for row in rows if row["sample_id"] not in traces]
    if missing:
        return {
            **common,
            "decision": "stop_missing_segment_traces",
            "reason": (
                "the quality manifest carries no anchor-error trace for every scored "
                "request; rerun collection with P2b instrumentation enabled"
            ),
            "untraced_sample_ids": missing,
        }

    joined: list[dict[str, Any]] = []
    schedules: set[tuple[str, ...]] = set()
    for row in rows:
        risks = segment_risks(traces[row["sample_id"]])
        if not risks:
            return {
                **common,
                "decision": "stop_missing_segment_traces",
                "reason": (
                    f"request {row['sample_id']} closed no cache segment, so it carries "
                    "no controller evidence"
                ),
                "untraced_sample_ids": [row["sample_id"]],
            }
        schedules.add(tuple(risk.segment_id for risk in risks))
        joined.append(
            {
                **row,
                **request_signals(risks),
                "segments": [risk.to_dict() for risk in risks],
            }
        )

    if len(schedules) != 1:
        return {
            **common,
            "decision": "stop_segment_schedule_not_fixed",
            "reason": (
                "per-segment thresholds are defined only for one fixed schedule; the "
                "studied requests closed different segment sequences"
            ),
            "observed_schedule_count": len(schedules),
            "observed_schedules": sorted(list(schedule) for schedule in schedules),
        }

    fail_closed = [row["sample_id"] for row in joined if row["fail_closed_by_numerics"]]
    request_separations = {
        signal: separation_diagnostic(
            [row[signal] for row in joined],
            [row["contract_failed"] for row in joined],
        )
        for signal in REQUEST_SIGNALS
    }
    diagnostics = {
        "fixed_segment_ids": sorted(schedules)[0],
        "numerically_fail_closed_sample_ids": fail_closed,
        "harm_distribution": {
            field: {
                "all": _distribution([row[field] for row in joined]),
                "contract_failure": _distribution(
                    [row[field] for row in joined if row["contract_failed"]]
                ),
            }
            for field in HARM_FIELDS
        },
        "request_signal_separation": request_separations,
        "signal_harm_correlation": signal_harm_correlations(joined),
        "segment_summaries": segment_summaries(joined),
    }

    if len(failures) < minimum_positives:
        return {
            **common,
            "decision": "stop_insufficient_quality_positives",
            "reason": (
                f"{len(failures)} contract failures is below the registered minimum of "
                f"{minimum_positives}; six per-segment thresholds may not be fitted to "
                "an incidental failure"
            ),
            "diagnostics": diagnostics,
            "requests": joined,
        }

    return {
        **common,
        "threshold_calibration_authorized": split_role == "development",
        "decision": "paired_diagnostics_complete",
        "reason": (
            "the paired study observed enough contract positives to report segment "
            "diagnostics; threshold selection remains a separate registered step and "
            "is not performed here"
        ),
        "diagnostics": diagnostics,
        "requests": joined,
    }


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    """Load, verify, and execute the paired study from registered artifacts."""

    evaluation_path = Path(args.natural_range_evaluation).expanduser().resolve()
    evaluation = _load_json(evaluation_path, "natural-range evaluation")
    _validate_content_hash(evaluation, "natural-range evaluation")
    if evaluation.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("natural-range evaluation schema is unsupported")

    binding = evaluation.get("quality_manifest")
    if not isinstance(binding, Mapping):
        raise ValueError("natural-range evaluation has no quality-manifest binding")
    manifest_path = Path(str(binding.get("path"))).expanduser().resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != binding.get("file_sha256"):
        raise ValueError("natural-range evaluation quality-manifest binding is invalid")
    manifest = _load_json(manifest_path, "quality manifest")
    if manifest.get("schema") != QUALITY_INPUT_SCHEMA:
        raise ValueError("quality manifest schema is unsupported")
    if manifest.get("hardware_measured") is not True:
        raise ValueError("segment-risk evidence must come from measured hardware runs")

    study = build_study(
        evaluation,
        manifest,
        candidate_id=str(args.candidate_id),
        split_role=str(args.split_role),
        minimum_positives=int(args.minimum_quality_positives),
    )
    payload = {
        "schema": RESULT_SCHEMA,
        "schema_revision": RESULT_SCHEMA_REVISION,
        "sources": {
            "natural_range_evaluation": _source_ref(evaluation_path, evaluation),
            "quality_manifest": _source_ref(manifest_path),
            "split": evaluation.get("split"),
            "bucket_id": evaluation.get("bucket_id"),
            "natural_range_limits": evaluation.get("natural_range_limits"),
            "implementation": implementation_bundle(
                (
                    Path(__file__).resolve(),
                    ROOT / "difflet" / "offline" / "cache_profile" / "composition.py",
                    ROOT / "difflet" / "offline" / "cache_profile" / "provenance.py",
                    ROOT / "difflet" / "offline" / "cache_profile" / "schedule.py",
                    ROOT / "difflet" / "pipeline" / "cache" / "control_error.py",
                    ROOT / "scripts" / "evaluate_flux_cache_segment_risk.py",
                ),
                root=ROOT,
            ),
        },
        "protocol": {
            "segment_signal": "endpoint_z = relative_l2(taylor_estimate, measured_output)",
            "scheduler_exposure": "W_s = sum over skipped steps of abs(delta_sigma)",
            "segment_risk": "r_s = W_s * z_s",
            "cumulative_risk_ledger": "D = sum of committed r_s over the request",
            "harm": "harm = baseline_score - candidate_score, per contract metric",
            "contract_utilization": "max over metrics of harm / natural_range_limit",
            "path_semantics": (
                "the trace is the logical post-restore path; physical rollback cost is "
                "not represented and must come from a separate execution receipt"
            ),
            "claim_boundary": (
                "development diagnostics for controller design; no serving claim and no "
                "threshold selection"
            ),
        },
        "study": study,
    }
    if args.out is not None:
        _write_hashed_json(Path(args.out).expanduser().resolve(), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Relate same-request cache-segment risk to decoded ImageReward/VQAScore harm."
        )
    )
    parser.add_argument(
        "--natural-range-evaluation",
        required=True,
        help="natural-range evaluation JSON produced by scripts/flux_cache_natural_range_gate.py",
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="the single fixed-schedule candidate under study",
    )
    parser.add_argument(
        "--split-role",
        required=True,
        choices=("development", "holdout"),
        help="development permits later threshold selection; holdout never does",
    )
    parser.add_argument(
        "--minimum-quality-positives",
        type=int,
        default=DEFAULT_MINIMUM_POSITIVES,
        help="registered minimum contract failures before thresholds may be fitted",
    )
    parser.add_argument("--out", default=None, help="write the hashed result to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_study(args)
    study = result["study"]
    print(
        f"{study['decision']}: {study['contract_failure_count']} contract failures in "
        f"{study['request_count']} requests; threshold calibration authorized="
        f"{study['threshold_calibration_authorized']}"
    )
    return 0


__all__ = [
    "DEFAULT_MINIMUM_POSITIVES",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_REVISION",
    "SegmentRisk",
    "build_parser",
    "build_study",
    "main",
    "request_signals",
    "run_study",
    "segment_risks",
    "segment_summaries",
    "separation_diagnostic",
    "signal_harm_correlations",
]
