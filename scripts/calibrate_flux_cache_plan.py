#!/usr/bin/env python3
"""Select a measured FLUX cache candidate and emit cache-plan-v1.

Selection is fail closed. A candidate must pass all five quality/speed gates,
be hardware measured, use PeriodicAnchorPolicy × TaylorSeer, and report zero
``consecutive_skip_vetoes``. Eligible candidates are ranked by lower LPIPS,
then higher PSNR, then higher measured speedup. Thresholds are never relaxed
when no candidate passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.pipeline.cache import (  # noqa: E402
    CACHE_PLAN_SCHEMA,
    CacheCompatibility,
    CachePlan,
    PeriodicAnchorPolicy,
    build_policy,
    build_predictor,
    resolve_cache_plan,
)
from difflet.pipeline.cache.predictors import TaylorSeerPredictor  # noqa: E402
from scripts.evaluate_cache_quality import (  # noqa: E402
    QUALITY_CURVE_SCHEMA,
    validate_metric_config,
)
from scripts.flux_cache_protocol import (  # noqa: E402
    validate_evaluation_protocol,
    validate_protocol_binding,
)

NO_ACCEPTABLE_CANDIDATE = "无可接受候选"
_IDENTITY_FIELDS = {
    "model",
    "model_id",
    "shape_label",
    "num_steps",
    "scheduler_class",
    "guidance_scale",
    "prompt_count",
    "seed_count",
    "sample_count",
}
_THRESHOLD_FIELDS = {
    "min_speedup",
    "min_trajectory_cosine",
    "min_final_latent_cosine",
    "min_psnr_db",
    "max_lpips",
}
_WORST_FIELDS = {
    "trajectory_cosine",
    "final_latent_cosine",
    "psnr_db",
    "ssim",
    "lpips",
}


class NoAcceptableCandidateError(ValueError):
    """Raised when strict gates leave no plan candidate."""


def _check_keys(
    value: Mapping[str, Any],
    name: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _load_quality_curve(
    source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        document = dict(source)
    else:
        path = Path(source)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"quality curve does not exist: {path}") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"quality curve is not valid JSON: {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("quality curve must contain a JSON object")
    return document


def _gate_failures(
    candidate: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> list[str]:
    worst = candidate["worst_sample"]
    failures: list[str] = []
    if candidate["hardware_measured"] is not True:
        failures.append("hardware_measured")
    checks = (
        (
            "measured_speedup",
            float(candidate["measured_speedup"]),
            float(thresholds["min_speedup"]),
            "min",
        ),
        (
            "trajectory_cosine",
            float(worst["trajectory_cosine"]),
            float(thresholds["min_trajectory_cosine"]),
            "min",
        ),
        (
            "final_latent_cosine",
            float(worst["final_latent_cosine"]),
            float(thresholds["min_final_latent_cosine"]),
            "min",
        ),
        (
            "psnr_db",
            float(worst["psnr_db"]),
            float(thresholds["min_psnr_db"]),
            "min",
        ),
        (
            "lpips",
            float(worst["lpips"]),
            float(thresholds["max_lpips"]),
            "max",
        ),
    )
    for name, value, threshold, direction in checks:
        if (direction == "min" and value < threshold) or (direction == "max" and value > threshold):
            failures.append(name)
    return failures


def _validate_curve(document: dict[str, Any]) -> list[dict[str, Any]]:
    _check_keys(
        document,
        "quality curve",
        required={
            "schema",
            *_IDENTITY_FIELDS,
            "hardware_measured",
            "aggregation",
            "metric_config",
            "thresholds",
            "candidates",
            "passing_candidate_ids",
        },
        optional={"protocol", "evaluation_protocol"},
    )
    if document["schema"] != QUALITY_CURVE_SCHEMA:
        raise ValueError(
            f"quality curve schema must be {QUALITY_CURVE_SCHEMA!r}, " f"got {document['schema']!r}"
        )
    if document["model"] != "flux":
        raise ValueError("calibrate_flux_cache_plan accepts model='flux' only")
    for field in ("model", "model_id", "shape_label", "scheduler_class"):
        _strict_string(document[field], f"quality curve.{field}")
    for field in ("num_steps", "prompt_count", "seed_count", "sample_count"):
        _strict_positive_int(document[field], f"quality curve.{field}")
    if document["sample_count"] != document["prompt_count"] * document["seed_count"]:
        raise ValueError("quality curve.sample_count must equal prompt_count * seed_count")
    _finite_float(document["guidance_scale"], "quality curve.guidance_scale")
    _strict_bool(document["hardware_measured"], "quality curve.hardware_measured")
    has_protocol = "protocol" in document
    evaluation_protocol = None
    if has_protocol:
        validate_protocol_binding(
            document["protocol"],
            document,
            name="quality curve.protocol",
        )
        if "evaluation_protocol" not in document:
            raise ValueError("protocol-v1 quality curve is missing evaluation_protocol")
        evaluation_protocol = validate_evaluation_protocol(
            document["evaluation_protocol"],
            name="quality curve.evaluation_protocol",
        )
        if evaluation_protocol["source"]["git_dirty"]:
            raise ValueError("quality curve.evaluation_protocol was created from a dirty worktree")
    elif "evaluation_protocol" in document:
        raise ValueError("quality curve.evaluation_protocol requires an experiment protocol")
    if document["aggregation"] != "worst-sample":
        raise ValueError("quality curve aggregation must be 'worst-sample'")
    metric_config = validate_metric_config(
        document["metric_config"],
        require_lpips_provenance=has_protocol,
    )
    if evaluation_protocol is not None and evaluation_protocol["metric_config"] != metric_config:
        raise ValueError("quality curve.evaluation_protocol does not match metric_config")

    thresholds_value = document["thresholds"]
    if not isinstance(thresholds_value, dict):
        raise ValueError("quality curve.thresholds must be a JSON object")
    _check_keys(
        thresholds_value,
        "quality curve.thresholds",
        required=_THRESHOLD_FIELDS,
    )
    thresholds = {
        field: _finite_float(thresholds_value[field], f"quality curve.thresholds.{field}")
        for field in _THRESHOLD_FIELDS
    }
    if thresholds["min_speedup"] <= 0.0:
        raise ValueError("quality curve.thresholds.min_speedup must be positive")
    for field in ("min_trajectory_cosine", "min_final_latent_cosine"):
        if not -1.0 <= thresholds[field] <= 1.0:
            raise ValueError(f"quality curve.thresholds.{field} must be between -1 and 1")
    for field in ("min_psnr_db", "max_lpips"):
        if thresholds[field] < 0.0:
            raise ValueError(f"quality curve.thresholds.{field} must be nonnegative")
    candidates_value = document["candidates"]
    if not isinstance(candidates_value, list) or not candidates_value:
        raise ValueError("quality curve.candidates must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, value in enumerate(candidates_value):
        name = f"quality curve.candidates[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        _check_keys(
            value,
            name,
            required={
                "candidate_id",
                "policy",
                "predictor",
                "hardware_measured",
                "measured_speedup",
                "runner_stats",
                "per_sample",
                "worst_sample",
                "passes_gate",
                "failed_gates",
            },
        )
        candidate_id = _strict_string(value["candidate_id"], f"{name}.candidate_id")
        if candidate_id in identifiers:
            raise ValueError(f"duplicate candidate_id: {candidate_id!r}")
        identifiers.add(candidate_id)
        if not isinstance(value["policy"], dict) or not isinstance(value["predictor"], dict):
            raise ValueError(f"{name}.policy and predictor must be JSON objects")
        policy = build_policy(value["policy"])
        predictor = build_predictor(value["predictor"])
        if not isinstance(policy, PeriodicAnchorPolicy):
            raise ValueError(f"{name} must use PeriodicAnchorPolicy")
        if not isinstance(predictor, TaylorSeerPredictor):
            raise ValueError(f"{name} must use TaylorSeerPredictor")
        if policy.require_final_anchor is not True:
            raise ValueError(f"{name} must require the final anchor")
        hardware_measured = _strict_bool(value["hardware_measured"], f"{name}.hardware_measured")
        measured_speedup = _finite_float(value["measured_speedup"], f"{name}.measured_speedup")
        if measured_speedup <= 0.0:
            raise ValueError(f"{name}.measured_speedup must be positive")
        if not isinstance(value["runner_stats"], dict):
            raise ValueError(f"{name}.runner_stats must be a JSON object")
        runner_stats = dict(value["runner_stats"])
        vetoes = _strict_nonnegative_int(
            runner_stats.get("consecutive_skip_vetoes"),
            f"{name}.runner_stats.consecutive_skip_vetoes",
        )
        if not isinstance(value["per_sample"], list) or not value["per_sample"]:
            raise ValueError(f"{name}.per_sample must be a non-empty list")
        if len(value["per_sample"]) != document["sample_count"]:
            raise ValueError(
                f"{name}.per_sample has {len(value['per_sample'])} rows, "
                f"expected {document['sample_count']}"
            )
        sample_metrics: list[dict[str, float]] = []
        sample_ids: set[str] = set()
        for sample_index, sample in enumerate(value["per_sample"]):
            sample_name = f"{name}.per_sample[{sample_index}]"
            if not isinstance(sample, dict):
                raise ValueError(f"{sample_name} must be a JSON object")
            _check_keys(
                sample,
                sample_name,
                required={
                    "sample_id",
                    "prompt_index",
                    "seed",
                    *_WORST_FIELDS,
                },
            )
            sample_id = _strict_string(sample["sample_id"], f"{sample_name}.sample_id")
            if sample_id in sample_ids:
                raise ValueError(f"{name}.per_sample has duplicate sample_id")
            sample_ids.add(sample_id)
            _strict_nonnegative_int(sample["prompt_index"], f"{sample_name}.prompt_index")
            _strict_nonnegative_int(sample["seed"], f"{sample_name}.seed")
            sample_metrics.append(
                {
                    field: _finite_float(sample[field], f"{sample_name}.{field}")
                    for field in _WORST_FIELDS
                }
            )
        worst_value = value["worst_sample"]
        if not isinstance(worst_value, dict):
            raise ValueError(f"{name}.worst_sample must be a JSON object")
        _check_keys(
            worst_value,
            f"{name}.worst_sample",
            required=_WORST_FIELDS,
        )
        worst = {
            field: _finite_float(worst_value[field], f"{name}.worst_sample.{field}")
            for field in _WORST_FIELDS
        }
        for field in ("trajectory_cosine", "final_latent_cosine", "ssim"):
            if not -1.0 <= worst[field] <= 1.0:
                raise ValueError(f"{name}.worst_sample.{field} must be between -1 and 1")
        if worst["lpips"] < 0.0:
            raise ValueError(f"{name}.worst_sample.lpips must be nonnegative")
        recomputed_worst = {
            "trajectory_cosine": min(sample["trajectory_cosine"] for sample in sample_metrics),
            "final_latent_cosine": min(sample["final_latent_cosine"] for sample in sample_metrics),
            "psnr_db": min(sample["psnr_db"] for sample in sample_metrics),
            "ssim": min(sample["ssim"] for sample in sample_metrics),
            "lpips": max(sample["lpips"] for sample in sample_metrics),
        }
        if any(
            not math.isclose(
                worst[field],
                recomputed_worst[field],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for field in _WORST_FIELDS
        ):
            raise ValueError(f"{name}.worst_sample disagrees with its per-sample metrics")
        passes_gate = _strict_bool(value["passes_gate"], f"{name}.passes_gate")
        if not isinstance(value["failed_gates"], list) or any(
            not isinstance(item, str) for item in value["failed_gates"]
        ):
            raise ValueError(f"{name}.failed_gates must be a list of strings")
        normalized_candidate = {
            "candidate_id": candidate_id,
            "policy": dict(value["policy"]),
            "predictor": dict(value["predictor"]),
            "hardware_measured": hardware_measured,
            "measured_speedup": measured_speedup,
            "runner_stats": runner_stats,
            "consecutive_skip_vetoes": vetoes,
            "worst_sample": worst,
        }
        expected_failures = _gate_failures(normalized_candidate, thresholds)
        if passes_gate != (not expected_failures):
            raise ValueError(f"{name}.passes_gate disagrees with the recorded metrics/thresholds")
        if list(value["failed_gates"]) != expected_failures:
            raise ValueError(f"{name}.failed_gates disagrees with the recorded metrics/thresholds")
        normalized_candidate["passes_gate"] = passes_gate
        normalized.append(normalized_candidate)

    passing_ids = document["passing_candidate_ids"]
    if not isinstance(passing_ids, list) or any(not isinstance(item, str) for item in passing_ids):
        raise ValueError("quality curve.passing_candidate_ids must be a list of strings")
    expected_ids = [
        candidate["candidate_id"] for candidate in normalized if candidate["passes_gate"]
    ]
    if passing_ids != expected_ids:
        raise ValueError("quality curve.passing_candidate_ids disagrees with candidate gates")
    return normalized


def calibrate(
    source: str | Path | Mapping[str, Any],
) -> tuple[CachePlan, dict[str, Any]]:
    """Return the strictly selected plan and normalized winning candidate."""

    document = _load_quality_curve(source)
    candidates = _validate_curve(document)
    eligible = [
        candidate
        for candidate in candidates
        if (
            document["hardware_measured"] is True
            and candidate["hardware_measured"] is True
            and candidate["passes_gate"] is True
            and candidate["consecutive_skip_vetoes"] == 0
        )
    ]
    if not eligible:
        raise NoAcceptableCandidateError(
            f"{NO_ACCEPTABLE_CANDIDATE}: all candidates failed a quality/speed "
            "gate, hardware evidence, or consecutive-skip veto check"
        )
    selected = min(
        eligible,
        key=lambda candidate: (
            float(candidate["worst_sample"]["lpips"]),
            -float(candidate["worst_sample"]["psnr_db"]),
            -float(candidate["measured_speedup"]),
            candidate["candidate_id"],
        ),
    )
    policy = build_policy(selected["policy"])
    predictor = build_predictor(selected["predictor"])
    num_steps = int(document["num_steps"])
    frozen_mask = policy.materialize_anchor_mask(num_steps)
    plan = CachePlan(
        compatibility=CacheCompatibility(
            model="flux",
            shape_label=document["shape_label"],
            num_steps=num_steps,
            scheduler_class=document["scheduler_class"],
        ),
        policy=selected["policy"],
        predictor=selected["predictor"],
        frozen_mask=frozen_mask,
    )
    # Apply the same identity/mask/safety checks the production FLUX path uses.
    resolve_cache_plan(
        plan,
        model="flux",
        shape_label=document["shape_label"],
        num_steps=num_steps,
        scheduler_class=document["scheduler_class"],
    )
    return plan, selected


def _write_plan(path: Path, plan: CachePlan, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-curve", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        plan, selected = calibrate(Path(args.quality_curve).expanduser().resolve())
        output = Path(args.out).expanduser().resolve()
        _write_plan(output, plan, force=bool(args.force))
    except (FileExistsError, NoAcceptableCandidateError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    print(
        f"[cache-calibrate] selected={selected['candidate_id']} "
        f"lpips={selected['worst_sample']['lpips']:.6f} "
        f"psnr={selected['worst_sample']['psnr_db']:.3f} "
        f"speedup={selected['measured_speedup']:.3f}x",
        flush=True,
    )
    print(
        f"[cache-calibrate] wrote {CACHE_PLAN_SCHEMA}: {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
