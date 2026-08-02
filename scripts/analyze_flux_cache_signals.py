#!/usr/bin/env python3
"""Test whether FLUX runtime cache measurements predict final image quality.

The analysis is offline. It joins the strict measurement files referenced by
``quality-input-v2.json`` with the per-sample metrics in
``quality-curve-v2.json``. The primary test ranks requests *within each fixed
cache candidate* before combining them, so a result cannot pass merely because
aggressive candidates are both noisier and lower quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.pipeline.cache import load_cache_measurements  # noqa: E402
from scripts.calibrate_flux_cache_plan import (  # noqa: E402
    _load_quality_curve,
    _validate_curve,
)
from scripts.collect_flux_cache_ab import build_candidate_arms  # noqa: E402
from scripts.evaluate_cache_quality import (  # noqa: E402
    _artifact_path,
    _load_json,
    _validate_cache_measurement_artifact,
    _validate_quality_input,
)
from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402

SIGNAL_STUDY_SCHEMA = "difflet-flux-cache-signal-study"
SIGNAL_STUDY_SCHEMA_REVISION = 1
SIGNAL_STUDY_PROTOCOL_SCHEMA = "difflet-flux-cache-signal-study-protocol"
SIGNAL_STUDY_PROTOCOL_REVISION = 1


def _check_keys(value: Mapping[str, Any], name: str, required: set[str]) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
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


def _strict_number_list(value: Any, name: str, *, integer: bool) -> list[int | float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if integer:
        return [_strict_int(item, f"{name}[{index}]") for index, item in enumerate(value)]
    return [_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)]


def load_signal_study_protocol(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and strictly validate the preregistered signal-study protocol."""

    if isinstance(source, Mapping):
        document = dict(source)
    else:
        document = _load_json(Path(source), "signal study protocol")
    _check_keys(
        document,
        "signal study protocol",
        {
            "schema",
            "schema_revision",
            "study_id",
            "created_at",
            "prompt_suite",
            "candidate_sweep",
            "primary_test",
            "exploratory_signals",
            "pilot_exclusions",
            "sha256",
        },
    )
    if document["schema"] != SIGNAL_STUDY_PROTOCOL_SCHEMA:
        raise ValueError("unsupported signal study protocol schema")
    if document["schema_revision"] != SIGNAL_STUDY_PROTOCOL_REVISION:
        raise ValueError("unsupported signal study protocol revision")
    _strict_string(document["study_id"], "signal study protocol.study_id")
    _strict_string(document["created_at"], "signal study protocol.created_at")

    suite = document["prompt_suite"]
    if not isinstance(suite, dict):
        raise ValueError("signal study protocol.prompt_suite must be an object")
    _check_keys(suite, "signal study protocol.prompt_suite", {"suite_id", "splits", "seeds"})
    _strict_string(suite["suite_id"], "signal study protocol.prompt_suite.suite_id")
    splits = suite["splits"]
    if not isinstance(splits, dict) or set(splits) != {"calibration", "holdout"}:
        raise ValueError("signal study protocol prompt splits must be calibration and holdout")
    for split, digest in splits.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"signal study protocol split {split!r} has invalid SHA-256")
    seeds = _strict_number_list(
        suite["seeds"], "signal study protocol.prompt_suite.seeds", integer=True
    )
    if len(seeds) != len(set(seeds)):
        raise ValueError("signal study protocol seeds must be unique")

    sweep = document["candidate_sweep"]
    if not isinstance(sweep, dict):
        raise ValueError("signal study protocol.candidate_sweep must be an object")
    _check_keys(
        sweep,
        "signal study protocol.candidate_sweep",
        {
            "warmup_steps",
            "anchor_intervals",
            "orders",
            "anchor_phase",
            "cooldown_steps",
            "require_final_anchor",
            "coordinate",
        },
    )
    for field in ("warmup_steps", "anchor_intervals", "orders"):
        _strict_number_list(
            sweep[field], f"signal study protocol.candidate_sweep.{field}", integer=True
        )
    _strict_int(sweep["anchor_phase"], "signal study protocol.candidate_sweep.anchor_phase")
    _strict_int(sweep["cooldown_steps"], "signal study protocol.candidate_sweep.cooldown_steps")
    if type(sweep["require_final_anchor"]) is not bool:
        raise ValueError("signal study protocol require_final_anchor must be a boolean")
    if sweep["coordinate"] not in ("index", "timestep", "sigma"):
        raise ValueError("signal study protocol coordinate is unsupported")

    primary = document["primary_test"]
    if not isinstance(primary, dict):
        raise ValueError("signal study protocol.primary_test must be an object")
    _check_keys(
        primary,
        "signal study protocol.primary_test",
        {
            "signal",
            "quality_target",
            "analysis",
            "minimum_prompts",
            "minimum_samples_per_candidate",
            "minimum_stratified_spearman",
            "minimum_positive_candidate_fraction",
            "bootstrap",
        },
    )
    if primary["signal"] not in {
        "maximum-anchor-estimate-relative-error",
        "mean-anchor-estimate-relative-error",
    }:
        raise ValueError("unsupported primary signal")
    if primary["quality_target"] != "lpips":
        raise ValueError("unsupported primary quality target")
    if primary["analysis"] != "candidate-stratified-spearman":
        raise ValueError("unsupported primary analysis")
    _strict_int(primary["minimum_prompts"], "primary_test.minimum_prompts", minimum=2)
    _strict_int(
        primary["minimum_samples_per_candidate"],
        "primary_test.minimum_samples_per_candidate",
        minimum=2,
    )
    minimum_rho = _finite_float(
        primary["minimum_stratified_spearman"],
        "primary_test.minimum_stratified_spearman",
    )
    positive_fraction = _finite_float(
        primary["minimum_positive_candidate_fraction"],
        "primary_test.minimum_positive_candidate_fraction",
    )
    if not -1.0 <= minimum_rho <= 1.0 or not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("signal study correlation thresholds are outside their valid ranges")
    bootstrap = primary["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise ValueError("signal study protocol.primary_test.bootstrap must be an object")
    _check_keys(
        bootstrap,
        "signal study protocol.primary_test.bootstrap",
        {"cluster", "resamples", "seed", "minimum_lower_95_confidence_bound"},
    )
    if bootstrap["cluster"] != "prompt":
        raise ValueError("signal study bootstrap cluster must be prompt")
    _strict_int(bootstrap["resamples"], "primary_test.bootstrap.resamples", minimum=100)
    _strict_int(bootstrap["seed"], "primary_test.bootstrap.seed")
    lower_bound = _finite_float(
        bootstrap["minimum_lower_95_confidence_bound"],
        "primary_test.bootstrap.minimum_lower_95_confidence_bound",
    )
    if not -1.0 <= lower_bound <= 1.0:
        raise ValueError("bootstrap lower-bound threshold must be between -1 and 1")

    exploratory = document["exploratory_signals"]
    supported = {
        "maximum-anchor-estimate-relative-error",
        "mean-anchor-estimate-relative-error",
        "maximum-anchor-output-change",
        "maximum-anchor-output-curvature",
        "maximum-latent-relative-update",
    }
    if (
        not isinstance(exploratory, list)
        or not exploratory
        or any(item not in supported for item in exploratory)
        or len(exploratory) != len(set(exploratory))
    ):
        raise ValueError("signal study exploratory_signals are invalid")
    exclusions = document["pilot_exclusions"]
    if not isinstance(exclusions, list) or any(not isinstance(item, dict) for item in exclusions):
        raise ValueError("signal study pilot_exclusions must be a list of objects")

    digest = document["sha256"]
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if digest != canonical_sha256(payload):
        raise ValueError("signal study protocol sha256 does not match its content")
    return document


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return max(
        -1.0,
        min(
            1.0,
            sum(a * b for a, b in zip(left_centered, right_centered)) / denominator,
        ),
    )


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Spearman rank correlation with average ranks for ties."""

    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Spearman inputs must be finite")
    return _pearson(_average_ranks(left), _average_ranks(right))


def candidate_stratified_spearman(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    target: str,
) -> float | None:
    """Rank within each candidate, then correlate the centered pooled ranks."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    left: list[float] = []
    right: list[float] = []
    for candidate_rows in grouped.values():
        signal_ranks = _average_ranks([float(row[signal]) for row in candidate_rows])
        target_ranks = _average_ranks([float(row[target]) for row in candidate_rows])
        signal_mean = sum(signal_ranks) / len(signal_ranks)
        target_mean = sum(target_ranks) / len(target_ranks)
        left.extend(value - signal_mean for value in signal_ranks)
        right.extend(value - target_mean for value in target_ranks)
    return _pearson(left, right)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    target: str,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None, int]:
    by_prompt: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[int(row["prompt_index"])].append(row)
    prompt_ids = sorted(by_prompt)
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled: list[Mapping[str, Any]] = []
        for prompt_id in (generator.choice(prompt_ids) for _ in prompt_ids):
            sampled.extend(by_prompt[prompt_id])
        estimate = candidate_stratified_spearman(
            sampled,
            signal=signal,
            target=target,
        )
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None, None, 0
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975), len(estimates)


def _measured_values(records: Sequence[Any], field: str) -> list[float]:
    values = [
        float(value)
        for record in records
        if record.estimate_status == "measured"
        for value in (getattr(record, field),)
        if value is not None and math.isfinite(float(value))
    ]
    if not values:
        raise ValueError(f"measurement report has no usable {field} values")
    return values


def _optional_values(records: Sequence[Any], field: str) -> list[float]:
    return [
        float(value)
        for record in records
        for value in (getattr(record, field),)
        if value is not None and math.isfinite(float(value))
    ]


def _signal_summaries(report: Any) -> dict[str, float]:
    estimate_errors = _measured_values(
        report.anchor_measurements,
        "estimate_relative_error",
    )
    output_changes = _optional_values(report.anchor_measurements, "relative_output_change")
    output_curvatures = _optional_values(
        report.anchor_measurements,
        "relative_output_curvature",
    )
    latent_updates = [
        float(record.relative_update)
        for record in report.latent_updates
        if record.relative_update is not None and math.isfinite(float(record.relative_update))
    ]
    if not output_changes or not output_curvatures or not latent_updates:
        raise ValueError("measurement report is missing an exploratory signal")
    return {
        "maximum-anchor-estimate-relative-error": max(estimate_errors),
        "mean-anchor-estimate-relative-error": sum(estimate_errors) / len(estimate_errors),
        "maximum-anchor-output-change": max(output_changes),
        "maximum-anchor-output-curvature": max(output_curvatures),
        "maximum-latent-relative-update": max(latent_updates),
    }


def _expected_candidate_definitions(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sweep = protocol["candidate_sweep"]
    arms = build_candidate_arms(
        warmup_steps=sweep["warmup_steps"],
        anchor_intervals=sweep["anchor_intervals"],
        orders=sweep["orders"],
        anchor_phase=sweep["anchor_phase"],
        cooldown_steps=sweep["cooldown_steps"],
        require_final_anchor=sweep["require_final_anchor"],
        coord=sweep["coordinate"],
    )
    return {
        arm.candidate_id: {
            "policy": arm.policy_spec(),
            "predictor": arm.predictor_spec(),
        }
        for arm in arms
    }


def _validate_study_binding(
    protocol: Mapping[str, Any],
    identity: Mapping[str, Any],
    definitions: Mapping[str, Mapping[str, Any]],
) -> str:
    experiment = identity.get("protocol")
    if not isinstance(experiment, dict):
        raise ValueError("formal signal analysis requires an experiment protocol")
    selection = experiment["prompt_selection"]
    expected_suite = protocol["prompt_suite"]
    if selection["suite_id"] != expected_suite["suite_id"]:
        raise ValueError("experiment prompt suite does not match the signal study")
    split = selection["split"]
    if split not in expected_suite["splits"]:
        raise ValueError("experiment prompt split is not calibration or holdout")
    if selection["sha256"] != expected_suite["splits"][split]:
        raise ValueError("experiment prompt split digest does not match the signal study")
    if experiment["rng"]["seeds"] != expected_suite["seeds"]:
        raise ValueError("experiment seeds do not match the signal study")
    expected = _expected_candidate_definitions(protocol)
    actual = {
        candidate_id: {
            "policy": definition["policy"],
            "predictor": definition["predictor"],
        }
        for candidate_id, definition in definitions.items()
    }
    if actual != expected:
        raise ValueError("experiment candidate sweep does not match the signal study")
    return str(split)


def _correlation_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal: str,
    target: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    per_candidate = []
    for candidate_id in sorted(grouped):
        candidate_rows = grouped[candidate_id]
        rho = spearman(
            [float(row[signal]) for row in candidate_rows],
            [float(row[target]) for row in candidate_rows],
        )
        per_candidate.append(
            {
                "candidate_id": candidate_id,
                "sample_count": len(candidate_rows),
                "spearman": rho,
            }
        )
    finite = [row["spearman"] for row in per_candidate if row["spearman"] is not None]
    return {
        "signal": signal,
        "quality_target": target,
        "candidate_stratified_spearman": candidate_stratified_spearman(
            rows,
            signal=signal,
            target=target,
        ),
        "pooled_spearman": spearman(
            [float(row[signal]) for row in rows],
            [float(row[target]) for row in rows],
        ),
        "positive_candidate_fraction": (sum(value > 0.0 for value in finite) / len(per_candidate)),
        "per_candidate": per_candidate,
    }


def analyze(
    quality_input_path: Path,
    quality_curve_path: Path,
    study_protocol_source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Join strict artifacts and evaluate the preregistered signal test."""

    study_protocol = load_signal_study_protocol(study_protocol_source)
    quality_input = _load_json(quality_input_path, "quality input")
    identity, definitions, comparisons = _validate_quality_input(quality_input)
    quality_curve = _load_quality_curve(quality_curve_path)
    _validate_curve(quality_curve)
    for field in (
        "model",
        "model_id",
        "shape_label",
        "num_steps",
        "scheduler_class",
        "guidance_scale",
        "prompt_count",
        "seed_count",
        "sample_count",
    ):
        if quality_curve[field] != identity[field]:
            raise ValueError(f"quality curve {field} does not match quality input")
    if quality_curve.get("protocol") != identity.get("protocol"):
        raise ValueError("quality curve experiment protocol does not match quality input")
    split = _validate_study_binding(study_protocol, identity, definitions)

    quality_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for candidate in quality_curve["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in definitions:
            raise ValueError("quality curve contains an unknown candidate")
        if (
            candidate["policy"] != definitions[candidate_id]["policy"]
            or candidate["predictor"] != definitions[candidate_id]["predictor"]
        ):
            raise ValueError("quality curve candidate definition does not match quality input")
        for sample in candidate["per_sample"]:
            key = (candidate_id, sample["sample_id"])
            if key in quality_by_key:
                raise ValueError("quality curve contains a duplicate candidate/sample row")
            quality_by_key[key] = sample

    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        artifacts = comparison["candidate"]
        if "cache_measurements" not in artifacts:
            raise ValueError("signal analysis requires candidate cache measurements")
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
        report = load_cache_measurements(measurement_path)
        quality = quality_by_key.get((comparison["candidate_id"], comparison["sample_id"]))
        if quality is None:
            raise ValueError("quality curve is missing a candidate/sample row")
        if (
            quality["prompt_index"] != comparison["prompt_index"]
            or quality["seed"] != comparison["seed"]
        ):
            raise ValueError("quality curve sample identity does not match quality input")
        rows.append(
            {
                "candidate_id": comparison["candidate_id"],
                "sample_id": comparison["sample_id"],
                "prompt_index": comparison["prompt_index"],
                "seed": comparison["seed"],
                "lpips": _finite_float(quality["lpips"], "quality LPIPS"),
                "psnr_db": _finite_float(quality["psnr_db"], "quality PSNR"),
                **_signal_summaries(report),
            }
        )
    if set(quality_by_key) != {(row["candidate_id"], row["sample_id"]) for row in rows}:
        raise ValueError("quality input and quality curve candidate/sample matrices differ")

    primary_config = study_protocol["primary_test"]
    primary = _correlation_report(
        rows,
        signal=primary_config["signal"],
        target=primary_config["quality_target"],
    )
    bootstrap = primary_config["bootstrap"]
    lower, upper, valid_resamples = _bootstrap_interval(
        rows,
        signal=primary_config["signal"],
        target=primary_config["quality_target"],
        resamples=bootstrap["resamples"],
        seed=bootstrap["seed"],
    )
    primary["prompt_cluster_bootstrap_95_confidence_interval"] = [lower, upper]
    primary["valid_bootstrap_resamples"] = valid_resamples

    candidate_sample_counts = [row["sample_count"] for row in primary["per_candidate"]]
    gates = {
        "minimum_prompts": len({row["prompt_index"] for row in rows})
        >= primary_config["minimum_prompts"],
        "minimum_samples_per_candidate": bool(candidate_sample_counts)
        and min(candidate_sample_counts) >= primary_config["minimum_samples_per_candidate"],
        "minimum_stratified_spearman": primary["candidate_stratified_spearman"] is not None
        and primary["candidate_stratified_spearman"]
        >= primary_config["minimum_stratified_spearman"],
        "minimum_positive_candidate_fraction": primary["positive_candidate_fraction"]
        >= primary_config["minimum_positive_candidate_fraction"],
        "minimum_lower_95_confidence_bound": lower is not None
        and lower >= bootstrap["minimum_lower_95_confidence_bound"],
        "complete_bootstrap": valid_resamples == bootstrap["resamples"],
    }
    primary["gates"] = gates
    primary["passes"] = all(gates.values())

    exploratory = [
        _correlation_report(rows, signal=signal, target="lpips")
        for signal in study_protocol["exploratory_signals"]
    ]
    result = {
        "schema": SIGNAL_STUDY_SCHEMA,
        "schema_revision": SIGNAL_STUDY_SCHEMA_REVISION,
        "study_id": study_protocol["study_id"],
        "study_protocol_sha256": study_protocol["sha256"],
        "experiment_protocol_sha256": identity["protocol"]["sha256"],
        "evaluation_protocol_sha256": quality_curve["evaluation_protocol"]["sha256"],
        "split": split,
        "model": identity["model"],
        "model_id": identity["model_id"],
        "shape_label": identity["shape_label"],
        "num_steps": identity["num_steps"],
        "prompt_count": identity["prompt_count"],
        "seed_count": identity["seed_count"],
        "candidate_count": len(definitions),
        "row_count": len(rows),
        "primary_test": primary,
        "exploratory": exploratory,
    }
    return {**result, "sha256": canonical_sha256(result)}


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
    parser.add_argument("--study-protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = Path(args.out).expanduser().resolve()
        result = analyze(
            Path(args.quality_input).expanduser().resolve(),
            Path(args.quality_curve).expanduser().resolve(),
            Path(args.study_protocol).expanduser().resolve(),
        )
        _write_json(output, result, force=bool(args.force))
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    primary = result["primary_test"]
    print(
        "[cache-signals] "
        f"split={result['split']} "
        f"rho={primary['candidate_stratified_spearman']:.6f} "
        f"passes={primary['passes']} -> {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
