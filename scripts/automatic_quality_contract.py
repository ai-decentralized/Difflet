"""Automatic semantic-quality contract helpers for FLUX cache experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.evaluate_flux_cache_semantics import REPORT_SCHEMA, REPORT_SCHEMA_REVISION
from scripts.flux_cache_protocol import canonical_sha256


PROTOCOL_SCHEMA = "difflet-flux-cache-automatic-quality-contract-protocol"
CONTRACT_SCHEMA = "difflet-flux-cache-automatic-quality-contract"
EVALUATION_SCHEMA = "difflet-flux-cache-automatic-quality-evaluation"
SCHEMA_REVISION = 1
METRICS = ("image_reward", "vqa_score")


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


def load_protocol(path: Path) -> dict[str, Any]:
    document = load_json(path, "automatic quality protocol")
    expected = {
        "schema",
        "schema_revision",
        "study_id",
        "created_at",
        "quality_claim",
        "controlled_generation",
        "margin_calibration",
        "static_candidate",
        "automatic_damage_rule",
        "signal_gate",
        "holdout",
        "limitations",
        "sha256",
    }
    if set(document) != expected:
        raise ValueError("automatic quality protocol fields do not match the protocol")
    if (
        document["schema"] != PROTOCOL_SCHEMA
        or document["schema_revision"] != SCHEMA_REVISION
    ):
        raise ValueError("automatic quality protocol schema is unsupported")
    _validate_digest(document, "automatic quality protocol")
    calibration = document["margin_calibration"]
    quantile = float(calibration["quantile"])
    if not 0.0 < quantile <= 1.0:
        raise ValueError("margin calibration quantile must be in (0, 1]")
    seeds = calibration["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("margin calibration requires at least two unique seeds")
    if calibration["candidate_images_excluded_from_margin_estimation"] is not True:
        raise ValueError("candidate images must be excluded from margin estimation")
    if document["automatic_damage_rule"] != {
        "metrics": list(METRICS),
        "paired_loss": "baseline_score_minus_candidate_score",
        "comparison": "strictly-greater-than-metric-margin",
        "combination": "fail-if-either-metric-fails",
        "weighted_average_forbidden": True,
    }:
        raise ValueError("automatic damage rule is unsupported")
    return document


def _without_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime_fields(item)
            for key, item in value.items()
            if key not in {"load_seconds"}
        }
    if isinstance(value, list):
        return [_without_runtime_fields(item) for item in value]
    return value


def metric_identity(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if set(metrics) != set(METRICS):
        raise ValueError("semantic report must contain ImageReward and VQAScore")
    normalized = _without_runtime_fields(copy.deepcopy(dict(metrics)))
    # Hugging Face cache bookkeeping changes whenever the same immutable
    # checkpoint is opened.  Lock files, download metadata, and the cache's
    # own .gitignore are not model identity; retaining them would reject two
    # reports that used byte-identical weights and preprocessing.
    transient_suffixes = (".lock", ".metadata", ".gitignore")
    for config in normalized.values():
        checkpoint_files = config.get("checkpoint_files")
        if isinstance(checkpoint_files, list):
            config["checkpoint_files"] = [
                row
                for row in checkpoint_files
                if not str(row.get("path", "")).endswith(transient_suffixes)
            ]
    return {"config": normalized, "sha256": canonical_sha256(normalized)}


def load_semantic_report(path: Path) -> dict[str, Any]:
    document = load_json(path, "semantic report")
    expected = {
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
    if set(document) != expected:
        raise ValueError("semantic report fields do not match the scoring protocol")
    if (
        document["schema"] != REPORT_SCHEMA
        or document["schema_revision"] != REPORT_SCHEMA_REVISION
        or document["complete"] is not True
    ):
        raise ValueError("semantic report is unsupported or incomplete")
    metric_identity(document["metrics"])
    return document


def semantic_source(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    sources = report["sources"]
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("semantic report must bind exactly one quality manifest")
    source = sources[0]
    manifest_path = Path(source["path"]).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != source["sha256"]:
        raise ValueError("semantic report quality-manifest binding is invalid")
    manifest = load_json(manifest_path, "quality manifest")
    try:
        selection = manifest["protocol"]["prompt_selection"]
    except (KeyError, TypeError) as error:
        raise ValueError("quality manifest has no prompt selection") from error
    if source["split"] != selection["split"]:
        raise ValueError("semantic report and quality manifest split differ")
    return manifest, selection, manifest_path


def validate_generation_identity(
    manifest: Mapping[str, Any],
    controlled: Mapping[str, Any],
) -> None:
    """Require evidence to use the exact generation path registered by the study."""
    try:
        experiment = manifest["protocol"]
        generation = experiment["generation"]
        model = experiment["model"]
        compile_inputs = experiment["compile"]["cache_inputs"]
        parallel = compile_inputs["parallel"]
        source = experiment["source"]
    except (KeyError, TypeError) as error:
        raise ValueError("quality manifest has no complete generation identity") from error
    observed = {
        "model_id": model["model_id"],
        "model_revision": model["resolved_revision"],
        "scheduler_class": generation["scheduler_class"],
        "scheduler_config_sha256": canonical_sha256(generation["scheduler_config"]),
        "num_steps": generation["num_steps"],
        "height": generation["height"],
        "width": generation["width"],
        "guidance_scale": generation["guidance_scale"],
        "dtype": generation["dtype"],
        "tp_degree": parallel["tp_degree"],
    }
    expected = {key: controlled[key] for key in observed}
    if observed != expected:
        raise ValueError("quality manifest generation identity differs from the protocol")
    if source.get("git_dirty") is not False:
        raise ValueError("quality evidence must come from a clean git worktree")


def validate_static_candidate(
    manifest: Mapping[str, Any],
    registered: Mapping[str, Any],
) -> None:
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("quality manifest has no candidate definitions")
    matching = [row for row in candidates if row.get("candidate_id") == registered["candidate_id"]]
    if len(matching) != 1:
        raise ValueError("quality manifest does not contain the frozen static candidate")
    candidate = matching[0]
    expected_policy = {
        "type": "periodic_anchor",
        "warmup_steps": registered["warmup_steps"],
        "anchor_interval": registered["anchor_interval"],
        "anchor_phase": 1,
        "cooldown_steps": 1,
        "require_final_anchor": True,
    }
    expected_predictor = {"type": "taylorseer", "order": 1, "coord": "index"}
    if candidate.get("policy") != expected_policy or candidate.get("predictor") != expected_predictor:
        raise ValueError("quality manifest static-candidate definition differs from the protocol")


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile from no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _finite_score(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite")
    return score


def calibrate_contract(
    protocol_path: Path,
    semantic_report_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    report = load_semantic_report(semantic_report_path)
    manifest, selection, manifest_path = semantic_source(report)
    calibration = protocol["margin_calibration"]
    validate_generation_identity(manifest, protocol["controlled_generation"])
    if (
        selection["split"] != calibration["split"]
        or selection["sha256"] != calibration["split_sha256"]
        or len(selection["prompts"]) != calibration["prompt_count"]
    ):
        raise ValueError("semantic report does not use the registered margin split")
    expected_seeds = tuple(calibration["seeds"])
    manifest_seeds = tuple(manifest["protocol"]["rng"]["seeds"])
    if manifest_seeds != expected_seeds:
        raise ValueError("margin-calibration seeds do not match the protocol")

    baseline_records = [row for row in report["images"] if row["role"] == "baseline"]
    if len(baseline_records) != calibration["sample_count"]:
        raise ValueError("margin calibration has the wrong number of baseline images")
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_records:
        if row["candidate_id"] is not None or row["split"] != calibration["split"]:
            raise ValueError("margin calibration contains an invalid baseline identity")
        grouped[(int(row["prompt_index"]), str(row["prompt"]))].append(row)
    if len(grouped) != calibration["prompt_count"]:
        raise ValueError("margin calibration prompt count is incorrect")

    differences: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for rows in grouped.values():
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        if tuple(int(row["seed"]) for row in rows) != expected_seeds:
            raise ValueError("each margin prompt must contain every registered seed")
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                for metric in METRICS:
                    left_score = _finite_score(left["scores"][metric], metric)
                    right_score = _finite_score(right["scores"][metric], metric)
                    differences[metric].append(abs(left_score - right_score))

    quantile = float(calibration["quantile"])
    summaries = {}
    margins = {}
    for metric, values in differences.items():
        margin = _nearest_rank(values, quantile)
        margins[metric] = margin
        summaries[metric] = {
            "pair_count": len(values),
            "minimum": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "maximum": max(values),
            "selected_quantile": quantile,
            "margin": margin,
        }

    payload = {
        "schema": CONTRACT_SCHEMA,
        "schema_revision": SCHEMA_REVISION,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "content_sha256": protocol["sha256"],
            "study_id": protocol["study_id"],
        },
        "quality_claim": protocol["quality_claim"],
        "controlled_generation": protocol["controlled_generation"],
        "static_candidate": protocol["static_candidate"],
        "automatic_damage_rule": protocol["automatic_damage_rule"],
        "metric_identity": metric_identity(report["metrics"]),
        "margin_calibration": {
            "semantic_report": str(semantic_report_path),
            "semantic_report_sha256": sha256_file(semantic_report_path),
            "quality_manifest": str(manifest_path),
            "quality_manifest_sha256": sha256_file(manifest_path),
            "split": calibration["split"],
            "split_sha256": calibration["split_sha256"],
            "prompt_count": calibration["prompt_count"],
            "seeds": list(expected_seeds),
            "method": calibration["method"],
            "candidate_images_used": False,
            "summaries": summaries,
        },
        "margins": margins,
        "signal_gate": protocol["signal_gate"],
        "holdout": protocol["holdout"],
        "limitations": protocol["limitations"],
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def load_contract(path: Path) -> dict[str, Any]:
    document = load_json(path, "automatic quality contract")
    if document.get("schema") != CONTRACT_SCHEMA or document.get("schema_revision") != SCHEMA_REVISION:
        raise ValueError("automatic quality contract schema is unsupported")
    _validate_digest(document, "automatic quality contract")
    if set(document["margins"]) != set(METRICS):
        raise ValueError("automatic quality contract metric margins are incomplete")
    return document


def clopper_pearson_upper(failures: int, samples: int, confidence: float) -> float:
    if samples <= 0 or failures < 0 or failures > samples or not 0.0 < confidence < 1.0:
        raise ValueError("invalid binomial confidence inputs")
    if failures == samples:
        return 1.0
    alpha = 1.0 - confidence

    def cdf(probability: float) -> float:
        return sum(
            math.comb(samples, index)
            * probability**index
            * (1.0 - probability) ** (samples - index)
            for index in range(failures + 1)
        )

    lower, upper = 0.0, 1.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if cdf(middle) > alpha:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def evaluate_contract(
    contract_path: Path,
    semantic_report_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    report = load_semantic_report(semantic_report_path)
    manifest, selection, manifest_path = semantic_source(report)
    validate_generation_identity(manifest, contract["controlled_generation"])
    if metric_identity(report["metrics"])["sha256"] != contract["metric_identity"]["sha256"]:
        raise ValueError("semantic metric identity differs from the calibrated contract")

    comparisons = report["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("semantic report contains no comparisons")
    rows = []
    for comparison in comparisons:
        deltas = comparison["candidate_minus_baseline"]
        harms = {
            metric: -_finite_score(deltas[metric], f"{metric} delta") for metric in METRICS
        }
        failed_metrics = [
            metric for metric in METRICS if harms[metric] > float(contract["margins"][metric])
        ]
        rows.append(
            {
                "candidate_id": comparison["candidate_id"],
                "sample_id": comparison["sample_id"],
                "prompt_index": int(comparison["prompt_index"]),
                "seed": int(comparison["seed"]),
                "prompt": comparison["prompt"],
                "harms": harms,
                "failed_metrics": failed_metrics,
                "passes": not failed_metrics,
            }
        )

    holdout = contract["holdout"]
    is_registered_holdout = selection["split"] == holdout["split"]
    if is_registered_holdout:
        if (
            selection["sha256"] != holdout["split_sha256"]
            or len(selection["prompts"]) != holdout["prompt_count"]
            or tuple(manifest["protocol"]["rng"]["seeds"]) != tuple(holdout["seeds"])
        ):
            raise ValueError("holdout prompt or seed identity differs from the contract")
        validate_static_candidate(manifest, contract["static_candidate"])
        candidate_ids = {row["candidate_id"] for row in rows}
        if contract["static_candidate"]["candidate_id"] not in candidate_ids:
            raise ValueError("holdout must include the frozen static candidate")
        for candidate_id in candidate_ids:
            candidate_rows = [
                row for row in rows if row["candidate_id"] == candidate_id
            ]
            if len(candidate_rows) != holdout["sample_count"]:
                raise ValueError(
                    f"holdout candidate {candidate_id!r} sample count differs "
                    "from the contract"
                )

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    confidence = float(holdout["confidence"])
    target_upper = float(holdout["maximum_failure_rate_upper_bound"])
    summaries = []
    for candidate_id in sorted(grouped):
        candidate_rows = grouped[candidate_id]
        failures = sum(not row["passes"] for row in candidate_rows)
        upper = clopper_pearson_upper(failures, len(candidate_rows), confidence)
        summaries.append(
            {
                "candidate_id": candidate_id,
                "sample_count": len(candidate_rows),
                "failure_count": failures,
                "observed_failure_rate": failures / len(candidate_rows),
                "failure_rate_upper_bound": upper,
                "confidence": confidence,
                "target_upper_bound": target_upper,
                "passes_statistical_gate": upper <= target_upper,
            }
        )

    payload = {
        "schema": EVALUATION_SCHEMA,
        "schema_revision": SCHEMA_REVISION,
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "content_sha256": contract["sha256"],
        },
        "semantic_report": {
            "path": str(semantic_report_path),
            "sha256": sha256_file(semantic_report_path),
        },
        "quality_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "split": selection["split"],
        "split_sha256": selection["sha256"],
        "registered_holdout": is_registered_holdout,
        "margins": contract["margins"],
        "candidate_summaries": summaries,
        "rows": rows,
    }
    return {**payload, "sha256": canonical_sha256(payload)}
