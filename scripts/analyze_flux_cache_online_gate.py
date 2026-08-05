#!/usr/bin/env python3
"""Fit a tiny development-only FLUX cache-risk gate from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

IMAGE_REWARD_MAX_HARM = 0.7824214100837708
VQA_SCORE_MAX_HARM = 0.25

MODULATION_FEATURES = (
    "max_region_level",
    "max_top2_region_level",
    "max_level_cv",
    "max_region_acceleration",
    "max_top2_region_acceleration",
    "max_acceleration_range",
    "max_acceleration_cv",
    "peak_acceleration_level_cv_percent",
)
OUTPUT_FEATURES = (
    "output_max_region_relative_l1",
    "output_max_top2_region_relative_l1",
    "output_max_region_velocity_turn",
    "output_max_top2_region_velocity_turn",
    "output_max_region_acceleration_ratio",
    "output_max_top2_region_acceleration_ratio",
)
FEATURE_GROUPS = {
    "modulation_only": MODULATION_FEATURES,
    "output_only": OUTPUT_FEATURES,
    "combined": MODULATION_FEATURES + OUTPUT_FEATURES,
}


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be a JSON object")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _semantic_index(document: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if not document.get("complete"):
        raise ValueError("semantic report is incomplete")
    comparisons = document.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("semantic report contains no comparisons")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in comparisons:
        key = (str(row["candidate_id"]), str(row["sample_id"]))
        if key in result:
            raise ValueError("semantic report contains duplicate candidate/sample rows")
        result[key] = row
    return result


def build_rows(
    quality_document: Mapping[str, Any],
    semantic_document: Mapping[str, Any],
    *,
    quality_root: Path,
) -> list[dict[str, Any]]:
    """Join semantic labels to serving-available feature artifacts."""

    comparisons = quality_document.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("quality manifest contains no comparisons")
    semantic = _semantic_index(semantic_document)
    expected_features = set(FEATURE_GROUPS["combined"])
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        candidate_id = str(comparison["candidate_id"])
        sample_id = str(comparison["sample_id"])
        key = (candidate_id, sample_id)
        semantic_row = semantic.get(key)
        if semantic_row is None:
            raise ValueError("quality and semantic comparison matrices differ")
        if (
            semantic_row["prompt"] != comparison["prompt"]
            or int(semantic_row["seed"]) != int(comparison["seed"])
        ):
            raise ValueError("semantic comparison identity mismatch")

        candidate_artifacts = comparison.get("candidate")
        if not isinstance(candidate_artifacts, dict):
            raise ValueError("candidate artifacts are missing")
        relative = candidate_artifacts.get("online_signal")
        expected_sha256 = candidate_artifacts.get("online_signal_sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise ValueError("candidate online-signal artifact is missing")
        signal_path = (quality_root / relative).resolve()
        if _sha256_file(signal_path) != expected_sha256:
            raise ValueError("candidate online-signal sha256 mismatch")
        signal = _load_json(signal_path, "online-signal artifact")
        features = signal.get("features")
        if not isinstance(features, dict) or set(features) != expected_features:
            raise ValueError("online-signal feature fields do not match the frozen set")
        feature_values = {
            name: _finite_float(features[name], f"features.{name}")
            for name in FEATURE_GROUPS["combined"]
        }

        baseline_scores = semantic_row["baseline_scores"]
        candidate_scores = semantic_row["candidate_scores"]
        image_reward_harm = _finite_float(
            baseline_scores["image_reward"], "baseline ImageReward"
        ) - _finite_float(candidate_scores["image_reward"], "candidate ImageReward")
        vqa_score_harm = _finite_float(
            baseline_scores["vqa_score"], "baseline VQAScore"
        ) - _finite_float(candidate_scores["vqa_score"], "candidate VQAScore")
        failed = bool(
            image_reward_harm > IMAGE_REWARD_MAX_HARM
            or vqa_score_harm > VQA_SCORE_MAX_HARM
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "prompt_index": int(comparison["prompt_index"]),
                "prompt": comparison["prompt"],
                "seed": int(comparison["seed"]),
                "features": feature_values,
                "image_reward_harm": image_reward_harm,
                "vqa_score_harm": vqa_score_harm,
                "failed": failed,
            }
        )
    if set(semantic) != {
        (row["candidate_id"], row["sample_id"]) for row in rows
    }:
        raise ValueError("quality and semantic comparison matrices differ")
    return rows


def threshold_at_full_recall(
    labels: Sequence[int], scores: Sequence[float]
) -> dict[str, Any]:
    """Choose the highest inclusive threshold retaining every positive."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and equal length")
    positive_scores = [float(score) for label, score in zip(labels, scores) if label]
    if not positive_scores:
        raise ValueError("full-recall threshold requires at least one positive")
    threshold = min(positive_scores)
    predicted = [float(score) >= threshold for score in scores]
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    true_positives = sum(bool(label) and prediction for label, prediction in zip(labels, predicted))
    false_positives = sum(not bool(label) and prediction for label, prediction in zip(labels, predicted))
    return {
        "threshold": threshold,
        "failure_recall": true_positives / positives,
        "passing_false_brake_rate": false_positives / negatives if negatives else None,
        "passing_false_brake_count": false_positives,
        "passing_count": negatives,
    }


def _metric_summary(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(set(labels)) != 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "full_recall_operating_point": threshold_at_full_recall(labels, scores),
    }


def grouped_oof_logistic(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> dict[str, Any]:
    """Return leave-one-prompt-out probabilities and a final all-data fit."""

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray([int(bool(row["failed"])) for row in rows], dtype=np.int64)
    groups = np.asarray([int(row["prompt_index"]) for row in rows], dtype=np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("gate fitting requires both passing and failing samples")
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    splitter = LeaveOneGroupOut()
    fold_count = 0
    for train_indices, test_indices in splitter.split(x, y, groups):
        if len(set(y[train_indices].tolist())) != 2:
            raise ValueError("a grouped training fold contains only one class")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                l1_ratio=0.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=0,
            ),
        )
        model.fit(x[train_indices], y[train_indices])
        oof[test_indices] = model.predict_proba(x[test_indices])[:, 1]
        fold_count += 1
    if np.isnan(oof).any():
        raise RuntimeError("grouped cross-validation did not score every row")

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
        ),
    )
    final_model.fit(x, y)
    scaler = final_model.named_steps["standardscaler"]
    logistic = final_model.named_steps["logisticregression"]
    metrics = _metric_summary(y.tolist(), oof.tolist())
    by_candidate: dict[str, Any] = {}
    candidate_ids = sorted({str(row["candidate_id"]) for row in rows})
    for candidate_id in candidate_ids:
        indices = [
            index for index, row in enumerate(rows)
            if row["candidate_id"] == candidate_id
        ]
        labels = y[indices].tolist()
        scores = oof[indices].tolist()
        by_candidate[candidate_id] = {
            "sample_count": len(indices),
            "failure_count": sum(labels),
            **_metric_summary(labels, scores),
        }
    return {
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "fold_count": fold_count,
        "metrics": metrics,
        "by_candidate": by_candidate,
        "oof_probabilities": oof.tolist(),
        "final_fit": {
            "standardization_mean": scaler.mean_.tolist(),
            "standardization_scale": scaler.scale_.tolist(),
            "coefficients": logistic.coef_[0].tolist(),
            "intercept": float(logistic.intercept_[0]),
            "probability": "sigmoid(intercept + sum(coefficients[i] * ((x[i]-mean[i])/scale[i])))",
        },
    }


def analyze(quality_path: Path, semantic_path: Path) -> dict[str, Any]:
    quality = _load_json(quality_path, "quality manifest")
    semantic = _load_json(semantic_path, "semantic report")
    rows = build_rows(quality, semantic, quality_root=quality_path.parent)
    labels = [int(row["failed"]) for row in rows]
    models = {
        name: grouped_oof_logistic(rows, feature_names)
        for name, feature_names in FEATURE_GROUPS.items()
    }
    selected_name = max(
        models,
        key=lambda name: (
            models[name]["metrics"]["average_precision"],
            -models[name]["feature_count"],
        ),
    )
    for index, row in enumerate(rows):
        row["oof_probabilities"] = {
            name: float(model["oof_probabilities"][index])
            for name, model in models.items()
        }
    failure_count = sum(labels)
    candidate_counts = {
        candidate_id: sum(
            row["failed"] for row in rows if row["candidate_id"] == candidate_id
        )
        for candidate_id in sorted({row["candidate_id"] for row in rows})
    }
    target_id = "adaptive-stage-oil-p30-e1p40-w6-i8-k12-o1-index"
    return {
        "schema": "difflet-flux-cache-online-gate-development",
        "schema_revision": 1,
        "sources": {
            "quality_manifest": {
                "path": str(quality_path),
                "sha256": _sha256_file(quality_path),
            },
            "semantic_report": {
                "path": str(semantic_path),
                "sha256": _sha256_file(semantic_path),
            },
        },
        "contract": {
            "image_reward_max_harm": IMAGE_REWARD_MAX_HARM,
            "vqa_score_max_harm": VQA_SCORE_MAX_HARM,
            "combination": "fail if either paired harm is strictly above its limit",
        },
        "sample_count": len(rows),
        "failure_count": failure_count,
        "failure_count_by_candidate": candidate_counts,
        "stopping_rules": {
            "at_least_five_total_failures": failure_count >= 5,
            "at_least_two_target_policy_failures": candidate_counts.get(target_id, 0) >= 2,
        },
        "validation": "leave-one-prompt-index-out; all seeds and candidate strata for one prompt are held out together",
        "models": models,
        "selection": {
            "primary_metric": "grouped out-of-fold average precision",
            "selected_model": selected_name,
            "selected_feature_names": models[selected_name]["feature_names"],
            "development_only": True,
            "serving_claim": False,
            "qualification_required": "freeze this model and its thresholds, then run a new independent positive-containing holdout",
        },
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = analyze(
        args.quality_input.expanduser().resolve(),
        args.semantic_report.expanduser().resolve(),
    )
    _write_json(args.out.expanduser().resolve(), result)
    selected = result["selection"]["selected_model"]
    print(json.dumps(result["models"][selected]["metrics"], indent=2, sort_keys=True))
    print(f"selected={selected} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
