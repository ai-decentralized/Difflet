#!/usr/bin/env python3
"""Evaluate the frozen warmup VQA terminal router on a prospective holdout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_flux_cache_online_gate import build_rows  # noqa: E402
from scripts.flux_cache_brake_intervention import (  # noqa: E402
    _quality_failed,
    _write_json,
    canonical_sha256,
    sha256_file,
)
from scripts.flux_cache_protocol import python_source_sha256  # noqa: E402

PROTOCOL_SCHEMA = "difflet-flux-cache-warmup-vqa-router-holdout"
PROTOCOL_SCHEMA_REVISION = 1
RESULT_SCHEMA = "difflet-flux-cache-warmup-vqa-router-holdout-result"
RESULT_SCHEMA_REVISION = 1


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return document


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_json(path)
    digest = protocol.get("sha256")
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    if digest != canonical_sha256(payload):
        raise ValueError("holdout protocol digest does not match")
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_revision") != PROTOCOL_SCHEMA_REVISION
    ):
        raise ValueError("holdout protocol schema is unsupported")
    if protocol.get("serving_claim") is not False or protocol.get("prospective") is not True:
        raise ValueError("holdout must be prospective and disable serving claims")
    for name in (
        "prompt_source",
        "candidate",
        "development_protocol",
        "development_result",
        "evaluator",
    ):
        binding = protocol[name]
        if sha256_file(_rooted(binding["path"])) != binding["file_sha256"]:
            raise ValueError(f"registered {name} file hash does not match")
    signal = protocol["router"]["online_signal"]
    if (
        signal["feature"] != "max_acceleration_cv"
        or signal["window"] != [2, 5]
        or signal["risk_direction"] != "lower_is_riskier"
    ):
        raise ValueError("holdout signal does not match the supported frozen router")
    if protocol["router"]["action"] != "disable_cache_from_step_6":
        raise ValueError("holdout action does not match the supported terminal brake")
    if protocol["collection"]["python_source_sha256"] != python_source_sha256(ROOT):
        raise ValueError("registered Python source hash does not match the worktree")
    return protocol


def _speed_indices(
    document: Mapping[str, Any], candidate_id: str
) -> tuple[dict[str, float], dict[str, float]]:
    baseline = {
        str(row["sample_id"]): float(row["elapsed_s"])
        for row in document["baseline"]["samples"]
    }
    matches = [row for row in document["candidates"] if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("speed manifest does not contain the frozen candidate exactly once")
    candidate = {
        str(row["sample_id"]): float(row["elapsed_s"]) for row in matches[0]["samples"]
    }
    return baseline, candidate


def _rank_metrics(labels: list[bool], risks: list[float]) -> dict[str, float | None]:
    if set(labels) != {False, True}:
        return {"roc_auc": None, "average_precision": None}
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "roc_auc": float(roc_auc_score(labels, risks)),
        "average_precision": float(average_precision_score(labels, risks)),
    }


def evaluate(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    quality_path = Path(args.quality_input).expanduser().resolve()
    semantic_path = Path(args.semantic_report).expanduser().resolve()
    speed_path = Path(args.speed_manifest).expanduser().resolve()
    quality = _load_json(quality_path)
    semantic = _load_json(semantic_path)
    speed = _load_json(speed_path)
    rows = build_rows(quality, semantic, quality_root=quality_path.parent)
    expected = int(protocol["sample_matrix"]["expected_comparisons"])
    if len(rows) != expected:
        raise ValueError(f"holdout expected {expected} comparisons, got {len(rows)}")
    candidate_id = protocol["candidate"]["candidate_id"]
    if {row["candidate_id"] for row in rows} != {candidate_id}:
        raise ValueError("holdout comparison candidate differs from frozen candidate")
    baseline_times, candidate_times = _speed_indices(speed, candidate_id)
    threshold = float(protocol["router"]["online_signal"]["threshold"])
    vqa_margin = float(protocol["offline_label"]["paired_harm_threshold"])
    quality_margins = {
        "image_reward": float(protocol["quality_contract"]["image_reward_max_harm"]),
        "vqa_score": vqa_margin,
    }
    evaluated = []
    for row in rows:
        signal = float(row["features"]["max_acceleration_cv"])
        if not math.isfinite(signal):
            raise ValueError("holdout signal must be finite")
        triggered = signal <= threshold
        vqa_failed = float(row["vqa_score_harm"]) > vqa_margin
        semantic_row = next(
            item
            for item in semantic["comparisons"]
            if item["candidate_id"] == candidate_id and item["sample_id"] == row["sample_id"]
        )
        original_failed, original_harm = _quality_failed(
            baseline=semantic_row["baseline_scores"],
            candidate=semantic_row["candidate_scores"],
            margins=quality_margins,
        )
        if triggered:
            routed_scores = semantic_row["baseline_scores"]
            elapsed = baseline_times[row["sample_id"]]
            route = "terminal"
        else:
            routed_scores = semantic_row["candidate_scores"]
            elapsed = candidate_times[row["sample_id"]]
            route = "oil"
        routed_failed, routed_harm = _quality_failed(
            baseline=semantic_row["baseline_scores"],
            candidate=routed_scores,
            margins=quality_margins,
        )
        evaluated.append(
            {
                "sample_id": row["sample_id"],
                "prompt_index": row["prompt_index"],
                "seed": row["seed"],
                "signal": signal,
                "risk_score": -signal,
                "triggered": triggered,
                "route": route,
                "vqa_score_harm": row["vqa_score_harm"],
                "vqa_failed": vqa_failed,
                "original_failed": original_failed,
                "original_harm": original_harm,
                "routed_failed": routed_failed,
                "routed_harm": routed_harm,
                "original_elapsed_s": candidate_times[row["sample_id"]],
                "routed_elapsed_s_diagnostic": elapsed,
            }
        )
    positives = [row for row in evaluated if row["vqa_failed"]]
    negatives = [row for row in evaluated if not row["vqa_failed"]]
    missed = [row for row in positives if not row["triggered"]]
    false_routes = [row for row in negatives if row["triggered"]]
    introduced = [row for row in evaluated if not row["original_failed"] and row["routed_failed"]]
    positive_groups = len({row["prompt_index"] for row in positives})
    failure_recall = 1.0 - len(missed) / len(positives) if positives else None
    false_route_rate = len(false_routes) / len(negatives) if negatives else None
    ranking = _rank_metrics(
        [row["vqa_failed"] for row in evaluated],
        [row["risk_score"] for row in evaluated],
    )
    criteria = protocol["decision_rule"]
    enough_positives = positive_groups >= int(criteria["minimum_positive_prompt_groups"])
    metrics_pass = bool(
        enough_positives
        and failure_recall is not None
        and failure_recall >= float(criteria["minimum_failure_recall"])
        and false_route_rate is not None
        and false_route_rate <= float(criteria["maximum_passing_false_route_rate"])
        and ranking["roc_auc"] is not None
        and ranking["roc_auc"] >= float(criteria["minimum_roc_auc"])
        and not introduced
    )
    if not enough_positives:
        status = "prospective_inconclusive_too_few_positive_groups"
    elif metrics_pass:
        status = "prospective_replication_passed"
    else:
        status = "prospective_replication_rejected"
    original_total = sum(row["original_elapsed_s"] for row in evaluated)
    routed_total = sum(row["routed_elapsed_s_diagnostic"] for row in evaluated)
    document = {
        "schema": RESULT_SCHEMA,
        "schema_revision": RESULT_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "prospective": True,
        "serving_claim": False,
        "inputs": {
            "protocol_path": str(protocol_path),
            "protocol_file_sha256": sha256_file(protocol_path),
            "protocol_content_sha256": protocol["sha256"],
            "quality_input_path": str(quality_path),
            "quality_input_file_sha256": sha256_file(quality_path),
            "semantic_report_path": str(semantic_path),
            "semantic_report_file_sha256": sha256_file(semantic_path),
            "speed_manifest_path": str(speed_path),
            "speed_manifest_file_sha256": sha256_file(speed_path),
        },
        "router": protocol["router"],
        "offline_label": protocol["offline_label"],
        "summary": {
            "comparison_count": len(evaluated),
            "vqa_failure_count": len(positives),
            "vqa_failure_prompt_group_count": positive_groups,
            "failure_recall": failure_recall,
            "missed_failure_count": len(missed),
            "passing_false_route_count": len(false_routes),
            "passing_false_route_rate": false_route_rate,
            "triggered_count": sum(row["triggered"] for row in evaluated),
            "original_contract_failure_count": sum(row["original_failed"] for row in evaluated),
            "routed_contract_failure_count": sum(row["routed_failed"] for row in evaluated),
            "introduced_contract_failure_count": len(introduced),
            "ranking": ranking,
            "original_total_s": original_total,
            "routed_total_s_diagnostic": routed_total,
            "routing_cost_increase_fraction_diagnostic": routed_total / original_total - 1.0,
        },
        "decision": {
            "status": status,
            "criteria": criteria,
            "serving_qualified": False,
            "reason": (
                "The frozen signal replicated on enough independent positive prompt groups."
                if metrics_pass
                else "The frozen prospective decision rule was not satisfied."
            ),
        },
        "rows": evaluated,
    }
    destination = Path(args.out).expanduser().resolve()
    _write_json(destination, document, add_digest=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--quality-input", required=True)
    parser.add_argument("--semantic-report", required=True)
    parser.add_argument("--speed-manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    destination = evaluate(args)
    print(f"[warmup-router-holdout] result={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
