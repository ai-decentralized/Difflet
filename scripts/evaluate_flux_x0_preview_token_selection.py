#!/usr/bin/env python3
"""Test in-distribution CLIP patch selection for the x0-preview XL-VQA brake.

Unlike average pooling, this diagnostic only removes tokens: every retained
visual embedding is an unchanged CLIP patch embedding.  The frozen primary
keeps the largest-L2 patch in every 2x2 spatial cell (144 of 576 tokens).
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluate_flux_x0_preview_token_pooling as base


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_VARIANT = "local_maxnorm_1of4"
VARIANTS = (
    (PRIMARY_VARIANT, 1),
    ("local_medoid_1of4", 1),
    ("local_maxnorm_2of4", 2),
)


def _local_select(features: Any, *, keep: int, mode: str, torch: Any) -> Any:
    batch, tokens, channels = features.shape
    source_grid = math.isqrt(tokens)
    if source_grid != 24 or tokens != source_grid * source_grid:
        raise ValueError(f"expected a 24x24 CLIP patch grid, got {tokens} tokens")
    target_grid = source_grid // 2
    blocks = (
        features.reshape(batch, target_grid, 2, target_grid, 2, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, target_grid, target_grid, 4, channels)
    )
    if mode == "maxnorm":
        importance = blocks.float().square().sum(dim=-1)
        selected = importance.topk(keep, dim=-1, largest=True, sorted=False).indices
    elif mode == "medoid":
        if keep != 1:
            raise ValueError("medoid selector only supports one token per cell")
        center = blocks.float().mean(dim=-2, keepdim=True)
        distance = (blocks.float() - center).square().sum(dim=-1)
        selected = distance.argmin(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unsupported selector: {mode}")
    selected = selected.sort(dim=-1).values
    gather_index = selected.unsqueeze(-1).expand(*selected.shape, channels)
    retained = torch.gather(blocks, dim=-2, index=gather_index)
    return retained.reshape(batch, target_grid * target_grid * keep, channels)


def _threshold_at_false_brake_budget(
    labels: list[int], scores: list[float], false_brake_budget: int
) -> dict[str, Any]:
    candidates = sorted(set(float(score) for score in scores))
    operating = [
        base._lower_risk_operating_point(labels, scores, threshold)
        for threshold in candidates
    ]
    qualified = [row for row in operating if row["false_positive_count"] <= false_brake_budget]
    if not qualified:
        raise ValueError("no threshold satisfies the false-brake budget")
    return max(
        qualified,
        key=lambda row: (row["true_positive_count"], row["threshold"]),
    )


def _run(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.torch_threads)
    distribution = base.semantic.importlib.metadata.distribution("t2v-metrics")
    package_root = Path(distribution.locate_file("t2v_metrics")).resolve()
    module = base.semantic._load_clip_flant5_module(package_root)
    load_started = time.perf_counter()
    wrapper = module.CLIPT5Model(
        base.semantic.VQA_MODEL,
        device="cpu",
        cache_dir=str(args.model_cache),
    )
    original_load_images = wrapper.load_images
    wrapper.load_images = lambda images: original_load_images(images).to(dtype=torch.bfloat16)
    load_seconds = time.perf_counter() - load_started

    def one(row: dict[str, Any], *, record: bool) -> dict[str, Any]:
        preprocess_started = time.perf_counter()
        image_tensor = wrapper.load_images([row["image_path"]])
        preprocess_seconds = time.perf_counter() - preprocess_started
        tokenize_started = time.perf_counter()
        tokenized = base._tokenize(module, wrapper, row["prompt"], torch)
        tokenize_seconds = time.perf_counter() - tokenize_started
        vision_started = time.perf_counter()
        vision_features = wrapper.model.get_vision_tower()(image_tensor)
        vision_seconds = time.perf_counter() - vision_started
        variants = {}
        for name, keep in VARIANTS:
            selector = "medoid" if "medoid" in name else "maxnorm"
            projector_started = time.perf_counter()
            retained = _local_select(vision_features, keep=keep, mode=selector, torch=torch)
            projected = wrapper.model.mm_projector(retained)
            projector_seconds = time.perf_counter() - projector_started
            score, timing = base._score_projected(
                wrapper, tokenized, image_tensor, projected, torch
            )
            timing.update(
                {
                    "preprocess": preprocess_seconds,
                    "tokenize": tokenize_seconds,
                    "vision_tower": vision_seconds,
                    "select_and_projector": projector_seconds,
                    "estimated_end_to_end": (
                        preprocess_seconds
                        + tokenize_seconds
                        + vision_seconds
                        + projector_seconds
                        + timing["model_total"]
                    ),
                }
            )
            visual_tokens = 12 * 12 * keep
            variants[name] = {
                "score": score,
                "visual_token_count": visual_tokens,
                "encoder_sequence_tokens": visual_tokens + tokenized["text_token_count"],
                "timing_seconds": timing,
            }
        if not record:
            return {}
        return {
            **row,
            "text_token_count": tokenized["text_token_count"],
            "answer_token_count": tokenized["answer_token_count"],
            "variants": variants,
        }

    print("[x0-vqa-select] warmup", flush=True)
    one(rows[0], record=False)
    one(rows[0], record=False)
    scored = []
    for index, row in enumerate(rows, start=1):
        scored.append(one(row, record=True))
        print(f"[x0-vqa-select] {index}/{len(rows)} {row['cohort']} {row['sample_id']}", flush=True)
    return {
        "load_seconds": load_seconds,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
        "rows": scored,
    }


def _analyze(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["rows"]
    stress = [row for row in rows if row["cohort"] == "stress_step28"]
    independent = [row for row in rows if row["cohort"] == "independent_step28"]
    variants = {}
    for name, keep in VARIANTS:
        stress_scores = [float(row["variants"][name]["score"]) for row in stress]
        independent_scores = [float(row["variants"][name]["score"]) for row in independent]
        stress_actionable = [int(row["terminal29_actionable"]) for row in stress]
        stress_failure = [int(row["continue_failure"]) for row in stress]
        independent_failure = [int(row["continue_failure"]) for row in independent]
        threshold = _threshold_at_false_brake_budget(stress_actionable, stress_scores, 9)
        independent_operating = base._lower_risk_operating_point(
            independent_failure, independent_scores, float(threshold["threshold"])
        )
        timing: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            for field, value in row["variants"][name]["timing_seconds"].items():
                timing[field].append(float(value))
        variants[name] = {
            "visual_token_count": 12 * 12 * keep,
            "stress": {
                "actionable_roc_auc_lower_is_riskier": base._binary_auc(
                    stress_actionable, [-score for score in stress_scores]
                ),
                "continue_failure_roc_auc_lower_is_riskier": base._binary_auc(
                    stress_failure, [-score for score in stress_scores]
                ),
                "false_brake_budget_operating_point": threshold,
                "spearman_vs_stored_canonical_preview_vqa": base._spearman(
                    stress_scores, [float(row["stored_full_vqa"]) for row in stress]
                ),
            },
            "independent": {
                "continue_failure_roc_auc_lower_is_riskier": base._binary_auc(
                    independent_failure, [-score for score in independent_scores]
                ),
                "frozen_stress_threshold_operating_point": independent_operating,
                "spearman_vs_stored_canonical_preview_vqa": base._spearman(
                    independent_scores,
                    [float(row["stored_full_vqa"]) for row in independent],
                ),
            },
            "timing_seconds": {
                field: base._summary(values) for field, values in sorted(timing.items())
            },
        }
    primary = variants[PRIMARY_VARIANT]
    return {
        "variants": variants,
        "primary": {
            "variant": PRIMARY_VARIANT,
            "median_judge_seconds": primary["timing_seconds"]["estimated_end_to_end"][
                "median"
            ],
            "stress_actionable_roc_auc": primary["stress"][
                "actionable_roc_auc_lower_is_riskier"
            ],
            "stress_actionable_caught_at_nine_false_brakes": primary["stress"][
                "false_brake_budget_operating_point"
            ]["true_positive_count"],
            "independent_continue_failure_roc_auc": primary["independent"][
                "continue_failure_roc_auc_lower_is_riskier"
            ],
            "independent_failures_caught_at_frozen_threshold": primary["independent"][
                "frozen_stress_threshold_operating_point"
            ]["true_positive_count"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.set_defaults(
        out=ROOT / "benchmark/flux_cache/x0-preview-vqa-token-selection-result.json"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = base._load_study_rows(args)
    run = _run(args, rows)
    result = {
        "schema": "difflet-flux-x0-preview-vqa-token-selection-result",
        "schema_revision": 1,
        "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "frozen_primary": {
            "variant": PRIMARY_VARIANT,
            "selection": "largest L2-norm unchanged CLIP patch in each non-overlapping 2x2 cell",
            "visual_tokens": 144,
            "risk_direction": "lower_is_riskier",
            "development_threshold_rule": (
                "maximize actionable recall subject to at most nine stress false brakes"
            ),
            "independent_use": "apply the numeric stress threshold unchanged",
        },
        "exploratory": ["local_medoid_1of4", "local_maxnorm_2of4"],
        "runtime": {key: value for key, value in run.items() if key != "rows"},
        "analysis": _analyze(run),
        "rows": run["rows"],
    }
    base._write_json(args.out.resolve(), result)
    print(json.dumps(result["analysis"]["primary"], indent=2, sort_keys=True), flush=True)
    print(f"[x0-vqa-select] result={args.out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
