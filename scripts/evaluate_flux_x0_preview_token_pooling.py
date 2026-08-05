#!/usr/bin/env python3
"""Measure whether visual-token pooling makes x0-preview XL-VQA affordable.

This is a serving-cost diagnostic, not a replacement quality metric.  It keeps
the frozen CLIP-FlanT5-XL checkpoint and question/answer contract, and only
average-pools the CLIP 24x24 patch grid before the learned multimodal
projector.  The 12x12 grid is the frozen primary variant; 8x8 is exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_flux_cache_semantics as semantic  # noqa: E402


PRIMARY_GRID = 12
EXPLORATORY_GRIDS = (8,)
FULL_GRID = 24
VQA_HARM_MARGIN = 0.25
STRESS_CANDIDATE = "adaptive-vqa-stress-w6-i32-m16-k40-o1-index"
INDEPENDENT_CANDIDATE = "adaptive-oil-e1p80-w6-i8-k12-o1-index"


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    body = dict(document)
    body.pop("sha256", None)
    encoded = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    body["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    encoded = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman requires equal nontrivial vectors")
    x = _rank(left)
    y = _rank(right)
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_norm = sum((a - x_mean) ** 2 for a in x)
    y_norm = sum((b - y_mean) ** 2 for b in y)
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return numerator / math.sqrt(x_norm * y_norm)


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive = [float(score) for label, score in zip(labels, scores, strict=True) if label]
    negative = [float(score) for label, score in zip(labels, scores, strict=True) if not label]
    if not positive or not negative:
        raise ValueError("AUC requires positive and negative rows")
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            wins += float(positive_score > negative_score)
            wins += 0.5 * float(positive_score == negative_score)
    return wins / (len(positive) * len(negative))


def _lower_risk_operating_point(
    labels: Sequence[int], scores: Sequence[float], threshold: float | None = None
) -> dict[str, Any]:
    if threshold is None:
        threshold = max(float(score) for label, score in zip(labels, scores, strict=True) if label)
    triggered = [float(score) <= threshold for score in scores]
    true_positive = sum(bool(label) and hit for label, hit in zip(labels, triggered, strict=True))
    false_positive = sum(not bool(label) and hit for label, hit in zip(labels, triggered, strict=True))
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    return {
        "threshold": threshold,
        "comparison": "score <= threshold",
        "positive_count": positive_count,
        "negative_count": negative_count,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "recall": true_positive / positive_count if positive_count else None,
        "false_brake_rate": false_positive / negative_count if negative_count else None,
        "trigger_count": sum(triggered),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def _comparison_rows(path: Path, candidate_id: str | None = None) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _load_json(path)["comparisons"]:
        if candidate_id is not None and str(row["candidate_id"]) != candidate_id:
            continue
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate sample {sample_id} in {path}")
        rows[sample_id] = row
    return rows


def _preview_rows(root: Path) -> dict[str, dict[str, Any]]:
    quality_path = root / "quality-input.json"
    semantic_path = root / "semantic-scores.json"
    quality = _load_json(quality_path)
    stored = _comparison_rows(semantic_path)
    rows = {}
    for comparison in quality["comparisons"]:
        sample_id = str(comparison["sample_id"])
        score = stored[sample_id]
        rows[sample_id] = {
            "sample_id": sample_id,
            "prompt": str(comparison["prompt"]),
            "image_path": str(Path(comparison["candidate"]["image"]).resolve()),
            "stored_full_vqa": float(score["candidate_scores"]["vqa_score"]),
        }
    return rows


def _load_study_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    stress_preview = _preview_rows(args.stress_preview_root)
    stress_source = _comparison_rows(args.stress_semantic, STRESS_CANDIDATE)
    terminal = {}
    for path in args.terminal_semantic:
        terminal.update(_comparison_rows(path, "terminal-step-29"))
    if set(stress_preview) != set(stress_source) or set(stress_preview) != set(terminal):
        raise ValueError("stress preview/source/terminal sample matrices differ")

    rows = []
    for sample_id, preview in stress_preview.items():
        source = stress_source[sample_id]
        terminal_row = terminal[sample_id]
        continue_harm = float(source["baseline_scores"]["vqa_score"]) - float(
            source["candidate_scores"]["vqa_score"]
        )
        terminal_harm = float(terminal_row["baseline_scores"]["vqa_score"]) - float(
            terminal_row["candidate_scores"]["vqa_score"]
        )
        rows.append(
            {
                **preview,
                "cohort": "stress_step28",
                "continue_harm": continue_harm,
                "continue_failure": continue_harm > VQA_HARM_MARGIN,
                "terminal29_harm": terminal_harm,
                "terminal29_actionable": (
                    continue_harm > VQA_HARM_MARGIN and terminal_harm <= VQA_HARM_MARGIN
                ),
            }
        )

    independent_preview = _preview_rows(args.independent_preview_root)
    independent_source = _comparison_rows(args.independent_semantic, INDEPENDENT_CANDIDATE)
    if set(independent_preview) != set(independent_source):
        raise ValueError("independent preview/source sample matrices differ")
    for sample_id, preview in independent_preview.items():
        source = independent_source[sample_id]
        continue_harm = float(source["baseline_scores"]["vqa_score"]) - float(
            source["candidate_scores"]["vqa_score"]
        )
        rows.append(
            {
                **preview,
                "cohort": "independent_step28",
                "continue_harm": continue_harm,
                "continue_failure": continue_harm > VQA_HARM_MARGIN,
                "terminal29_harm": None,
                "terminal29_actionable": None,
            }
        )
    return rows


class _StageTimer:
    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.elapsed: defaultdict[str, float] = defaultdict(float)

    def pre(self, name: str):
        def hook(_module, _inputs):
            self.started[name] = time.perf_counter()

        return hook

    def post(self, name: str):
        def hook(_module, _inputs, _output):
            self.elapsed[name] += time.perf_counter() - self.started.pop(name)

        return hook


def _tokenize(module: Any, wrapper: Any, prompt: str, torch: Any) -> dict[str, Any]:
    question = module.default_question_template.format(prompt)
    question = module.format_question(question, conversation_style=wrapper.conversational_style)
    answer = module.format_answer(
        module.default_answer_template.format(prompt),
        conversation_style=wrapper.conversational_style,
    )
    input_ids = module.t5_tokenizer_image_token(question, wrapper.tokenizer, return_tensors="pt")
    labels = module.t5_tokenizer_image_token(answer, wrapper.tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(wrapper.tokenizer.pad_token_id),
        "decoder_attention_mask": labels.ne(module.IGNORE_INDEX),
        "labels": labels,
        "text_token_count": int(input_ids.shape[1] - 1),
        "answer_token_count": int(labels.shape[1]),
    }


def _pool_features(features: Any, grid: int, torch: Any) -> Any:
    if grid == FULL_GRID:
        return features
    batch, tokens, channels = features.shape
    source_grid = math.isqrt(tokens)
    if source_grid * source_grid != tokens or source_grid % grid:
        raise ValueError(f"cannot pool {tokens} visual tokens to {grid}x{grid}")
    spatial = features.reshape(batch, source_grid, source_grid, channels).permute(0, 3, 1, 2)
    ratio = source_grid // grid
    spatial = torch.nn.functional.avg_pool2d(spatial, kernel_size=ratio, stride=ratio)
    return spatial.permute(0, 2, 3, 1).reshape(batch, grid * grid, channels)


def _score_projected(
    wrapper: Any,
    tokenized: dict[str, Any],
    images: Any,
    projected: Any,
    torch: Any,
) -> tuple[float, dict[str, float]]:
    model = wrapper.model
    original_encode_images = model.encode_images
    model.encode_images = lambda _images: projected
    timer = _StageTimer()
    handles = []
    for name, layer in (("t5_encoder", model.encoder), ("t5_decoder", model.decoder), ("lm_head", model.lm_head)):
        handles.append(layer.register_forward_pre_hook(timer.pre(name)))
        handles.append(layer.register_forward_hook(timer.post(name)))
    started = time.perf_counter()
    try:
        outputs = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            decoder_attention_mask=tokenized["decoder_attention_mask"],
            labels=tokenized["labels"],
            images=images,
            return_dict=True,
        )
    finally:
        model.encode_images = original_encode_images
        for handle in handles:
            handle.remove()
    model_seconds = time.perf_counter() - started
    loss = torch.nn.functional.cross_entropy(
        outputs.logits[0], tokenized["labels"][0], reduction="mean"
    )
    score = float((-loss).exp().float().item())
    timing = dict(timer.elapsed)
    timing["model_total"] = model_seconds
    timing["model_other"] = model_seconds - sum(timer.elapsed.values())
    return score, timing


def _run_scores(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.torch_threads)
    distribution = semantic.importlib.metadata.distribution("t2v-metrics")
    package_root = Path(distribution.locate_file("t2v_metrics")).resolve()
    module = semantic._load_clip_flant5_module(package_root)
    load_started = time.perf_counter()
    wrapper = module.CLIPT5Model(
        semantic.VQA_MODEL,
        device="cpu",
        cache_dir=str(args.model_cache),
    )
    original_load_images = wrapper.load_images
    wrapper.load_images = lambda images: original_load_images(images).to(dtype=torch.bfloat16)
    load_seconds = time.perf_counter() - load_started

    grids = (FULL_GRID, PRIMARY_GRID, *EXPLORATORY_GRIDS)

    def one(row: dict[str, Any], *, record: bool) -> dict[str, Any]:
        preprocess_started = time.perf_counter()
        image_tensor = wrapper.load_images([row["image_path"]])
        preprocess_seconds = time.perf_counter() - preprocess_started
        tokenize_started = time.perf_counter()
        tokenized = _tokenize(module, wrapper, row["prompt"], torch)
        tokenize_seconds = time.perf_counter() - tokenize_started
        vision_started = time.perf_counter()
        vision_features = wrapper.model.get_vision_tower()(image_tensor)
        vision_seconds = time.perf_counter() - vision_started
        variants = {}
        for grid in grids:
            projector_started = time.perf_counter()
            pooled = _pool_features(vision_features, grid, torch)
            projected = wrapper.model.mm_projector(pooled)
            projector_seconds = time.perf_counter() - projector_started
            score, timing = _score_projected(wrapper, tokenized, image_tensor, projected, torch)
            timing.update(
                {
                    "preprocess": preprocess_seconds,
                    "tokenize": tokenize_seconds,
                    "vision_tower": vision_seconds,
                    "pool_and_projector": projector_seconds,
                    "estimated_end_to_end": (
                        preprocess_seconds
                        + tokenize_seconds
                        + vision_seconds
                        + projector_seconds
                        + timing["model_total"]
                    ),
                }
            )
            variants[str(grid)] = {
                "score": score,
                "visual_token_count": grid * grid,
                "encoder_sequence_tokens": grid * grid + tokenized["text_token_count"],
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

    print("[x0-vqa-pool] warmup", flush=True)
    one(rows[0], record=False)
    one(rows[0], record=False)
    scored = []
    for index, row in enumerate(rows, start=1):
        scored.append(one(row, record=True))
        print(f"[x0-vqa-pool] {index}/{len(rows)} {row['cohort']} {row['sample_id']}", flush=True)
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
    results = {}
    for grid in (FULL_GRID, PRIMARY_GRID, *EXPLORATORY_GRIDS):
        key = str(grid)
        stress_scores = [float(row["variants"][key]["score"]) for row in stress]
        independent_scores = [float(row["variants"][key]["score"]) for row in independent]
        stress_actionable = [int(row["terminal29_actionable"]) for row in stress]
        stress_failures = [int(row["continue_failure"]) for row in stress]
        independent_failures = [int(row["continue_failure"]) for row in independent]
        operating = _lower_risk_operating_point(stress_actionable, stress_scores)
        independent_operating = _lower_risk_operating_point(
            independent_failures,
            independent_scores,
            float(operating["threshold"]),
        )
        timing_fields: defaultdict[str, list[float]] = defaultdict(list)
        for row in rows:
            for field, value in row["variants"][key]["timing_seconds"].items():
                timing_fields[field].append(float(value))
        results[key] = {
            "visual_grid": [grid, grid],
            "visual_token_count": grid * grid,
            "stress": {
                "sample_count": len(stress),
                "actionable_count": sum(stress_actionable),
                "actionable_roc_auc_lower_is_riskier": _binary_auc(
                    stress_actionable, [-score for score in stress_scores]
                ),
                "continue_failure_count": sum(stress_failures),
                "continue_failure_roc_auc_lower_is_riskier": _binary_auc(
                    stress_failures, [-score for score in stress_scores]
                ),
                "full_actionable_recall_operating_point": operating,
                "spearman_vs_stored_canonical_preview_vqa": _spearman(
                    stress_scores, [float(row["stored_full_vqa"]) for row in stress]
                ),
            },
            "independent": {
                "sample_count": len(independent),
                "continue_failure_count": sum(independent_failures),
                "continue_failure_roc_auc_lower_is_riskier": _binary_auc(
                    independent_failures, [-score for score in independent_scores]
                ),
                "frozen_stress_threshold_operating_point": independent_operating,
                "spearman_vs_stored_canonical_preview_vqa": _spearman(
                    independent_scores,
                    [float(row["stored_full_vqa"]) for row in independent],
                ),
            },
            "timing_seconds": {
                field: _summary(values) for field, values in sorted(timing_fields.items())
            },
        }

    full_errors = [
        abs(float(row["variants"][str(FULL_GRID)]["score"]) - float(row["stored_full_vqa"]))
        for row in rows
    ]
    primary = results[str(PRIMARY_GRID)]
    full = results[str(FULL_GRID)]
    primary_e2e = primary["timing_seconds"]["estimated_end_to_end"]["median"]
    full_e2e = full["timing_seconds"]["estimated_end_to_end"]["median"]
    return {
        "variants": results,
        "full_score_reproduction": {
            "maximum_absolute_error": max(full_errors),
            "mean_absolute_error": statistics.fmean(full_errors),
        },
        "primary": {
            "grid": PRIMARY_GRID,
            "visual_token_reduction": (FULL_GRID * FULL_GRID) / (PRIMARY_GRID * PRIMARY_GRID),
            "median_judge_speedup": full_e2e / primary_e2e,
            "median_judge_seconds": primary_e2e,
            "stress_actionable_roc_auc": primary["stress"][
                "actionable_roc_auc_lower_is_riskier"
            ],
            "stress_full_recall_false_brake_count": primary["stress"][
                "full_actionable_recall_operating_point"
            ]["false_positive_count"],
            "independent_continue_failure_roc_auc": primary["independent"][
                "continue_failure_roc_auc_lower_is_riskier"
            ],
            "independent_failures_caught_at_frozen_threshold": primary["independent"][
                "frozen_stress_threshold_operating_point"
            ]["true_positive_count"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stress-preview-root",
        type=Path,
        default=Path("/home/ubuntu/difflet-artifacts/flux-cache-x0-preview-step28-20260804"),
    )
    parser.add_argument(
        "--independent-preview-root",
        type=Path,
        default=Path(
            "/home/ubuntu/difflet-artifacts/flux-cache-x0-preview-step28-independent-oil-20260804"
        ),
    )
    parser.add_argument(
        "--stress-semantic",
        type=Path,
        default=Path(
            "/home/ubuntu/difflet-artifacts/flux-cache-warmup-vqa-stress-extreme-20260804/semantic-scores.json"
        ),
    )
    parser.add_argument(
        "--terminal-semantic",
        type=Path,
        nargs=2,
        default=(
            Path(
                "/home/ubuntu/difflet-artifacts/flux-cache-terminal-brake-timing-failures-20260804/semantic-scores.json"
            ),
            Path(
                "/home/ubuntu/difflet-artifacts/flux-cache-terminal-brake-step29-completion-20260804/semantic-scores.json"
            ),
        ),
    )
    parser.add_argument(
        "--independent-semantic",
        type=Path,
        default=Path(
            "/home/ubuntu/difflet-artifacts/flux-cache-online-signal-failure-enriched-20260803/semantic-scores.json"
        ),
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("/home/ubuntu/.cache/diffcache-semantic/vqascore"),
    )
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark/flux_cache/x0-preview-vqa-token-pooling-result.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = [
        args.stress_preview_root / "quality-input.json",
        args.stress_preview_root / "semantic-scores.json",
        args.independent_preview_root / "quality-input.json",
        args.independent_preview_root / "semantic-scores.json",
        args.stress_semantic,
        *args.terminal_semantic,
        args.independent_semantic,
    ]
    rows = _load_study_rows(args)
    run = _run_scores(args, rows)
    result = {
        "schema": "difflet-flux-x0-preview-vqa-token-pooling-result",
        "schema_revision": 1,
        "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "frozen_primary": {
            "checkpoint": semantic.VQA_MODEL,
            "question_template": "official t2v-metrics CLIP-FlanT5 VQAScore",
            "visual_pooling": "24x24 CLIP patches average-pooled to 12x12 before mm_projector",
            "grid": PRIMARY_GRID,
            "risk_direction": "lower_is_riskier",
            "development_label": "terminal29_actionable",
            "threshold_selection": "maximum primary score among actionable stress positives",
            "independent_use": "apply the numeric stress threshold unchanged",
        },
        "exploratory": [
            "unpooled 24x24 score reproduction and cost reference",
            "8x8 visual-token pooling",
        ],
        "input_files": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}
            for path in inputs
        ],
        "runtime": {key: value for key, value in run.items() if key != "rows"},
        "analysis": _analyze(run),
        "rows": run["rows"],
    }
    _write_json(args.out.resolve(), result)
    print(json.dumps(result["analysis"]["primary"], indent=2, sort_keys=True), flush=True)
    print(f"[x0-vqa-pool] result={args.out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
