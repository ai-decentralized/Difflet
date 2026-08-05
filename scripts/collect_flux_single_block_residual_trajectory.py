#!/usr/bin/env python3
"""Pair a cheap FLUX block-input signal with real MLP cache error.

The collector reuses registered cached denoising trajectories.  For every
sample it evaluates a truncated offline teacher at the last real anchor
(step 7) and at decisions available before terminal brakes at steps 21 and 29
(component states from steps 20 and 28).  It never runs the remaining 37
single blocks, final projection, scheduler, VAE, or quality scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flux_cache_brake_intervention import (  # noqa: E402
    _git_identity,
    _write_json,
    sha256_file,
)

MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
DEFAULT_SOURCE_ROOT = Path(
    "/home/ubuntu/difflet-artifacts/flux-cache-warmup-vqa-stress-extreme-20260804"
)
DEFAULT_TEACHER_CACHE = Path(
    "/home/ubuntu/difflet-artifacts/flux-single-block-real-trajectory-20260804/compiled-normalized-samples"
)
HARDWARE_ACK = "I am collecting real FLUX single-block residual trajectories"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--teacher-cache", default=str(DEFAULT_TEACHER_CACHE))
    parser.add_argument("--scope", choices=("failures", "all"), default="failures")
    parser.add_argument(
        "--decision-steps",
        default="20,28",
        help="Comma-separated component signal steps; each must follow the step-7 anchor.",
    )
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _roc_auc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return 0.0
    favorable = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return favorable / (len(positives) * len(negatives))


def _full_recall_diagnostic(labels: list[bool], scores: list[float]) -> dict[str, Any]:
    positives = [score for label, score in zip(labels, scores) if label]
    if not positives:
        return {"available": False}
    threshold = min(positives)
    triggered = [score >= threshold for score in scores]
    return {
        "available": True,
        "risk_direction": "higher",
        "threshold": threshold,
        "positive_count": sum(labels),
        "true_brake_count": sum(label and trigger for label, trigger in zip(labels, triggered)),
        "false_brake_count": sum(
            (not label) and trigger for label, trigger in zip(labels, triggered)
        ),
        "negative_count": sum(not label for label in labels),
        "trigger_count": sum(triggered),
    }


def _decision_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("decision-steps must contain integers") from exc
    if not steps or tuple(sorted(set(steps))) != steps:
        raise ValueError("decision-steps must be unique and increasing")
    if any(step <= 7 or step >= 50 for step in steps):
        raise ValueError("decision-steps must be between 8 and 49")
    return steps


def _relative_rms(current, anchor) -> float:
    current = current.detach().float().cpu()
    anchor = anchor.detach().float().cpu()
    numerator = (current - anchor).square().mean().sqrt()
    denominator = anchor.square().mean().sqrt().clamp_min(1e-12)
    return float(numerator / denominator)


def _taylor1(previous, latest, *, previous_step: int, latest_step: int, target_step: int):
    denominator = latest_step - previous_step
    if denominator == 0:
        raise ValueError("Taylor history steps must differ")
    result = latest.float() + (latest.float() - previous.float()) * (
        (target_step - latest_step) / denominator
    )
    return result.to(dtype=latest.dtype)


def _clone_probe_outputs(value):
    import torch

    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise TypeError(
            "residual teacher must return raw samples, moments, normalized samples, and MLP output"
        )
    if not all(torch.is_tensor(item) for item in value):
        raise TypeError("residual teacher outputs must be tensors")
    return tuple(item.detach().cpu().clone() for item in value)


def _terminal_vqa_rows() -> dict[str, dict[str, Any]]:
    paths = (
        Path(
            "/home/ubuntu/difflet-artifacts/flux-cache-terminal-brake-timing-failures-20260804/semantic-scores.json"
        ),
        Path(
            "/home/ubuntu/difflet-artifacts/flux-cache-terminal-brake-step29-completion-20260804/semantic-scores.json"
        ),
    )
    result = {}
    for path in paths:
        for row in _load_json(path)["comparisons"]:
            if row["candidate_id"] == "terminal-step-29":
                result[str(row["sample_id"])] = row
    if len(result) != 48:
        raise RuntimeError("terminal@29 VQA matrix must contain 48 samples")
    return result


def _source_samples(source_root: Path, *, scope: str) -> list[dict[str, Any]]:
    quality = _load_json(source_root / "quality-input-v2.json")
    semantic = _load_json(source_root / "semantic-scores.json")
    score_rows = {str(row["sample_id"]): row for row in semantic["comparisons"]}
    terminal_rows = _terminal_vqa_rows()
    samples = []
    for comparison in quality["comparisons"]:
        sample_id = str(comparison["sample_id"])
        score = score_rows[sample_id]
        baseline_vqa = float(score["baseline_scores"]["vqa_score"])
        continue_vqa = float(score["candidate_scores"]["vqa_score"])
        terminal_vqa = float(terminal_rows[sample_id]["candidate_scores"]["vqa_score"])
        continue_failed = baseline_vqa - continue_vqa > 0.25
        if scope == "failures" and not continue_failed:
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "prompt_index": int(comparison["prompt_index"]),
                "prompt": str(comparison["prompt"]),
                "seed": int(comparison["seed"]),
                "trajectory": source_root / comparison["candidate"]["trajectory"],
                "continue_failed": continue_failed,
                "actionable_terminal29": bool(
                    continue_failed and baseline_vqa - terminal_vqa <= 0.25
                ),
                "terminal29_vqa_benefit": terminal_vqa - continue_vqa,
            }
        )
    return samples


def _prepare_runtime(args: argparse.Namespace):
    import numpy as np
    import torch

    from difflet import DiffletParallelConfig, DiffletPipeline
    from difflet.backends.trainium.flux.single_block_residual_probe import (
        NeuronFluxSingleBlockResidualProbeApplication,
    )
    from difflet.models.flux.pipeline import calculate_shift, retrieve_timesteps

    pipe = DiffletPipeline.from_pretrained(
        MODEL_ID,
        model_type="flux",
        revision=MODEL_REVISION,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        load=False,
        skip_compile=True,
    )
    pipe.app.load(
        str(pipe.compiled_path),
        skip_warmup=bool(args.skip_warmup),
        select={"text_encoder", "text_encoder_2"},
    )
    teacher_cache = Path(args.teacher_cache).expanduser().resolve()
    if not (teacher_cache / "model.pt").exists():
        raise FileNotFoundError(f"compiled residual teacher is missing: {teacher_cache}")
    teacher = NeuronFluxSingleBlockResidualProbeApplication(
        model_path=pipe.app.transformer_path,
        config=pipe.app.backbone_config,
    )
    teacher.load(str(teacher_cache), skip_warmup=bool(args.skip_warmup))

    flux_pipeline = pipe.app.pipe
    sigmas = np.linspace(1.0, 1 / 50, 50)
    image_seq_len = 4096
    mu = calculate_shift(
        image_seq_len,
        flux_pipeline.scheduler.config.base_image_seq_len,
        flux_pipeline.scheduler.config.max_image_seq_len,
        flux_pipeline.scheduler.config.base_shift,
        flux_pipeline.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        flux_pipeline.scheduler,
        50,
        torch.device("cpu"),
        sigmas=sigmas,
        mu=mu,
    )
    image_ids = flux_pipeline._prepare_latent_image_ids(
        1, 64, 64, torch.device("cpu"), torch.bfloat16
    )
    guidance = torch.full([1], 3.5, dtype=torch.float32)
    return pipe, teacher, timesteps, image_ids, guidance


def _encode_prompt(flux_pipeline, prompt: str):
    prompt_embeds, pooled, text_ids = flux_pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=torch_device_cpu(),
        num_images_per_prompt=1,
        max_sequence_length=512,
        lora_scale=None,
    )
    return prompt_embeds, pooled, text_ids


def torch_device_cpu():
    import torch

    return torch.device("cpu")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    if args.sample_limit < 0:
        raise ValueError("sample-limit must be nonnegative")
    decision_steps = _decision_steps(args.decision_steps)
    source_root = Path(args.source_root).expanduser().resolve()
    samples = _source_samples(source_root, scope=args.scope)
    if args.sample_limit:
        samples = samples[: args.sample_limit]
    if not samples:
        raise RuntimeError("no samples selected")

    import torch

    pipe, teacher, timesteps, image_ids, guidance = _prepare_runtime(args)
    flux_pipeline = pipe.app.pipe
    step_specs = tuple((step, step - 1, step) for step in decision_steps)
    rows = []
    determinism = None
    started = time.time()
    for sample_index, sample in enumerate(samples):
        trajectory = torch.load(sample["trajectory"], map_location="cpu", weights_only=True)
        if tuple(trajectory.shape) != (50, 1, 4096, 64):
            raise RuntimeError(f"unexpected trajectory shape for {sample['sample_id']}")
        prompt_embeds, pooled, text_ids = _encode_prompt(
            flux_pipeline, sample["prompt"]
        )
        sample_row_start = len(rows)

        def probe(trajectory_index: int, timestep_index: int):
            return _clone_probe_outputs(
                teacher(
                    trajectory[trajectory_index],
                    prompt_embeds,
                    pooled,
                    timesteps[timestep_index : timestep_index + 1] / 1000,
                    image_ids,
                    text_ids,
                    guidance,
                )
            )

        sample_started = time.perf_counter()
        (
            previous_samples,
            previous_moments,
            previous_normalized_samples,
            previous_mlp,
        ) = probe(4, 5)
        anchor_samples, anchor_moments, anchor_normalized_samples, anchor_mlp = probe(6, 7)
        if sample_index == 0:
            (
                repeated_samples,
                repeated_moments,
                repeated_normalized_samples,
                repeated_mlp,
            ) = probe(6, 7)
            determinism = {
                "sample_tensor_equal": bool(torch.equal(anchor_samples, repeated_samples)),
                "moments_tensor_equal": bool(
                    torch.equal(anchor_moments, repeated_moments)
                ),
                "normalized_samples_tensor_equal": bool(
                    torch.equal(
                        anchor_normalized_samples, repeated_normalized_samples
                    )
                ),
                "mlp_tensor_equal": bool(torch.equal(anchor_mlp, repeated_mlp)),
                "sample_relative_rms": _relative_rms(repeated_samples, anchor_samples),
                "moments_relative_rms": _relative_rms(
                    repeated_moments, anchor_moments
                ),
                "normalized_samples_relative_rms": _relative_rms(
                    repeated_normalized_samples, anchor_normalized_samples
                ),
                "mlp_relative_rms": _relative_rms(repeated_mlp, anchor_mlp),
            }
            if not all(
                determinism[key]
                for key in (
                    "sample_tensor_equal",
                    "moments_tensor_equal",
                    "normalized_samples_tensor_equal",
                    "mlp_tensor_equal",
                )
            ):
                raise RuntimeError("residual teacher repeat determinism gate failed")
        for decision_step, trajectory_index, timestep_index in step_specs:
            (
                current_samples,
                current_moments,
                current_normalized_samples,
                current_mlp,
            ) = probe(trajectory_index, timestep_index)
            predicted_samples = _taylor1(
                previous_samples,
                anchor_samples,
                previous_step=5,
                latest_step=7,
                target_step=decision_step,
            )
            predicted_moments = _taylor1(
                previous_moments,
                anchor_moments,
                previous_step=5,
                latest_step=7,
                target_step=decision_step,
            )
            predicted_normalized_samples = _taylor1(
                previous_normalized_samples,
                anchor_normalized_samples,
                previous_step=5,
                latest_step=7,
                target_step=decision_step,
            )
            predicted_mlp = _taylor1(
                previous_mlp,
                anchor_mlp,
                previous_step=5,
                latest_step=7,
                target_step=decision_step,
            )
            rows.append(
                {
                    **{key: value for key, value in sample.items() if key != "trajectory"},
                    "anchor_step": 7,
                    "decision_signal_step": decision_step,
                    "raw_input_sample_drift": _relative_rms(
                        current_samples, anchor_samples
                    ),
                    "raw_input_moments_drift": _relative_rms(
                        current_moments, anchor_moments
                    ),
                    "normalized_input_sample_drift": _relative_rms(
                        current_normalized_samples, anchor_normalized_samples
                    ),
                    "true_mlp_cache_error": _relative_rms(current_mlp, anchor_mlp),
                    "raw_input_sample_prediction_error": _relative_rms(
                        predicted_samples, current_samples
                    ),
                    "raw_input_moments_prediction_error": _relative_rms(
                        predicted_moments, current_moments
                    ),
                    "normalized_input_sample_prediction_error": _relative_rms(
                        predicted_normalized_samples, current_normalized_samples
                    ),
                    "true_mlp_prediction_error": _relative_rms(
                        predicted_mlp, current_mlp
                    ),
                }
            )
        elapsed = time.perf_counter() - sample_started
        sample_summary = " ".join(
            f"s{row['decision_signal_step']}="
            f"{row['raw_input_sample_drift']:.6f}/"
            f"{row['true_mlp_cache_error']:.6f} "
            f"pred={row['raw_input_sample_prediction_error']:.6f}/"
            f"{row['true_mlp_prediction_error']:.6f}"
            for row in rows[sample_row_start:]
        )
        print(
            f"[single-block-real] {sample['sample_id']} elapsed={elapsed:.3f}s "
            f"{sample_summary}",
            flush=True,
        )

    by_step = {}
    for step, _, _ in step_specs:
        step_rows = [row for row in rows if row["decision_signal_step"] == step]
        by_step[str(step)] = {
            "sample_count": len(step_rows),
            "signal_vs_true_mlp_error_spearman": _spearman(
                [float(row["raw_input_sample_drift"]) for row in step_rows],
                [float(row["true_mlp_cache_error"]) for row in step_rows],
            ),
            "moments_vs_true_mlp_error_spearman": _spearman(
                [float(row["raw_input_moments_drift"]) for row in step_rows],
                [float(row["true_mlp_cache_error"]) for row in step_rows],
            ),
            "normalized_samples_vs_true_mlp_error_spearman": _spearman(
                [
                    float(row["normalized_input_sample_drift"])
                    for row in step_rows
                ],
                [float(row["true_mlp_cache_error"]) for row in step_rows],
            ),
            "taylor_aligned": {
                "raw_samples_vs_true_mlp_prediction_error_spearman": _spearman(
                    [
                        float(row["raw_input_sample_prediction_error"])
                        for row in step_rows
                    ],
                    [float(row["true_mlp_prediction_error"]) for row in step_rows],
                ),
                "raw_moments_vs_true_mlp_prediction_error_spearman": _spearman(
                    [
                        float(row["raw_input_moments_prediction_error"])
                        for row in step_rows
                    ],
                    [float(row["true_mlp_prediction_error"]) for row in step_rows],
                ),
                "normalized_samples_vs_true_mlp_prediction_error_spearman": _spearman(
                    [
                        float(row["normalized_input_sample_prediction_error"])
                        for row in step_rows
                    ],
                    [float(row["true_mlp_prediction_error"]) for row in step_rows],
                ),
                "true_mlp_prediction_error_range": [
                    min(float(row["true_mlp_prediction_error"]) for row in step_rows),
                    max(float(row["true_mlp_prediction_error"]) for row in step_rows),
                ],
            },
            "signal_range": [
                min(float(row["raw_input_sample_drift"]) for row in step_rows),
                max(float(row["raw_input_sample_drift"]) for row in step_rows),
            ],
            "teacher_error_range": [
                min(float(row["true_mlp_cache_error"]) for row in step_rows),
                max(float(row["true_mlp_cache_error"]) for row in step_rows),
            ],
        }
        if args.scope == "all":
            labels = [bool(row["continue_failed"]) for row in step_rows]
            online_scores = [
                float(row["raw_input_sample_prediction_error"]) for row in step_rows
            ]
            teacher_scores = [
                float(row["true_mlp_prediction_error"]) for row in step_rows
            ]
            by_step[str(step)]["quality_mapping"] = {
                "label": (
                    "continue-cache VQA harm > 0.25; at step20 all 12 positives are "
                    "known terminal@21 rescues"
                ),
                "positive_count": sum(labels),
                "online_signal_failure_auc": _roc_auc(labels, online_scores),
                "offline_teacher_failure_auc": _roc_auc(labels, teacher_scores),
                "online_signal_full_recall": _full_recall_diagnostic(
                    labels, online_scores
                ),
                "offline_teacher_full_recall": _full_recall_diagnostic(
                    labels, teacher_scores
                ),
                "online_signal_vs_terminal29_benefit_spearman": _spearman(
                    online_scores,
                    [float(row["terminal29_vqa_benefit"]) for row in step_rows],
                ),
            }
    return {
        "schema": "difflet-flux-single-block-real-trajectory",
        "schema_revision": 6,
        "serving_claim": False,
        "opened_data": True,
        "scope": args.scope,
        "decision_steps": list(decision_steps),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source": {
            "root": str(source_root),
            "quality_sha256": sha256_file(source_root / "quality-input-v2.json"),
            "semantic_sha256": sha256_file(source_root / "semantic-scores.json"),
        },
        "teacher": {
            "compiled_path": str(Path(args.teacher_cache).expanduser().resolve()),
            "truncation": "embedding + 19 double blocks + single-block-0 MLP only",
            "full_dit_calls": 0,
            "predictor_alignment": {
                "type": "taylorseer",
                "order": 1,
                "coord": "index",
                "history_steps": [5, 7],
                "relative_error_denominator": "actual tensor L2 norm",
            },
        },
        "signal": {
            "candidates": {
                "raw_input_samples": {
                    "state_shape": [1, 24, 32],
                    "state_bytes_bfloat16": 1536,
                },
                "raw_input_moments": {
                    "state_shape": [1, 24, 32, 2],
                    "state_bytes_bfloat16": 3072,
                },
                "normalized_input_samples": {
                    "state_shape": [1, 24, 32],
                    "state_bytes_bfloat16": 1536,
                },
            },
            "host_transfer_required_online": False,
        },
        "determinism": determinism,
        "analysis": {
            "by_decision_signal_step": by_step,
            "failure_scope_gates": {
                "raw_input_samples": (
                    "Require Spearman >= 0.8 at both step 20 and step 28 before "
                    "collecting controls."
                ),
                "raw_input_moments_followup": (
                    "After the sparse estimator misses its gate, require the same >=0.8 "
                    "at both steps before an AOT stability test and control collection."
                ),
                "normalized_input_samples_followup": (
                    "After raw moments fail, require >=0.8 at both steps, then "
                    "measure its same-step AOT floor before controls."
                ),
            },
        },
        "execution": {
            **_git_identity(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "skip_warmup": bool(args.skip_warmup),
        },
        "rows": rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
        status = 0
    except Exception as exc:
        result = {
            "schema": "difflet-flux-single-block-real-trajectory",
            "schema_revision": 6,
            "serving_claim": False,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-40:],
        }
        status = 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, flush=True)
    if args.out:
        destination = Path(args.out).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(destination, result, add_digest=True)
        print(f"[single-block-real] result={destination}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
