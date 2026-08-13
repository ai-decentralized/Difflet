#!/usr/bin/env python3
"""Collect Gate B labels on a real FLUX cache rollout.

At every naturally skipped step, the experiment computes a shadow full-DiT
output on the identical pre-step cache latent, records only scalar L2
summaries, and advances the original cached prediction.  At every real anchor,
it records the two-anchor counterfactual error available to an online Gram tap.
The shadow calls make this an offline diagnostic; timings are never serving
speed evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet import DiffletParallelConfig, DiffletPipeline  # noqa: E402
from difflet.offline.cache_profile.schedule import scheduler_sigmas  # noqa: E402
from difflet.pipeline.cache.profile import load_phased_candidate  # noqa: E402
from scripts.evaluate_flux_gram_transfer import (  # noqa: E402
    _correlation_summary,
    _pearson,
    _rankdata,
    _sha256_file,
    _spearman,
)
from scripts.flux_cache_protocol import load_prompt_suite  # noqa: E402


MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tensor_error(estimate: Any, actual: Any) -> tuple[float, float, float]:
    import torch

    estimate_working = estimate.detach().float()
    actual_working = actual.detach().float()
    summary = torch.stack(
        (
            torch.linalg.vector_norm(
                (estimate_working - actual_working).reshape(-1), ord=2
            ),
            torch.linalg.vector_norm(actual_working.reshape(-1), ord=2),
        )
    )
    error_norm, reference_norm = summary.detach().cpu().tolist()
    relative = float(error_norm / max(reference_norm, 1e-12))
    if not all(math.isfinite(value) for value in (error_norm, reference_norm, relative)):
        raise RuntimeError("non-finite shadow error measurement")
    return float(error_norm), float(reference_norm), relative


class _ShadowTrace:
    def __init__(self) -> None:
        self.anchors: list[tuple[int, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.shadow_calls = 0

    def __call__(
        self,
        *,
        step_index: int,
        timestep: Any,
        latents: Any,
        predicted: Any,
        used_cache_prediction: bool,
        compute_actual: Any,
    ) -> Any:
        del timestep, latents
        if used_cache_prediction:
            actual = compute_actual()
            absolute, reference, relative = _tensor_error(predicted, actual)
            self.shadow_calls += 1
            self.rows.append(
                {
                    "step": int(step_index),
                    "kind": "skip",
                    "error_norm": absolute,
                    "reference_norm": reference,
                    "relative_error": relative,
                }
            )
            return predicted

        row: dict[str, Any] = {
            "step": int(step_index),
            "kind": "anchor",
            "error_norm": None,
            "reference_norm": None,
            "relative_error": None,
            "source_anchors": None,
        }
        if len(self.anchors) >= 2:
            (a, va), (b, vb) = self.anchors[-2:]
            ratio = (step_index - b) / (b - a)
            # Match TaylorSeer: construct in FP32, quantize the prediction back
            # to the anchor dtype, then measure in FP32.
            estimate = (
                vb.detach().float()
                + (vb.detach().float() - va.detach().float()) * ratio
            ).to(dtype=predicted.dtype)
            absolute, reference, relative = _tensor_error(estimate, predicted)
            row.update(
                {
                    "error_norm": absolute,
                    "reference_norm": reference,
                    "relative_error": relative,
                    "source_anchors": [int(a), int(b)],
                }
            )
        self.rows.append(row)
        self.anchors = (self.anchors + [(int(step_index), predicted.detach())])[-2:]
        return predicted


def _integrated_ratio(rows: Iterable[dict[str, Any]]) -> float:
    values = list(rows)
    numerator = sum(row["delta_sigma_abs"] * row["error_norm"] for row in values)
    denominator = sum(row["delta_sigma_abs"] * row["reference_norm"] for row in values)
    return float(numerator / max(denominator, 1e-12))


def _weighted_relative_mean(rows: Iterable[dict[str, Any]]) -> float:
    values = list(rows)
    numerator = sum(row["delta_sigma_abs"] * row["relative_error"] for row in values)
    denominator = sum(row["delta_sigma_abs"] for row in values)
    return float(numerator / max(denominator, 1e-12))


def _segment_rows(
    samples: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        rows = sample["steps"]
        anchors = [row for row in rows if row["kind"] == "anchor"]
        for index, anchor in enumerate(anchors[:-1]):
            if anchor["relative_error"] is None:
                continue
            next_anchor = anchors[index + 1]
            skipped = [
                row
                for row in rows
                if row["kind"] == "skip"
                and anchor["step"] < row["step"] < next_anchor["step"]
            ]
            if not skipped:
                continue
            result.setdefault(int(anchor["step"]), []).append(
                {
                    "sample_id": sample["sample_id"],
                    "anchor_relative_error": anchor["relative_error"],
                    "anchor_absolute_error": anchor["error_norm"],
                    "next_anchor_step": int(next_anchor["step"]),
                    "skip_count": len(skipped),
                    "next_segment_integrated_ratio": _integrated_ratio(skipped),
                    "next_segment_relative_weighted_mean": _weighted_relative_mean(skipped),
                    "next_segment_absolute_damage": float(
                        sum(row["delta_sigma_abs"] * row["error_norm"] for row in skipped)
                    ),
                }
            )
    return result


def _fixed_segment_summary(
    segments: dict[int, list[dict[str, Any]]],
    *,
    left_key: str,
    right_key: str,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    steps = sorted(segments)
    sample_ids = [row["sample_id"] for row in segments[steps[0]]]
    if any([row["sample_id"] for row in segments[step]] != sample_ids for step in steps):
        raise RuntimeError("fixed segments do not share an identical sample order")

    def statistic(indices: np.ndarray | None = None, *, permute: bool = False) -> float:
        correlations = []
        for step in steps:
            x = np.asarray([row[left_key] for row in segments[step]], dtype=np.float64)
            y = np.asarray([row[right_key] for row in segments[step]], dtype=np.float64)
            if indices is not None:
                x, y = x[indices], y[indices]
            if permute:
                y = rng.permutation(y)
            correlations.append(_spearman(x, y))
        return float(np.mean(correlations))

    observed = statistic()
    bootstrap = []
    for _ in range(replicates):
        indices = rng.integers(0, len(sample_ids), len(sample_ids))
        value = statistic(indices)
        if math.isfinite(value):
            bootstrap.append(value)
    permutation = [statistic(permute=True) for _ in range(replicates)]

    pooled_x, pooled_y = [], []
    for step in steps:
        pooled_x.extend(_rankdata(np.asarray([row[left_key] for row in segments[step]])))
        pooled_y.extend(_rankdata(np.asarray([row[right_key] for row in segments[step]])))
    return {
        "estimand": "equal-weight mean of within-fixed-segment Spearman correlations",
        "segment_count": len(steps),
        "sample_count_per_segment": len(sample_ids),
        "mean_spearman_rho": observed,
        "pooled_within_segment_rank_correlation": _pearson(pooled_x, pooled_y),
        "bootstrap_95pct_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "positive_association_permutation_p_one_sided": float(
            (1 + sum(value >= observed for value in permutation)) / (replicates + 1)
        ),
        "replicates": replicates,
    }


def _analyze(
    samples: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, Any]:
    segments = _segment_rows(samples)
    fixed = {
        str(step): _correlation_summary(
            [row["anchor_relative_error"] for row in rows],
            [row["next_segment_integrated_ratio"] for row in rows],
            rng=rng,
            replicates=replicates,
        )
        for step, rows in sorted(segments.items())
    }
    primary = _fixed_segment_summary(
        segments,
        left_key="anchor_relative_error",
        right_key="next_segment_integrated_ratio",
        rng=rng,
        replicates=replicates,
    )
    raw_sensitivity = _fixed_segment_summary(
        segments,
        left_key="anchor_absolute_error",
        right_key="next_segment_absolute_damage",
        rng=rng,
        replicates=replicates,
    )
    positive = sum(row["spearman_rho"] > 0.0 for row in fixed.values())
    advance = bool(
        primary["mean_spearman_rho"] >= 0.4
        and primary["bootstrap_95pct_ci"][0] > 0.0
        and primary["positive_association_permutation_p_one_sided"] < 0.05
        and positive >= math.ceil(len(fixed) / 2)
    )
    return {
        "primary_fixed_segment": primary,
        "fixed_segment_diagnostics": fixed,
        "raw_absolute_sensitivity": raw_sensitivity,
        "positive_fixed_segment_count": positive,
        "fixed_segment_count": len(fixed),
        "advance_to_conditional_bound_fit": advance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--prompt-suite", required=True, type=Path)
    parser.add_argument("--prompt-split", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--compile-cache-dir", default=None)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.max_prompts is not None and args.max_prompts < 1:
        raise SystemExit("--max-prompts must be positive")

    import torch

    arm = load_phased_candidate(args.candidate)
    if arm.predictor_spec() != {"type": "taylorseer", "order": 1, "coord": "index"}:
        raise SystemExit("Gate B currently requires order-1 index TaylorSeer")
    selection = load_prompt_suite(args.prompt_suite.resolve(), args.prompt_split)
    prompt_rows = selection.descriptor["prompts"]
    if args.max_prompts is not None:
        prompt_rows = prompt_rows[: args.max_prompts]
    output_root = args.out_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    pipe = DiffletPipeline.from_pretrained(
        MODEL_ID,
        model_type="flux",
        revision=MODEL_REVISION,
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        compile_cache_dir=args.compile_cache_dir,
        height=args.height,
        width=args.width,
        skip_warmup=args.skip_warmup,
    )
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False
    flux_pipeline._tc_output_dynamics_record = False
    flux_pipeline.teacache_controller = None
    generation = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "num_steps": args.num_steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "dtype": "bfloat16",
        "tp_degree": args.tp_degree,
        "scheduler_class": type(flux_pipeline.scheduler).__name__,
        "scheduler_config": dict(flux_pipeline.scheduler.config),
    }
    sigmas = np.asarray(scheduler_sigmas(generation), dtype=np.float64)
    delta_sigma = np.diff(sigmas)

    samples: list[dict[str, Any]] = []
    started = time.time()
    for index, prompt_row in enumerate(prompt_rows):
        trace = _ShadowTrace()
        flux_pipeline._cache_counterfactual_hook = trace
        session = arm.build_session(args.num_steps)
        sample_started = time.perf_counter()
        pipe(
            prompt=prompt_row["text"],
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            output_type="latent",
            generator=torch.Generator().manual_seed(args.seed),
            cache_session=session,
        )
        elapsed = time.perf_counter() - sample_started
        if len(trace.rows) != args.num_steps:
            raise RuntimeError(f"incomplete trace for prompt {index}: {len(trace.rows)} rows")
        for row in trace.rows:
            row["delta_sigma_abs"] = abs(float(delta_sigma[row["step"]]))
        stats = session.statistics()
        if trace.shadow_calls != stats["skipped_steps"]:
            raise RuntimeError("shadow calls do not equal naturally skipped steps")
        sample = {
            "sample_id": f"p{index:03d}-s{args.seed}",
            "prompt_id": prompt_row["prompt_id"],
            "category": prompt_row["category"],
            "seed": args.seed,
            "elapsed_s": elapsed,
            "runner_stats": stats,
            "shadow_full_dit_calls": trace.shadow_calls,
            "steps": trace.rows,
        }
        samples.append(sample)
        _write_json(output_root / "partial-trace.json", {"samples": samples})
        print(
            f"[gram-gate-b] {sample['sample_id']} {elapsed:.3f}s "
            f"anchors={stats['full_steps']} shadows={trace.shadow_calls}",
            flush=True,
        )
    delattr(flux_pipeline, "_cache_counterfactual_hook")

    analysis = _analyze(
        samples,
        rng=np.random.default_rng(20260811),
        replicates=args.bootstrap_replicates,
    )
    result = {
        "schema": "difflet-flux-gram-transfer-cache-rollout-gate-result",
        "schema_revision": 1,
        "study_id": "flux-gram-transfer-cache-rollout-gate-20260811",
        "status": (
            "advance_to_conditional_bound_fit"
            if analysis["advance_to_conditional_bound_fit"]
            else "cache_rollout_gate_rejected"
        ),
        "serving_claim": False,
        "timing_claim": False,
        "evidence_scope": "real_cache_rollout_with_offline_shadow_full_dit_labels",
        "sources": {
            "candidate": {
                "path": str(args.candidate.resolve()),
                "file_sha256": _sha256_file(args.candidate),
                "candidate_id": arm.candidate_id,
            },
            "prompt_suite": {
                "path": str(args.prompt_suite.resolve()),
                "file_sha256": _sha256_file(args.prompt_suite),
                "split": args.prompt_split,
                "descriptor_sha256": selection.descriptor["sha256"],
            },
            "compiled_path": str(pipe.compiled_path),
            "compiled_manifest_sha256": _sha256_file(pipe.compiled_path / "manifest.json"),
        },
        "generation_identity": generation,
        "sample_count": len(samples),
        "total_elapsed_s": time.time() - started,
        "measurement": {
            "anchor_signal": "relative L2 of two-anchor counterfactual versus actual anchor",
            "next_segment_label": "sum |delta_sigma| error_norm / sum |delta_sigma| shadow_actual_norm",
            "shadow_semantics": "compute actual on identical pre-step cache latent; advance original prediction",
            "extra_transformer_calls": sum(row["shadow_full_dit_calls"] for row in samples),
        },
        "frozen_gate": {
            "minimum_mean_fixed_segment_spearman_rho": 0.4,
            "bootstrap_95pct_lower_bound_strictly_positive": True,
            "maximum_one_sided_permutation_p": 0.05,
            "minimum_positive_fixed_segment_fraction": 0.5,
            "interpretation": "pass only authorizes a separately trained and held-out conditional upper-bound experiment",
        },
        "analysis": analysis,
        "samples": samples,
        "decision": {
            "quality_budget_claim_supported": False,
            "reason": (
                "Correlation is not a calibrated conditional upper bound; a disjoint fit/holdout study is still required."
            ),
        },
    }
    result_path = output_root / "gate-b-result.json"
    _write_json(result_path, result)
    print(
        f"[gram-gate-b] {result['status']}: "
        f"mean fixed-segment rho={analysis['primary_fixed_segment']['mean_spearman_rho']:.4f}, "
        f"95% CI={analysis['primary_fixed_segment']['bootstrap_95pct_ci']}",
        flush=True,
    )
    print(f"[gram-gate-b] result={result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
