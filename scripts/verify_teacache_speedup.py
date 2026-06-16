#!/usr/bin/env python3
"""Collect or normalize TeaCache speedup evidence.

The default verifier consumes explicit candidate measurements. Hardware T0b/T1
collection is available only behind explicit foreground acknowledgement flags.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from scripts.calibrate_teacache import (
    _build_hv_bundle,
    _dtype_from_name,
    _hv_shape_label,
    _load_hv_bundle,
    _mark_step,
    _require_hv_teacache_mod_input,
)


INPUT_SCHEMA = "nova-m9-teacache-speedup-candidates-v1"
OUTPUT_SCHEMA = "nova-m9-teacache-speedup-curve-v1"
INTEGRATION_SCHEMA = "nova-m9-teacache-integration-v1"
DEFAULT_MIN_SPEEDUP = 1.5
DEFAULT_MIN_TRAJECTORY_COSINE = 0.9999
DEFAULT_MIN_FINAL_COSINE = 0.9995
FOREGROUND_ACK = "I am running TeaCache T0b in the foreground"


def _candidate_passes(
    candidate: dict[str, Any],
    *,
    min_speedup: float,
    min_trajectory_cosine: float,
    min_final_cosine: float,
) -> bool:
    measured_speedup = candidate.get("measured_speedup", candidate.get("target_speedup"))
    trajectory_cosine = candidate.get("trajectory_cosine")
    final_cosine = candidate.get("final_cosine")
    return bool(
        candidate.get("hardware_measured") is True
        and measured_speedup is not None
        and float(measured_speedup) >= min_speedup
        and trajectory_cosine is not None
        and float(trajectory_cosine) >= min_trajectory_cosine
        and final_cosine is not None
        and float(final_cosine) >= min_final_cosine
    )


def verify(
    doc: dict[str, Any],
    *,
    min_speedup: float,
    min_trajectory_cosine: float,
    min_final_cosine: float,
) -> dict[str, Any]:
    if doc.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"unsupported speedup candidate schema: {doc.get('schema')!r}")
    candidates = doc.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate JSON must contain a non-empty candidates list")

    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row["passes_gate"] = _candidate_passes(
            row,
            min_speedup=min_speedup,
            min_trajectory_cosine=min_trajectory_cosine,
            min_final_cosine=min_final_cosine,
        )
        normalized.append(row)

    passing = [row for row in normalized if row["passes_gate"]]
    return {
        "schema": OUTPUT_SCHEMA,
        "model": doc.get("model"),
        "shape_label": doc.get("shape_label"),
        "num_steps": doc.get("num_steps"),
        "thresholds": {
            "min_speedup": min_speedup,
            "min_trajectory_cosine": min_trajectory_cosine,
            "min_final_cosine": min_final_cosine,
        },
        "candidates": normalized,
        "passing_targets": [row.get("target_speedup") for row in passing],
        "can_unlock_t1": bool(passing),
    }


def _load_hv_app(
    args: argparse.Namespace,
    meta: dict[str, Any],
    *,
    teacache_speedup: float | None = None,
    teacache_calibration_path: str | None = None,
):
    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from nova.pipeline.parallel_config import NovaParallelConfig

    return NeuronHunyuanVideoApplication(
        model_path=args.source_dir,
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=_dtype_from_name(args.dtype),
        shape={
            "height": int(meta["height"]),
            "width": int(meta["width"]),
            "num_frames": int(meta["num_frames"]),
        },
        text_seq_len=int(meta["text_seq_len"]),
        enable_vae_decoder=False,
        teacache_speedup=teacache_speedup,
        teacache_calibration_path=teacache_calibration_path,
    )


def _tensor_cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            lhs.detach().float().cpu().reshape(1, -1),
            rhs.detach().float().cpu().reshape(1, -1),
            dim=1,
        ).item()
    )


def _trajectory_cosine(
    lhs: list[torch.Tensor] | None,
    rhs: list[torch.Tensor] | None,
) -> float:
    if lhs is None or rhs is None or len(lhs) != len(rhs):
        return 0.0
    return min(_tensor_cosine(left, right) for left, right in zip(lhs, rhs))


def _run_hv_bundle(
    app,
    *,
    meta: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    num_steps: int,
) -> tuple[Any, float]:
    bundle = _build_hv_bundle(meta, tensors)
    timesteps = tensors["timesteps"][:num_steps]
    start = time.perf_counter()
    output = app.pipeline(
        bundle=bundle,
        timesteps=timesteps,
        output_type="latent",
        return_trajectory=True,
    )
    _mark_step()
    return output, time.perf_counter() - start


def _parse_candidate_calibration(value: str) -> tuple[float, str]:
    target, sep, path = value.partition(":")
    if sep != ":" or not target or not path:
        raise ValueError(
            "--candidate-calibration must use TARGET_SPEEDUP:CALIBRATION_JSON"
        )
    return float(target), path


def _collect_hunyuan_video_candidates(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_hardware:
        raise RuntimeError("--collect-hv-candidates-out requires --allow-hardware")
    if args.foreground_ack != FOREGROUND_ACK:
        raise RuntimeError(f"--foreground-ack must equal {FOREGROUND_ACK!r}")
    if not args.source_dir or not args.compiled_dir:
        raise ValueError("--source-dir and --compiled-dir are required for HV collection")
    if not args.bundle:
        raise ValueError("--bundle is required for HV collection")
    if not args.candidate_calibration:
        raise ValueError("--candidate-calibration is required for HV collection")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    first_meta: dict[str, Any] | None = None
    bundle_records: list[tuple[dict[str, Any], dict[str, torch.Tensor]]] = []
    for bundle_path in args.bundle:
        meta, tensors = _load_hv_bundle(Path(bundle_path))
        first_meta = first_meta or meta
        bundle_records.append((meta, tensors))
    if first_meta is None:
        raise ValueError("no bundles were provided")
    num_steps = int(args.num_inference_steps or first_meta["num_inference_steps"])

    # Load the application ONCE — reloading per candidate triggers c10::Error
    # after ~5 iterations (cclog 75 issue 2). Baseline runs with controller=None;
    # candidates mount a fresh TeaCacheController on the same loaded app.
    from nova.pipeline.teacache import (
        TeaCacheController,
        load_teacache_calibration_or_raise,
    )

    shared_app = _load_hv_app(args, first_meta)
    _require_hv_teacache_mod_input(shared_app)
    shared_app.load(args.compiled_dir, skip_warmup=bool(args.skip_warmup))

    # Baseline: controller=None
    shared_app.pipeline.teacache_controller = None
    shared_app.pipeline.teacache_speedup = None
    baselines = []
    for meta, tensors in bundle_records:
        output, elapsed = _run_hv_bundle(
            shared_app,
            meta=meta,
            tensors=tensors,
            num_steps=num_steps,
        )
        baselines.append((output, elapsed))

    candidates: list[dict[str, Any]] = []
    for spec in args.candidate_calibration:
        target_speedup, calibration_path = _parse_candidate_calibration(spec)
        # Mount a fresh controller on the shared app (no reload).
        shape_label_str = "{}x{}x{}".format(
            int(first_meta["height"]),
            int(first_meta["width"]),
            int(first_meta["num_frames"]),
        )
        calibration = load_teacache_calibration_or_raise(
            calibration_path,
            model="hunyuan_video",
            shape_label=shape_label_str,
        )
        shared_app.pipeline.teacache_controller = TeaCacheController(calibration)
        shared_app.pipeline.teacache_speedup = float(target_speedup)
        elapsed_values: list[float] = []
        trajectory_cosines: list[float] = []
        final_cosines: list[float] = []
        skipped_steps = 0
        full_steps = 0
        for bundle_index, (meta, tensors) in enumerate(bundle_records):
            shared_app.pipeline.teacache_controller.reset()
            output, elapsed = _run_hv_bundle(
                shared_app,
                meta=meta,
                tensors=tensors,
                num_steps=num_steps,
            )
            baseline_output, baseline_elapsed = baselines[bundle_index]
            elapsed_values.append(elapsed)
            trajectory_cosines.append(
                _trajectory_cosine(output.trajectory, baseline_output.trajectory)
            )
            final_cosines.append(_tensor_cosine(output.latents, baseline_output.latents))
            stats = shared_app.pipeline.teacache_controller.stats()
            skipped_steps += int(stats["skipped_steps"])
            full_steps += int(stats["full_steps"])
            print(
                f"[m9-t0b] target={target_speedup} bundle={bundle_index + 1} "
                f"elapsed={elapsed:.4f}s baseline={baseline_elapsed:.4f}s",
                flush=True,
            )
        baseline_elapsed_total = sum(item[1] for item in baselines)
        candidate_elapsed_total = sum(elapsed_values)
        candidates.append(
            {
                "target_speedup": target_speedup,
                "measured_speedup": baseline_elapsed_total / candidate_elapsed_total,
                "baseline_elapsed_s": baseline_elapsed_total,
                "teacache_elapsed_s": candidate_elapsed_total,
                "trajectory_cosine": min(trajectory_cosines),
                "final_cosine": min(final_cosines),
                "calibration": calibration_path,
                "full_steps": full_steps,
                "skipped_steps": skipped_steps,
                "hardware_measured": True,
            }
        )

    return {
        "schema": INPUT_SCHEMA,
        "model": "hunyuan_video",
        "shape_label": _hv_shape_label(first_meta),
        "num_steps": num_steps,
        "started_at": started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates": candidates,
    }


def _write_integration_if_requested(
    *,
    args: argparse.Namespace,
    curve: dict[str, Any],
) -> None:
    if not args.integration_out:
        return
    passing = [row for row in curve["candidates"] if row["passes_gate"]]
    if not passing:
        raise RuntimeError("--integration-out requested but no candidate passed the T1 gate")
    best = max(passing, key=lambda row: float(row["measured_speedup"]))
    doc = {
        "schema": INTEGRATION_SCHEMA,
        "model": curve["model"],
        "shape_label": curve.get("shape_label"),
        "num_steps": curve.get("num_steps"),
        "wallclock_speedup": best["measured_speedup"],
        "trajectory_cosine": best["trajectory_cosine"],
        "final_cosine": best["final_cosine"],
        "default_off": True,
        "hardware_measured": True,
        "source_candidate": best,
    }
    path = Path(args.integration_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-json", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    parser.add_argument(
        "--min-trajectory-cosine",
        type=float,
        default=DEFAULT_MIN_TRAJECTORY_COSINE,
    )
    parser.add_argument("--min-final-cosine", type=float, default=DEFAULT_MIN_FINAL_COSINE)
    parser.add_argument("--collect-hv-candidates-out", default=None)
    parser.add_argument("--integration-out", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--compiled-dir", default=None)
    parser.add_argument("--bundle", action="append", default=[])
    parser.add_argument(
        "--candidate-calibration",
        action="append",
        default=[],
        help="TARGET_SPEEDUP:CALIBRATION_JSON; may be repeated",
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.collect_hv_candidates_out:
        doc = _collect_hunyuan_video_candidates(args)
        collect_out = Path(args.collect_hv_candidates_out)
        collect_out.parent.mkdir(parents=True, exist_ok=True)
        collect_out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    else:
        if args.candidates_json is None:
            raise ValueError(
                "--candidates-json is required unless --collect-hv-candidates-out is used"
            )
        doc = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
    result = verify(
        doc,
        min_speedup=args.min_speedup,
        min_trajectory_cosine=args.min_trajectory_cosine,
        min_final_cosine=args.min_final_cosine,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_integration_if_requested(args=args, curve=result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
