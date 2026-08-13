#!/usr/bin/env python3
"""Compile and run the H1c resident predictor+scheduler+latent experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--black-forest-labs--FLUX.1-dev/snapshots/"
    "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)
ANCHORS = (0, 1, 2, 3, 4, 5, 9, 15, 21, 31, 41, 49)


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
        import torch_neuronx  # noqa: F401
    except (ModuleNotFoundError, OSError):
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = (
                f"{NEURON_VENV / 'bin'}:/opt/aws/neuron/bin:{env.get('PATH', '')}"
            )
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.pipelines.flux.pipeline_flux import (  # noqa: E402
    calculate_shift,
    retrieve_timesteps,
)

from difflet.backends.trainium.flux.resident_cache_step import (  # noqa: E402
    NeuronFluxResidentCacheStepApplication,
)
from difflet.backends.trainium.flux.resident_cache_step_barrier import (  # noqa: E402
    NeuronFluxResidentCacheStepBarrierApplication,
)
from difflet.backends.trainium.flux.resident_cache_step_split import (  # noqa: E402
    NeuronFluxResidentCacheStepSplitApplication,
)
from difflet.models.flux.application import create_flux_config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact_summary(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "model_pt_sha256": _sha256(path / "model.pt"),
    }


def _read_rank0(ranked_output) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(ranked_output, list) or not ranked_output:
        raise TypeError(f"unexpected ranked output: {type(ranked_output)!r}")
    rank0 = ranked_output[0]
    if not isinstance(rank0, list) or len(rank0) != 2:
        raise TypeError(f"unexpected per-rank output: {type(rank0)!r}")
    return rank0[0].cpu(), rank0[1].cpu()


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float() - expected.float()
    absolute = float(difference.abs().max().item())
    relative = float(
        torch.linalg.vector_norm(difference).item()
        / max(torch.linalg.vector_norm(expected.float()).item(), 1e-30)
    )
    return absolute, relative


def _schedule(model_path: Path, steps: int, seq_len: int):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_path), subfolder="scheduler"
    )
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(
        seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, count = retrieve_timesteps(
        scheduler, steps, "cpu", sigmas=sigmas, mu=mu
    )
    if count != steps or len(scheduler.sigmas) != steps + 1:
        raise RuntimeError("unexpected FlowMatch Euler schedule length")
    deltas = scheduler.sigmas[1:] - scheduler.sigmas[:-1]
    return timesteps, scheduler.sigmas, deltas


def _coefficients(step: int, anchor_steps: list[int]) -> torch.Tensor:
    previous, latest = anchor_steps[-2:]
    ratio = (step - latest) / (latest - previous)
    return torch.tensor([-ratio, 1.0 + ratio], dtype=torch.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--compiled-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile-schedule", action="store_true")
    parser.add_argument("--split-precision-boundary", action="store_true")
    parser.add_argument("--optimization-barrier", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-skips", type=int, default=10)
    parser.add_argument("--timed-skips", type=int, default=100)
    return parser.parse_args()


def _make_app(args):
    _, _, config, _ = create_flux_config(
        model_path=str(args.model_path),
        world_size=args.tp_degree,
        backbone_tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
    )
    config.neuron_config.skip_sharding = True
    config.neuron_config.save_sharded_checkpoint = False
    if args.split_precision_boundary and args.optimization_barrier:
        raise ValueError("split precision boundary and optimization barrier are exclusive")
    if args.split_precision_boundary:
        application_cls = NeuronFluxResidentCacheStepSplitApplication
    elif args.optimization_barrier:
        application_cls = NeuronFluxResidentCacheStepBarrierApplication
    else:
        application_cls = NeuronFluxResidentCacheStepApplication
    app = application_cls(
        model_path=str(args.model_path / "transformer"), config=config
    )
    return config, app


def _run_schedule(
    app, initial, noises, deltas, *, read_each_step: bool, split_precision_boundary: bool
):
    request_token = (
        torch.tensor([20260812, 0, 0, 0], dtype=torch.int32)
        if split_precision_boundary
        else torch.tensor([20260812, 0], dtype=torch.int32)
    )
    app.ranked_forward(initial, request_token)
    host_latent = initial.clone()
    fused_reference_latent = initial.clone()
    host_anchors: list[tuple[int, torch.Tensor]] = []
    maximum_absolute_error = 0.0
    maximum_relative_l2_error = 0.0
    fused_maximum_absolute_error = 0.0
    fused_maximum_relative_l2_error = 0.0
    records = []
    for step, (actual_noise, delta) in enumerate(zip(noises, deltas)):
        delta_value = float(delta.item())
        delta_tensor = torch.tensor([delta_value], dtype=torch.float32)
        if step in ANCHORS:
            anchor_delta = (
                torch.tensor([delta_value, 0.0], dtype=torch.float32)
                if split_precision_boundary
                else delta_tensor
            )
            ranked = app.ranked_forward(actual_noise, anchor_delta)
            noise = actual_noise
            fused_noise = actual_noise.float()
            host_anchors.append((step, actual_noise))
            host_anchors = host_anchors[-2:]
            action = "anchor"
        else:
            coefficients = _coefficients(step, [item[0] for item in host_anchors])
            if split_precision_boundary:
                predicted_ranked = app.ranked_forward(coefficients)
                ranked = app.ranked_scheduler(predicted_ranked, delta_tensor)
            else:
                ranked = app.ranked_forward(coefficients, delta_tensor)
            noise = (
                host_anchors[0][1].float() * float(coefficients[0])
                + host_anchors[1][1].float() * float(coefficients[1])
            ).to(torch.bfloat16)
            fused_noise = (
                host_anchors[0][1].float() * float(coefficients[0])
                + host_anchors[1][1].float() * float(coefficients[1])
            )
            action = "skip"
        host_latent = (
            host_latent.float() + delta_value * noise.float()
        ).to(torch.bfloat16)
        fused_reference_latent = (
            fused_reference_latent.float() + delta_value * fused_noise
        ).to(torch.bfloat16)
        if read_each_step:
            actual, _ = _read_rank0(ranked)
            absolute, relative = _errors(actual, host_latent)
            fused_absolute, fused_relative = _errors(actual, fused_reference_latent)
            maximum_absolute_error = max(maximum_absolute_error, absolute)
            maximum_relative_l2_error = max(maximum_relative_l2_error, relative)
            fused_maximum_absolute_error = max(
                fused_maximum_absolute_error, fused_absolute
            )
            fused_maximum_relative_l2_error = max(
                fused_maximum_relative_l2_error, fused_relative
            )
            records.append(
                {
                    "step": step,
                    "action": action,
                    "maximum_absolute_error": absolute,
                    "relative_l2_error": relative,
                    "fused_reference_maximum_absolute_error": fused_absolute,
                    "fused_reference_relative_l2_error": fused_relative,
                }
            )
    final, checksum = _read_rank0(
        app.ranked_forward(torch.tensor([0, 0, 1], dtype=torch.int32))
    )
    final_absolute, final_relative = _errors(final, host_latent)
    fused_final_absolute, fused_final_relative = _errors(final, fused_reference_latent)
    maximum_absolute_error = max(maximum_absolute_error, final_absolute)
    maximum_relative_l2_error = max(maximum_relative_l2_error, final_relative)
    return {
        "records": records,
        "final_maximum_absolute_error": final_absolute,
        "final_relative_l2_error": final_relative,
        "fused_reference_final_maximum_absolute_error": fused_final_absolute,
        "fused_reference_final_relative_l2_error": fused_final_relative,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_l2_error": maximum_relative_l2_error,
        "fused_reference_maximum_absolute_error": fused_maximum_absolute_error,
        "fused_reference_maximum_relative_l2_error": fused_maximum_relative_l2_error,
        "final_checksum": float(checksum.item()),
    }


def _latency(app, shape, generator, args):
    initial = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    anchor_a = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    anchor_b = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    request_token = (
        torch.tensor([19, 0, 0, 0], dtype=torch.int32)
        if args.split_precision_boundary
        else torch.tensor([19, 0], dtype=torch.int32)
    )
    app.ranked_forward(initial, request_token)
    anchor_delta = torch.tensor(
        [-0.01, 0.0] if args.split_precision_boundary else [-0.01],
        dtype=torch.float32,
    )
    app.ranked_forward(anchor_a, anchor_delta)
    app.ranked_forward(anchor_b, anchor_delta)
    coefficients = torch.tensor([-0.5, 1.5], dtype=torch.float32)
    delta = torch.tensor([-0.01], dtype=torch.float32)
    for _ in range(args.warmup_skips):
        if args.split_precision_boundary:
            output = app.ranked_scheduler(app.ranked_forward(coefficients), delta)
        else:
            output = app.ranked_forward(coefficients, delta)
        _ = output[0][1].cpu().item()
    device_ms = []
    for _ in range(args.timed_skips):
        started = time.perf_counter()
        if args.split_precision_boundary:
            output = app.ranked_scheduler(app.ranked_forward(coefficients), delta)
        else:
            output = app.ranked_forward(coefficients, delta)
        _ = output[0][1].cpu().item()
        device_ms.append((time.perf_counter() - started) * 1000.0)

    host_latent = initial
    host_ms = []
    for _ in range(args.timed_skips):
        started = time.perf_counter()
        predicted = (
            anchor_a.float() * -0.5 + anchor_b.float() * 1.5
        ).to(torch.bfloat16)
        host_latent = (host_latent.float() + -0.01 * predicted.float()).to(
            torch.bfloat16
        )
        _ = host_latent.reshape(-1)[0].item()
        host_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "warmup_skips": args.warmup_skips,
        "measured_skips": args.timed_skips,
        "device_p50_ms": statistics.median(device_ms),
        "device_p95_ms": float(torch.quantile(torch.tensor(device_ms), 0.95).item()),
        "host_reference_p50_ms": statistics.median(host_ms),
        "host_reference_p95_ms": float(
            torch.quantile(torch.tensor(host_ms), 0.95).item()
        ),
    }


def main() -> int:
    args = _parse_args()
    if args.steps != 50:
        raise ValueError("the frozen H1c protocol requires exactly 50 steps")
    args.model_path = args.model_path.expanduser().resolve()
    args.compiled_dir = args.compiled_dir.expanduser().resolve()
    args.result = args.result.expanduser().resolve()
    config, app = _make_app(args)
    compile_seconds = None
    if args.compile:
        if args.compiled_dir.exists() and any(args.compiled_dir.iterdir()):
            raise FileExistsError(f"compiled directory is not empty: {args.compiled_dir}")
        args.compiled_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        app.compile(str(args.compiled_dir))
        compile_seconds = time.perf_counter() - started
    if not (args.compiled_dir / "model.pt").is_file():
        raise FileNotFoundError(args.compiled_dir / "model.pt")
    started = time.perf_counter()
    app.load(str(args.compiled_dir), skip_warmup=True)
    load_seconds = time.perf_counter() - started

    seq_len = args.height * args.width // (16 * 16)
    shape = (1, seq_len, int(config.in_channels))
    timesteps, sigmas, deltas = _schedule(args.model_path, args.steps, seq_len)
    generator = torch.Generator().manual_seed(20260812)
    initial = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    noises = [
        torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        for _ in range(args.steps)
    ]
    schedule = _run_schedule(
        app,
        initial,
        noises,
        deltas,
        read_each_step=not args.profile_schedule,
        split_precision_boundary=args.split_precision_boundary,
    )
    details: dict[str, Any] = {
        "mode": "schedule_profile" if args.profile_schedule else "mechanism",
        "execution_contract": (
            "split_predict_bf16_ranked_then_scheduler"
            if args.split_precision_boundary
            else "single_fused_predictor_barrier_scheduler"
            if args.optimization_barrier
            else "single_fused_predictor_scheduler"
        ),
        "anchors": list(ANCHORS),
        "anchor_count": len(ANCHORS),
        "skip_count": args.steps - len(ANCHORS),
        "sigma_first": float(sigmas[0].item()),
        "sigma_last": float(sigmas[-1].item()),
        "timestep_first": float(timesteps[0].item()),
        "timestep_last": float(timesteps[-1].item()),
        "schedule_parity": schedule,
    }
    if not args.profile_schedule:
        details["latency"] = _latency(app, shape, generator, args)
    passed = (
        schedule["maximum_absolute_error"] == 0.0
        and schedule["maximum_relative_l2_error"] == 0.0
    )
    payload = {
        "schema": (
            "difflet-flux-h1f-optimization-barrier-result"
            if args.optimization_barrier
            else "difflet-flux-h1c-resident-cache-step-result"
        ),
        "schema_revision": 1,
        "study_id": (
            "flux-h1c-split-precision-boundary-20260812"
            if args.split_precision_boundary
            else "flux-h1f-optimization-barrier-contract-20260812"
            if args.optimization_barrier
            else "flux-h1c-resident-predictor-scheduler-latent-20260812"
        ),
        "status": (
            "profile_captured"
            if args.profile_schedule
            else "mechanism_passed"
            if passed
            else "mechanism_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "shape": list(shape),
        "tensor_bytes": int(initial.numel() * initial.element_size()),
        "tp_degree": args.tp_degree,
        "compile_seconds": compile_seconds,
        "load_seconds": load_seconds,
        "compiled_artifact": _artifact_summary(args.compiled_dir),
        "details": details,
    }
    _write_json(args.result, payload)
    print(
        f"[h1c] {payload['status']} mode={details['mode']} "
        f"max_abs={schedule['maximum_absolute_error']:.6g}",
        flush=True,
    )
    print(f"[h1c] result={args.result}", flush=True)
    return 0 if payload["status"] != "mechanism_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
