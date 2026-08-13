#!/usr/bin/env python3
"""Compile and run the H1g BF16 XLA While/scan segment experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
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
MAX_SEGMENT_STEPS = 9


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

from difflet.backends.trainium.flux.resident_cache_step_scan import (  # noqa: E402
    NeuronFluxResidentCacheStepScanApplication,
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
    return timesteps, scheduler.sigmas, scheduler.sigmas[1:] - scheduler.sigmas[:-1]


def _coefficients(step: int, anchor_steps: list[int]) -> torch.Tensor:
    previous, latest = anchor_steps[-2:]
    ratio = (step - latest) / (latest - previous)
    return torch.tensor([-ratio, 1.0 + ratio], dtype=torch.float32)


def _packet(
    steps: list[int], anchor_steps: list[int], deltas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = torch.zeros((MAX_SEGMENT_STEPS, 2), dtype=torch.float32)
    delta_packet = torch.zeros((MAX_SEGMENT_STEPS, 1), dtype=torch.float32)
    for row, step in enumerate(steps):
        coefficients[row] = _coefficients(step, anchor_steps)
        delta_packet[row, 0] = float(deltas[step].item())
    return coefficients, delta_packet


def _host_skip(
    latent: torch.Tensor,
    anchor_noises: list[torch.Tensor],
    coefficients: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    prediction = (
        anchor_noises[0].float() * float(coefficients[0])
        + anchor_noises[1].float() * float(coefficients[1])
    ).to(torch.bfloat16)
    return (latent.float() + delta * prediction.float()).to(torch.bfloat16)


def _update_error(
    records: list[dict[str, Any]],
    *,
    label: str,
    step: int,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    absolute, relative = _errors(actual, expected)
    records.append(
        {
            "label": label,
            "step": step,
            "maximum_absolute_error": absolute,
            "relative_l2_error": relative,
        }
    )
    return absolute, relative


def _run_single_step_audit(app, initial, noises, deltas):
    app.ranked_forward(initial, torch.tensor([20260812, 1], dtype=torch.int32))
    host_latent = initial.clone()
    anchor_steps: list[int] = []
    anchor_noises: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for step, actual_noise in enumerate(noises):
        delta = float(deltas[step].item())
        if step in ANCHORS:
            host_latent = (
                host_latent.float() + delta * actual_noise.float()
            ).to(torch.bfloat16)
            ranked = app.ranked_forward(
                actual_noise, torch.tensor([delta], dtype=torch.float32)
            )
            anchor_steps.append(step)
            anchor_noises.append(actual_noise)
            anchor_steps = anchor_steps[-2:]
            anchor_noises = anchor_noises[-2:]
            label = "anchor"
        else:
            coefficients, delta_packet = _packet([step], anchor_steps, deltas)
            host_latent = _host_skip(
                host_latent, anchor_noises, coefficients[0], delta
            )
            ranked = app.ranked_forward(coefficients, delta_packet)
            label = "single_active_scan"
        actual, _ = _read_rank0(ranked)
        absolute, relative = _update_error(
            records,
            label=label,
            step=step,
            actual=actual,
            expected=host_latent,
        )
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    final, checksum = _read_rank0(
        app.ranked_forward(torch.tensor([0, 0, 1], dtype=torch.int32))
    )
    final_absolute, final_relative = _errors(final, host_latent)
    return {
        "records": records,
        "maximum_absolute_error": max(maximum_absolute, final_absolute),
        "maximum_relative_l2_error": max(maximum_relative, final_relative),
        "final_maximum_absolute_error": final_absolute,
        "final_relative_l2_error": final_relative,
        "final_checksum": float(checksum.item()),
        "cache_entry_points_per_rank": 52,
        "scan_invocations": 38,
    }


def _run_segment_audit(app, initial, noises, deltas):
    app.ranked_forward(initial, torch.tensor([20260812, 2], dtype=torch.int32))
    host_latent = initial.clone()
    anchor_steps: list[int] = []
    anchor_noises: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    segment_lengths: list[int] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    step = 0
    while step < len(noises):
        if step in ANCHORS:
            delta = float(deltas[step].item())
            host_latent = (
                host_latent.float() + delta * noises[step].float()
            ).to(torch.bfloat16)
            ranked = app.ranked_forward(
                noises[step], torch.tensor([delta], dtype=torch.float32)
            )
            anchor_steps.append(step)
            anchor_noises.append(noises[step])
            anchor_steps = anchor_steps[-2:]
            anchor_noises = anchor_noises[-2:]
            label = "anchor"
            endpoint = step
            step += 1
        else:
            segment = []
            while step < len(noises) and step not in ANCHORS:
                segment.append(step)
                step += 1
            coefficients, delta_packet = _packet(segment, anchor_steps, deltas)
            for row, skip_step in enumerate(segment):
                host_latent = _host_skip(
                    host_latent,
                    anchor_noises,
                    coefficients[row],
                    float(deltas[skip_step].item()),
                )
            ranked = app.ranked_forward(coefficients, delta_packet)
            segment_lengths.append(len(segment))
            label = "multi_active_scan_endpoint"
            endpoint = segment[-1]
        actual, _ = _read_rank0(ranked)
        absolute, relative = _update_error(
            records,
            label=label,
            step=endpoint,
            actual=actual,
            expected=host_latent,
        )
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    final, checksum = _read_rank0(
        app.ranked_forward(torch.tensor([0, 0, 1], dtype=torch.int32))
    )
    final_absolute, final_relative = _errors(final, host_latent)
    return {
        "records": records,
        "segment_lengths": segment_lengths,
        "maximum_absolute_error": max(maximum_absolute, final_absolute),
        "maximum_relative_l2_error": max(maximum_relative, final_relative),
        "final_maximum_absolute_error": final_absolute,
        "final_relative_l2_error": final_relative,
        "final_checksum": float(checksum.item()),
        "cache_entry_points_per_rank": 20,
        "projected_full_resident_entry_points_per_rank": 33,
        "scan_invocations": len(segment_lengths),
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
    app.ranked_forward(initial, torch.tensor([19, 0], dtype=torch.int32))
    app.ranked_forward(anchor_a, torch.tensor([-0.01], dtype=torch.float32))
    app.ranked_forward(anchor_b, torch.tensor([-0.01], dtype=torch.float32))
    coefficients = torch.stack(
        [torch.tensor([-float(i), 1.0 + float(i)]) for i in range(1, 10)]
    )
    deltas = torch.full((MAX_SEGMENT_STEPS, 1), -0.01, dtype=torch.float32)
    for _ in range(args.warmup_segments):
        output = app.ranked_forward(coefficients, deltas)
        _ = output[0][1].cpu().reshape(-1)[-1].item()
    device_ms = []
    for _ in range(args.timed_segments):
        started = time.perf_counter()
        output = app.ranked_forward(coefficients, deltas)
        _ = output[0][1].cpu().reshape(-1)[-1].item()
        device_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "segment_steps": MAX_SEGMENT_STEPS,
        "warmup_segments": args.warmup_segments,
        "measured_segments": args.timed_segments,
        "device_segment_p50_ms": statistics.median(device_ms),
        "device_segment_p95_ms": float(
            torch.quantile(torch.tensor(device_ms), 0.95).item()
        ),
        "device_effective_step_p50_ms": statistics.median(device_ms)
        / MAX_SEGMENT_STEPS,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--compiled-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup-segments", type=int, default=10)
    parser.add_argument("--timed-segments", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.steps != 50:
        raise ValueError("the frozen H1g protocol requires exactly 50 steps")
    args.model_path = args.model_path.expanduser().resolve()
    args.compiled_dir = args.compiled_dir.expanduser().resolve()
    args.result = args.result.expanduser().resolve()
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
    app = NeuronFluxResidentCacheStepScanApplication(
        model_path=str(args.model_path / "transformer"), config=config
    )
    compile_seconds = None
    if args.compile:
        if args.compiled_dir.exists() and any(args.compiled_dir.iterdir()):
            raise FileExistsError(f"compiled directory is not empty: {args.compiled_dir}")
        args.compiled_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            # Neuron's INFO-level HLO naming TorchDispatchMode does not implement
            # HigherOrderOperator dispatch.  Disabling naming metadata leaves the
            # executable/optimization contract unchanged and allows XLA While to
            # reach the backend compiler.
            app.compile(str(args.compiled_dir), debug="none")
            compile_seconds = time.perf_counter() - started
        except Exception as error:
            compile_seconds = time.perf_counter() - started
            failure = {
                "schema": "difflet-flux-h1g-scan-segment-result",
                "schema_revision": 1,
                "study_id": "flux-h1g-bf16-while-carry-segment-20260812",
                "status": "compile_failed",
                "serving_claim": False,
                "architecture_speed_claim": False,
                "shape": [1, args.height * args.width // (16 * 16), int(config.in_channels)],
                "tp_degree": args.tp_degree,
                "compile_seconds": compile_seconds,
                "details": {
                    "execution_contract": "bf16_xla_while_carry_fixed_9_step_scan",
                    "hlo_metadata_level": "none",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "gate": "G1_compile_while",
                },
            }
            _write_json(args.result, failure)
            print(
                f"[h1g] compile_failed type={type(error).__name__} "
                f"result={args.result}",
                flush=True,
            )
            return 1
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
    single_step = _run_single_step_audit(app, initial, noises, deltas)
    segment = _run_segment_audit(app, initial, noises, deltas)
    latency = _latency(app, shape, generator, args)
    exact = all(
        value == 0.0
        for value in (
            single_step["maximum_absolute_error"],
            single_step["maximum_relative_l2_error"],
            segment["maximum_absolute_error"],
            segment["maximum_relative_l2_error"],
        )
    )
    payload = {
        "schema": "difflet-flux-h1g-scan-segment-result",
        "schema_revision": 1,
        "study_id": "flux-h1g-bf16-while-carry-segment-20260812",
        "status": "mechanism_passed" if exact else "mechanism_failed",
        "serving_claim": False,
        "architecture_speed_claim": False,
        "shape": list(shape),
        "tensor_bytes": int(initial.numel() * initial.element_size()),
        "tp_degree": args.tp_degree,
        "compile_seconds": compile_seconds,
        "load_seconds": load_seconds,
        "compiled_artifact": _artifact_summary(args.compiled_dir),
        "details": {
            "execution_contract": "bf16_xla_while_carry_fixed_9_step_scan",
            "anchors": list(ANCHORS),
            "anchor_count": len(ANCHORS),
            "skip_count": args.steps - len(ANCHORS),
            "sigma_first": float(sigmas[0].item()),
            "sigma_last": float(sigmas[-1].item()),
            "timestep_first": float(timesteps[0].item()),
            "timestep_last": float(timesteps[-1].item()),
            "single_step_audit": single_step,
            "segment_audit": segment,
            "latency": latency,
        },
    }
    _write_json(args.result, payload)
    print(
        f"[h1g] {payload['status']} single_abs="
        f"{single_step['maximum_absolute_error']:.6g} "
        f"segment_abs={segment['maximum_absolute_error']:.6g}",
        flush=True,
    )
    print(f"[h1g] result={args.result}", flush=True)
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
