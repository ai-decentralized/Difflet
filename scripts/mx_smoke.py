#!/usr/bin/env python3
"""Smoke test for Nova MX kernels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nova.backends.cpu.ops_impl import mx as cpu_mx


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1),
        b.float().reshape(1, -1),
    ).item()


def _build_inputs(seed: int, k_tiles: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    if k_tiles == 1:
        stationary_shape = (128, 512)
        moving_shape = (128, 2048)
    else:
        stationary_shape = (k_tiles, 128, 512)
        moving_shape = (k_tiles, 128, 2048)
    stationary = (0.1 * torch.randn(stationary_shape, generator=generator)).to(
        torch.bfloat16
    )
    moving = (0.1 * torch.randn(moving_shape, generator=generator)).to(torch.bfloat16)
    return stationary, moving


def _cpu_reference(
    stationary: torch.Tensor,
    moving: torch.Tensor,
    mx_dtype: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    stationary_mx, stationary_scale = cpu_mx.quantize_mx(stationary, dtype=mx_dtype)
    moving_mx, moving_scale = cpu_mx.quantize_mx(moving, dtype=mx_dtype)
    if stationary_mx.ndim == 3:
        out = cpu_mx.matmul_mx_k_tiles_reference(
            stationary_mx,
            stationary_scale,
            moving_mx,
            moving_scale,
            dtype=mx_dtype,
            out_dtype=torch.float32,
        )
    else:
        out = cpu_mx.matmul_mx_single_tile_reference(
            stationary_mx,
            stationary_scale,
            moving_mx,
            moving_scale,
            dtype=mx_dtype,
            out_dtype=torch.float32,
        )
    return out, (stationary_mx, stationary_scale, moving_mx, moving_scale)


def _run_simulator(
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    mx_dtype: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    from nki.simulator import simulate_kernel

    from nova.backends.trainium.nki_kernels.mx import (
        matmul_mx_k_tiles_kernel,
        matmul_mx_single_tile_kernel,
    )

    stationary_mx, stationary_scale, moving_mx, moving_scale = packed
    kernel = (
        matmul_mx_k_tiles_kernel
        if stationary_mx.ndim == 3
        else matmul_mx_single_tile_kernel
    )
    kernel_args = (
        stationary_mx.numpy(),
        stationary_scale.numpy(),
        moving_mx.numpy(),
        moving_scale.numpy(),
    )
    if stationary_mx.ndim == 3:
        kernel_args = (*kernel_args, int(stationary_mx.shape[0]))
    start = time.perf_counter()
    out = simulate_kernel(
        kernel,
        kernel_args,
        {"mx_dtype": mx_dtype},
    )
    elapsed = time.perf_counter() - start
    return torch.from_numpy(out).float(), {"simulator_time_s": elapsed}


def _run_trainium(
    stationary: torch.Tensor,
    moving: torch.Tensor,
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    mx_dtype: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    import torch_xla.core.xla_model as xm

    from nova.backends.trainium.ops_impl import mx as trainium_mx

    device = xm.xla_device()

    start = time.perf_counter()
    if stationary.ndim == 2:
        stationary_xla = stationary.to(device)
        moving_xla = moving.to(device)
    else:
        stationary_mx, stationary_scale, moving_mx, moving_scale = packed
        stationary_mx = stationary_mx.to(device)
        stationary_scale = stationary_scale.to(device)
        moving_mx = moving_mx.to(device)
        moving_scale = moving_scale.to(device)
    xm.mark_step()
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    if stationary.ndim == 2:
        stationary_mx, stationary_scale = trainium_mx.quantize_mx(
            stationary_xla, dtype=mx_dtype
        )
        moving_mx, moving_scale = trainium_mx.quantize_mx(moving_xla, dtype=mx_dtype)
    out = trainium_mx.matmul_mx(
        stationary_mx,
        stationary_scale,
        moving_mx,
        moving_scale,
        dtype=mx_dtype,
        out_dtype=torch.bfloat16,
    )
    xm.mark_step()
    forward_time = time.perf_counter() - start

    start = time.perf_counter()
    out_cpu = out.cpu().float()
    xm.mark_step()
    transfer_time = time.perf_counter() - start

    return out_cpu, {
        "load_time_s": load_time,
        "forward_time_s": forward_time,
        "transfer_time_s": transfer_time,
    }


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("simulator", "trainium"),
        default="simulator",
        help="execution backend for the MX kernel",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--k-tiles",
        type=int,
        default=1,
        help="number of leading K tiles to accumulate",
    )
    parser.add_argument(
        "--mx-dtype",
        choices=("float8_e4m3fn_x4", "float8_e5m2_x4"),
        default="float8_e4m3fn_x4",
        help="packed MXFP8 format for quantize + matmul",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("/tmp/nova_mx_smoke_metrics.json"),
    )
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs", type=float, default=0.01)
    parser.add_argument("--max-max-abs", type=float, default=0.05)
    args = parser.parse_args()

    if args.k_tiles < 1:
        parser.error("--k-tiles must be >= 1")

    stationary, moving = _build_inputs(args.seed, args.k_tiles)

    cpu_start = time.perf_counter()
    cpu_out, packed = _cpu_reference(stationary, moving, args.mx_dtype)
    cpu_time = time.perf_counter() - cpu_start

    if args.mode == "trainium":
        actual_out, timings = _run_trainium(stationary, moving, packed, args.mx_dtype)
    else:
        actual_out, timings = _run_simulator(packed, args.mx_dtype)

    expected_out = cpu_out.to(torch.bfloat16).float()
    observed_out = actual_out.to(torch.bfloat16).float()
    diff = (expected_out - observed_out).abs()
    metrics = {
        "mode": args.mode,
        "seed": args.seed,
        "k_tiles": args.k_tiles,
        "mx_dtype": args.mx_dtype,
        "shape": {
            "stationary_native": list(stationary.shape),
            "moving_native": list(moving.shape),
            "output": list(cpu_out.shape),
        },
        "cpu_reference_time_s": cpu_time,
        **timings,
        "cosine": _cosine(expected_out, observed_out),
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "passed": False,
    }
    metrics["passed"] = (
        metrics["cosine"] >= args.min_cosine
        and metrics["mean_abs"] <= args.max_mean_abs
        and metrics["max_abs"] <= args.max_max_abs
    )

    _write_metrics(args.metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
