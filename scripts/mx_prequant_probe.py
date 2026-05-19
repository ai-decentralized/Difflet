#!/usr/bin/env python3
"""Single-op probe for Trainium ``linear_mx_prequant``.

This isolates the path used by the LTX-2 all-E4M3 experiment:

* host prequantizes the weight into MX ``weight_mx`` / ``weight_scale``;
* the Trainium kernel quantizes activations at runtime;
* ``nc_matmul_mx`` accumulates K tiles into one 512-wide output tile.

The script also runs ``linear_mx`` as a control, because that path quantizes
both activation and weight through the older public API.  For N > 512 it also
runs the Tier-1 hoisted-activation path, which quantizes activation once and
reuses it across all 512-wide N tiles.
"""

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


def _build_inputs(seed: int, k_dim: int, n_dim: int, scale: float, dtype: torch.dtype):
    generator = torch.Generator().manual_seed(seed)
    input_bf16 = (scale * torch.randn((128, k_dim), generator=generator)).to(dtype)
    weight_k_n = (scale * torch.randn((k_dim, n_dim), generator=generator)).to(dtype)
    bias = (scale * torch.randn((n_dim,), generator=generator)).to(dtype)
    return input_bf16, weight_k_n, bias


def _prequantize_weight(weight_k_n: torch.Tensor, mx_dtype: str):
    n_tiles = []
    n_scales = []
    for n_start in range(0, weight_k_n.shape[1], 512):
        moving_tiles = []
        moving_scales = []
        for k_start in range(0, weight_k_n.shape[0], 512):
            weight_tile = weight_k_n[k_start : k_start + 512, n_start : n_start + 512]
            weight_tile = weight_tile.contiguous()
            moving_native = (
                weight_tile.reshape(128, 4, 512)
                .permute(0, 2, 1)
                .reshape(128, 512 * 4)
                .contiguous()
            )
            moving_mx, moving_scale = cpu_mx.quantize_mx(moving_native, dtype=mx_dtype)
            moving_tiles.append(moving_mx.view(torch.int32))
            moving_scales.append(moving_scale)
        n_tiles.append(torch.stack(moving_tiles, dim=0))
        n_scales.append(torch.stack(moving_scales, dim=0))
    return torch.stack(n_tiles, dim=0), torch.stack(n_scales, dim=0)


def _diff_metrics(expected: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    expected = expected.to(torch.bfloat16).float()
    observed = observed.to(torch.bfloat16).float()
    diff = (expected - observed).abs()
    return {
        "cosine": _cosine(expected, observed),
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
    }


def _run_trainium(
    input_bf16: torch.Tensor,
    weight_k_n: torch.Tensor,
    weight_mx: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    mx_dtype: str,
):
    import torch_xla.core.xla_model as xm

    from nova.backends.trainium.ops_impl import mx as trainium_mx

    device = xm.xla_device()

    start = time.perf_counter()
    input_device = input_bf16.to(device)
    weight_device = weight_k_n.to(device)
    weight_mx_device = weight_mx.to(device)
    weight_scale_device = weight_scale.to(device)
    bias_device = bias.to(device) if bias is not None else None
    xm.mark_step()
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    cached_out = trainium_mx.linear_mx_prequant_cached_activation(
        input_device,
        weight_mx_device,
        weight_scale_device,
        bias_device,
        dtype=mx_dtype,
    )
    cached_cpu = cached_out.cpu()
    xm.mark_step()
    cached_time = time.perf_counter() - start

    start = time.perf_counter()
    prequant_outputs = []
    for n_idx in range(weight_mx.shape[0]):
        n_start = n_idx * 512
        prequant_outputs.append(
            trainium_mx.linear_mx_prequant(
                input_device,
                weight_mx_device[n_idx],
                weight_scale_device[n_idx],
                bias_device[n_start : n_start + 512].contiguous()
                if bias_device is not None
                else None,
                dtype=mx_dtype,
            )
        )
    prequant_out = torch.cat(prequant_outputs, dim=1)
    prequant_cpu = prequant_out.cpu()
    xm.mark_step()
    prequant_time = time.perf_counter() - start

    start = time.perf_counter()
    public_out = trainium_mx.linear_mx(
        input_device,
        weight_device,
        bias_device,
        dtype=mx_dtype,
    )
    public_cpu = public_out.cpu()
    xm.mark_step()
    public_time = time.perf_counter() - start

    return cached_cpu, prequant_cpu, public_cpu, {
        "load_time_s": load_time,
        "cached_forward_time_s": cached_time,
        "prequant_forward_time_s": prequant_time,
        "public_forward_time_s": public_time,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-dim", type=int, default=2048)
    parser.add_argument("--n-dim", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument(
        "--mx-dtype",
        choices=("float8_e4m3fn_x4", "float8_e5m2_x4"),
        default="float8_e4m3fn_x4",
    )
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs", type=float, default=0.01)
    parser.add_argument("--max-max-abs", type=float, default=0.05)
    args = parser.parse_args()

    if args.k_dim % 512 != 0:
        parser.error("--k-dim must be a multiple of 512")
    if args.n_dim % 512 != 0:
        parser.error("--n-dim must be a multiple of 512")

    input_bf16, weight_k_n, bias = _build_inputs(
        args.seed,
        args.k_dim,
        args.n_dim,
        args.scale,
        torch.bfloat16,
    )
    weight_mx, weight_scale = _prequantize_weight(weight_k_n, args.mx_dtype)

    cpu_start = time.perf_counter()
    expected = cpu_mx.linear_mx_outer_n_reference(
        input_bf16,
        weight_k_n,
        bias,
        dtype=args.mx_dtype,
        out_dtype=torch.float32,
    )
    cpu_time = time.perf_counter() - cpu_start

    cached_out, prequant_out, public_out, timings = _run_trainium(
        input_bf16,
        weight_k_n,
        weight_mx,
        weight_scale,
        bias,
        args.mx_dtype,
    )

    metrics = {
        "seed": args.seed,
        "k_dim": args.k_dim,
        "n_dim": args.n_dim,
        "scale": args.scale,
        "mx_dtype": args.mx_dtype,
        "cpu_reference_time_s": cpu_time,
        **timings,
        "cached_vs_cpu": _diff_metrics(expected, cached_out),
        "prequant_vs_cpu": _diff_metrics(expected, prequant_out),
        "public_vs_cpu": _diff_metrics(expected, public_out),
        "cached_vs_prequant": _diff_metrics(prequant_out, cached_out),
        "prequant_vs_public": _diff_metrics(public_out, prequant_out),
        "passed": False,
    }
    metrics["passed"] = all(
        section["cosine"] >= args.min_cosine
        and section["mean_abs"] <= args.max_mean_abs
        and section["max_abs"] <= args.max_max_abs
        for section in (
            metrics["cached_vs_cpu"],
            metrics["prequant_vs_cpu"],
            metrics["public_vs_cpu"],
            metrics["cached_vs_prequant"],
            metrics["prequant_vs_public"],
        )
    )

    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
