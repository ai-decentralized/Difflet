#!/usr/bin/env python3
"""HV-1.5 block-0 to_q real-weight MX parity probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import safe_open

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hunyuan15_attention_boundary_probe import _prepare_frontend  # noqa: E402
from hunyuan15_transformer_parity import _load_bundle_inputs, _load_config  # noqa: E402
from nova.backends.cpu.ops_impl import mx as cpu_mx  # noqa: E402


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1),
        b.float().reshape(1, -1),
    ).item()


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _load_to_q_shard(
    model_dir: Path,
    *,
    rank: int,
    tp_degree: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    transformer_path = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    weight_key = "transformer_blocks.0.attn.to_q.weight"
    bias_key = "transformer_blocks.0.attn.to_q.bias"
    with safe_open(transformer_path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = sorted({weight_key, bias_key} - available)
        if missing:
            raise KeyError(f"{transformer_path} is missing keys: {', '.join(missing)}")
        weight = handle.get_tensor(weight_key)
        bias = handle.get_tensor(bias_key)

    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError(
            f"unexpected to_q shapes: weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
        )
    if weight.shape[0] != bias.shape[0]:
        raise ValueError(
            f"to_q weight/bias output mismatch: {weight.shape[0]} != {bias.shape[0]}"
        )
    if weight.shape[0] % tp_degree != 0:
        raise ValueError(f"to_q output dim {weight.shape[0]} is not divisible by tp={tp_degree}")
    if rank < 0 or rank >= tp_degree:
        raise ValueError(f"--rank must be in [0, {tp_degree}), got {rank}")

    shard = weight.shape[0] // tp_degree
    start = rank * shard
    end = start + shard
    weight_slice = weight[start:end, :].to(dtype=dtype).contiguous()
    bias_slice = bias[start:end].to(dtype=dtype).contiguous()
    weight_k_n = weight_slice.T.contiguous()
    meta = {
        "path": str(transformer_path),
        "weight_key": weight_key,
        "bias_key": bias_key,
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "bias_shape": list(bias.shape),
        "bias_dtype": str(bias.dtype),
        "rank": rank,
        "tp_degree": tp_degree,
        "weight_slice_shape": list(weight_slice.shape),
        "weight_k_n_shape": list(weight_k_n.shape),
        "bias_slice_shape": list(bias_slice.shape),
    }
    return weight_k_n, bias_slice, meta


def _derive_block0_to_q_input(
    model_dir: Path,
    bundle_path: Path,
    *,
    dtype: torch.dtype,
    timestep_index: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    transformer_dir = model_dir / "transformer"
    cfg = _load_config(transformer_dir)
    bundle_args = SimpleNamespace(bundle=str(bundle_path), timestep_index=timestep_index)
    inputs = _load_bundle_inputs(bundle_args, cfg, dtype)
    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=dtype,
    ).eval()
    with torch.no_grad():
        hidden_states, _encoder_hidden_states, temb, _attention_mask, _rotary = _prepare_frontend(
            model,
            inputs,
            use_meanflow=bool(cfg.get("use_meanflow", False)),
        )
        norm_hidden_states, *_ = model.transformer_blocks[0].norm1(hidden_states, emb=temb)
    meta = {
        "bundle": str(bundle_path),
        "bundle_schema": "transformer-boundary",
        "hidden_states_shape": list(inputs["hidden_states"].shape),
        "derived_norm_hidden_states_shape": list(norm_hidden_states.shape),
        "derived_norm_hidden_states_dtype": str(norm_hidden_states.dtype),
        "image_seq_len": int(getattr(bundle_args, "image_seq_len")),
        "text_seq_len": int(getattr(bundle_args, "text_seq_len")),
        "text_seq_len_2": int(getattr(bundle_args, "text_seq_len_2")),
    }
    return norm_hidden_states.to(dtype=torch.bfloat16).contiguous(), meta


def _baseline_linear(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out = input_slice.float() @ weight_k_n.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


def _run_cpu(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    chunks = []
    start = time.perf_counter()
    for offset in range(0, input_slice.shape[0], 128):
        chunks.append(
            cpu_mx.linear_mx_reference(
                input_slice[offset : offset + 128],
                weight_k_n,
                bias,
            )
        )
    elapsed = time.perf_counter() - start
    return torch.cat(chunks, dim=0), {"cpu_mx_time_s": elapsed}


def _run_simulator(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    from nki.simulator import simulate_kernel

    from nova.backends.trainium.nki_kernels.mx import matmul_mx_k_tiles_kernel

    chunks = []
    start = time.perf_counter()
    for offset in range(0, input_slice.shape[0], 128):
        packed = cpu_mx.pack_linear_mx_inputs(
            input_slice[offset : offset + 128],
            weight_k_n,
            bias,
        )
        out = simulate_kernel(
            matmul_mx_k_tiles_kernel,
            tuple(tensor.numpy() for tensor in packed),
            {},
        )
        out_tensor = torch.from_numpy(out).to(torch.bfloat16)
        if bias is not None:
            out_tensor = (out_tensor.float() + bias.float()).to(torch.bfloat16)
        chunks.append(out_tensor)
    elapsed = time.perf_counter() - start
    return torch.cat(chunks, dim=0), {"simulator_time_s": elapsed}


def _run_trainium(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    import torch_xla.core.xla_model as xm

    from nova.backends.trainium.ops_impl import mx as trainium_mx

    device = xm.xla_device()
    start = time.perf_counter()
    weight_device = weight_k_n.to(device)
    bias_device = bias.to(device) if bias is not None else None
    chunks = []
    xm.mark_step()
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    for offset in range(0, input_slice.shape[0], 128):
        out = trainium_mx.linear_mx(
            input_slice[offset : offset + 128].to(device),
            weight_device,
            bias_device,
        )
        chunks.append(out.cpu())
    xm.mark_step()
    elapsed = time.perf_counter() - start
    return torch.cat(chunks, dim=0).to(torch.bfloat16), {
        "trainium_load_time_s": load_time,
        "trainium_forward_time_s": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("cpu", "simulator", "trainium"),
        default="cpu",
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--m-slice", type=int, default=128)
    parser.add_argument("--m-offset", type=int, default=0)
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs", type=float, default=0.03)
    parser.add_argument("--max-max-abs", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.m_slice < 1 or args.m_slice % 128 != 0:
        raise ValueError(f"--m-slice must be a positive multiple of 128, got {args.m_slice}")
    if args.m_offset < 0:
        raise ValueError(f"--m-offset must be non-negative, got {args.m_offset}")

    weight_k_n, bias, weight_meta = _load_to_q_shard(
        args.model_dir,
        rank=args.rank,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
    )
    if args.no_bias:
        bias = None
    activation, activation_meta = _derive_block0_to_q_input(
        args.model_dir,
        args.bundle,
        dtype=torch.bfloat16,
        timestep_index=args.timestep_index,
    )
    activation_2d = activation.squeeze(0)
    end = args.m_offset + args.m_slice
    if end > activation_2d.shape[0]:
        raise ValueError(
            f"requested M slice [{args.m_offset}, {end}) exceeds activation "
            f"sequence length {activation_2d.shape[0]}"
        )
    input_slice = activation_2d[args.m_offset:end, :].contiguous()

    probe = {
        "weight": weight_meta,
        "activation": activation_meta,
        "m_offset": args.m_offset,
        "m_slice": args.m_slice,
        "input_slice_shape": list(input_slice.shape),
        "bias_enabled": bias is not None,
    }
    print(json.dumps(probe, indent=2, sort_keys=True))
    if args.probe_only:
        return 0

    baseline_start = time.perf_counter()
    baseline = _baseline_linear(input_slice, weight_k_n, bias)
    baseline_time = time.perf_counter() - baseline_start

    if args.mode == "trainium":
        observed, timings = _run_trainium(input_slice, weight_k_n, bias)
    elif args.mode == "simulator":
        observed, timings = _run_simulator(input_slice, weight_k_n, bias)
    else:
        observed, timings = _run_cpu(input_slice, weight_k_n, bias)

    observed = observed.to(torch.bfloat16)
    diff = (baseline.float() - observed.float()).abs()
    metrics = {
        **probe,
        "mode": args.mode,
        "baseline_time_s": baseline_time,
        **timings,
        "output_shape": list(observed.shape),
        "cosine": _cosine(baseline, observed),
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "passed": False,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_mean_abs": args.max_mean_abs,
            "max_max_abs": args.max_max_abs,
        },
    }
    metrics["passed"] = (
        metrics["cosine"] >= args.min_cosine
        and metrics["mean_abs"] <= args.max_mean_abs
        and metrics["max_abs"] <= args.max_max_abs
    )

    if args.metrics_out is not None:
        _write_metrics(args.metrics_out, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
