#!/usr/bin/env python3
"""HV-1.5 block-0 RowParallel real-weight MX outer-N parity probe."""

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


class TargetSpec:
    def __init__(
        self,
        *,
        name: str,
        weight_key: str,
        bias_key: str,
        hook_name: str,
        input_dim: int,
    ) -> None:
        self.name = name
        self.weight_key = weight_key
        self.bias_key = bias_key
        self.hook_name = hook_name
        self.input_dim = input_dim


TARGETS = {
    "to_out.0": TargetSpec(
        name="to_out.0",
        weight_key="transformer_blocks.0.attn.to_out.0.weight",
        bias_key="transformer_blocks.0.attn.to_out.0.bias",
        hook_name="to_out.0",
        input_dim=2048,
    ),
    "ff.net.2": TargetSpec(
        name="ff.net.2",
        weight_key="transformer_blocks.0.ff.net.2.weight",
        bias_key="transformer_blocks.0.ff.net.2.bias",
        hook_name="ff.net.2",
        input_dim=8192,
    ),
}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1),
        b.float().reshape(1, -1),
    ).item()


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _target_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    block = model.transformer_blocks[0]
    return {
        "to_out.0": block.attn.to_out[0],
        "ff.net.2": block.ff.net[2],
    }


def _capture_block0_inputs(
    model_dir: Path,
    bundle_path: Path,
    *,
    dtype: torch.dtype,
    timestep_index: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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

    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module, module_inputs):
            captured[name] = module_inputs[0].detach().cpu().to(torch.bfloat16).contiguous()

        return hook

    for name, module in _target_modules(model).items():
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        with torch.no_grad():
            hidden_states, encoder_hidden_states, temb, attention_mask, rotary = _prepare_frontend(
                model,
                inputs,
                use_meanflow=bool(cfg.get("use_meanflow", False)),
            )
            model.transformer_blocks[0](
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask=attention_mask,
                freqs_cis=rotary,
            )
    finally:
        for handle in hooks:
            handle.remove()

    missing = sorted(set(TARGETS) - set(captured))
    if missing:
        raise RuntimeError(f"forward hooks did not capture targets: {', '.join(missing)}")

    meta = {
        "bundle": str(bundle_path),
        "bundle_schema": "transformer-boundary",
        "hidden_states_shape": list(inputs["hidden_states"].shape),
        "captured_shapes": {name: list(tensor.shape) for name, tensor in captured.items()},
        "captured_dtypes": {name: str(tensor.dtype) for name, tensor in captured.items()},
        "image_seq_len": int(getattr(bundle_args, "image_seq_len")),
        "text_seq_len": int(getattr(bundle_args, "text_seq_len")),
        "text_seq_len_2": int(getattr(bundle_args, "text_seq_len_2")),
    }
    return captured, meta


def _load_rowparallel_shard(
    model_dir: Path,
    spec: TargetSpec,
    *,
    rank: int,
    tp_degree: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    transformer_path = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    with safe_open(transformer_path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = sorted({spec.weight_key, spec.bias_key} - available)
        if missing:
            raise KeyError(f"{transformer_path} is missing keys: {', '.join(missing)}")
        weight = handle.get_tensor(spec.weight_key)
        bias = handle.get_tensor(spec.bias_key)

    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError(
            f"unexpected {spec.name} shapes: weight={tuple(weight.shape)} "
            f"bias={tuple(bias.shape)}"
        )
    if weight.shape[0] != bias.shape[0]:
        raise ValueError(
            f"{spec.name} weight/bias output mismatch: {weight.shape[0]} != {bias.shape[0]}"
        )
    if weight.shape[1] % tp_degree != 0:
        raise ValueError(
            f"{spec.name} input dim {weight.shape[1]} is not divisible by tp={tp_degree}"
        )
    if rank < 0 or rank >= tp_degree:
        raise ValueError(f"--rank must be in [0, {tp_degree}), got {rank}")

    in_local = weight.shape[1] // tp_degree
    start = rank * in_local
    end = start + in_local
    weight_slice = weight[:, start:end].to(dtype=dtype).contiguous()
    weight_k_n = weight_slice.T.contiguous()
    bias_full = bias.to(dtype=dtype).contiguous()
    meta = {
        "path": str(transformer_path),
        "target": spec.name,
        "weight_key": spec.weight_key,
        "bias_key": spec.bias_key,
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "bias_shape": list(bias.shape),
        "bias_dtype": str(bias.dtype),
        "rank": rank,
        "tp_degree": tp_degree,
        "input_slice": [start, end],
        "weight_slice_shape": list(weight_slice.shape),
        "weight_k_n_shape": list(weight_k_n.shape),
        "bias_shape_full": list(bias_full.shape),
        "k_tiles": weight_k_n.shape[0] // 512,
        "n_tiles": weight_k_n.shape[1] // 512,
    }
    return weight_k_n, bias_full, meta


def _baseline_partial(input_slice: torch.Tensor, weight_k_n: torch.Tensor) -> torch.Tensor:
    return (input_slice.float() @ weight_k_n.float()).to(torch.bfloat16)


def _run_cpu(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    start = time.perf_counter()
    out = cpu_mx.linear_mx_outer_n_reference(input_slice, weight_k_n)
    elapsed = time.perf_counter() - start
    return out, {"cpu_mx_time_s": elapsed}


def _run_simulator(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    from nki.simulator import simulate_kernel

    from nova.backends.trainium.nki_kernels.mx import matmul_mx_k_tiles_kernel

    outputs = []
    start = time.perf_counter()
    for n_start in range(0, weight_k_n.shape[1], 512):
        packed = cpu_mx.pack_linear_mx_inputs(
            input_slice,
            weight_k_n[:, n_start : n_start + 512].contiguous(),
        )
        out = simulate_kernel(
            matmul_mx_k_tiles_kernel,
            (*tuple(tensor.numpy() for tensor in packed), packed[0].shape[0]),
            {},
        )
        outputs.append(torch.from_numpy(out).to(torch.bfloat16))
    elapsed = time.perf_counter() - start
    return torch.cat(outputs, dim=1), {"simulator_time_s": elapsed}


def _run_trainium(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    import torch_xla.core.xla_model as xm

    from nova.backends.trainium.ops_impl import mx as trainium_mx

    device = xm.xla_device()
    start = time.perf_counter()
    input_device = input_slice.to(device)
    weight_device = weight_k_n.to(device)
    xm.mark_step()
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    out = trainium_mx.linear_mx(input_device, weight_device)
    out_cpu = out.cpu().to(torch.bfloat16)
    xm.mark_step()
    elapsed = time.perf_counter() - start
    return out_cpu, {
        "trainium_load_time_s": load_time,
        "trainium_forward_time_s": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--target", choices=sorted(TARGETS), default="to_out.0")
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

    captured, activation_meta = _capture_block0_inputs(
        args.model_dir,
        args.bundle,
        dtype=torch.bfloat16,
        timestep_index=args.timestep_index,
    )
    spec = TARGETS[args.target]
    weight_k_n, bias_full, weight_meta = _load_rowparallel_shard(
        args.model_dir,
        spec,
        rank=args.rank,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
    )
    activation = captured[spec.hook_name].squeeze(0)
    in_start, in_end = weight_meta["input_slice"]
    end = args.m_offset + args.m_slice
    if end > activation.shape[0]:
        raise ValueError(
            f"requested M slice [{args.m_offset}, {end}) exceeds activation "
            f"sequence length {activation.shape[0]}"
        )
    input_slice = activation[args.m_offset:end, in_start:in_end].contiguous()

    probe = {
        "target": args.target,
        "weight": weight_meta,
        "activation": activation_meta,
        "m_offset": args.m_offset,
        "m_slice": args.m_slice,
        "input_slice_shape": list(input_slice.shape),
        "bias_policy": "excluded_from_per_rank_partial_gate",
        "bias_shape_full": list(bias_full.shape),
    }
    print(json.dumps(probe, indent=2, sort_keys=True))
    if args.probe_only:
        return 0

    baseline_start = time.perf_counter()
    baseline = _baseline_partial(input_slice, weight_k_n)
    baseline_time = time.perf_counter() - baseline_start

    if args.mode == "trainium":
        observed, timings = _run_trainium(input_slice, weight_k_n)
    elif args.mode == "simulator":
        observed, timings = _run_simulator(input_slice, weight_k_n)
    else:
        observed, timings = _run_cpu(input_slice, weight_k_n)

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
