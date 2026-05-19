#!/usr/bin/env python3
"""HV-1.5 per-block MX calibration sweep."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hunyuan15_transformer_parity import _load_bundle_inputs, _load_config  # noqa: E402
from nova.backends.cpu.ops_impl import mx as cpu_mx  # noqa: E402


TARGETS: dict[str, str] = {
    "to_q": "column",
    "to_k": "column",
    "to_v": "column",
    "to_out.0": "row",
    "ff.net.0.proj": "column",
    "ff.net.2": "row",
}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1),
        b.float().reshape(1, -1),
    ).item()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _target_module(block: torch.nn.Module, linear: str) -> torch.nn.Module:
    if linear == "to_q":
        return block.attn.to_q
    if linear == "to_k":
        return block.attn.to_k
    if linear == "to_v":
        return block.attn.to_v
    if linear == "to_out.0":
        return block.attn.to_out[0]
    if linear == "ff.net.0.proj":
        return block.ff.net[0].proj
    if linear == "ff.net.2":
        return block.ff.net[2]
    raise KeyError(linear)


def _slice_linear(
    module: torch.nn.Module,
    activation: torch.Tensor,
    parallel: str,
    *,
    rank: int,
    tp_degree: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    weight = module.weight.detach().cpu().to(torch.bfloat16).contiguous()
    bias = (
        module.bias.detach().cpu().to(torch.bfloat16).contiguous()
        if getattr(module, "bias", None) is not None
        else None
    )
    if weight.ndim != 2:
        raise ValueError(f"expected 2D Linear weight, got {tuple(weight.shape)}")
    if rank < 0 or rank >= tp_degree:
        raise ValueError(f"--rank must be in [0, {tp_degree}), got {rank}")

    if parallel == "column":
        if weight.shape[0] % tp_degree != 0:
            raise ValueError(f"column output dim {weight.shape[0]} not divisible by {tp_degree}")
        shard = weight.shape[0] // tp_degree
        out_start = rank * shard
        out_end = out_start + shard
        weight_slice = weight[out_start:out_end, :]
        bias_slice = bias[out_start:out_end] if bias is not None else None
        input_slice = activation
        shard_meta = {
            "parallel": parallel,
            "output_slice": [out_start, out_end],
            "input_slice": [0, weight.shape[1]],
            "bias_policy": "column_local_bias_included",
        }
    elif parallel == "row":
        if weight.shape[1] % tp_degree != 0:
            raise ValueError(f"row input dim {weight.shape[1]} not divisible by {tp_degree}")
        shard = weight.shape[1] // tp_degree
        in_start = rank * shard
        in_end = in_start + shard
        weight_slice = weight[:, in_start:in_end]
        bias_slice = None
        input_slice = activation[:, in_start:in_end].contiguous()
        shard_meta = {
            "parallel": parallel,
            "output_slice": [0, weight.shape[0]],
            "input_slice": [in_start, in_end],
            "bias_policy": "row_post_reduce_bias_excluded",
        }
    else:
        raise ValueError(f"unsupported parallel type: {parallel!r}")

    return (
        input_slice.contiguous(),
        weight_slice.T.contiguous(),
        bias_slice.contiguous() if bias_slice is not None else None,
        {
            **shard_meta,
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape) if bias is not None else None,
            "weight_k_n_shape": [weight_slice.shape[1], weight_slice.shape[0]],
            "k_tiles": weight_slice.shape[1] // 512,
            "n_tiles": weight_slice.shape[0] // 512,
        },
    )


def _baseline(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out = input_slice.float() @ weight_k_n.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(torch.bfloat16)


_MX_DTYPES = ("float8_e4m3fn_x4", "float8_e5m2_x4")
# Short keys used in the per-cell metrics dict and synthesizer.
_DTYPE_KEY = {"float8_e4m3fn_x4": "e4m3", "float8_e5m2_x4": "e5m2"}


def _run_mx(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
    bias: torch.Tensor | None,
    mx_dtype: str,
) -> torch.Tensor:
    chunks = []
    for offset in range(0, input_slice.shape[0], 128):
        chunks.append(
            cpu_mx.linear_mx(
                input_slice[offset : offset + 128].contiguous(),
                weight_k_n,
                bias,
                dtype=mx_dtype,
            )
        )
    return torch.cat(chunks, dim=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--m-slice", type=int, default=128)
    parser.add_argument("--m-offset", type=int, default=0)
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.m_slice < 1 or args.m_slice % 128 != 0:
        raise ValueError(f"--m-slice must be a positive multiple of 128, got {args.m_slice}")

    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    transformer_dir = args.model_dir / args.transformer_subfolder
    cfg = _load_config(transformer_dir)
    bundle_args = SimpleNamespace(bundle=str(args.bundle), timestep_index=args.timestep_index)
    inputs = _load_bundle_inputs(bundle_args, cfg, torch.bfloat16)
    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=torch.bfloat16,
    ).eval()

    block_count = len(model.transformer_blocks)
    if args.max_blocks is not None:
        block_count = min(block_count, args.max_blocks)
    captures: dict[tuple[int, str], torch.Tensor] = {}
    hooks = []

    end = args.m_offset + args.m_slice

    def make_hook(block_idx: int, linear: str):
        def hook(_module, module_inputs):
            activation = module_inputs[0].detach().cpu().to(torch.bfloat16).squeeze(0)
            if end > activation.shape[0]:
                raise ValueError(
                    f"requested M slice [{args.m_offset}, {end}) exceeds "
                    f"{block_idx}:{linear} activation length {activation.shape[0]}"
                )
            captures[(block_idx, linear)] = activation[args.m_offset:end, :].contiguous()

        return hook

    for block_idx, block in enumerate(model.transformer_blocks[:block_count]):
        for linear in TARGETS:
            hook = _target_module(block, linear).register_forward_pre_hook(
                make_hook(block_idx, linear)
            )
            hooks.append(hook)

    start = time.perf_counter()
    try:
        with torch.no_grad():
            model(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                encoder_attention_mask=inputs["encoder_attention_mask"],
                timestep_r=inputs["timestep_r"] if bool(cfg.get("use_meanflow", False)) else None,
                encoder_hidden_states_2=inputs["encoder_hidden_states_2"],
                encoder_attention_mask_2=inputs["encoder_attention_mask_2"],
                image_embeds=inputs["image_embeds"],
                return_dict=False,
            )
    finally:
        for handle in hooks:
            handle.remove()
    capture_time_s = time.perf_counter() - start

    rows = []
    mx_start = time.perf_counter()
    for block_idx in range(block_count):
        block = model.transformer_blocks[block_idx]
        for linear, parallel in TARGETS.items():
            activation = captures[(block_idx, linear)]
            input_slice, weight_k_n, bias, shard_meta = _slice_linear(
                _target_module(block, linear),
                activation,
                parallel,
                rank=args.rank,
                tp_degree=args.tp_degree,
            )
            baseline = _baseline(input_slice, weight_k_n, bias)
            metrics: dict[str, dict[str, float]] = {}
            for mx_dtype in _MX_DTYPES:
                observed = _run_mx(
                    input_slice, weight_k_n, bias, mx_dtype
                ).to(torch.bfloat16)
                diff = (baseline.float() - observed.float()).abs()
                metrics[_DTYPE_KEY[mx_dtype]] = {
                    "cosine": _cosine(baseline, observed),
                    "mean_abs": diff.mean().item(),
                    "max_abs": diff.max().item(),
                }
            e4m3 = metrics["e4m3"]
            rows.append(
                {
                    "block": block_idx,
                    "linear": linear,
                    "rank": args.rank,
                    "tp_degree": args.tp_degree,
                    "m_offset": args.m_offset,
                    "m_slice": args.m_slice,
                    "input_shape": list(input_slice.shape),
                    "output_shape": list(input_slice.shape[:1])
                    + [weight_k_n.shape[1]],
                    # Back-compat: top-level keys remain the E4M3 values so
                    # the M5.4.0 reader still parses (Decision D2). The
                    # three-dtype data lives under "metrics".
                    "cosine": e4m3["cosine"],
                    "mean_abs": e4m3["mean_abs"],
                    "max_abs": e4m3["max_abs"],
                    "metrics": metrics,
                    **shard_meta,
                }
            )
    mx_time_s = time.perf_counter() - mx_start

    expected_rows = block_count * len(TARGETS)

    def _dtype_summary(key: str) -> dict[str, float]:
        cosines = [row["metrics"][key]["cosine"] for row in rows]
        return {
            "min_cosine": min(cosines),
            "mean_cosine": sum(cosines) / len(cosines),
            "max_mean_abs": max(row["metrics"][key]["mean_abs"] for row in rows),
            "max_max_abs": max(row["metrics"][key]["max_abs"] for row in rows),
        }

    has_nan = any(
        row["metrics"][k]["cosine"] != row["metrics"][k]["cosine"]
        for row in rows
        for k in ("e4m3", "e5m2")
    )
    table = {
        "schema_version": 2,
        "model_dir": str(args.model_dir),
        "transformer_subfolder": args.transformer_subfolder,
        "bundle": str(args.bundle),
        "rank": args.rank,
        "tp_degree": args.tp_degree,
        "m_offset": args.m_offset,
        "m_slice": args.m_slice,
        "block_count": block_count,
        "targets": list(TARGETS),
        "expected_rows": expected_rows,
        "row_count": len(rows),
        "complete": len(rows) == expected_rows,
        "has_nan": has_nan,
        "capture_time_s": capture_time_s,
        "mx_time_s": mx_time_s,
        # Back-compat: "summary" stays E4M3 (the M5.4.0 default).
        "summary": _dtype_summary("e4m3"),
        "summary_by_dtype": {
            "e4m3": _dtype_summary("e4m3"),
            "e5m2": _dtype_summary("e5m2"),
        },
        "rows": rows,
    }
    _write_json(args.out, table)
    print(
        json.dumps(
            {
                k: table[k]
                for k in (
                    "row_count",
                    "complete",
                    "has_nan",
                    "summary_by_dtype",
                )
            },
            indent=2,
        )
    )
    return 0 if table["complete"] and not table["has_nan"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
