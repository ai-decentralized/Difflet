#!/usr/bin/env python3
"""LTX-2 dual-stream per-block MX calibration sweep (M5.4.5).

Mirrors ``hv15_precision_calibration_sweep.py`` but tags every cell with
its stream (``video`` / ``audio``) so the synthesizer can compare the
two streams' selective frontiers separately. The open question
(Decision D5, cclog 48 §10): is the audio stream — head_dim 64, the
known BF16-drift modality — exponent-bound (where E5M2 would help) or
mantissa-bound like the HV-1.5 video stream (where E5M2 was refuted on
324/324 cells)?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Reuse the HV-1.5 sweep's stream-agnostic primitives (DRY).
from hv15_precision_calibration_sweep import (  # noqa: E402
    _DTYPE_KEY,
    _MX_DTYPES,
    _baseline,
    _cosine,
    _slice_linear,
)
from nova.backends.cpu.ops_impl import mx as cpu_mx  # noqa: E402


# (attribute path on the block, parallel kind). Per LTX-2 dual-stream
# block layout (diffusers transformer_ltx2.LTX2VideoTransformerBlock):
# the video stream owns attn1 + ff; the audio stream owns audio_attn1 +
# audio_ff. Self-attention only — cross-modal/text attns are a separate
# axis, out of M5.4.5 scope.
VIDEO_TARGETS: dict[str, str] = {
    "attn1.to_q": "column",
    "attn1.to_k": "column",
    "attn1.to_v": "column",
    "attn1.to_out.0": "row",
    "ff.net.0.proj": "column",
    "ff.net.2": "row",
}
AUDIO_TARGETS: dict[str, str] = {
    "audio_attn1.to_q": "column",
    "audio_attn1.to_k": "column",
    "audio_attn1.to_v": "column",
    "audio_attn1.to_out.0": "row",
    "audio_ff.net.0.proj": "column",
    "audio_ff.net.2": "row",
}
STREAM_TARGETS = {"video": VIDEO_TARGETS, "audio": AUDIO_TARGETS}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _resolve_module(block: torch.nn.Module, dotted: str) -> torch.nn.Module:
    module: Any = block
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _run_mx_padded(
    input_slice: torch.Tensor,
    weight_k_n: torch.Tensor,
    bias: torch.Tensor | None,
    mx_dtype: str,
) -> torch.Tensor:
    """MX linear over an arbitrary-M slice.

    ``cpu_mx.linear_mx`` requires M to be a multiple of 128. The audio
    stream has only 126 latent tokens, so pad with zeros up to the next
    multiple of 128, run, then truncate back to the real row count
    (padded rows are bias-only and identical on both sides).
    """

    m = input_slice.shape[0]
    padded_m = ((m + 127) // 128) * 128
    if padded_m != m:
        pad = torch.zeros(
            (padded_m - m, input_slice.shape[1]),
            dtype=input_slice.dtype,
        )
        work = torch.cat([input_slice, pad], dim=0).contiguous()
    else:
        work = input_slice
    chunks = []
    for offset in range(0, work.shape[0], 128):
        chunks.append(
            cpu_mx.linear_mx(
                work[offset : offset + 128].contiguous(),
                weight_k_n,
                bias,
                dtype=mx_dtype,
            )
        )
    return torch.cat(chunks, dim=0)[:m]


def _latent_dims_from_meta(meta: dict[str, Any]) -> tuple[int, int, int, int]:
    height = int(meta["height"])
    width = int(meta["width"])
    num_frames = int(meta["num_frames"])
    # Mirrors scripts/ltx_2_cache_dit_inputs.py:_latent_dims.
    latent_num_frames = (num_frames - 1) // 8 + 1
    return (
        latent_num_frames,
        height // 32,
        width // 32,
        int(meta["audio_num_frames"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--m-slice", type=int, default=128)
    parser.add_argument("--m-offset", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.m_slice < 1:
        raise ValueError(f"--m-slice must be positive, got {args.m_slice}")

    import diffusers.utils.import_utils as import_utils

    import_utils._torch_xla_available = False
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
    )
    from safetensors.torch import load_file as load_safetensors_file

    transformer_dir = args.model_dir / args.transformer_subfolder
    meta_path = Path(str(args.bundle) + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    latent_nf, latent_h, latent_w, audio_nf = _latent_dims_from_meta(meta)

    tensors = load_safetensors_file(str(args.bundle), device="cpu")
    timesteps = tensors["timesteps"].reshape(-1)
    timestep = timesteps[args.step_index].reshape(1).to(torch.bfloat16)
    inputs = {
        "hidden_states": tensors["latents_init"].to(torch.bfloat16).contiguous(),
        "audio_hidden_states": tensors["audio_latents_init"]
        .to(torch.bfloat16)
        .contiguous(),
        "encoder_hidden_states": tensors["encoder_hidden_states"]
        .to(torch.bfloat16)
        .contiguous(),
        "audio_encoder_hidden_states": tensors["audio_encoder_hidden_states"]
        .to(torch.bfloat16)
        .contiguous(),
        "timestep": timestep.contiguous(),
        "sigma": timestep.clone().contiguous(),
        "encoder_attention_mask": tensors["encoder_attention_mask"]
        .to(torch.bool)
        .contiguous(),
        "audio_encoder_attention_mask": tensors["audio_encoder_attention_mask"]
        .to(torch.bool)
        .contiguous(),
        "video_coords": tensors["video_coords"].to(torch.float32).contiguous(),
        "audio_coords": tensors["audio_coords"].to(torch.float32).contiguous(),
    }

    model = LTX2VideoTransformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()

    block_count = len(model.transformer_blocks)
    if args.max_blocks is not None:
        block_count = min(block_count, args.max_blocks)

    captures: dict[tuple[int, str, str], torch.Tensor] = {}
    hooks = []
    end = args.m_offset + args.m_slice

    def make_hook(block_idx: int, stream: str, linear: str):
        def hook(_module, module_inputs):
            act = module_inputs[0].detach().cpu().to(torch.bfloat16)
            if act.ndim == 3:
                act = act.squeeze(0)
            real_end = min(end, act.shape[0])
            captures[(block_idx, stream, linear)] = act[
                args.m_offset : real_end, :
            ].contiguous()

        return hook

    for block_idx, block in enumerate(model.transformer_blocks[:block_count]):
        for stream, targets in STREAM_TARGETS.items():
            for linear in targets:
                handle = _resolve_module(block, linear).register_forward_pre_hook(
                    make_hook(block_idx, stream, linear)
                )
                hooks.append(handle)

    start = time.perf_counter()
    try:
        with torch.no_grad():
            model(
                hidden_states=inputs["hidden_states"],
                audio_hidden_states=inputs["audio_hidden_states"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                audio_encoder_hidden_states=inputs["audio_encoder_hidden_states"],
                timestep=inputs["timestep"],
                audio_timestep=inputs["timestep"],
                sigma=inputs["sigma"],
                audio_sigma=inputs["sigma"],
                encoder_attention_mask=inputs["encoder_attention_mask"],
                audio_encoder_attention_mask=inputs["audio_encoder_attention_mask"],
                num_frames=latent_nf,
                height=latent_h,
                width=latent_w,
                fps=float(meta.get("frame_rate", 24.0)),
                audio_num_frames=audio_nf,
                video_coords=inputs["video_coords"],
                audio_coords=inputs["audio_coords"],
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=False,
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
        for stream, targets in STREAM_TARGETS.items():
            for linear, parallel in targets.items():
                activation = captures[(block_idx, stream, linear)]
                input_slice, weight_k_n, bias, shard_meta = _slice_linear(
                    _resolve_module(block, linear),
                    activation,
                    parallel,
                    rank=args.rank,
                    tp_degree=args.tp_degree,
                )
                baseline = _baseline(input_slice, weight_k_n, bias)
                metrics: dict[str, dict[str, float]] = {}
                for mx_dtype in _MX_DTYPES:
                    observed = _run_mx_padded(
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
                        "stream": stream,
                        "linear": linear,
                        "rank": args.rank,
                        "tp_degree": args.tp_degree,
                        "m_offset": args.m_offset,
                        "m_rows": input_slice.shape[0],
                        "cosine": e4m3["cosine"],
                        "mean_abs": e4m3["mean_abs"],
                        "max_abs": e4m3["max_abs"],
                        "metrics": metrics,
                        **shard_meta,
                    }
                )
    mx_time_s = time.perf_counter() - mx_start

    expected_rows = block_count * (len(VIDEO_TARGETS) + len(AUDIO_TARGETS))

    def _summary(predicate) -> dict[str, float] | None:
        sel = [r for r in rows if predicate(r)]
        if not sel:
            return None
        out = {}
        for key in ("e4m3", "e5m2"):
            cosines = [r["metrics"][key]["cosine"] for r in sel]
            out[key] = {
                "min_cosine": min(cosines),
                "mean_cosine": sum(cosines) / len(cosines),
                "max_mean_abs": max(r["metrics"][key]["mean_abs"] for r in sel),
                "max_max_abs": max(r["metrics"][key]["max_abs"] for r in sel),
            }
        return out

    has_nan = any(
        row["metrics"][k]["cosine"] != row["metrics"][k]["cosine"]
        for row in rows
        for k in ("e4m3", "e5m2")
    )
    table = {
        "schema_version": 2,
        "model": "ltx_2",
        "model_dir": str(args.model_dir),
        "transformer_subfolder": args.transformer_subfolder,
        "bundle": str(args.bundle),
        "rank": args.rank,
        "tp_degree": args.tp_degree,
        "m_offset": args.m_offset,
        "m_slice": args.m_slice,
        "block_count": block_count,
        "streams": list(STREAM_TARGETS),
        "targets_by_stream": {s: list(t) for s, t in STREAM_TARGETS.items()},
        "expected_rows": expected_rows,
        "row_count": len(rows),
        "complete": len(rows) == expected_rows,
        "has_nan": has_nan,
        "capture_time_s": capture_time_s,
        "mx_time_s": mx_time_s,
        "summary": _summary(lambda r: True),
        "summary_by_stream": {
            "video": _summary(lambda r: r["stream"] == "video"),
            "audio": _summary(lambda r: r["stream"] == "audio"),
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
                    "summary_by_stream",
                )
            },
            indent=2,
        )
    )
    return 0 if table["complete"] and not table["has_nan"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
