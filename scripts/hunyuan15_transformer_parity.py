#!/usr/bin/env python3
"""HunyuanVideo 1.5 transformer parity: Trainium artifact vs diffusers CPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import torch  # noqa: E402


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--cache-dir", default="/tmp/difflet_hunyuan15_transformer_parity_cache")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--text-seq-len", type=int, default=7)
    parser.add_argument("--text-seq-len-2", type=int, default=3)
    parser.add_argument("--image-seq-len", type=int, default=4)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--reference-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional safetensors artifact from scripts/hunyuan15_cache_dit_inputs.py.",
    )
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--transformer-runtime",
        choices=("monolithic", "segmented"),
        default="monolithic",
        help="Difflet HunyuanVideo 1.5 transformer runtime to validate.",
    )
    parser.add_argument("--segmented-query-tile-size", type=int, default=2051)
    parser.add_argument("--segmented-key-tile-size", type=int, default=2051)
    parser.add_argument(
        "--segmented-block-load-mode",
        choices=("all", "streaming", "process"),
        default="all",
        help=(
            "Load every segmented block component, reuse one pre/post NEFF in-process, "
            "or run each block in a fresh worker process."
        ),
    )
    parser.add_argument("--segmented-block-compiler-args", default=None)
    parser.add_argument("--segmented-attention-compiler-args", default=None)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Run only the Trainium path and report a runtime gate.",
    )
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--save-trainium", default=None)
    parser.add_argument("--save-reference", default=None)
    return parser


def _load_config(transformer_dir: Path) -> dict[str, Any]:
    return json.loads((transformer_dir / "config.json").read_text(encoding="utf-8"))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.detach().float().reshape(-1),
        b.detach().float().reshape(-1),
        dim=0,
    ).item()


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("sample", "images", "frames", "latents"):
            item = value.get(key)
            if torch.is_tensor(item):
                return item
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if hasattr(value, "sample") and torch.is_tensor(value.sample):
        return value.sample
    raise TypeError(f"could not extract tensor from {type(value)!r}")


def _make_inputs(args: argparse.Namespace, cfg: dict[str, Any], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent_frames = (int(args.num_frames) - 1) // 4 + 1
    latent_height = int(args.height) // 16
    latent_width = int(args.width) // 16
    return {
        "hidden_states": torch.randn(
            [1, int(cfg["in_channels"]), latent_frames, latent_height, latent_width],
            generator=generator,
            dtype=dtype,
        ),
        "timestep": torch.ones([1], dtype=dtype),
        "encoder_hidden_states": torch.randn(
            [1, args.text_seq_len, int(cfg["text_embed_dim"])],
            generator=generator,
            dtype=dtype,
        ),
        "encoder_attention_mask": torch.ones([1, args.text_seq_len], dtype=torch.int64),
        "timestep_r": torch.ones([1], dtype=dtype),
        "encoder_hidden_states_2": torch.randn(
            [1, args.text_seq_len_2, int(cfg["text_embed_2_dim"])],
            generator=generator,
            dtype=dtype,
        ),
        "encoder_attention_mask_2": torch.ones([1, args.text_seq_len_2], dtype=torch.int64),
        "image_embeds": torch.zeros([1, args.image_seq_len, int(cfg["image_embed_dim"])], dtype=dtype),
    }


def _first_timestep(timesteps: torch.Tensor, index: int, dtype: torch.dtype) -> torch.Tensor:
    if timesteps.ndim == 0:
        return timesteps.reshape(1).to(dtype=dtype)
    if index < 0 or index >= timesteps.numel():
        raise IndexError(f"--timestep-index {index} out of range for timesteps shape {tuple(timesteps.shape)}")
    return timesteps.reshape(-1)[index].reshape(1).to(dtype=dtype)


def _load_bundle_inputs(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    tensors = load_file(args.bundle, device="cpu")
    required = {
        "hidden_states",
        "timesteps",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "encoder_hidden_states_2",
        "encoder_attention_mask_2",
        "image_embeds",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise KeyError(f"{args.bundle} is missing required tensors: {', '.join(missing)}")

    hidden_states = tensors["hidden_states"].to(dtype=dtype).contiguous()
    if hidden_states.shape[1] != int(cfg["in_channels"]):
        raise ValueError(
            f"bundle hidden_states has {hidden_states.shape[1]} channels, "
            f"but transformer config expects {cfg['in_channels']}"
        )
    args.text_seq_len = int(tensors["encoder_hidden_states"].shape[1])
    args.text_seq_len_2 = int(tensors["encoder_hidden_states_2"].shape[1])
    args.image_seq_len = int(tensors["image_embeds"].shape[1])
    return {
        "hidden_states": hidden_states,
        "timestep": _first_timestep(tensors["timesteps"], args.timestep_index, dtype),
        "encoder_hidden_states": tensors["encoder_hidden_states"].to(dtype=dtype).contiguous(),
        "encoder_attention_mask": tensors["encoder_attention_mask"].to(dtype=torch.int64).contiguous(),
        "timestep_r": tensors.get("timestep_r", torch.ones([hidden_states.shape[0]], dtype=dtype)).to(dtype=dtype),
        "encoder_hidden_states_2": tensors["encoder_hidden_states_2"].to(dtype=dtype).contiguous(),
        "encoder_attention_mask_2": tensors["encoder_attention_mask_2"].to(dtype=torch.int64).contiguous(),
        "image_embeds": tensors["image_embeds"].to(dtype=dtype).contiguous(),
    }


def _run_trainium(args: argparse.Namespace, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any, float]:
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    from difflet import DiffletParallelConfig, DiffletPipeline
    from difflet.models.hunyuan_video.application import HunyuanVideo15DiTInputBundle

    t0 = time.perf_counter()
    app_kwargs = {
        "text_seq_len": args.text_seq_len,
        "text_seq_len_2": args.text_seq_len_2,
        "image_seq_len": args.image_seq_len,
        "transformer_runtime": args.transformer_runtime,
    }
    if args.transformer_subfolder != "transformer":
        app_kwargs["transformer_subfolder"] = args.transformer_subfolder
    if args.transformer_runtime == "segmented":
        app_kwargs.update(
            {
                "segmented_query_tile_size": args.segmented_query_tile_size,
                "segmented_key_tile_size": args.segmented_key_tile_size,
                "segmented_block_load_mode": args.segmented_block_load_mode,
            }
        )
        if args.segmented_block_compiler_args is not None:
            app_kwargs["segmented_block_compiler_args"] = args.segmented_block_compiler_args
        if args.segmented_attention_compiler_args is not None:
            app_kwargs["segmented_attention_compiler_args"] = args.segmented_attention_compiler_args

    pipe = DiffletPipeline.from_pretrained(
        args.model_dir,
        model_type="hunyuan_video_15",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=args.dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=args.cache_dir,
        force_compile=args.force_compile,
        skip_compile=args.skip_compile,
        load=True,
        skip_warmup=args.skip_warmup,
        application_kwargs=app_kwargs,
    )
    load_elapsed = time.perf_counter() - t0
    bundle = HunyuanVideo15DiTInputBundle(**inputs)
    t1 = time.perf_counter()
    with torch.no_grad():
        output = _first_tensor(pipe(bundle)).detach().cpu()
    forward_elapsed = time.perf_counter() - t1
    print(f"[hunyuan15-parity] compiled_path = {pipe.compiled_path}", flush=True)
    print(f"[hunyuan15-parity] trainium load elapsed = {load_elapsed:.3f}s", flush=True)
    print(f"[hunyuan15-parity] trainium forward elapsed = {forward_elapsed:.3f}s", flush=True)
    return output, pipe, forward_elapsed


def _run_reference(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, float]:
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    transformer_dir = Path(args.model_dir) / args.transformer_subfolder
    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=args.reference_dtype,
    ).eval()
    use_meanflow = bool(cfg.get("use_meanflow", False))
    ref_inputs = {
        key: value.to(dtype=args.reference_dtype) if torch.is_floating_point(value) else value
        for key, value in inputs.items()
    }
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(
            hidden_states=ref_inputs["hidden_states"],
            timestep=ref_inputs["timestep"],
            encoder_hidden_states=ref_inputs["encoder_hidden_states"],
            encoder_attention_mask=ref_inputs["encoder_attention_mask"],
            timestep_r=ref_inputs["timestep_r"] if use_meanflow else None,
            encoder_hidden_states_2=ref_inputs["encoder_hidden_states_2"],
            encoder_attention_mask_2=ref_inputs["encoder_attention_mask_2"],
            image_embeds=ref_inputs["image_embeds"],
            return_dict=False,
        )[0].detach().cpu()
    elapsed = time.perf_counter() - t0
    print(f"[hunyuan15-parity] reference elapsed = {elapsed:.3f}s", flush=True)
    return output, elapsed


def main() -> int:
    args = build_parser().parse_args()
    transformer_dir = Path(args.model_dir) / args.transformer_subfolder
    cfg = _load_config(transformer_dir)
    inputs = (
        _load_bundle_inputs(args, cfg, args.dtype)
        if args.bundle is not None
        else _make_inputs(args, cfg, args.dtype)
    )
    metrics = {
        "model_dir": args.model_dir,
        "transformer_subfolder": args.transformer_subfolder,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "text_seq_len": args.text_seq_len,
        "text_seq_len_2": args.text_seq_len_2,
        "image_seq_len": args.image_seq_len,
        "tp_degree": args.tp_degree,
        "transformer_runtime": args.transformer_runtime,
        "reference_skipped": args.skip_reference,
    }
    if args.transformer_runtime == "segmented":
        metrics.update(
            {
                "segmented_query_tile_size": args.segmented_query_tile_size,
                "segmented_key_tile_size": args.segmented_key_tile_size,
                "segmented_block_load_mode": args.segmented_block_load_mode,
            }
        )
    trainium, pipe, trainium_forward_elapsed = _run_trainium(args, inputs)
    metrics.update(
        {
            "compiled_path": str(pipe.compiled_path),
            "trainium_forward_elapsed_s": trainium_forward_elapsed,
            "trainium_shape": list(trainium.shape),
            "trainium_mean": float(trainium.float().mean()),
            "trainium_absmax": float(trainium.float().abs().max()),
            "trainium_checksum": float(trainium.float().sum()),
        }
    )
    last_segmented_metrics = getattr(getattr(pipe, "app", None), "transformer", None)
    last_segmented_metrics = getattr(last_segmented_metrics, "last_metrics", None)
    if last_segmented_metrics is not None:
        metrics.update(
            {
                "stream_tile_calls": int(last_segmented_metrics.stream_tile_calls),
                "stream_forward_elapsed_s": float(last_segmented_metrics.stream_forward_elapsed_s),
                "stream_block_count": int(last_segmented_metrics.block_count),
                "stream_blocks_elapsed_s": float(last_segmented_metrics.blocks_elapsed_s),
            }
        )
    if not args.skip_reference:
        reference, reference_elapsed = _run_reference(args, cfg, inputs)
        diff = (trainium.float() - reference.float()).abs()
        metrics.update(
            {
                "cosine": _cosine(trainium, reference),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "reference_elapsed_s": reference_elapsed,
                "reference_shape": list(reference.shape),
                "reference_mean": float(reference.float().mean()),
            }
        )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-parity] metrics -> {metrics_path}", flush=True)
    if args.save_trainium:
        torch.save(trainium, args.save_trainium)
    if args.save_reference:
        torch.save(reference, args.save_reference)
    if args.skip_reference:
        print("[hunyuan15-parity] runtime gate = PASS", flush=True)
        return 0
    pass_gate = metrics["cosine"] >= args.min_cosine
    print(f"[hunyuan15-parity] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
