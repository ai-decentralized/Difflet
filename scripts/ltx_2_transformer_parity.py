#!/usr/bin/env python3
"""LTX-2 transformer parity: Trainium artifact vs CPU reference."""

from __future__ import annotations

import argparse
import gc
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
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402


def _disable_xla_lazy_import() -> None:
    import diffusers.utils.import_utils as import_utils

    import_utils._torch_xla_available = False


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Parent dir containing transformer/")
    parser.add_argument("--bundle", required=True, help="Cached LTX-2 DiT inputs safetensors")
    parser.add_argument("--cache-dir", default=".nova-cache/ltx_2_transformer_parity")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--audio-num-frames", type=int, default=None)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--reference-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument(
        "--transformer-mode",
        choices=("single", "segmented"),
        default="single",
        help="Nova LTX-2 transformer backend mode.",
    )
    parser.add_argument(
        "--segmented-block-load-mode",
        choices=("streaming", "process"),
        default="streaming",
        help="Block reload mode when --transformer-mode segmented.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("trace", "diffusers", "segmented-cpu"),
        default="trace",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--min-video-cosine", type=float, default=0.999)
    parser.add_argument("--min-audio-cosine", type=float, default=0.999)
    parser.add_argument("--max-video-mean-abs", type=float, default=None)
    parser.add_argument("--max-audio-mean-abs", type=float, default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--save-trainium", default=None)
    parser.add_argument("--save-reference", default=None)
    parser.add_argument("--num-threads", type=int, default=0)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_meta(bundle_path: Path) -> dict[str, Any]:
    meta_path = Path(str(bundle_path) + ".meta.json")
    return _load_json(meta_path) if meta_path.exists() else {}


def _meta_or_arg(meta: dict[str, Any], args: argparse.Namespace, name: str) -> int:
    attr = name.replace("-", "_")
    value = getattr(args, attr)
    if value is not None:
        return int(value)
    if name in meta:
        return int(meta[name])
    raise ValueError(f"--{name.replace('_', '-')} is required when bundle meta is missing")


def _first_timestep(timesteps: torch.Tensor, index: int, dtype: torch.dtype) -> torch.Tensor:
    if timesteps.ndim == 0:
        return timesteps.reshape(1).to(dtype=dtype)
    if index < 0 or index >= timesteps.numel():
        raise IndexError(f"--step-index {index} outside timesteps shape {tuple(timesteps.shape)}")
    return timesteps.reshape(-1)[index].reshape(1).to(dtype=dtype)


def _load_bundle(bundle_path: Path, step_index: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    tensors = load_safetensors_file(str(bundle_path), device="cpu")
    required = {
        "latents_init",
        "audio_latents_init",
        "timesteps",
        "encoder_hidden_states",
        "audio_encoder_hidden_states",
        "encoder_attention_mask",
        "audio_encoder_attention_mask",
        "video_coords",
        "audio_coords",
    }
    missing = sorted(required.difference(tensors))
    if missing:
        raise KeyError(f"bundle {bundle_path} is missing tensors: {missing}")
    timestep = _first_timestep(tensors["timesteps"], step_index, dtype)
    return {
        "hidden_states": tensors["latents_init"].to(dtype=dtype).contiguous(),
        "audio_hidden_states": tensors["audio_latents_init"].to(dtype=dtype).contiguous(),
        "encoder_hidden_states": tensors["encoder_hidden_states"].to(dtype=dtype).contiguous(),
        "audio_encoder_hidden_states": tensors["audio_encoder_hidden_states"].to(
            dtype=dtype
        ).contiguous(),
        "timestep": timestep.contiguous(),
        "sigma": timestep.clone().contiguous(),
        "encoder_attention_mask": tensors["encoder_attention_mask"].to(torch.bool).contiguous(),
        "audio_encoder_attention_mask": tensors["audio_encoder_attention_mask"].to(
            torch.bool
        ).contiguous(),
        "video_coords": tensors["video_coords"].to(dtype=torch.float32).contiguous(),
        "audio_coords": tensors["audio_coords"].to(dtype=torch.float32).contiguous(),
    }


def _iter_safetensor_shards(component_dir: Path) -> list[Path]:
    index_names = (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
    )
    for index_name in index_names:
        index_path = component_dir / index_name
        if index_path.exists():
            index = _load_json(index_path)
            return [component_dir / name for name in sorted(set(index["weight_map"].values()))]
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        path = component_dir / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"no safetensors shard/index found under {component_dir}")


def _load_trace_state_dict(
    model: torch.nn.Module,
    transformer_dir: Path,
    dtype: torch.dtype,
) -> None:
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    unexpected_all: set[str] = set()
    for shard in _iter_safetensor_shards(transformer_dir):
        print(f"[ltx2-parity] reference shard = {shard.name}", flush=True)
        state = load_safetensors_file(str(shard), device="cpu")
        state = {
            (key if key.startswith("transformer.") else f"transformer.{key}"): (
                value.to(dtype) if torch.is_floating_point(value) else value
            )
            for key, value in state.items()
        }
        _, unexpected = model.load_state_dict(state, strict=False)
        loaded.update(state)
        unexpected_all.update(unexpected)
        del state
        gc.collect()
    missing = sorted(expected.difference(loaded))
    unexpected = sorted(unexpected_all)
    if missing or unexpected:
        raise RuntimeError(
            f"trace reference load mismatch: missing={missing[:8]} (n={len(missing)}), "
            f"unexpected={unexpected[:8]} (n={len(unexpected)})"
        )


def _extract_pair(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, dict):
        sample = output.get("sample")
        audio_sample = output.get("audio_sample")
        if torch.is_tensor(sample) and torch.is_tensor(audio_sample):
            return sample, audio_sample
    sample = getattr(output, "sample", None)
    audio_sample = getattr(output, "audio_sample", None)
    if torch.is_tensor(sample) and torch.is_tensor(audio_sample):
        return sample, audio_sample
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        first, second = output[0], output[1]
        if torch.is_tensor(first) and torch.is_tensor(second):
            return first, second
    raise TypeError(f"cannot extract LTX-2 tensor pair from {type(output)!r}")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    lhs = a.detach().reshape(-1)
    rhs = b.detach().reshape(-1)
    if torch.equal(lhs, rhs):
        return 1.0
    lhs = lhs.to(dtype=torch.float64)
    rhs = rhs.to(dtype=torch.float64)
    denom = torch.linalg.vector_norm(lhs) * torch.linalg.vector_norm(rhs)
    if denom == 0:
        return 1.0 if torch.equal(lhs, rhs) else 0.0
    value = torch.dot(lhs, rhs) / denom
    return float(torch.clamp(value, -1.0, 1.0).item())


def _run_trainium(
    args: argparse.Namespace,
    meta: dict[str, Any],
    inputs: dict[str, torch.Tensor],
):
    os.environ.setdefault("NOVA_BACKEND", "trainium")

    from nova import NovaParallelConfig, NovaPipeline
    from nova.models.ltx_2.application import LTX2DiTInputBundle

    height = _meta_or_arg(meta, args, "height")
    width = _meta_or_arg(meta, args, "width")
    num_frames = _meta_or_arg(meta, args, "num_frames")
    audio_num_frames = _meta_or_arg(meta, args, "audio_num_frames")
    text_seq_len = int(inputs["encoder_hidden_states"].shape[1])
    audio_text_seq_len = int(inputs["audio_encoder_hidden_states"].shape[1])
    print(
        f"[ltx2-parity] trainium shape={height}x{width}x{num_frames} "
        f"audio_frames={audio_num_frames} tp={args.tp_degree}",
        flush=True,
    )
    t0 = time.perf_counter()
    app_kwargs = {
        "transformer_mode": args.transformer_mode,
        "text_seq_len": text_seq_len,
        "audio_num_frames": audio_num_frames,
    }
    if args.transformer_mode == "segmented" and args.segmented_block_load_mode != "process":
        app_kwargs["segmented_block_load_mode"] = args.segmented_block_load_mode
    if audio_text_seq_len != text_seq_len:
        app_kwargs["audio_text_seq_len"] = audio_text_seq_len
    process_segmented = (
        args.transformer_mode == "segmented"
        and args.segmented_block_load_mode == "process"
    )
    pipe = NovaPipeline.from_pretrained(
        args.model_dir,
        model_type="ltx_2",
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=args.dtype,
        height=height,
        width=width,
        num_frames=num_frames,
        compile_cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        force_compile=args.force_compile,
        skip_compile=args.skip_compile,
        load=not process_segmented,
        skip_warmup=args.skip_warmup,
        application_kwargs=app_kwargs,
    )
    print(f"[ltx2-parity] trainium load elapsed = {time.perf_counter() - t0:.3f}s", flush=True)
    print(f"[ltx2-parity] compiled_path = {pipe.compiled_path}", flush=True)
    set_compiled_model_path = getattr(pipe.app.transformer, "set_compiled_model_path", None)
    if set_compiled_model_path is not None:
        set_compiled_model_path(str(pipe.compiled_path))
    if process_segmented:
        pipe.app.transformer.block_load_mode = "process"
    bundle = LTX2DiTInputBundle(**inputs)
    t1 = time.perf_counter()
    with torch.no_grad():
        video, audio = _extract_pair(pipe(bundle))
    print(f"[ltx2-parity] trainium forward elapsed = {time.perf_counter() - t1:.3f}s", flush=True)
    return video.detach().cpu(), audio.detach().cpu(), pipe


def _to_device(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_floating_point(value):
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def _run_trace_reference(
    args: argparse.Namespace,
    pipe,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    _disable_xla_lazy_import()
    from nova.backends.trainium.ltx_2.transformer import _LTX2TransformerTraceModule

    device = torch.device(args.reference_device)
    dtype = args.reference_dtype
    model = _LTX2TransformerTraceModule(pipe.app.transformer.config).eval()
    _load_trace_state_dict(model, Path(args.model_dir) / "transformer", dtype)
    model.to(device=device, dtype=dtype)
    model_inputs = [_to_device(value, device=device, dtype=dtype) for value in inputs.values()]
    t0 = time.perf_counter()
    with torch.no_grad():
        video, audio = _extract_pair(model(*model_inputs))
    print(f"[ltx2-parity] trace reference elapsed = {time.perf_counter() - t0:.3f}s", flush=True)
    return video.detach().cpu(), audio.detach().cpu()


def _run_diffusers_reference(
    args: argparse.Namespace,
    pipe,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    _disable_xla_lazy_import()
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    device = torch.device(args.reference_device)
    dtype = args.reference_dtype
    model = LTX2VideoTransformer3DModel.from_pretrained(
        Path(args.model_dir) / "transformer",
        torch_dtype=dtype,
        local_files_only=True,
    ).eval()
    model.to(device=device, dtype=dtype)
    cfg = pipe.app.transformer.config
    ref_inputs = {
        key: _to_device(value, device=device, dtype=dtype)
        for key, value in inputs.items()
    }
    t0 = time.perf_counter()
    with torch.no_grad():
        video, audio = _extract_pair(
            model(
                hidden_states=ref_inputs["hidden_states"],
                audio_hidden_states=ref_inputs["audio_hidden_states"],
                encoder_hidden_states=ref_inputs["encoder_hidden_states"],
                audio_encoder_hidden_states=ref_inputs["audio_encoder_hidden_states"],
                timestep=ref_inputs["timestep"],
                audio_timestep=ref_inputs["timestep"],
                sigma=ref_inputs["sigma"],
                audio_sigma=ref_inputs["sigma"],
                encoder_attention_mask=ref_inputs["encoder_attention_mask"],
                audio_encoder_attention_mask=ref_inputs["audio_encoder_attention_mask"],
                num_frames=int(cfg.latent_num_frames),
                height=int(cfg.latent_height),
                width=int(cfg.latent_width),
                fps=float(getattr(cfg, "frame_rate", 24.0)),
                audio_num_frames=int(cfg.audio_num_frames),
                video_coords=ref_inputs["video_coords"],
                audio_coords=ref_inputs["audio_coords"],
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=bool(getattr(cfg, "use_cross_timestep", False)),
                return_dict=False,
            )
        )
    elapsed = time.perf_counter() - t0
    print(f"[ltx2-parity] diffusers reference elapsed = {elapsed:.3f}s", flush=True)
    return video.detach().cpu(), audio.detach().cpu()


def _run_segmented_cpu_reference(
    args: argparse.Namespace,
    pipe,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.transformer_mode != "segmented":
        raise ValueError("reference-mode=segmented-cpu requires --transformer-mode segmented")
    segmented = pipe.app.transformer
    model = segmented._load_cpu_transformer()
    device = torch.device(args.reference_device)
    dtype = args.reference_dtype
    model.to(device=device, dtype=dtype)
    cfg = segmented.config
    ref_inputs = {
        key: _to_device(value, device=device, dtype=dtype)
        for key, value in inputs.items()
    }
    t0 = time.perf_counter()
    with torch.no_grad():
        video, audio = _extract_pair(
            model(
                hidden_states=ref_inputs["hidden_states"],
                audio_hidden_states=ref_inputs["audio_hidden_states"],
                encoder_hidden_states=ref_inputs["encoder_hidden_states"],
                audio_encoder_hidden_states=ref_inputs["audio_encoder_hidden_states"],
                timestep=ref_inputs["timestep"],
                audio_timestep=ref_inputs["timestep"],
                sigma=ref_inputs["sigma"],
                audio_sigma=ref_inputs["sigma"],
                encoder_attention_mask=ref_inputs["encoder_attention_mask"],
                audio_encoder_attention_mask=ref_inputs["audio_encoder_attention_mask"],
                num_frames=int(cfg.latent_num_frames),
                height=int(cfg.latent_height),
                width=int(cfg.latent_width),
                fps=float(getattr(cfg, "frame_rate", 24.0)),
                audio_num_frames=int(cfg.audio_num_frames),
                video_coords=ref_inputs["video_coords"],
                audio_coords=ref_inputs["audio_coords"],
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=bool(getattr(cfg, "use_cross_timestep", False)),
                return_dict=False,
            )
        )
    elapsed = time.perf_counter() - t0
    print(f"[ltx2-parity] segmented CPU reference elapsed = {elapsed:.3f}s", flush=True)
    return video.detach().cpu(), audio.detach().cpu()


def _save_pair(path: str, video: torch.Tensor, audio: torch.Tensor) -> None:
    torch.save({"video": video, "audio": audio}, path)
    print(f"[ltx2-parity] tensor pair -> {path}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    _disable_xla_lazy_import()

    bundle_path = Path(args.bundle)
    meta = _bundle_meta(bundle_path)
    inputs = _load_bundle(bundle_path, args.step_index, args.dtype)
    print(f"[ltx2-parity] bundle = {bundle_path}", flush=True)
    print(f"[ltx2-parity] prompt = {meta.get('prompt')}", flush=True)
    print(f"[ltx2-parity] step_index = {args.step_index}", flush=True)

    trainium_video, trainium_audio, pipe = _run_trainium(args, meta, inputs)
    if args.reference_mode == "trace":
        reference_video, reference_audio = _run_trace_reference(args, pipe, inputs)
    elif args.reference_mode == "diffusers":
        reference_video, reference_audio = _run_diffusers_reference(args, pipe, inputs)
    else:
        reference_video, reference_audio = _run_segmented_cpu_reference(args, pipe, inputs)

    if trainium_video.shape != reference_video.shape:
        raise RuntimeError(
            f"video shape mismatch: trainium={tuple(trainium_video.shape)} "
            f"reference={tuple(reference_video.shape)}"
        )
    if trainium_audio.shape != reference_audio.shape:
        raise RuntimeError(
            f"audio shape mismatch: trainium={tuple(trainium_audio.shape)} "
            f"reference={tuple(reference_audio.shape)}"
        )

    video_diff = (trainium_video.float() - reference_video.float()).abs()
    audio_diff = (trainium_audio.float() - reference_audio.float()).abs()
    metrics = {
        "bundle": str(bundle_path),
        "compiled_path": str(pipe.compiled_path),
        "model_dir": args.model_dir,
        "transformer_mode": args.transformer_mode,
        "reference_mode": args.reference_mode,
        "step_index": args.step_index,
        "height": pipe.shape["height"],
        "width": pipe.shape["width"],
        "num_frames": pipe.shape["num_frames"],
        "text_seq_len": int(pipe.app.text_seq_len),
        "audio_text_seq_len": int(pipe.app.audio_text_seq_len),
        "audio_num_frames": int(pipe.app.audio_num_frames),
        "tp_degree": args.tp_degree,
        "video_cosine": _cosine(trainium_video, reference_video),
        "audio_cosine": _cosine(trainium_audio, reference_audio),
        "video_max_abs": float(video_diff.max()),
        "audio_max_abs": float(audio_diff.max()),
        "video_mean_abs": float(video_diff.mean()),
        "audio_mean_abs": float(audio_diff.mean()),
        "trainium_video_shape": list(trainium_video.shape),
        "trainium_audio_shape": list(trainium_audio.shape),
        "reference_video_shape": list(reference_video.shape),
        "reference_audio_shape": list(reference_audio.shape),
        "trainium_video_mean": float(trainium_video.float().mean()),
        "trainium_audio_mean": float(trainium_audio.float().mean()),
        "reference_video_mean": float(reference_video.float().mean()),
        "reference_audio_mean": float(reference_audio.float().mean()),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[ltx2-parity] metrics -> {metrics_path}", flush=True)
    if args.save_trainium:
        _save_pair(args.save_trainium, trainium_video, trainium_audio)
    if args.save_reference:
        _save_pair(args.save_reference, reference_video, reference_audio)

    pass_gate = (
        metrics["video_cosine"] >= args.min_video_cosine
        and metrics["audio_cosine"] >= args.min_audio_cosine
    )
    if args.max_video_mean_abs is not None:
        pass_gate = pass_gate and metrics["video_mean_abs"] <= args.max_video_mean_abs
    if args.max_audio_mean_abs is not None:
        pass_gate = pass_gate and metrics["audio_mean_abs"] <= args.max_audio_mean_abs
    print(f"[ltx2-parity] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
