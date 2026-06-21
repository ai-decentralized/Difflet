#!/usr/bin/env python3
"""LTX-2 scheduler trajectory parity for Trainium segmented runtime.

This compares the Difflet host scheduler loop step-by-step from the same cached
DiT input bundle. The Trainium side can use the segmented process-isolated block
runtime; the reference side uses the same segmented front/back-end stitching
with the upstream CPU transformer blocks.
"""

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
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Local LTX-2 snapshot dir")
    parser.add_argument("--bundle", required=True, help="Cached LTX-2 DiT inputs safetensors")
    parser.add_argument("--cache-dir", default=".difflet-cache/ltx_2_trajectory_parity")
    parser.add_argument("--compiled-model-path", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--audio-num-frames", type=int, default=None)
    parser.add_argument("--text-seq-len", type=int, default=None)
    parser.add_argument("--audio-text-seq-len", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--audio-guidance-scale", type=float, default=1.0)
    parser.add_argument("--transformer-mode", choices=("segmented",), default="segmented")
    parser.add_argument(
        "--segmented-block-load-mode",
        choices=("streaming", "process"),
        default="process",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--min-video-cosine", type=float, default=0.999)
    parser.add_argument("--min-audio-cosine", type=float, default=0.999)
    parser.add_argument("--metrics-out", default=None)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_meta(bundle_path: Path) -> dict[str, Any]:
    meta_path = Path(str(bundle_path) + ".meta.json")
    return _load_json(meta_path) if meta_path.exists() else {}


def _meta_or_arg(meta: dict[str, Any], args: argparse.Namespace, name: str) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    if name in meta:
        return int(meta[name])
    raise ValueError(f"--{name.replace('_', '-')} is required when bundle meta is missing")


def _float_meta_or_arg(
    meta: dict[str, Any],
    args: argparse.Namespace,
    name: str,
    default: float,
) -> float:
    value = getattr(args, name)
    if value is not None:
        return float(value)
    if name in meta:
        return float(meta[name])
    return float(default)


def _load_bundle(bundle_path: Path, dtype: torch.dtype):
    from difflet.models.ltx_2.application import LTX2DiTInputBundle

    tensors = load_safetensors_file(str(bundle_path), device="cpu")
    required = {
        "latents_init",
        "audio_latents_init",
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
    batch_size = int(tensors["latents_init"].shape[0])
    timestep = torch.zeros((batch_size,), dtype=dtype)
    return LTX2DiTInputBundle(
        hidden_states=tensors["latents_init"].to(dtype=dtype).contiguous(),
        audio_hidden_states=tensors["audio_latents_init"].to(dtype=dtype).contiguous(),
        encoder_hidden_states=tensors["encoder_hidden_states"].to(dtype=dtype).contiguous(),
        audio_encoder_hidden_states=tensors["audio_encoder_hidden_states"]
        .to(dtype=dtype)
        .contiguous(),
        timestep=timestep,
        sigma=timestep.clone(),
        encoder_attention_mask=tensors["encoder_attention_mask"].to(torch.bool).contiguous(),
        audio_encoder_attention_mask=tensors["audio_encoder_attention_mask"]
        .to(torch.bool)
        .contiguous(),
        video_coords=tensors["video_coords"].to(dtype=torch.float32).contiguous(),
        audio_coords=tensors["audio_coords"].to(dtype=torch.float32).contiguous(),
    )


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


class _SegmentedCPUAdapter:
    supports_ltx_2_extra_kwargs = False

    def __init__(self, segmented: Any, dtype: torch.dtype) -> None:
        self.segmented = segmented
        self.dtype = dtype

    def __call__(self, bundle):
        cfg = self.segmented.config
        model = self.segmented._load_cpu_transformer()
        with torch.no_grad():
            return model(
                hidden_states=bundle.hidden_states.to(dtype=self.dtype),
                audio_hidden_states=bundle.audio_hidden_states.to(dtype=self.dtype),
                encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=self.dtype),
                audio_encoder_hidden_states=bundle.audio_encoder_hidden_states.to(
                    dtype=self.dtype
                ),
                timestep=bundle.timestep.to(dtype=self.dtype),
                audio_timestep=bundle.timestep.to(dtype=self.dtype),
                sigma=bundle.sigma.to(dtype=self.dtype),
                audio_sigma=bundle.sigma.to(dtype=self.dtype),
                encoder_attention_mask=bundle.encoder_attention_mask,
                audio_encoder_attention_mask=bundle.audio_encoder_attention_mask,
                num_frames=int(cfg.latent_num_frames),
                height=int(cfg.latent_height),
                width=int(cfg.latent_width),
                fps=float(getattr(cfg, "frame_rate", 24.0)),
                audio_num_frames=int(cfg.audio_num_frames),
                video_coords=bundle.video_coords,
                audio_coords=bundle.audio_coords,
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=bool(getattr(cfg, "use_cross_timestep", False)),
                return_dict=False,
            )


def _make_pipe(args: argparse.Namespace, meta: dict[str, Any]):
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    from difflet import DiffletParallelConfig, DiffletPipeline

    height = _meta_or_arg(meta, args, "height")
    width = _meta_or_arg(meta, args, "width")
    num_frames = _meta_or_arg(meta, args, "num_frames")
    audio_num_frames = _meta_or_arg(meta, args, "audio_num_frames")
    app_kwargs = {
        "transformer_mode": args.transformer_mode,
        "text_seq_len": int(
            args.text_seq_len
            if args.text_seq_len is not None
            else meta.get("tensor_shapes", {})
            .get("encoder_hidden_states", [None, 1024])[1]
        ),
        "audio_num_frames": audio_num_frames,
        "frame_rate": _float_meta_or_arg(meta, args, "frame_rate", 24.0),
    }
    if args.audio_text_seq_len is not None:
        app_kwargs["audio_text_seq_len"] = int(args.audio_text_seq_len)
    elif "tensor_shapes" in meta and "audio_encoder_hidden_states" in meta["tensor_shapes"]:
        app_kwargs["audio_text_seq_len"] = int(meta["tensor_shapes"]["audio_encoder_hidden_states"][1])
    if args.segmented_block_load_mode != "process":
        app_kwargs["segmented_block_load_mode"] = args.segmented_block_load_mode
    process_segmented = args.segmented_block_load_mode == "process"
    pipe = DiffletPipeline.from_pretrained(
        args.model_dir,
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
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
    compiled_model_path = Path(args.compiled_model_path) if args.compiled_model_path else pipe.compiled_path
    set_compiled_model_path = getattr(pipe.app.transformer, "set_compiled_model_path", None)
    if set_compiled_model_path is not None:
        set_compiled_model_path(str(compiled_model_path))
    if args.transformer_mode == "segmented":
        pipe.app.transformer.block_load_mode = args.segmented_block_load_mode
    return pipe, compiled_model_path


def _run_trainium(args: argparse.Namespace, pipe, bundle):
    t0 = time.perf_counter()
    with torch.no_grad():
        output = pipe(
            bundle=bundle,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            audio_guidance_scale=args.audio_guidance_scale,
            output_type="latent",
            return_trajectory=True,
        )
    print(f"[ltx2-trajectory] trainium elapsed = {time.perf_counter() - t0:.3f}s", flush=True)
    return output


def _run_reference(args: argparse.Namespace, pipe, bundle, meta: dict[str, Any]):
    from difflet.models.ltx_2.pipeline import LTX2Orchestrator

    adapter = _SegmentedCPUAdapter(pipe.app.transformer, args.dtype)
    t0 = time.perf_counter()
    ref = LTX2Orchestrator(
        model_path=args.model_dir,
        transformer=adapter,
        dtype=args.dtype,
        height=_meta_or_arg(meta, args, "height"),
        width=_meta_or_arg(meta, args, "width"),
        num_frames=_meta_or_arg(meta, args, "num_frames"),
        text_seq_len=int(pipe.app.text_seq_len),
        audio_text_seq_len=int(pipe.app.audio_text_seq_len),
        audio_num_frames=int(pipe.app.audio_num_frames),
        frame_rate=float(pipe.app.frame_rate),
    )
    with torch.no_grad():
        output = ref(
            bundle=bundle,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            audio_guidance_scale=args.audio_guidance_scale,
            output_type="latent",
            return_trajectory=True,
        )
    print(f"[ltx2-trajectory] reference elapsed = {time.perf_counter() - t0:.3f}s", flush=True)
    return output


def _trajectory_metrics(trainium, reference) -> list[dict[str, Any]]:
    if trainium.trajectory is None or reference.trajectory is None:
        raise RuntimeError("trajectory parity requires both outputs to include trajectories")
    if len(trainium.trajectory) != len(reference.trajectory):
        raise RuntimeError(
            f"trajectory length mismatch: trainium={len(trainium.trajectory)} "
            f"reference={len(reference.trajectory)}"
        )
    metrics = []
    for index, ((tv, ta), (rv, ra)) in enumerate(zip(trainium.trajectory, reference.trajectory)):
        if tv.shape != rv.shape:
            raise RuntimeError(f"video trajectory shape mismatch at {index}: {tv.shape} != {rv.shape}")
        if ta.shape != ra.shape:
            raise RuntimeError(f"audio trajectory shape mismatch at {index}: {ta.shape} != {ra.shape}")
        video_diff = (tv.float() - rv.float()).abs()
        audio_diff = (ta.float() - ra.float()).abs()
        metrics.append(
            {
                "index": index,
                "video_cosine": _cosine(tv, rv),
                "audio_cosine": _cosine(ta, ra),
                "video_max_abs": float(video_diff.max()),
                "audio_max_abs": float(audio_diff.max()),
                "video_mean_abs": float(video_diff.mean()),
                "audio_mean_abs": float(audio_diff.mean()),
                "video_shape": list(tv.shape),
                "audio_shape": list(ta.shape),
            }
        )
    return metrics


def main() -> int:
    args = build_parser().parse_args()
    _disable_xla_lazy_import()
    bundle_path = Path(args.bundle)
    meta = _bundle_meta(bundle_path)
    bundle = _load_bundle(bundle_path, args.dtype)
    pipe, compiled_model_path = _make_pipe(args, meta)
    print(f"[ltx2-trajectory] bundle = {bundle_path}", flush=True)
    print(f"[ltx2-trajectory] compiled_model_path = {compiled_model_path}", flush=True)
    trainium = _run_trainium(args, pipe, bundle)
    reference = _run_reference(args, pipe, bundle, meta)
    per_step = _trajectory_metrics(trainium, reference)
    min_video_cosine = min(item["video_cosine"] for item in per_step)
    min_audio_cosine = min(item["audio_cosine"] for item in per_step)
    metrics = {
        "bundle": str(bundle_path),
        "compiled_model_path": str(compiled_model_path),
        "model_dir": args.model_dir,
        "dtype": str(args.dtype),
        "num_inference_steps": args.num_inference_steps,
        "effective_steps": len(per_step) - 1,
        "trajectory_length": len(per_step),
        "transformer_mode": args.transformer_mode,
        "segmented_block_load_mode": args.segmented_block_load_mode,
        "min_video_cosine": min_video_cosine,
        "min_audio_cosine": min_audio_cosine,
        "per_step": per_step,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[ltx2-trajectory] metrics -> {metrics_path}", flush=True)
    pass_gate = (
        min_video_cosine >= args.min_video_cosine
        and min_audio_cosine >= args.min_audio_cosine
    )
    print(f"[ltx2-trajectory] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
