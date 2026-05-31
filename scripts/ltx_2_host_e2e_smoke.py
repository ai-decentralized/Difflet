#!/usr/bin/env python3
"""Run a local LTX-2 host-orchestrated Nova smoke.

This is the end-to-end closure entrypoint for a local diffusers-format
``Lightricks/LTX-2`` snapshot. It exercises Nova's prompt -> connector ->
Trainium transformer -> optional VAE/audio decode path. It intentionally does
not download weights; use ``NovaPipeline.from_pretrained`` or HF tooling to
materialize the snapshot first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    if Path(sys.executable) == NEURON_PYTHON or not NEURON_PYTHON.exists():
        return
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        env = os.environ.copy()
        env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("NOVA_LTX_2_MODEL_DIR", ""),
        help="Local LTX-2 snapshot dir. Env: NOVA_LTX_2_MODEL_DIR.",
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
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
    parser.add_argument("--cache-dir", default=".nova-cache/ltx_2_host_e2e")
    parser.add_argument(
        "--compiled-model-path",
        default=None,
        help=(
            "Optional compiled artifact root to use at runtime. This is useful "
            "for segmented process mode because host-only app kwargs do not "
            "change the block artifact."
        ),
    )
    parser.add_argument("--teacache-calibration", default=None,
                        help="path to a TeaCache calibration JSON; enables adaptive step skipping")
    parser.add_argument("--output-type", choices=("latent", "pt"), default="latent")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--audio-guidance-scale", type=float, default=None)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--audio-guidance-rescale", type=float, default=None)
    parser.add_argument("--decode-timestep", type=float, default=0.0)
    parser.add_argument("--decode-noise-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host-device", default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--metrics-out", default="/tmp/nova_ltx_2_host_e2e_smoke.json")
    parser.add_argument(
        "--save-tensors",
        default=None,
        help="Optional .pt path for raw output frames/audio tensors.",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="Optional .gif path for the first decoded video sample.",
    )
    parser.add_argument(
        "--save-audio",
        default=None,
        help="Optional .wav path for the first decoded audio sample.",
    )
    return parser


def _has_any(parent: Path, names: tuple[str, ...]) -> bool:
    return any((parent / name).exists() for name in names)


def _has_weights(parent: Path) -> bool:
    return _has_any(
        parent,
        (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.safetensors.index.json",
            "model.safetensors",
            "model.safetensors.index.json",
        ),
    )


def _snapshot_preflight(model_dir: Path, *, require_decode: bool) -> tuple[bool, list[str]]:
    required = [
        ("snapshot dir", model_dir.exists()),
        ("model_index.json", (model_dir / "model_index.json").exists()),
        ("transformer/config.json", (model_dir / "transformer" / "config.json").exists()),
        ("transformer weights", _has_weights(model_dir / "transformer")),
        ("scheduler/scheduler_config.json", (model_dir / "scheduler" / "scheduler_config.json").exists()),
        ("text_encoder/config.json", (model_dir / "text_encoder" / "config.json").exists()),
        ("text_encoder weights", _has_weights(model_dir / "text_encoder")),
        (
            "tokenizer files",
            _has_any(
                model_dir / "tokenizer",
                ("tokenizer.json", "tokenizer_config.json", "tokenizer.model"),
            ),
        ),
        ("connectors/config.json", (model_dir / "connectors" / "config.json").exists()),
        ("connectors weights", _has_weights(model_dir / "connectors")),
    ]
    decode = [
        ("vae/config.json", (model_dir / "vae" / "config.json").exists()),
        ("vae weights", _has_weights(model_dir / "vae")),
        ("audio_vae/config.json", (model_dir / "audio_vae" / "config.json").exists()),
        ("audio_vae weights", _has_weights(model_dir / "audio_vae")),
        ("vocoder/config.json", (model_dir / "vocoder" / "config.json").exists()),
        ("vocoder weights", _has_weights(model_dir / "vocoder")),
    ]
    checks = required + decode
    lines = [f"{'OK' if passed else 'MISSING'} {name}" for name, passed in checks]
    required_ok = all(passed for _name, passed in required)
    decode_ok = all(passed for _name, passed in decode)
    return required_ok and (decode_ok or not require_decode), lines


def _default_prompt() -> list[str]:
    return ["a close-up cinematic shot of a glass teapot on a wooden table"]


def _shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def _tensor_stats(value: Any) -> dict[str, Any] | None:
    if value is None or not hasattr(value, "detach"):
        return None
    import torch

    tensor = value.detach()
    finite = torch.isfinite(tensor)
    stats: dict[str, Any] = {
        "finite": bool(finite.all().item()),
        "finite_count": int(finite.sum().item()),
        "numel": int(tensor.numel()),
    }
    if bool(finite.any().item()):
        finite_tensor = tensor[finite].float()
        stats.update(
            {
                "min": float(finite_tensor.min().item()),
                "max": float(finite_tensor.max().item()),
                "mean": float(finite_tensor.mean().item()),
            }
        )
    return stats


def _save_tensor_outputs(path: str, output: Any) -> None:
    import torch

    tensor_path = Path(path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frames": output.frames.detach().cpu() if hasattr(output.frames, "detach") else output.frames,
            "audio": output.audio.detach().cpu() if hasattr(output.audio, "detach") else output.audio,
            "latents": (
                output.latents.detach().cpu() if hasattr(output.latents, "detach") else output.latents
            ),
            "audio_latents": (
                output.audio_latents.detach().cpu()
                if hasattr(output.audio_latents, "detach")
                else output.audio_latents
            ),
        },
        tensor_path,
    )


def _save_video_gif(path: str, frames: Any, *, fps: float) -> None:
    if not str(path).lower().endswith(".gif"):
        raise ValueError("LTX-2 --save-video currently writes GIF previews; use a .gif path.")
    from PIL import Image

    tensor = frames.detach().cpu().float()
    if tensor.ndim != 5:
        raise ValueError(f"LTX-2 --save-video expected frames shape (B,T,C,H,W), got {tuple(tensor.shape)}")
    tensor = tensor[0]
    if tensor.shape[1] not in {1, 3, 4}:
        raise ValueError(f"LTX-2 --save-video expected channel dim at index 1, got {tuple(tensor.shape)}")
    tensor = tensor.clamp(0.0, 1.0).permute(0, 2, 3, 1)
    array = (tensor.numpy() * 255.0).round().astype("uint8")
    pil_frames = [Image.fromarray(frame.squeeze(-1) if frame.shape[-1] == 1 else frame) for frame in array]
    video_path = Path(path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(round(1000.0 / max(float(fps), 1.0))), 1)
    pil_frames[0].save(
        video_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _save_audio_wav(path: str, audio: Any, *, sample_rate: int = 16000) -> None:
    import numpy as np
    from scipy.io import wavfile

    tensor = audio.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"LTX-2 --save-audio expected audio shape (B,C,N), got {tuple(tensor.shape)}")
    tensor = tensor[0].transpose(0, 1).clamp(-1.0, 1.0)
    array = (tensor.numpy() * 32767.0).round().astype(np.int16)
    audio_path = Path(path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(audio_path), int(sample_rate), array)


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()
    if not args.model_dir:
        print("[ltx2-e2e] --model-dir or NOVA_LTX_2_MODEL_DIR is required", file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir).expanduser().resolve()
    ok, lines = _snapshot_preflight(model_dir, require_decode=args.output_type == "pt")
    print(f"[ltx2-e2e] model_dir = {model_dir}")
    for line in lines:
        print(f"[ltx2-e2e] preflight {line}")
    if not ok:
        print("[ltx2-e2e] preflight FAIL: local LTX-2 snapshot is incomplete", file=sys.stderr)
        return 2
    if args.num_inference_steps < 2:
        print(
            "[ltx2-e2e] preflight FAIL: LTX-2 diffusers scheduler requires "
            "--num-inference-steps >= 2; one-step schedules produce non-finite timesteps.",
            file=sys.stderr,
        )
        return 2
    if args.preflight_only:
        print("[ltx2-e2e] preflight PASS")
        return 0

    import torch

    from nova import NovaParallelConfig, NovaPipeline

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    prompt = args.prompt or _default_prompt()

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="ltx_2",
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=args.cache_dir,
        force_compile=args.force_compile,
        skip_compile=args.skip_compile,
        load=not args.skip_load
        and not (
            args.transformer_mode == "segmented"
            and args.segmented_block_load_mode == "process"
        ),
        skip_warmup=args.skip_warmup,
        application_kwargs={
            "enable_host_pipeline": True,
            "enable_decode_components": args.output_type == "pt",
            "host_device": args.host_device,
            "text_seq_len": args.text_seq_len,
            "frame_rate": args.frame_rate,
            "transformer_mode": args.transformer_mode,
            "teacache_calibration_path": args.teacache_calibration,
        },
    )
    compiled_model_path = Path(args.compiled_model_path) if args.compiled_model_path else pipe.compiled_path
    set_compiled_model_path = getattr(pipe.app.transformer, "set_compiled_model_path", None)
    if set_compiled_model_path is not None:
        set_compiled_model_path(str(compiled_model_path))
    if args.transformer_mode == "segmented":
        pipe.app.transformer.block_load_mode = args.segmented_block_load_mode

    import time as _time
    _t0 = _time.monotonic()
    output = pipe(
        prompt=prompt[0] if len(prompt) == 1 else prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        audio_guidance_scale=args.audio_guidance_scale,
        guidance_rescale=args.guidance_rescale,
        audio_guidance_rescale=args.audio_guidance_rescale,
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        output_type=args.output_type,
        generator=generator,
    )
    print(f"[ltx2] forward elapsed = {_time.monotonic() - _t0:.3f}s", flush=True)

    metrics = {
        "model_dir": str(model_dir),
        "prompt_count": len(prompt),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "num_inference_steps": args.num_inference_steps,
        "output_type": args.output_type,
        "transformer_mode": args.transformer_mode,
        "segmented_block_load_mode": args.segmented_block_load_mode,
        "compiled_model_path": str(compiled_model_path),
        "frames_shape": _shape(output.frames),
        "audio_shape": _shape(output.audio),
        "latents_shape": _shape(output.latents),
        "audio_latents_shape": _shape(output.audio_latents),
        "frames_dtype": str(getattr(output.frames, "dtype", None)),
        "audio_dtype": str(getattr(output.audio, "dtype", None)),
        "latents_dtype": str(getattr(output.latents, "dtype", None)),
        "audio_latents_dtype": str(getattr(output.audio_latents, "dtype", None)),
        "frames_stats": _tensor_stats(output.frames),
        "audio_stats": _tensor_stats(output.audio),
        "latents_stats": _tensor_stats(output.latents),
        "audio_latents_stats": _tensor_stats(output.audio_latents),
        "saved_tensors": str(Path(args.save_tensors)) if args.save_tensors else None,
        "saved_video": str(Path(args.save_video)) if args.save_video else None,
        "saved_audio": str(Path(args.save_audio)) if args.save_audio else None,
    }
    if args.save_tensors:
        _save_tensor_outputs(args.save_tensors, output)
        print(f"[ltx2-e2e] tensors -> {args.save_tensors}")
    if args.save_video:
        _save_video_gif(args.save_video, output.frames, fps=args.frame_rate)
        print(f"[ltx2-e2e] video -> {args.save_video}")
    if args.save_audio:
        _save_audio_wav(args.save_audio, output.audio)
        print(f"[ltx2-e2e] audio -> {args.save_audio}")
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ltx2-e2e] wrote {metrics_path}")
    print(
        "[ltx2-e2e] PASS "
        f"frames_shape={metrics['frames_shape']} audio_shape={metrics['audio_shape']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
