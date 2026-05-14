#!/usr/bin/env python3
"""Cache LTX-2 dual-stream transformer-boundary input tensors.

This helper runs the host-side LTX-2 text encoder, connector stack, scheduler
setup, and latent initialization. It writes the fixed tensors consumed by the
Nova LTX-2 Trainium transformer boundary. Video VAE, audio VAE, and vocoder
decode remain outside this artifact.
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
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from nova.models.ltx_2.pipeline import (  # noqa: E402
    ltx_2_scheduler_mu,
    make_ltx_2_audio_coords,
    make_ltx_2_video_coords,
)


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def _prompt_arg(prompts: list[str]) -> str | list[str]:
    return prompts[0] if len(prompts) == 1 else prompts


def _disable_xla_lazy_import() -> None:
    # CPU-side caching should not initialize PJRT just because torch_xla is
    # present in the Neuron venv.
    import diffusers.utils.import_utils as import_utils

    import_utils._torch_xla_available = False


def _load_host_pipeline(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    revision: str | None,
    local_files_only: bool,
):
    _disable_xla_lazy_import()
    from diffusers import LTX2Pipeline

    load_kwargs = _host_pipeline_load_kwargs(
        dtype=dtype,
        revision=revision,
        local_files_only=local_files_only,
    )
    pipe = LTX2Pipeline.from_pretrained(model_id, **load_kwargs)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _host_pipeline_load_kwargs(
    *,
    dtype: torch.dtype,
    revision: str | None,
    local_files_only: bool,
) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "torch_dtype": dtype,
        "transformer": None,
        "vae": None,
        "audio_vae": None,
        "vocoder": None,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    return load_kwargs


def _latent_dims(*, height: int, width: int, num_frames: int) -> tuple[int, int, int]:
    return (num_frames - 1) // 8 + 1, height // 32, width // 32


def _make_cache_coords(
    *,
    batch_size: int,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    audio_num_frames: int,
    device: torch.device | str,
    frame_rate: float,
    audio_sampling_rate: int,
    audio_hop_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_coords = make_ltx_2_video_coords(
        batch_size=batch_size,
        num_frames=latent_num_frames,
        height=latent_height,
        width=latent_width,
        device=device,
        fps=frame_rate,
    )
    audio_coords = make_ltx_2_audio_coords(
        batch_size=batch_size,
        audio_num_frames=audio_num_frames,
        device=device,
        sampling_rate=audio_sampling_rate,
        hop_length=audio_hop_length,
    )
    return video_coords, audio_coords


def cache_dit_inputs(args: argparse.Namespace) -> None:
    _disable_xla_lazy_import()
    from diffusers.pipelines.ltx2.pipeline_ltx2 import retrieve_timesteps

    device = torch.device(args.device)
    prompt = _prompt_arg(args.prompt)
    pipe = _load_host_pipeline(
        model_id=args.model_id,
        dtype=args.model_dtype,
        device=device,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=args.text_seq_len,
            device=device,
            dtype=args.model_dtype,
        )
        (
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            connector_attention_mask,
        ) = pipe.connectors(
            prompt_embeds,
            prompt_attention_mask,
            padding_side=args.tokenizer_padding_side,
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        batch_size = prompt_embeds.shape[0]
        latents = pipe.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=args.video_latent_channels,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            noise_scale=args.noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        duration_s = args.num_frames / args.frame_rate
        audio_latents_per_second = args.audio_sampling_rate / args.audio_hop_length / 4.0
        audio_num_frames = (
            args.audio_num_frames
            if args.audio_num_frames is not None
            else round(duration_s * audio_latents_per_second)
        )
        audio_latents = pipe.prepare_audio_latents(
            batch_size=batch_size,
            num_channels_latents=args.audio_latent_channels,
            audio_latent_length=audio_num_frames,
            num_mel_bins=args.audio_mel_bins,
            noise_scale=args.noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        sigmas = (
            np.linspace(1.0, 1.0 / args.num_inference_steps, args.num_inference_steps)
            if args.sigmas is None
            else args.sigmas
        )
        mu = ltx_2_scheduler_mu(pipe.scheduler.config)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            args.num_inference_steps,
            "cpu",
            sigmas=sigmas,
            mu=mu,
        )

        latent_num_frames, latent_height, latent_width = _latent_dims(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        video_coords, audio_coords = _make_cache_coords(
            batch_size=batch_size,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            audio_num_frames=audio_num_frames,
            device=device,
            frame_rate=args.frame_rate,
            audio_sampling_rate=args.audio_sampling_rate,
            audio_hop_length=args.audio_hop_length,
        )

    tensors = {
        "latents_init": latents.to(dtype=args.output_dtype, device="cpu").contiguous(),
        "audio_latents_init": audio_latents.to(dtype=args.output_dtype, device="cpu").contiguous(),
        "timesteps": timesteps.to(dtype=args.output_dtype, device="cpu").contiguous(),
        "encoder_hidden_states": connector_prompt_embeds.to(
            dtype=args.output_dtype,
            device="cpu",
        ).contiguous(),
        "audio_encoder_hidden_states": connector_audio_prompt_embeds.to(
            dtype=args.output_dtype,
            device="cpu",
        ).contiguous(),
        "encoder_attention_mask": connector_attention_mask.to(
            dtype=torch.bool,
            device="cpu",
        ).contiguous(),
        "audio_encoder_attention_mask": connector_attention_mask.to(
            dtype=torch.bool,
            device="cpu",
        ).contiguous(),
        "video_coords": video_coords.to(dtype=torch.float32, device="cpu").contiguous(),
        "audio_coords": audio_coords.to(dtype=torch.float32, device="cpu").contiguous(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"format": "nova-ltx-2-dit-inputs-v1"})

    meta = {
        "schema": "nova-ltx-2-dit-inputs-v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "audio_num_frames": audio_num_frames,
        "num_inference_steps": args.num_inference_steps,
        "video_latent_channels": args.video_latent_channels,
        "audio_latent_channels": args.audio_latent_channels,
        "audio_mel_bins": args.audio_mel_bins,
        "model_dtype": str(args.model_dtype),
        "output_dtype": str(args.output_dtype),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Lightricks/LTX-2")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--video-latent-channels", type=int, default=128)
    parser.add_argument("--audio-latent-channels", type=int, default=8)
    parser.add_argument("--audio-num-frames", type=int, default=None)
    parser.add_argument("--audio-mel-bins", type=int, default=64)
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)
    parser.add_argument("--audio-hop-length", type=int, default=160)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument("--tokenizer-padding-side", choices=("left", "right"), default="left")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigmas", type=float, nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--output-dtype", type=_parse_dtype, default=torch.bfloat16)
    return parser


def main() -> None:
    cache_dit_inputs(build_parser().parse_args())


if __name__ == "__main__":
    main()
