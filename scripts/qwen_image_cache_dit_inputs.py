#!/usr/bin/env python3
"""Cache Qwen-Image DiT input tensors from the HF text/scheduler path.

The M4a Trainium boundary consumes packed latents plus Qwen2.5-VL prompt
embeddings. This helper produces the host-side artifact used by transformer
parity and Trainium smoke tests once a full Qwen-Image snapshot is available.
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


def ensure_runtime_python() -> None:
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{Path(__file__).resolve().parents[1]}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


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


def pad_or_truncate_prompt_embeds(
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    *,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-length prompt embeddings for the Trainium transformer graph."""

    batch_size, seq_len, hidden_size = prompt_embeds.shape
    if prompt_embeds_mask is None:
        prompt_embeds_mask = torch.ones(
            batch_size,
            seq_len,
            dtype=torch.bool,
            device=prompt_embeds.device,
        )
    else:
        prompt_embeds_mask = prompt_embeds_mask.to(device=prompt_embeds.device, dtype=torch.bool)

    if seq_len > max_sequence_length:
        return (
            prompt_embeds[:, :max_sequence_length, :].contiguous(),
            prompt_embeds_mask[:, :max_sequence_length].contiguous(),
        )
    if seq_len == max_sequence_length:
        return prompt_embeds.contiguous(), prompt_embeds_mask.contiguous()

    pad_len = max_sequence_length - seq_len
    embed_pad = torch.zeros(
        batch_size,
        pad_len,
        hidden_size,
        dtype=prompt_embeds.dtype,
        device=prompt_embeds.device,
    )
    mask_pad = torch.zeros(
        batch_size,
        pad_len,
        dtype=torch.bool,
        device=prompt_embeds.device,
    )
    return (
        torch.cat([prompt_embeds, embed_pad], dim=1).contiguous(),
        torch.cat([prompt_embeds_mask, mask_pad], dim=1).contiguous(),
    )


def normalize_prompt_embeds(
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    *,
    max_sequence_length: int,
    pad_to_max_sequence_length: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pad_to_max_sequence_length:
        return pad_or_truncate_prompt_embeds(
            prompt_embeds,
            prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )
    if prompt_embeds_mask is None:
        prompt_embeds_mask = torch.ones(
            prompt_embeds.shape[:2],
            dtype=torch.bool,
            device=prompt_embeds.device,
        )
    else:
        prompt_embeds_mask = prompt_embeds_mask.to(device=prompt_embeds.device, dtype=torch.bool)
    return prompt_embeds.contiguous(), prompt_embeds_mask.contiguous()


def _load_text_pipeline(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    revision: str | None,
    local_files_only: bool,
):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
    from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        **load_kwargs,
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        **load_kwargs,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        **load_kwargs,
    )
    pipe = QwenImagePipeline(
        scheduler=scheduler,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def cache_dit_inputs(args: argparse.Namespace) -> None:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
        calculate_shift,
        retrieve_timesteps,
    )

    device = torch.device(args.device)
    prompt = _prompt_arg(args.prompt)
    model_dtype = args.model_dtype
    output_dtype = args.output_dtype
    pipe = _load_text_pipeline(
        model_id=args.model_id,
        dtype=model_dtype,
        device=device,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=args.text_seq_len,
        )
        prompt_embeds, prompt_embeds_mask = normalize_prompt_embeds(
            prompt_embeds,
            prompt_embeds_mask,
            max_sequence_length=args.text_seq_len,
            pad_to_max_sequence_length=args.pad_to_text_seq_len,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        batch_size = prompt_embeds.shape[0]
        latents = pipe.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=args.latent_channels,
            height=args.height,
            width=args.width,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        sigmas = (
            np.linspace(1.0, 1.0 / args.num_inference_steps, args.num_inference_steps)
            if args.sigmas is None
            else args.sigmas
        )
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            args.num_inference_steps,
            "cpu",
            sigmas=sigmas,
            mu=mu,
        )
        guidance = torch.full(
            [batch_size],
            args.guidance_scale,
            dtype=output_dtype,
            device=device,
        )

    tensors = {
        "latents_init": latents.to(dtype=output_dtype, device="cpu").contiguous(),
        "timesteps": timesteps.to(dtype=output_dtype, device="cpu").contiguous(),
        "encoder_hidden_states": prompt_embeds.to(dtype=output_dtype, device="cpu").contiguous(),
        "encoder_hidden_states_mask": prompt_embeds_mask.to(dtype=torch.bool, device="cpu").contiguous(),
        "guidance": guidance.to(device="cpu").contiguous(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"format": "difflet-qwen-image-dit-inputs-v1"})

    meta = {
        "schema": "difflet-qwen-image-dit-inputs-v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "text_seq_len": prompt_embeds.shape[1],
        "max_text_seq_len": args.text_seq_len,
        "pad_to_text_seq_len": args.pad_to_text_seq_len,
        "latent_channels": args.latent_channels,
        "guidance_scale": args.guidance_scale,
        "model_dtype": str(model_dtype),
        "output_dtype": str(output_dtype),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    print(f"wrote {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen-Image")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument(
        "--pad-to-text-seq-len",
        action="store_true",
        help="Pad prompt embeddings to --text-seq-len; otherwise save the active length.",
    )
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--output-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--sigmas", type=float, nargs="*", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cache_dit_inputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
