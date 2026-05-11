#!/usr/bin/env python3
"""Cache HunyuanVideo DiT input tensors from the HF text/scheduler path.

M3 v0 keeps Llama3, CLIP, and VAE outside Trainium. This helper produces the
host-side DiT input artifact consumed by Nova's HunyuanVideo backbone smoke and
trajectory tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file


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


def _load_text_pipeline(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    revision: str | None,
    local_files_only: bool,
):
    from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline
    from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    text_encoder = LlamaModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        **load_kwargs,
    )
    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_id,
        subfolder="tokenizer",
        **load_kwargs,
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        **load_kwargs,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer_2",
        **load_kwargs,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        **load_kwargs,
    )
    pipe = HunyuanVideoPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
        vae=None,
        scheduler=scheduler,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def cache_dit_inputs(args: argparse.Namespace) -> None:
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps

    device = torch.device(args.device)
    prompt = _prompt_arg(args.prompt)
    prompt_2 = args.prompt_2 if args.prompt_2 is not None else None
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
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            max_sequence_length=args.text_seq_len,
            num_videos_per_prompt=1,
            device=device,
            dtype=model_dtype,
        )
        sigmas = np.linspace(1.0, 0.0, args.num_inference_steps + 1)[:-1]
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            args.num_inference_steps,
            "cpu",
            sigmas=sigmas,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        batch_size = prompt_embeds.shape[0]
        latents = pipe.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=args.latent_channels,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        guidance = torch.full(
            [batch_size],
            args.guidance_scale * 1000.0,
            dtype=output_dtype,
            device=device,
        )

    tensors = {
        "latents_init": latents.to(dtype=output_dtype, device="cpu").contiguous(),
        "timesteps": timesteps.to(dtype=output_dtype, device="cpu").contiguous(),
        "encoder_hidden_states": prompt_embeds.to(dtype=output_dtype, device="cpu").contiguous(),
        "encoder_attention_mask": prompt_attention_mask.to(dtype=torch.int64, device="cpu").contiguous(),
        "pooled_projections": pooled_prompt_embeds.to(dtype=output_dtype, device="cpu").contiguous(),
        "guidance": guidance.to(device="cpu").contiguous(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"format": "nova-hunyuan-video-dit-inputs-v1"})

    meta = {
        "schema": "nova-hunyuan-video-dit-inputs-v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "prompt": args.prompt,
        "prompt_2": prompt_2,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "text_seq_len": args.text_seq_len,
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
    parser.add_argument("--model-id", default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--prompt-2", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--text-seq-len", type=int, default=256)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--output-dtype", type=_parse_dtype, default=torch.bfloat16)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cache_dit_inputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
