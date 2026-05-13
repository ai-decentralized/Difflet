#!/usr/bin/env python3
"""Cache HunyuanVideo 1.5 transformer-boundary input tensors.

This helper runs the host-side HunyuanVideo 1.5 Qwen2.5-VL + ByT5 text path
and writes the fixed tensors consumed by ``scripts/hunyuan15_transformer_parity.py``.
It intentionally caches a single conditional transformer call; classifier-free
guidance and VAE decode remain host/orchestrator concerns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
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
    # The Neuron venv has torch_xla installed, but CPU text caching should not
    # initialize PJRT. Diffusers checks this flag while importing the pipeline.
    import diffusers.utils.import_utils as import_utils

    import_utils._torch_xla_available = False


class _VaeShapeStub:
    temporal_compression_ratio = 4
    spatial_compression_ratio = 16

    def __init__(self) -> None:
        self.config = SimpleNamespace(latent_channels=32)


def _load_text_pipeline(
    *,
    model_id: str,
    dtype: torch.dtype,
    device: torch.device,
    revision: str | None,
    local_files_only: bool,
):
    _disable_xla_lazy_import()
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import (
        HunyuanVideo15Pipeline,
    )
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2Tokenizer, T5EncoderModel

    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
    }
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    text_encoder = Qwen2_5_VLTextModel.from_pretrained(
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
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        **load_kwargs,
    )
    tokenizer_2 = ByT5Tokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer_2",
        **load_kwargs,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        **load_kwargs,
    )
    pipe = HunyuanVideo15Pipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
        vae=_VaeShapeStub(),
        scheduler=scheduler,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        guider=ClassifierFreeGuidance(enabled=False),
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _latent_model_input(latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    cond_latents = torch.zeros_like(latents, dtype=dtype)
    mask = torch.zeros(
        latents.shape[0],
        1,
        latents.shape[2],
        latents.shape[3],
        latents.shape[4],
        dtype=dtype,
        device=latents.device,
    )
    return torch.cat([latents.to(dtype=dtype), cond_latents, mask], dim=1).contiguous()


def cache_dit_inputs(args: argparse.Namespace) -> None:
    _disable_xla_lazy_import()
    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import retrieve_timesteps

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
        prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2 = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=model_dtype,
            batch_size=len(args.prompt),
            num_videos_per_prompt=1,
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
        latents = pipe.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=args.latent_channels,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        image_embeds = torch.zeros(
            prompt_embeds.shape[0],
            args.image_seq_len,
            args.image_embed_dim,
            dtype=model_dtype,
            device=device,
        )
        timestep_r = torch.ones([prompt_embeds.shape[0]], dtype=output_dtype, device=device)

    tensors = {
        "latents_init": latents.to(dtype=output_dtype, device="cpu").contiguous(),
        "hidden_states": _latent_model_input(latents, output_dtype).to(device="cpu"),
        "timesteps": timesteps.to(dtype=output_dtype, device="cpu").contiguous(),
        "timestep_r": timestep_r.to(device="cpu").contiguous(),
        "encoder_hidden_states": prompt_embeds.to(dtype=output_dtype, device="cpu").contiguous(),
        "encoder_attention_mask": prompt_mask.to(dtype=torch.int64, device="cpu").contiguous(),
        "encoder_hidden_states_2": prompt_embeds_2.to(dtype=output_dtype, device="cpu").contiguous(),
        "encoder_attention_mask_2": prompt_mask_2.to(dtype=torch.int64, device="cpu").contiguous(),
        "image_embeds": image_embeds.to(dtype=output_dtype, device="cpu").contiguous(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"format": "nova-hunyuan-video15-dit-inputs-v1"})

    meta = {
        "schema": "nova-hunyuan-video15-dit-inputs-v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "latent_channels": args.latent_channels,
        "image_seq_len": args.image_seq_len,
        "image_embed_dim": args.image_embed_dim,
        "model_dtype": str(model_dtype),
        "output_dtype": str(output_dtype),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--image-seq-len", type=int, default=729)
    parser.add_argument("--image-embed-dim", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--output-dtype", type=_parse_dtype, default=torch.bfloat16)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dit_inputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
