#!/usr/bin/env python3
"""Generate multiple HV DiT input bundles from a prompts file.

Loads the HF text encoder ONCE and iterates over (split, prompt) tuples,
producing one .safetensors bundle per prompt. Skips bundles whose output
files already exist.

Used to materialize 16 multi-prompt 50-step bundles for cclog 76 m9 T0+T0b.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def _parse_prompts_file(path: Path) -> list[tuple[str, str]]:
    """Return list of (split, prompt) tuples from the cclog 67 prompts file."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    result = []
    for line in lines[1:]:  # skip header
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        split, prompt = parts[0].strip(), parts[1].strip()
        result.append((split, prompt))
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default="hunyuanvideo-community/HunyuanVideo")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=61)
    p.add_argument("--text-seq-len", type=int, default=256)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0, help="base seed (offset by prompt index)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline
    from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps
    from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast

    prompts = _parse_prompts_file(Path(args.prompts_file))
    print(f"[gen] {len(prompts)} prompts to generate", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    t_load = time.perf_counter()
    print("[gen] loading text encoders...", flush=True)
    text_encoder = LlamaModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder_2 = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder_2", torch_dtype=dtype
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer_2")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
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
    print(f"[gen] text pipeline loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    for idx, (split, prompt) in enumerate(prompts):
        # Create a slug for output filename
        slug = "_".join(prompt.split()[:5]).lower()
        slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)[:48]
        out_path = out_dir / f"{split}_{idx:02d}_{slug}_{args.num_inference_steps}step.safetensors"
        meta_path = Path(str(out_path) + ".meta.json")

        if out_path.exists() and meta_path.exists():
            print(f"[gen] {idx + 1}/{len(prompts)} SKIP (exists): {out_path.name}", flush=True)
            continue

        t_step = time.perf_counter()
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                max_sequence_length=args.text_seq_len,
                num_videos_per_prompt=1,
                device=device,
                dtype=dtype,
            )
            sigmas = np.linspace(1.0, 0.0, args.num_inference_steps + 1)[:-1]
            timesteps, _ = retrieve_timesteps(
                pipe.scheduler,
                args.num_inference_steps,
                "cpu",
                sigmas=sigmas,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + idx)
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
                dtype=dtype,
                device=device,
            )

        tensors = {
            "latents_init": latents.to(dtype=dtype, device="cpu").contiguous(),
            "timesteps": timesteps.to(dtype=dtype, device="cpu").contiguous(),
            "encoder_hidden_states": prompt_embeds.to(dtype=dtype, device="cpu").contiguous(),
            "encoder_attention_mask": prompt_attention_mask.to(dtype=torch.int64, device="cpu").contiguous(),
            "pooled_projections": pooled_prompt_embeds.to(dtype=dtype, device="cpu").contiguous(),
            "guidance": guidance.to(device="cpu").contiguous(),
        }
        save_file(tensors, str(out_path), metadata={"format": "nova-hunyuan-video-dit-inputs-v1"})

        meta = {
            "schema": "nova-hunyuan-video-dit-inputs-v1",
            "model_id": args.model_id,
            "prompt": [prompt],
            "prompt_2": None,
            "seed": int(args.seed + idx),
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "num_inference_steps": int(args.num_inference_steps),
            "text_seq_len": int(args.text_seq_len),
            "latent_channels": int(args.latent_channels),
            "guidance_scale": float(args.guidance_scale),
            "model_dtype": "torch.bfloat16",
            "output_dtype": "torch.bfloat16",
            "split": split,
            "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()},
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        elapsed = time.perf_counter() - t_step
        print(
            f"[gen] {idx + 1}/{len(prompts)} ({split}) done in {elapsed:.1f}s "
            f"→ {out_path.name}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
