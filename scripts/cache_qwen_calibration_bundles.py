#!/usr/bin/env python3
"""Generate multiple Qwen-Image DiT input bundles from a prompts file (cclog 82).

Loads the Qwen2.5-VL text encoder ONCE and iterates over (split, prompt) tuples,
producing one .safetensors bundle per prompt at the production 50-step shape.
Embeddings are padded to --text-seq-len so they match the compiled DiT graph.

Qwen analog of scripts/cache_hv_calibration_bundles.py (cclog 76). Used to
materialize the 8 calib + 8 holdout 50-step bundles for the Qwen m9 e2e A/B.
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

from scripts.qwen_image_cache_dit_inputs import (  # noqa: E402
    _load_text_pipeline,
    normalize_prompt_embeds,
)


def _parse_prompts_file(path: Path) -> list[tuple[str, str]]:
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
    p.add_argument("--model-id", default="/home/ubuntu/.cache/huggingface/hub/qwen-image-real")
    p.add_argument("--local-files-only", action="store_true", default=True)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--text-seq-len", type=int, default=1024)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0, help="base seed (offset by prompt index)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
        calculate_shift,
        retrieve_timesteps,
    )

    prompts = _parse_prompts_file(Path(args.prompts_file))
    print(f"[gen] {len(prompts)} prompts to generate", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    t_load = time.perf_counter()
    print("[gen] loading Qwen2.5-VL text encoder...", flush=True)
    pipe = _load_text_pipeline(
        model_id=args.model_id,
        dtype=dtype,
        device=device,
        revision=None,
        local_files_only=bool(args.local_files_only),
    )
    print(f"[gen] text pipeline loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    for idx, (split, prompt) in enumerate(prompts):
        slug = "_".join(prompt.split()[:5]).lower()
        slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)[:48]
        out_path = out_dir / f"{split}_{idx:02d}_{slug}_{args.num_inference_steps}step.safetensors"
        meta_path = Path(str(out_path) + ".meta.json")

        if out_path.exists() and meta_path.exists():
            print(f"[gen] {idx + 1}/{len(prompts)} SKIP (exists): {out_path.name}", flush=True)
            continue

        t_step = time.perf_counter()
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
                pad_to_max_sequence_length=True,
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + idx)
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
            sigmas = np.linspace(1.0, 1.0 / args.num_inference_steps, args.num_inference_steps)
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
                dtype=dtype,
                device=device,
            )

        tensors = {
            "latents_init": latents.to(dtype=dtype, device="cpu").contiguous(),
            "timesteps": timesteps.to(dtype=dtype, device="cpu").contiguous(),
            "encoder_hidden_states": prompt_embeds.to(dtype=dtype, device="cpu").contiguous(),
            "encoder_hidden_states_mask": prompt_embeds_mask.to(dtype=torch.bool, device="cpu").contiguous(),
            "guidance": guidance.to(device="cpu").contiguous(),
        }
        save_file(tensors, str(out_path), metadata={"format": "nova-qwen-image-dit-inputs-v1"})

        meta = {
            "schema": "nova-qwen-image-dit-inputs-v1",
            "model_id": args.model_id,
            "prompt": [prompt],
            "seed": int(args.seed + idx),
            "height": int(args.height),
            "width": int(args.width),
            "num_inference_steps": int(args.num_inference_steps),
            "text_seq_len": int(prompt_embeds.shape[1]),
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
