#!/usr/bin/env python3
"""diffusers' own HunyuanVideo15Pipeline on the CPU, from saved inputs: an ``embeds.pt`` (any
Difflet hv15 encode stage) and the initial noise, returning the final latents (``output_type="latent"``).

The third reference for an AOT vs native latents comparison: both backends are fed the same
embeds and noise, and this is what diffusers computes from them (bf16 by default, as the
backends run; ``--dtype fp32`` for the precision ceiling). CPU only; it touches no NeuronCore.

    python scripts/hv15_diffusers_cpu_latents.py --embeds embeds.pt --noise-seed-fp32-round 42 \
        --height 320 --width 512 --num-frames 61 --steps 3 --out latents.pt
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    p.add_argument("--revision", default=None)
    p.add_argument("--embeds", required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--steps", type=int, default=50)
    noise = p.add_mutually_exclusive_group(required=True)
    noise.add_argument("--noise", default=None, help="a saved initial-noise tensor")
    noise.add_argument("--noise-seed-fp32-round", type=int, default=None,
                       help="draw fp32 noise with this seed and round it to bf16 (the native stack's draw)")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import torch
    from diffusers import HunyuanVideo15Pipeline
    from diffusers.utils.torch_utils import randn_tensor

    from difflet.pipeline.path_resolver import resolve_model_path

    if a.threads:
        torch.set_num_threads(a.threads)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    root = resolve_model_path(a.model_id, revision=a.revision, local_files_only=True)
    started = time.perf_counter()
    pipe = HunyuanVideo15Pipeline.from_pretrained(root, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                                  tokenizer_2=None, torch_dtype=dtype)
    print(f"[hv15-diffusers] load {time.perf_counter() - started:.1f} s", flush=True)
    shape = (1, int(pipe.transformer.config.out_channels), (a.num_frames - 1) // 4 + 1, a.height // 16, a.width // 16)
    if a.noise:
        latents = torch.load(a.noise, map_location="cpu", weights_only=True)
    else:
        latents = randn_tensor(shape, generator=torch.Generator().manual_seed(a.noise_seed_fp32_round),
                               device=torch.device("cpu"), dtype=torch.float32).to(torch.bfloat16)
    embeds = {k: v.to(dtype) if v.is_floating_point() else v
              for k, v in torch.load(a.embeds, map_location="cpu", weights_only=True).items()}
    started = time.perf_counter()
    with torch.no_grad():
        out = pipe(**embeds, height=a.height, width=a.width, num_frames=a.num_frames, num_inference_steps=a.steps,
                   latents=latents.to(dtype), output_type="latent", return_dict=False)[0]
    print(f"[hv15-diffusers] denoise {time.perf_counter() - started:.1f} s, {a.steps} steps", flush=True)
    torch.save(out.cpu(), a.out)
    print(f"[hv15-diffusers] latents {tuple(out.shape)} {out.dtype} saved to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
