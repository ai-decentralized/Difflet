"""Wan 2.2 T2V end to end with diffusers on the CPU: the fp32 reference video and the bf16 baseline.

Matched to the backends' runs: the high-noise expert for every step (no transformer_2, no expert
boundary), 480x832, 9 frames, 50 steps, guidance 4.0, the empty negative prompt, seed 42. The
initial noise is drawn in fp32 from the seeded CPU generator (as the shared orchestrator draws it;
fp32 draws are identical across PyTorch versions). bf16 runs every component in bf16, VAE
included, as the backends do at 9 frames. The decoded frames are saved as a [B, C, F, H, W]
tensor in [0, 1], free of video-encoding loss.

    python tests/manual/wan_e2e_reference.py --model-dir $WAN --out-dir ~/tn/q --dtype fp32
    python tests/manual/wan_e2e_reference.py --model-dir $WAN --out-dir ~/tn/q --dtype bf16
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROMPT = "A cat walking along a sunny beach, cinematic"
HEIGHT, WIDTH, FRAMES = 480, 832, 9


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()
    from diffusers import WanPipeline

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    t = time.perf_counter()
    pipe = WanPipeline.from_pretrained(args.model_dir, transformer_2=None, torch_dtype=dtype)
    pipe.register_to_config(boundary_ratio=None)  # the high-noise expert for every step
    print(f"load {time.perf_counter() - t:.0f} s, threads {torch.get_num_threads()}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    z_dim = pipe.vae.config.z_dim
    latents = torch.randn(1, z_dim, (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8, generator=g,
                          dtype=torch.float32)
    t = time.perf_counter()
    with torch.no_grad():
        frames = pipe(prompt=PROMPT, negative_prompt="", height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                      num_inference_steps=args.steps, guidance_scale=4.0, latents=latents,
                      output_type="pt").frames
    video = frames.float().permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
    out = os.path.join(os.path.expanduser(args.out_dir), f"diffusers_{args.dtype}_wan_s{args.seed}.pt")
    torch.save(video, out)
    print(f"generate {time.perf_counter() - t:.0f} s -> {out} {tuple(video.shape)} "
          f"range [{float(video.min()):.3f}, {float(video.max()):.3f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
