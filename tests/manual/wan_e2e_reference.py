"""Wan 2.2 T2V end to end with diffusers on the CPU: the fp32 reference video and the bf16 baseline.

Matched to the backends' runs: by default the high-noise expert for every step (no
transformer_2, no expert boundary); with --dual-expert both experts, switched at the checkpoint's
boundary_ratio as WanPipeline and the backends' --dual-expert do. 480x832, 9 frames, 50 steps,
guidance 4.0, the empty negative prompt, seed 42. The initial noise is drawn in fp32 from the
seeded CPU generator (as the shared orchestrator draws it; fp32 draws are identical across
PyTorch versions). bf16 runs every component in bf16, VAE included, as the backends do at 9
frames. The decoded frames are saved as a [B, C, F, H, W] tensor in [0, 1], free of
video-encoding loss, and the final latents (before the VAE) beside them.

    python tests/manual/wan_e2e_reference.py --model-dir $WAN --out-dir ~/tn/q --dtype fp32
    python tests/manual/wan_e2e_reference.py --model-dir $WAN --out-dir ~/tn/q --dtype bf16
    python tests/manual/wan_e2e_reference.py --model-dir $WAN --out-dir ~/tn/q --dtype bf16 \\
        --dual-expert --steps 40 --guidance-scale-2 3.0 --prompt "..."

Two bf16 experts and umT5 need about 69 GB of host memory (two fp32 experts: about 137 GB).
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
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--guidance-scale-2", type=float, default=None,
                   help="the low-noise expert's guidance (with --dual-expert)")
    p.add_argument("--dual-expert", action="store_true",
                   help="keep transformer_2 and the checkpoint's boundary_ratio")
    args = p.parse_args()
    from diffusers import WanPipeline

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    t = time.perf_counter()
    if args.dual_expert:
        pipe = WanPipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    else:
        pipe = WanPipeline.from_pretrained(args.model_dir, transformer_2=None, torch_dtype=dtype)
        pipe.register_to_config(boundary_ratio=None)  # the high-noise expert for every step
    print(f"load {time.perf_counter() - t:.0f} s, threads {torch.get_num_threads()}, "
          f"boundary_ratio {pipe.config.boundary_ratio}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    z_dim = pipe.vae.config.z_dim
    latents = torch.randn(1, z_dim, (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8, generator=g,
                          dtype=torch.float32)
    t = time.perf_counter()
    with torch.no_grad():
        final = pipe(prompt=args.prompt, negative_prompt="", height=HEIGHT, width=WIDTH,
                     num_frames=FRAMES, num_inference_steps=args.steps,
                     guidance_scale=args.guidance_scale, guidance_scale_2=args.guidance_scale_2,
                     latents=latents, output_type="latent").frames
        frames = _decode(pipe, final)
    tag = f"diffusers_{args.dtype}_wan{'_dual' if args.dual_expert else ''}_s{args.seed}"
    out_dir = os.path.expanduser(args.out_dir)
    torch.save(final.float().cpu(), os.path.join(out_dir, f"{tag}_latents.pt"))
    video = frames.float()
    out = os.path.join(out_dir, f"{tag}.pt")
    torch.save(video, out)
    print(f"generate {time.perf_counter() - t:.0f} s -> {out} {tuple(video.shape)} "
          f"range [{float(video.min()):.3f}, {float(video.max()):.3f}]", flush=True)
    return 0


def _decode(pipe, latents: torch.Tensor) -> torch.Tensor:
    """WanPipeline's own decode of ``latents`` (un-normalise, VAE, postprocess to [0, 1]) as a
    [B, C, F, H, W] tensor."""
    vae = pipe.vae
    z = latents.to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
    video = vae.decode(z * std + mean, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="pt")  # [B, F, C, H, W]
    return frames.permute(0, 2, 1, 3, 4).contiguous()


if __name__ == "__main__":
    sys.exit(main())
