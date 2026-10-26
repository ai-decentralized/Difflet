"""HunyuanVideo 1.5 T2V end to end with diffusers on the CPU: the fp32 reference and the bf16 baseline.

A 9-frame clip (the 121-frame default is ~45 h per fp32 run on the CPU), otherwise as the backend
runs it: 480x848, 50 steps, the checkpoint's guider (CFG 6.0), the empty negative prompt, seed 42.
The noise is the pipeline's own seeded draw in its dtype, as the backend's loop draws it. The
decoded frames are saved as a [B, C, F, H, W] tensor in [0, 1].

    python tests/manual/hv15_e2e_reference.py --model-dir $HV15 --out-dir ~/tn/q --dtype fp32
    python tests/manual/hv15_e2e_reference.py --model-dir $HV15 --out-dir ~/tn/q --dtype bf16
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROMPT = "A cat walking along a sunny beach, cinematic"
HEIGHT, WIDTH, FRAMES = 480, 848, 9


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()
    from diffusers import HunyuanVideo15Pipeline

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    t = time.perf_counter()
    pipe = HunyuanVideo15Pipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    print(f"load {time.perf_counter() - t:.0f} s, threads {torch.get_num_threads()}", flush=True)
    t = time.perf_counter()
    with torch.no_grad():
        frames = pipe(prompt=PROMPT, height=HEIGHT, width=WIDTH, num_frames=FRAMES, num_inference_steps=args.steps,
                      generator=torch.Generator().manual_seed(args.seed), output_type="pt").frames
    video = frames.float().permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
    out = os.path.join(os.path.expanduser(args.out_dir), f"diffusers_{args.dtype}_hv15_s{args.seed}.pt")
    torch.save(video, out)
    print(f"generate {time.perf_counter() - t:.0f} s -> {out} {tuple(video.shape)} "
          f"range [{float(video.min()):.3f}, {float(video.max()):.3f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
