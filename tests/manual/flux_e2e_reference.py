"""FLUX.1-dev end to end with diffusers on the CPU: the fp32 reference image and the bf16 floor.

Same prompt, seed, size, steps and guidance as the backend runs. The initial noise is the seeded
CPU draw; fp32 draws are identical across PyTorch versions, and PyTorch 2.14's bf16 draw is the fp32
draw rounded, so both images start from the noise the backends used (with the Trainium run drawing
in fp32). Each image is a lossless PNG; the step progress goes to the log.

    python tests/manual/flux_e2e_reference.py --model-dir $FLUX --out-dir ~/tn/q --dtype fp32
    python tests/manual/flux_e2e_reference.py --model-dir $FLUX --out-dir ~/tn/q --dtype bf16
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROMPT = "A cat holding a sign that says hello world"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()
    from diffusers import FluxPipeline

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    t = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(args.model_dir, torch_dtype=dtype)
    print(f"load {time.perf_counter() - t:.0f} s, threads {torch.get_num_threads()}", flush=True)
    t = time.perf_counter()
    with torch.no_grad():
        image = pipe(PROMPT, height=1024, width=1024, num_inference_steps=args.steps, guidance_scale=3.5,
                     max_sequence_length=512, generator=torch.Generator().manual_seed(args.seed)).images[0]
    out = os.path.join(os.path.expanduser(args.out_dir), f"diffusers_{args.dtype}_flux_s{args.seed}.png")
    image.save(out)
    print(f"generate {time.perf_counter() - t:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
