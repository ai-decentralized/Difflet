"""Device smoke run: the composed Flux application, end to end, one image.

    torchrun --standalone --nproc-per-node 4 tests/manual/run_neuron_flux_smoke.py \
        --model-dir $FLUX --out ~/tn/flux_smoke.png

TP = world size; DiT bf16, T5 --t5-dtype (bf16 by default), VAE fp32 on rank 0, CLIP on the host.
Prints load and generate wall times (orientation only: matched numbers come from the CLI path).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="the FLUX.1-dev snapshot directory")
    p.add_argument("--out", required=True, help="PNG path (written by rank 0)")
    p.add_argument("--prompt", default="A cat holding a sign that says hello world")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t5-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--exec-mode", choices=("compile", "eager"), default="compile")
    args = p.parse_args()

    from difflet.backends.registry import get_backend
    from difflet.models.flux.neuron_application import TorchNeuronFluxApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0

    app = TorchNeuronFluxApplication(
        model_path=args.model_dir, parallel=parallel, dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width}, exec_mode=args.exec_mode,
        t5_dtype=getattr(torch, args.t5_dtype))
    t = time.perf_counter()
    app.load()
    if rank == 0:
        print(f"load {time.perf_counter() - t:.1f} s (DiT {app.dit.phase_seconds}, "
              f"T5 {app.t5.phase_seconds})", flush=True)

    t = time.perf_counter()
    out = app(prompt=args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
              generator=torch.Generator().manual_seed(args.seed))
    elapsed = time.perf_counter() - t
    if rank == 0:
        image = out.images[0]
        image.save(os.path.expanduser(args.out))
        print(f"generate {elapsed:.1f} s for {args.steps} steps -> {args.out} ({image.size})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
