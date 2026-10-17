"""Release 1 smoke: one Qwen-Image picture on one GPU through DiffletPipeline + the CUDA backend.

Task 9 of docs/superpowers/plans/2026-10-16-cuda-pr3-qwen-tp1.md. Prints per-phase wall
times and peak memory so the numbers go straight into the plan's status log.

    DIFFLET_BACKEND=cuda python scripts/cuda_smoke.py --prompt "..." --steps 4 --output /tmp/qwen.png
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def _gb(n: int) -> float:
    return round(n / 2**30, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen-Image")
    ap.add_argument("--prompt", default="a photorealistic cat sitting in a sunlit garden")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--output", default="/tmp/qwen.png")
    ap.add_argument("--compile-cache-dir", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--torch-compile", action="store_true", help="F4 task 3: compiled DiT")
    ap.add_argument("--teacache-cadence", type=int, default=None, help="F4 task 4: skip every Nth")
    args = ap.parse_args()

    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    from difflet import DiffletParallelConfig, DiffletPipeline

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    pipe = DiffletPipeline.from_pretrained(
        args.model_id,
        backend="cuda",
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        height=args.height,
        width=args.width,
        compile_cache_dir=args.compile_cache_dir,
        local_files_only=args.local_files_only,
        application_kwargs=(
            {
                **({"torch_compile": True} if args.torch_compile else {}),
                **({"teacache_cadence": args.teacache_cadence} if args.teacache_cadence else {}),
            }
            or None
        ),
    )
    torch.cuda.synchronize()
    timings["load_s"] = round(time.perf_counter() - t0, 1)
    print(
        f"[smoke] loaded in {timings['load_s']} s; peak alloc {_gb(torch.cuda.max_memory_allocated())} GB"
    )

    app = pipe.app
    t1 = time.perf_counter()
    enc, mask = app.encode_prompt(args.prompt)
    torch.cuda.synchronize()
    timings["encode_s"] = round(time.perf_counter() - t1, 2)
    print(
        f"[smoke] prompt encoded: {tuple(enc.shape)} valid={int(mask.sum())} in {timings['encode_s']} s"
    )

    t2 = time.perf_counter()
    images = app.generate(
        args.prompt,
        num_inference_steps=args.steps,
        seed=args.seed,
        output_type="pil",
    )
    torch.cuda.synchronize()
    timings["generate_s"] = round(time.perf_counter() - t2, 1)
    timings["per_step_s"] = round(timings["generate_s"] / max(args.steps, 1), 2)

    image = images[0]
    image.save(args.output)
    timings["peak_alloc_gb"] = _gb(torch.cuda.max_memory_allocated())
    timings["peak_reserved_gb"] = _gb(torch.cuda.max_memory_reserved())
    timings["device"] = torch.cuda.get_device_name(0)
    timings["torch"] = torch.__version__
    timings["steps"] = args.steps
    timings["shape"] = [args.height, args.width]
    timings["output"] = args.output
    print(f"[smoke] wrote {args.output} ({image.size[0]}x{image.size[1]})")
    print("[smoke] " + json.dumps(timings))


if __name__ == "__main__":
    main()
