#!/usr/bin/env python
"""Compare standard VAE vs TAEF1 on Flux: speed + output quality (same seed).

Each variant runs in its own subprocess for clean NeuronCore state and
timing, then the parent reports load/generate times and image quality
metrics (MSE / PSNR / SSIM) between the two outputs.

Usage:
    python scripts/bench_taef1_compare.py [--steps 28] [--seed 42] [--tp-degree 4]

Notes:
    - The first run of each variant includes AOT compilation if its cache
      key is cold (standard VAE ~10-15 min; TAEF1 ~1 min). Warm reruns are
      fast.
    - Same seed + same denoise loop -> identical latents; the image diff is
      purely the decoder difference (standard VAE vs TAEF1).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MODEL = "black-forest-labs/FLUX.1-dev"
TAEF1_REPO = "madebyollin/taef1"


def worker(variant: str, args: argparse.Namespace) -> None:
    import torch

    from difflet import DiffletPipeline, DiffletParallelConfig

    kwargs = {}
    if variant == "taef1":
        kwargs["application_kwargs"] = {"taef1": True, "taef1_path": TAEF1_REPO}

    t0 = time.monotonic()
    pipe = DiffletPipeline.from_pretrained(
        MODEL,
        model_type="flux",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
        **kwargs,
    )
    load_s = time.monotonic() - t0

    t0 = time.monotonic()
    out = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(args.seed),
    )
    generate_s = time.monotonic() - t0

    image_path = Path(args.out_dir) / f"{variant}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    out.images[0].save(str(image_path))

    print(json.dumps({
        "variant": variant,
        "load_s": round(load_s, 2),
        "generate_s": round(generate_s, 2),
        "image": str(image_path),
    }))


def _run_variant(variant: str, args: argparse.Namespace) -> dict:
    print(f"[{variant}] running… (first run may include AOT compile)", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", variant,
         "--steps", str(args.steps), "--seed", str(args.seed),
         "--prompt", args.prompt, "--out-dir", str(args.out_dir),
         "--tp-degree", str(args.tp_degree)],
        capture_output=True, text=True,
    )
    wall = time.monotonic() - t0
    last_line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
    try:
        result = json.loads(last_line)
    except json.JSONDecodeError:
        print(f"[{variant}] FAILED (exit {proc.returncode}) — stderr tail:")
        print("\n".join((proc.stderr or "").strip().splitlines()[-15:]))
        return {"variant": variant, "failed": True}
    result["wall_s"] = round(wall, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["std", "taef1"], default=None,
                        help=argparse.SUPPRESS)  # internal
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--prompt", default="a red fox sitting in a snowy forest, sharp detail")
    parser.add_argument("--out-dir", default="/tmp/taef1_cmp")
    args = parser.parse_args()

    if args.worker:
        worker(args.worker, args)
        return

    results = [_run_variant(v, args) for v in ("std", "taef1")]
    results = [r for r in results if not r.get("failed")]
    if len(results) < 2:
        print("One or both variants failed — see output above.")
        sys.exit(1)

    print("\n=== timing ===")
    print(f"{'variant':<8}{'load (s)':<12}{'generate (s)':<15}{'total wall (s)':<16}")
    for r in results:
        print(f"{r['variant']:<8}{r['load_s']:<12}{r['generate_s']:<15}{r['wall_s']:<16}")
    std, taef1 = results
    if taef1["load_s"] and std["load_s"]:
        print(f"load speedup: {std['load_s'] / taef1['load_s']:.2f}x "
              f"({std['load_s']:.1f}s -> {taef1['load_s']:.1f}s)")
    print(f"generate speedup: {std['generate_s'] / taef1['generate_s']:.2f}x "
          f"({std['generate_s']:.1f}s -> {taef1['generate_s']:.1f}s)")

    print("\n=== quality (same seed -> same latents, decoder-only diff) ===")
    try:#测试指标
        import numpy as np
        from PIL import Image

        a = np.asarray(Image.open(std["image"]).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(taef1["image"]).convert("RGB"), dtype=np.float64)
        mse = ((a - b) ** 2).mean()
        psnr = 10 * np.log10(255.0**2 / max(mse, 1e-12))
        maxdiff = np.abs(a - b).max()
        print(f"MSE:    {mse:.2f}")
        print(f"PSNR:   {psnr:.2f} dB")
        print(f"max abs diff: {maxdiff:.0f} / 255")
        try:
            from skimage.metrics import structural_similarity as ssim_fn
            print(f"SSIM:   {ssim_fn(a.astype(np.uint8), b.astype(np.uint8), channel_axis=2):.4f}")
        except ImportError:
            print("SSIM:   (scikit-image not installed — skip)")
    except Exception as exc:
        print(f"quality metrics failed: {exc}")
    print(f"\nimages: {std['image']}  |  {taef1['image']}")


if __name__ == "__main__":
    main()
