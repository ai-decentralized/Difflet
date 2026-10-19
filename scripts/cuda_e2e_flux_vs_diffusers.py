"""Whole-pipeline oracle: Difflet's CUDA Flux vs stock diffusers' FluxPipeline (F5). GPU-only.

Prompt in, image out, on one GPU, from the SAME packed initial noise, same steps, same
distilled guidance scale. Both sides are ``FluxPipeline`` (CLIP, T5, scheduler, VAE); the only
difference is the transformer — Difflet's DiT behind ``FluxDiTAdapter`` vs diffusers'
``FluxTransformer2DModel``. Difflet runs first, frees, then diffusers runs with the saved
noise. Pixel-space metrics go to a JSON the gate reads
(tests/numerical/test_cuda_flux_e2e_vs_diffusers.py):

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_flux_vs_diffusers.py \\
        --model-path /path/to/FLUX.1-dev-snapshot [--steps 28] [--guidance-scale 3.5] [--seed 0]

Metrics and PNG writing are scripts/cuda_e2e_vs_diffusers.py's (one definition for both e2e
gates). Its ``_to_unit`` is not used: ``FluxPipeline``'s ``output_type="pt"`` is already
post-processed to [0, 1] on both sides. Since F4b the JSON carries
``timings.{difflet,diffusers}.{load_s, generate_s, peak_alloc_gb}`` as the Wan oracle does.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_e2e_vs_diffusers import _save, image_metrics  # noqa: E402

MAX_SEQUENCE_LENGTH = 512


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="local HF snapshot of FLUX.1-dev")
    ap.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--prompt", default="a photorealistic cat sitting in a sunlit garden")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance-scale", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/logs/cuda_e2e_flux.json")
    ap.add_argument("--images-dir", default="/tmp/logs")
    ap.add_argument("--compile-cache-dir", default=None)
    ap.add_argument(
        "--noise-floor",
        action="store_true",
        help="also run diffusers against itself with bf16 gemm reductions in reduced precision "
        "flipped (a rounding-level perturbation): the metric's own sensitivity on this seed (F9's "
        "Wan recipe, F4f for Flux)",
    )
    return ap


def timings_record(*, load_s: float, generate_s: float, peak_bytes: int | None) -> dict:
    """One side's ``{load_s, generate_s, peak_alloc_gb}`` (GB = 1e9 bytes, as the Wan oracle)."""
    return {
        "load_s": round(float(load_s), 1),
        "generate_s": round(float(generate_s), 1),
        "peak_alloc_gb": None if peak_bytes is None else round(int(peak_bytes) / 1e9, 2),
    }


def _peak_bytes() -> int | None:
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None


def _unit_image(images: torch.Tensor) -> torch.Tensor:
    """FluxPipeline ``output_type="pt"`` (B, C, H, W) in [0, 1] → (C, H, W) float on CPU."""
    return images[0].float().clamp(0, 1).cpu()


def run_difflet(args, device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    from difflet import DiffletParallelConfig, DiffletPipeline

    torch.cuda.reset_peak_memory_stats()
    mark = time.monotonic()
    pipe = DiffletPipeline.from_pretrained(
        args.model_id,
        backend="cuda",
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        height=args.height,
        width=args.width,
        compile_cache_dir=args.compile_cache_dir,
        local_files_only=True,
        model_path_override=args.model_path,
        resolved_source_id=f"local:{args.model_path}",
        compiled_path_override=str(Path(args.compile_cache_dir or "/tmp/logs/cuda_e2e_flux_cache")),
    )
    app = pipe.app
    load_s = time.monotonic() - mark
    print(f"difflet loaded in {load_s:.1f}s", flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise, _ = app.pipe.prepare_latents(
        1,
        app.pipe.transformer.config.in_channels // 4,
        args.height,
        args.width,
        torch.bfloat16,
        device,
        generator,
    )
    mark = time.monotonic()
    out = app.generate(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        latents=noise,
        height=args.height,
        width=args.width,
        output_type="pt",
    )
    torch.cuda.synchronize()
    generate_s = time.monotonic() - mark
    print(f"difflet generated in {generate_s:.1f}s", flush=True)
    timings = timings_record(load_s=load_s, generate_s=generate_s, peak_bytes=_peak_bytes())
    img = _unit_image(out.images)
    noise = noise.cpu()
    del out, app, pipe
    # Free everything before the diffusers side loads its own ~34 GB (dynamo caches included).
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return img, noise, timings


def _diffusers_generate(pipe, args, device, noise: torch.Tensor) -> tuple[torch.Tensor, float]:
    mark = time.monotonic()
    with torch.no_grad():
        out = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            latents=noise.to(device),
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            output_type="pt",
        )
    torch.cuda.synchronize()
    generate_s = time.monotonic() - mark
    print(f"diffusers generated in {generate_s:.1f}s", flush=True)
    return _unit_image(out.images), generate_s


def run_diffusers(
    args, device, noise: torch.Tensor, *, noise_floor: bool = False
) -> tuple[torch.Tensor, dict, dict | None]:
    """One ``FluxPipeline`` load; the exact run and, with ``noise_floor``, a second run with
    ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` flipped — every
    bf16 gemm accumulates differently, a rounding-level perturbation of diffusers itself.
    Returns the exact image, its timings, and the perturbed run's ``{metrics, timings}`` or None."""
    from diffusers import FluxPipeline

    torch.cuda.reset_peak_memory_stats()
    mark = time.monotonic()
    pipe = FluxPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    load_s = time.monotonic() - mark
    print(f"diffusers loaded in {load_s:.1f}s", flush=True)
    img, generate_s = _diffusers_generate(pipe, args, device, noise)
    timings = timings_record(load_s=load_s, generate_s=generate_s, peak_bytes=_peak_bytes())
    floor = None
    if noise_floor:
        flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not flag
        try:
            perturbed, perturbed_s = _diffusers_generate(pipe, args, device, noise)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        floor = {
            "perturbation": f"allow_bf16_reduced_precision_reduction={not flag} (exact: {flag})",
            "metrics": image_metrics(img, perturbed),
            "timings": timings_record(load_s=load_s, generate_s=perturbed_s, peak_bytes=None),
        }
        print(f"diffusers vs its perturbed self (pixels in [0,1]): {floor['metrics']}", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return img, timings, floor


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = torch.device("cuda")
    images = Path(args.images_dir)
    difflet_png = images / "flux_e2e_difflet.png"
    diffusers_png = images / "flux_e2e_diffusers.png"

    img_d, noise, timings_d = run_difflet(args, device)
    _save(img_d, difflet_png)
    img_r, timings_r, floor = run_diffusers(args, device, noise, noise_floor=args.noise_floor)
    _save(img_r, diffusers_png)

    metrics = image_metrics(img_d, img_r)
    payload = {
        "model": "flux",
        "model_id": args.model_id,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "shape": [args.height, args.width],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "metrics": metrics,
        "images": {"difflet": str(difflet_png), "diffusers": str(diffusers_png)},
        "timings": {"difflet": timings_d, "diffusers": timings_r},
        "noise_floor": floor,
    }
    print(f"difflet vs diffusers (pixels in [0,1]): {metrics}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
