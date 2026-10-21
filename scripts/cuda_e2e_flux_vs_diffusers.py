"""Whole-pipeline oracle: Difflet's CUDA Flux vs stock diffusers' FluxPipeline. GPU-only.

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
post-processed to [0, 1] on both sides.
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
        "--torch-compile",
        action="store_true",
        help="compile the DiT blocks, as `difflet run --torch-compile` does (default: eager)",
    )
    return ap


def _unit_image(images: torch.Tensor) -> torch.Tensor:
    """FluxPipeline ``output_type="pt"`` (B, C, H, W) in [0, 1] → (C, H, W) float on CPU."""
    return images[0].float().clamp(0, 1).cpu()


def _application_kwargs(args: argparse.Namespace) -> dict | None:
    """The opt-in application options this run asks for (None: the defaults)."""
    kwargs = {k: True for k in ("torch_compile", "fuse_proj_out") if getattr(args, k, False)}
    return kwargs or None


def run_difflet(args, device) -> tuple[torch.Tensor, torch.Tensor]:
    from difflet import DiffletParallelConfig, DiffletPipeline

    mark = time.monotonic()
    pipe = DiffletPipeline.from_pretrained(
        args.model_id,
        backend="cuda",
        application_kwargs=_application_kwargs(args),
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
    print(f"difflet loaded in {time.monotonic() - mark:.1f}s", flush=True)
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
    print(f"difflet generated in {time.monotonic() - mark:.1f}s", flush=True)
    img = _unit_image(out.images)
    noise = noise.cpu()
    del out, app, pipe
    # Free everything before the diffusers side loads its own ~34 GB (dynamo caches included).
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return img, noise


def run_diffusers(args, device, noise: torch.Tensor) -> torch.Tensor:
    from diffusers import FluxPipeline

    mark = time.monotonic()
    pipe = FluxPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    print(f"diffusers loaded in {time.monotonic() - mark:.1f}s", flush=True)
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
    print(f"diffusers generated in {time.monotonic() - mark:.1f}s", flush=True)
    img = _unit_image(out.images)
    del out, pipe
    gc.collect()
    torch.cuda.empty_cache()
    return img


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = torch.device("cuda")
    images = Path(args.images_dir)
    difflet_png = images / "flux_e2e_difflet.png"
    diffusers_png = images / "flux_e2e_diffusers.png"

    img_d, noise = run_difflet(args, device)
    _save(img_d, difflet_png)
    img_r = run_diffusers(args, device, noise)
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
    }
    print(f"difflet vs diffusers (pixels in [0,1]): {metrics}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
