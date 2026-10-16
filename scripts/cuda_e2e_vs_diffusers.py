"""Whole-pipeline oracle: Difflet's CUDA Qwen-Image vs diffusers' QwenImagePipeline (F3).

Prompt in, image out, on one GPU, from the SAME packed initial noise, same steps and scheduler
settings, no classifier-free guidance on either side. The two models cannot be resident at
once on 96 GB (each side is ~58 GB with the text encoder), so Difflet runs first, frees, then
diffusers runs with the saved noise. Pixel-space metrics go to a JSON the gate reads
(tests/numerical/test_cuda_e2e_vs_diffusers.py):

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_vs_diffusers.py \\
        --model-path /path/to/Qwen-Image-snapshot --prompt "..." --steps 50 --seed 0

This is where the prompt padding mask shows: the DiT-level gate feeds an all-ones mask, a
real prompt is ~16 valid tokens of 1024.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch


def image_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Pixel metrics between two float images in [0, 1] of the same shape."""
    a64 = a.detach().double().flatten().cpu()
    b64 = b.detach().double().flatten().cpu()
    diff = (a64 - b64).abs()
    mse = float((diff**2).mean())
    return {
        "psnr_db": float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
        "cosine": float(torch.nn.functional.cosine_similarity(a64, b64, dim=0)),
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
    }


def _to_unit(images: torch.Tensor) -> torch.Tensor:
    """Difflet's VAE output (B, C[, 1], H, W) in [-1, 1] → (C, H, W) in [0, 1]."""
    if images.ndim == 5:
        images = images[:, :, 0]
    return (images[0].float() / 2 + 0.5).clamp(0, 1)


def _save(img: torch.Tensor, path: Path) -> None:
    from PIL import Image

    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def run_difflet(args, device) -> tuple[torch.Tensor, torch.Tensor]:
    from difflet import DiffletParallelConfig, DiffletPipeline

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
        compiled_path_override=str(Path(args.compile_cache_dir or "/tmp/logs/cuda_e2e_cache")),
    )
    app = pipe.app
    print(f"difflet loaded in {time.monotonic() - mark:.1f}s", flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = app.pipeline.prepare_latents(
        batch_size=1, dtype=torch.bfloat16, device=device, generator=generator
    )
    mark = time.monotonic()
    out = app.generate(args.prompt, num_inference_steps=args.steps, latents=noise, output_type="pt")
    torch.cuda.synchronize()
    print(f"difflet generated in {time.monotonic() - mark:.1f}s", flush=True)
    img = _to_unit(out.images)
    del out, app, pipe
    torch.cuda.empty_cache()
    return img.cpu(), noise.cpu()


def run_diffusers(args, device, noise: torch.Tensor) -> torch.Tensor:
    from diffusers import QwenImagePipeline

    mark = time.monotonic()
    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    print(f"diffusers loaded in {time.monotonic() - mark:.1f}s", flush=True)
    mark = time.monotonic()
    with torch.no_grad():
        out = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            latents=noise.to(device),
            true_cfg_scale=1.0,
            max_sequence_length=1024,
            output_type="pt",
        )
    torch.cuda.synchronize()
    print(f"diffusers generated in {time.monotonic() - mark:.1f}s", flush=True)
    img = out.images[0].float().clamp(0, 1).cpu()
    del out, pipe
    torch.cuda.empty_cache()
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="local HF snapshot of Qwen/Qwen-Image")
    ap.add_argument("--model-id", default="Qwen/Qwen-Image")
    ap.add_argument("--prompt", default="a photorealistic cat sitting in a sunlit garden")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/logs/cuda_e2e_qwen_image.json")
    ap.add_argument("--images-dir", default="/tmp/logs")
    ap.add_argument("--compile-cache-dir", default=None)
    args = ap.parse_args()
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = torch.device("cuda")
    images = Path(args.images_dir)

    img_d, noise = run_difflet(args, device)
    _save(img_d, images / "e2e_difflet.png")
    img_r = run_diffusers(args, device, noise)
    _save(img_r, images / "e2e_diffusers.png")

    metrics = image_metrics(img_d, img_r)
    payload = {
        "model": "qwen_image",
        "model_path": args.model_path,
        "prompt": args.prompt,
        "steps": args.steps,
        "seed": args.seed,
        "shape": [args.height, args.width],
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "metrics": metrics,
        "images": {"difflet": str(images / "e2e_difflet.png"), "diffusers": str(images / "e2e_diffusers.png")},
    }
    print(f"difflet vs diffusers (pixels in [0,1]): {metrics}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
