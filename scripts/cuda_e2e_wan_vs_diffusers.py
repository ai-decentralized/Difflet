"""Whole-pipeline oracle: Difflet's CUDA Wan 2.2 vs stock diffusers' WanPipeline (F1). GPU-only.

Prompt in, video out, on one GPU, from the SAME initial noise, same steps, same CFG scales
(4.0 above the expert boundary, 3.0 below), diffusers' empty negative prompt on both sides.
Difflet's side is ``CudaWanApplication`` (UMT5, both experts, VAE on the device, the
backend-neutral ``WanOrchestrator`` loop); diffusers' side is ``WanPipeline`` with its own
``WanTransformer3DModel`` experts. Difflet runs first and is freed (dynamo caches included),
then diffusers loads and runs with the saved noise. Pixel metrics over the whole video go to a
JSON the gate reads (tests/numerical/test_cuda_wan_e2e_vs_diffusers.py):

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_wan_vs_diffusers.py \\
        --model-path /path/to/Wan2.2-T2V-A14B-Diffusers-snapshot [--steps 40] [--num-frames 9]

The noise is ``WanOrchestrator.prepare_latents`` on the host from a seeded generator, shape
``(1, 16, latent_frames, H/8, W/8)``, fp32. ``WanPipeline.prepare_latents`` returns a given
``latents`` as is (``latents.to(device, float32)``, no re-draw) — diffusers 0.38.

Layouts: Difflet's ``output_type="pt"`` is the VAE's ``(B, 3, F, H, W)`` in [-1, 1];
``WanPipeline``'s is ``postprocess_video``'s ``(B, F, 3, H, W)`` in [0, 1]. Both become
``(F, H, W, 3)`` in [0, 1] before the metrics (scripts/cuda_e2e_vs_diffusers.py's
``image_metrics``, flattened over the whole clip). First-frame PNGs are written for the eye.
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
    ap.add_argument("--model-path", required=True, help="local HF snapshot of Wan 2.2 T2V-A14B")
    ap.add_argument("--model-id", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    ap.add_argument("--prompt", default="a photorealistic cat walking through a sunlit garden")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--guidance-scale-2", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--out", default="/tmp/logs/cuda_e2e_wan.json")
    ap.add_argument("--images-dir", default="/tmp/logs")
    ap.add_argument("--compile-cache-dir", default=None)
    return ap


def difflet_unit_video(frames: torch.Tensor) -> torch.Tensor:
    """Difflet's ``(B, 3, F, H, W)`` in [-1, 1] → ``(F, H, W, 3)`` in [0, 1], float, CPU."""
    return (frames[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0).cpu()


def diffusers_unit_video(video: torch.Tensor) -> torch.Tensor:
    """``WanPipeline(output_type="pt")``'s ``(B, F, 3, H, W)`` in [0, 1] → ``(F, H, W, 3)``."""
    return video[0].float().clamp(0, 1).permute(0, 2, 3, 1).cpu()


def first_frame(video: torch.Tensor) -> torch.Tensor:
    """``(F, H, W, 3)`` → the first frame as ``(3, H, W)``, what ``_save`` writes."""
    return video[0].permute(2, 0, 1)


def _gb(n: int) -> float:
    return round(n / 2**30, 2)


def run_difflet(args) -> tuple[torch.Tensor, torch.Tensor, dict]:
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
        num_frames=args.num_frames,
        compile_cache_dir=args.compile_cache_dir,
        local_files_only=True,
        model_path_override=args.model_path,
        resolved_source_id=f"local:{args.model_path}",
        compiled_path_override=str(Path(args.compile_cache_dir or "/tmp/logs/cuda_e2e_wan_cache")),
    )
    app = pipe.app
    timings = {"load_s": round(time.monotonic() - mark, 1)}
    print(f"difflet loaded in {timings['load_s']}s", flush=True)
    # Host fp32 noise from a seeded generator: the orchestrator's own draw, made explicit so
    # the diffusers side starts from the same tensor.
    noise = app.pipeline.prepare_latents(
        batch_size=1,
        channels=int(app.config.in_channels),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(args.seed),
    )
    mark = time.monotonic()
    frames = app.generate(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        latents=noise.clone(),  # the diffusers side gets the untouched original
        output_type="pt",
    )
    torch.cuda.synchronize()
    timings["generate_s"] = round(time.monotonic() - mark, 1)
    timings["peak_alloc_gb"] = _gb(torch.cuda.max_memory_allocated())
    print(f"difflet generated in {timings['generate_s']}s", flush=True)
    video = difflet_unit_video(frames)
    del frames, app, pipe
    # Free everything before diffusers loads its own two experts + UMT5 + VAE (~70 GB).
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return video, noise.cpu(), timings


def run_diffusers(args, device, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
    from diffusers import WanPipeline

    torch.cuda.reset_peak_memory_stats()
    mark = time.monotonic()
    pipe = WanPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    timings = {"load_s": round(time.monotonic() - mark, 1)}
    print(f"diffusers loaded in {timings['load_s']}s", flush=True)
    mark = time.monotonic()
    with torch.no_grad():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=None,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            latents=noise.to(device),
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            output_type="pt",
            return_dict=False,
        )[0]
    torch.cuda.synchronize()
    timings["generate_s"] = round(time.monotonic() - mark, 1)
    timings["peak_alloc_gb"] = _gb(torch.cuda.max_memory_allocated())
    print(f"diffusers generated in {timings['generate_s']}s", flush=True)
    out = diffusers_unit_video(video)
    del video, pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out, timings


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = torch.device("cuda")
    images = Path(args.images_dir)
    difflet_png = images / "wan_e2e_difflet_frame0.png"
    diffusers_png = images / "wan_e2e_diffusers_frame0.png"

    video_d, noise, timings_d = run_difflet(args)
    _save(first_frame(video_d), difflet_png)
    video_r, timings_r = run_diffusers(args, device, noise)
    _save(first_frame(video_r), diffusers_png)
    if video_d.shape != video_r.shape:
        raise SystemExit(
            f"video shapes differ: difflet {tuple(video_d.shape)} vs diffusers "
            f"{tuple(video_r.shape)}"
        )

    metrics = image_metrics(video_d, video_r)
    payload = {
        "model": "wan",
        "model_id": args.model_id,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "seed": args.seed,
        "shape": [args.height, args.width, args.num_frames],
        "latent_shape": list(noise.shape),
        "video_shape": list(video_d.shape),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "metrics": metrics,
        "timings": {"difflet": timings_d, "diffusers": timings_r},
        "images": {"difflet": str(difflet_png), "diffusers": str(diffusers_png)},
    }
    print(f"difflet vs diffusers (whole video, pixels in [0,1]): {metrics}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
