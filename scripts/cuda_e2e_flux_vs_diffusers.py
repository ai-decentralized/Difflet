"""Whole-pipeline oracle: Difflet's CUDA Flux vs stock diffusers' FluxPipeline. GPU-only.

Prompt in, image out, on one GPU, from the SAME packed initial noise, same steps, same
distilled guidance scale. Both sides are ``FluxPipeline`` (CLIP, T5, scheduler, VAE); the only
difference is the transformer — Difflet's DiT behind ``FluxDiTAdapter`` vs diffusers'
``FluxTransformer2DModel``. Difflet runs first, frees, then diffusers runs with the saved
noise. Pixel-space metrics go to a JSON the gate reads
(tests/numerical/test_cuda_flux_e2e_vs_diffusers.py):

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_flux_vs_diffusers.py \\
        --model-path /path/to/FLUX.1-dev-snapshot [--steps 28] [--guidance-scale 3.5] \\
        [--seed 0 | --seeds 0,1,2] [--noise-floor]

Metrics, PNG writing and the seed helpers are scripts/cuda_e2e_vs_diffusers.py's (one
definition for every e2e oracle). Its ``_to_unit`` is not used: ``FluxPipeline``'s
``output_type="pt"`` is already post-processed to [0, 1] on both sides. The JSON carries
``timings.{difflet,diffusers}.{load_s, generate_s, peak_alloc_gb}`` as the Wan oracle does.

**Seeds and the noise floor.** ``--noise-floor`` adds, per seed, a second diffusers
run with ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` flipped (every
bf16 gemm accumulates differently; restored after the run, errors included): exact vs perturbed
is diffusers' distance from itself, the scale the gate states Difflet's numbers against.
``--seeds 0,1,2`` (overrides ``--seed``; ``--seed N`` alone is ``--seeds N``) keeps one load per
side, as the Wan oracle does: Difflet is loaded once and generates every seed, each from the packed
noise its seeded ``torch.Generator(device)`` draws, and diffusers is loaded once and runs every
seed (exact, then perturbed) from that same noise. The JSON adds ``seeds``,
``per_seed.<seed>.{metrics, timings.{difflet, diffusers}, noise_floor | null, images}`` and
``aggregate.{difflet, noise_floor}.{psnr_db, cosine}.{min, mean, max}``; ``seed``, ``metrics``,
``timings``, ``images`` and ``noise_floor`` stay the first seed's (single-seed readers), and the
first seed's PNGs are also written under the single-seed names.
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
from cuda_e2e_vs_diffusers import (  # noqa: E402
    _save,
    aggregate_metrics,
    image_metrics,
    parse_seeds,
    run_seeds,
)

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
    ap.add_argument("--seed", type=int, default=0, help="one seed (default 0); see --seeds")
    ap.add_argument(
        "--seeds",
        type=parse_seeds,
        default=None,
        help="comma-separated seeds, overriding --seed: one load per side, every seed generated "
        "on it, per-seed and aggregate metrics in one JSON (e.g. 0,1,2)",
    )
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/logs/cuda_e2e_flux.json")
    ap.add_argument("--images-dir", default="/tmp/logs")
    ap.add_argument("--compile-cache-dir", default=None)
    ap.add_argument(
        "--noise-floor",
        action="store_true",
        help="per seed, also run diffusers against itself with bf16 gemm reductions in reduced "
        "precision flipped (a rounding-level perturbation): the metric's own sensitivity on that "
        "seed (the Wan oracle's recipe, applied to Flux)",
    )
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="compile the DiT blocks, as `difflet run --torch-compile` does (default: eager)",
    )
    ap.add_argument(
        "--fuse-proj-out",
        action="store_true",
        help="the single block's proj_out as one gemm (off by default; see "
        "difflet/backends/cuda/flux/proj_out.py)",
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


def _device() -> torch.device:
    return torch.device("cuda")


def _device_name() -> str:
    return torch.cuda.get_device_name(0)


def side_png(images_dir: Path, side: str, seed: int | None = None) -> Path:
    """One side's PNG (``side``: ``difflet`` or ``diffusers``); ``seed`` None: the single-seed
    name."""
    tag = "" if seed is None else f"_seed{seed}"
    return Path(images_dir) / f"flux_e2e_{side}{tag}.png"


def _unit_image(images: torch.Tensor) -> torch.Tensor:
    """FluxPipeline ``output_type="pt"`` (B, C, H, W) in [0, 1] → (C, H, W) float on CPU."""
    return images[0].float().clamp(0, 1).cpu()


def _application_kwargs(args: argparse.Namespace) -> dict | None:
    """The opt-in application options this run asks for (None: the defaults)."""
    kwargs = {k: True for k in ("torch_compile", "fuse_proj_out") if getattr(args, k, False)}
    return kwargs or None


def run_difflet(
    args, device, seeds: list[int]
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, dict]]:
    """One Difflet load, then one generate per seed from the packed noise its seeded generator
    draws. Returns per seed: the unit image, the noise (CPU), the timings (``load_s`` is the one
    shared load's; ``peak_alloc_gb`` the peak so far)."""
    from difflet import DiffletParallelConfig, DiffletPipeline

    torch.cuda.reset_peak_memory_stats()
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
    load_s = time.monotonic() - mark
    print(f"difflet loaded in {load_s:.1f}s", flush=True)
    images: dict[int, torch.Tensor] = {}
    noises: dict[int, torch.Tensor] = {}
    timings: dict[int, dict] = {}
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
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
            latents=noise.clone(),  # the diffusers side gets the untouched original
            height=args.height,
            width=args.width,
            output_type="pt",
        )
        torch.cuda.synchronize()
        generate_s = time.monotonic() - mark
        print(f"difflet (seed {seed}) generated in {generate_s:.1f}s", flush=True)
        timings[seed] = timings_record(
            load_s=load_s, generate_s=generate_s, peak_bytes=_peak_bytes()
        )
        images[seed] = _unit_image(out.images)
        noises[seed] = noise.cpu()
        del out
    del app, pipe
    # Free everything before the diffusers side loads its own ~34 GB (dynamo caches included).
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return images, noises, timings


def _diffusers_generate(
    pipe, args, device, noise: torch.Tensor, label: str
) -> tuple[torch.Tensor, float]:
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
    print(f"diffusers ({label}) generated in {generate_s:.1f}s", flush=True)
    return _unit_image(out.images), generate_s


def run_diffusers(
    args, device, noises: dict[int, torch.Tensor], *, noise_floor: bool = False
) -> tuple[dict[int, torch.Tensor], dict[int, dict], dict[int, dict]]:
    """One ``FluxPipeline`` load; per seed the exact run and, with ``noise_floor``, a second run
    with ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` flipped — every
    bf16 gemm accumulates differently, a rounding-level perturbation of diffusers itself — the
    flag restored after it (errors included). Returns per seed: the exact image, its timings
    (``load_s`` the one shared load's; ``peak_alloc_gb`` the peak so far, taken before that seed's
    perturbed run), and the perturbed run's ``{perturbation, metrics, timings}`` (empty without
    ``noise_floor``)."""
    from diffusers import FluxPipeline

    torch.cuda.reset_peak_memory_stats()
    mark = time.monotonic()
    pipe = FluxPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    load_s = time.monotonic() - mark
    print(f"diffusers loaded in {load_s:.1f}s", flush=True)
    images: dict[int, torch.Tensor] = {}
    timings: dict[int, dict] = {}
    floors: dict[int, dict] = {}
    for seed, noise in noises.items():
        images[seed], generate_s = _diffusers_generate(pipe, args, device, noise, f"seed {seed}")
        timings[seed] = timings_record(
            load_s=load_s, generate_s=generate_s, peak_bytes=_peak_bytes()
        )
        if not noise_floor:
            continue
        flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not flag
        try:
            perturbed, perturbed_s = _diffusers_generate(
                pipe, args, device, noise, f"seed {seed}, bf16 reductions flipped"
            )
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        floors[seed] = {
            "perturbation": f"allow_bf16_reduced_precision_reduction={not flag} (exact: {flag})",
            "metrics": image_metrics(images[seed], perturbed),
            "timings": timings_record(load_s=load_s, generate_s=perturbed_s, peak_bytes=None),
        }
        print(
            f"seed {seed}: diffusers vs its perturbed self (pixels in [0,1]): "
            f"{floors[seed]['metrics']}",
            flush=True,
        )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return images, timings, floors


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = _device()
    images_dir = Path(args.images_dir)
    seeds = run_seeds(args)
    first_seed = seeds[0]

    img_d, noises, timings_d = run_difflet(args, device, seeds)
    for seed, img in img_d.items():
        _save(img, side_png(images_dir, "difflet", seed=seed))
    _save(img_d[first_seed], side_png(images_dir, "difflet"))
    img_r, timings_r, floors = run_diffusers(args, device, noises, noise_floor=args.noise_floor)
    for seed, img in img_r.items():
        _save(img, side_png(images_dir, "diffusers", seed=seed))
    _save(img_r[first_seed], side_png(images_dir, "diffusers"))

    per_seed = {}
    for seed in seeds:
        metrics = image_metrics(img_d[seed], img_r[seed])
        print(f"seed {seed}: difflet vs diffusers (pixels in [0,1]): {metrics}", flush=True)
        per_seed[str(seed)] = {
            "metrics": metrics,
            "timings": {"difflet": timings_d[seed], "diffusers": timings_r[seed]},
            "noise_floor": floors.get(seed),
            "images": {
                side: str(side_png(images_dir, side, seed=seed))
                for side in ("difflet", "diffusers")
            },
        }
    entries = [per_seed[str(seed)] for seed in seeds]
    aggregate = {
        "difflet": aggregate_metrics(entries),
        "noise_floor": (
            aggregate_metrics([e["noise_floor"] for e in entries]) if args.noise_floor else None
        ),
    }
    first = per_seed[str(first_seed)]
    payload = {
        "model": "flux",
        "model_id": args.model_id,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": first_seed,
        "seeds": seeds,
        "shape": [args.height, args.width],
        "device": _device_name(),
        "torch": torch.__version__,
        # The single-seed keys: the first seed's.
        "metrics": first["metrics"],
        "images": {side: str(side_png(images_dir, side)) for side in ("difflet", "diffusers")},
        "timings": first["timings"],
        "noise_floor": first["noise_floor"],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    print(f"aggregate over {len(seeds)} seed(s), difflet: {aggregate['difflet']}", flush=True)
    if aggregate["noise_floor"] is not None:
        print(
            f"aggregate over {len(seeds)} seed(s), noise floor: {aggregate['noise_floor']}",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
