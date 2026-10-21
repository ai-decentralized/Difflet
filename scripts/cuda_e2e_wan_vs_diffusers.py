"""Whole-pipeline oracle: Difflet's CUDA Wan 2.2 vs stock diffusers' WanPipeline. GPU-only.

Prompt in, video out, on one GPU, from the SAME initial noise, same steps, same CFG scales
(4.0 above the expert boundary, 3.0 below), diffusers' empty negative prompt on both sides.
Difflet's side is ``CudaWanApplication`` (UMT5, both experts, VAE on the device, the
backend-neutral ``WanOrchestrator`` loop); diffusers' side is ``WanPipeline`` with its own
``WanTransformer3DModel`` experts. Difflet runs first and is freed (dynamo caches included),
then diffusers loads and runs with the saved noise. Pixel metrics over the whole video go to a
JSON the gate reads (tests/numerical/test_cuda_wan_e2e_vs_diffusers.py):

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_wan_vs_diffusers.py \\
        --model-path /path/to/Wan2.2-T2V-A14B-Diffusers-snapshot [--steps 40] [--num-frames 9] \\
        [--timestep-dtypes float32,model] [--seeds 0,1,2 --noise-floor]

**Timestep-dtype A/B.** ``--timestep-dtypes`` lists ``CudaWanApplication``'s
``timestep_dtype`` values to run (``float32``: production, UniPC's exact integer timestep;
``model``: the model-dtype bf16 cast, 999 → 1000), passed through ``DiffletPipeline``'s
``application_kwargs``. Difflet is loaded, run and freed once per variant, in order; the first
load draws the noise, which every later variant and the single diffusers run reuse. The JSON
holds ``variants.<name>.{metrics, timings, image}`` (one first-frame PNG per variant); the
top-level ``metrics``, ``timings.difflet`` and ``images.difflet`` are the first variant's, so
the gate's floors judge it.

**Seeds and the noise floor.** ``--seeds 0,1,2`` (overrides ``--seed``; ``--seed N``
alone is ``--seeds N``) keeps one load per side: each Difflet variant is loaded once and
generates every seed, and diffusers is loaded once. The first variant draws each seed's noise
(``torch.Generator().manual_seed(seed)``, as before); every later variant and diffusers reuse it.
``--noise-floor`` adds, per seed, a second diffusers run with both experts' ``forward`` wrapped so
the ``timestep`` (UniPC's int64 ``t.expand(B)``) is rounded through bf16 — the model-dtype cast,
999 → 1000, 750 → 752 — and every other argument passes untouched; the wrapper is removed before
the next seed's exact run. Exact vs perturbed is diffusers' distance from itself under that cast:
the scale the gate states the Difflet-vs-diffusers numbers against when it is recorded (the
mean over seeds within 1.15x, 1.21 dB, of the floor's mean; every seed within 1.21 dB of the
floor's worst seed; a loose 20 dB / 0.98 absolute floor), the 25 dB / 0.99 floors judging
otherwise. The JSON adds ``seeds``,
``per_seed.<seed>.{variants.<name>.{metrics, timings}, noise_floor: {metrics, timings} | null,
diffusers.timings, images}`` and ``aggregate.{variants.<name>, noise_floor}.{psnr_db, cosine}.
{min, mean, max}``; ``seed``, ``metrics``, ``variants``, ``timings``, ``images`` and
``noise_floor`` stay the first seed's (single-seed readers and the gate's fallbacks), and the
first seed's PNGs are also written under the single-seed names.

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
import contextlib
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
# CudaWanApplication's timestep_dtype values (difflet/backends/cuda/wan/application.py).
TIMESTEP_DTYPES = ("float32", "model")


def parse_timestep_dtypes(text: str) -> list[str]:
    """``"float32,model"`` → ``["float32", "model"]``: known values, no repeats, at least one."""
    names = [part.strip() for part in text.split(",")]
    for name in names:
        if name not in TIMESTEP_DTYPES:
            raise argparse.ArgumentTypeError(
                f"unknown timestep dtype {name!r}; expected a comma-separated list of "
                f"{', '.join(TIMESTEP_DTYPES)}"
            )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"repeated timestep dtype in {text!r}")
    return names


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
    ap.add_argument("--seed", type=int, default=0, help="one seed (default 0); see --seeds")
    ap.add_argument(
        "--seeds",
        type=parse_seeds,
        default=None,
        help="comma-separated seeds, overriding --seed: one load per side, every seed generated "
        "on it, per-seed and aggregate metrics in one JSON (e.g. 0,1,2)",
    )
    ap.add_argument(
        "--noise-floor",
        action="store_true",
        help="per seed, a second diffusers run with both experts' timestep rounded through bf16 "
        "(the model-dtype cast), compared against the exact run: diffusers' distance from itself, "
        "recorded next to the Difflet numbers, not gated",
    )
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--out", default="/tmp/logs/cuda_e2e_wan.json")
    ap.add_argument("--images-dir", default="/tmp/logs")
    ap.add_argument("--compile-cache-dir", default=None)
    ap.add_argument(
        "--timestep-dtypes",
        type=parse_timestep_dtypes,
        default=["float32"],
        help="comma-separated CudaWanApplication timestep_dtype values, one Difflet run each "
        "against one diffusers run (float32 = production, model = the bf16 model-dtype cast); the "
        "first is the gated one (default: float32)",
    )
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="compile the DiT blocks, as `difflet run --torch-compile` does (default: eager)",
    )
    return ap


def variant_png(images_dir: Path, name: str, seed: int | None = None) -> Path:
    """The first-frame PNG of one Difflet variant (``seed`` None: the single-seed name)."""
    tag = "" if seed is None else f"_seed{seed}"
    return Path(images_dir) / f"wan_e2e_difflet_{name}{tag}_frame0.png"


def diffusers_png(images_dir: Path, seed: int | None = None, *, bf16ts: bool = False) -> Path:
    """The first-frame PNG of diffusers' exact run, or (``bf16ts``) its bf16-rounded-timestep
    run; ``seed`` None: the single-seed name of the exact run."""
    if seed is None:
        assert not bf16ts, "the bf16-rounded run has per-seed names only"
        return Path(images_dir) / "wan_e2e_diffusers_frame0.png"
    tag = "_bf16ts" if bf16ts else ""
    return Path(images_dir) / f"wan_e2e_diffusers{tag}_seed{seed}_frame0.png"


def bf16_rounded_timestep(t: torch.Tensor) -> torch.Tensor:
    """``t`` rounded through bf16, as a float32 tensor (999 → 1000, 750 → 752): what the
    ``model`` cast feeds the experts. Through float32 first, so an int64 schedule rounds once;
    the result stays float, which ``WanTransformer3DModel``'s sinusoidal embedding takes as is
    (``timesteps[:, None].float()``)."""
    return t.to(torch.float32).to(torch.bfloat16).to(torch.float32)


@contextlib.contextmanager
def perturb_timestep(pipe):
    """Within the block, ``pipe.transformer`` and ``pipe.transformer_2`` (when present) see their
    ``timestep`` rounded through bf16; every other argument reaches ``forward`` untouched. On exit
    (errors included) each module's instance attributes are as before — an instance-level
    ``forward`` (a diffusers hook) restored, else removed so the class's runs again."""
    _missing = object()
    wrapped = []
    try:
        for module in (getattr(pipe, "transformer", None), getattr(pipe, "transformer_2", None)):
            if module is None or any(module is m for m, _ in wrapped):
                continue
            previous = vars(module).get("forward", _missing)
            original = module.forward

            def forward(*args, __original=original, **kwargs):
                if "timestep" in kwargs:
                    kwargs["timestep"] = bf16_rounded_timestep(kwargs["timestep"])
                elif len(args) >= 2:  # forward(hidden_states, timestep, ...)
                    args = (args[0], bf16_rounded_timestep(args[1]), *args[2:])
                return __original(*args, **kwargs)

            module.forward = forward
            wrapped.append((module, previous))
        yield
    finally:
        for module, previous in reversed(wrapped):
            if previous is _missing:
                del module.forward
            else:
                module.forward = previous


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


def _device_name() -> str:
    return torch.cuda.get_device_name(0)


def _device() -> torch.device:
    return torch.device("cuda")


def _application_kwargs(args: argparse.Namespace) -> dict | None:
    """The opt-in application options this run asks for (None: the defaults)."""
    kwargs = {k: True for k in ("torch_compile", "fuse_proj_out") if getattr(args, k, False)}
    return kwargs or None


def run_difflet(
    args, *, timestep_dtype: str, seeds: list[int], noises: dict[int, torch.Tensor] | None = None
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, dict]]:
    """One Difflet load with ``timestep_dtype``, then one generate per seed; draws each seed's
    noise unless ``noises`` is given. Returns per seed: the unit video, the noise, the timings
    (``load_s`` is the one shared load's)."""
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
        application_kwargs={"timestep_dtype": timestep_dtype, **(_application_kwargs(args) or {})},
    )
    app = pipe.app
    load_s = round(time.monotonic() - mark, 1)
    print(f"difflet (timestep_dtype={timestep_dtype}) loaded in {load_s}s", flush=True)
    videos: dict[int, torch.Tensor] = {}
    out_noises: dict[int, torch.Tensor] = {}
    timings: dict[int, dict] = {}
    for seed in seeds:
        if noises is None:
            # Host fp32 noise from a seeded generator: the orchestrator's own draw, made
            # explicit so every later variant and the diffusers side start from the same tensor.
            noise = app.pipeline.prepare_latents(
                batch_size=1,
                channels=int(app.config.in_channels),
                dtype=torch.float32,
                generator=torch.Generator().manual_seed(seed),
            )
        else:
            noise = noises[seed]
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
        timings[seed] = {
            "load_s": load_s,
            "generate_s": round(time.monotonic() - mark, 1),
            "peak_alloc_gb": _gb(torch.cuda.max_memory_allocated()),
        }
        print(
            f"difflet (timestep_dtype={timestep_dtype}, seed {seed}) generated in "
            f"{timings[seed]['generate_s']}s",
            flush=True,
        )
        videos[seed] = difflet_unit_video(frames)
        out_noises[seed] = noise.cpu()
        del frames
    del app, pipe
    # Free everything before the next variant or diffusers loads (two experts + UMT5 + VAE,
    # ~70 GB each).
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return videos, out_noises, timings


def _diffusers_generate(pipe, args, device, noise: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """One ``WanPipeline`` call from ``noise``: the unit video and its timings."""
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
    timings = {
        "generate_s": round(time.monotonic() - mark, 1),
        "peak_alloc_gb": _gb(torch.cuda.max_memory_allocated()),
    }
    out = diffusers_unit_video(video)
    del video
    return out, timings


def run_diffusers(
    args, device, noises: dict[int, torch.Tensor], *, noise_floor: bool = False
) -> tuple[dict[int, torch.Tensor], dict[int, dict], dict[int, torch.Tensor], dict[int, dict]]:
    """One ``WanPipeline`` load; per seed the exact run and, with ``noise_floor``, the run with
    both experts' timestep rounded through bf16. Returns (exact videos, exact timings, perturbed
    videos, perturbed timings), per seed; the perturbed dicts are empty without ``noise_floor``."""
    from diffusers import WanPipeline

    torch.cuda.reset_peak_memory_stats()
    mark = time.monotonic()
    pipe = WanPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    load_s = round(time.monotonic() - mark, 1)
    print(f"diffusers loaded in {load_s}s", flush=True)
    exact, exact_t, perturbed, perturbed_t = {}, {}, {}, {}
    for seed, noise in noises.items():
        exact[seed], timings = _diffusers_generate(pipe, args, device, noise)
        exact_t[seed] = {"load_s": load_s, **timings}
        print(f"diffusers (seed {seed}) generated in {timings['generate_s']}s", flush=True)
        if noise_floor:
            with perturb_timestep(pipe):
                perturbed[seed], timings = _diffusers_generate(pipe, args, device, noise)
            perturbed_t[seed] = {"load_s": load_s, **timings}
            print(
                f"diffusers (seed {seed}, bf16-rounded timestep) generated in "
                f"{timings['generate_s']}s",
                flush=True,
            )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return exact, exact_t, perturbed, perturbed_t


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("DIFFLET_BACKEND", "cuda")
    device = _device()
    images = Path(args.images_dir)
    seeds = run_seeds(args)
    first_seed = seeds[0]

    # videos[name][seed], timings_d[name][seed]
    videos: dict[str, dict[int, torch.Tensor]] = {}
    timings_d: dict[str, dict[int, dict]] = {}
    noises = None
    for name in args.timestep_dtypes:
        videos[name], noises, timings_d[name] = run_difflet(
            args, timestep_dtype=name, seeds=seeds, noises=noises
        )
        for seed, video in videos[name].items():
            _save(first_frame(video), variant_png(images, name, seed=seed))
        _save(first_frame(videos[name][first_seed]), variant_png(images, name))
    exact, exact_t, perturbed, perturbed_t = run_diffusers(
        args, device, noises, noise_floor=args.noise_floor
    )
    for seed in seeds:
        _save(first_frame(exact[seed]), diffusers_png(images, seed=seed))
        if seed in perturbed:
            _save(first_frame(perturbed[seed]), diffusers_png(images, seed=seed, bf16ts=True))
    _save(first_frame(exact[first_seed]), diffusers_png(images))
    for seed in seeds:
        others = [(f"difflet ({name})", videos[name][seed]) for name in videos]
        if seed in perturbed:
            others.append(("diffusers (bf16-rounded timestep)", perturbed[seed]))
        for label, video in others:
            if video.shape != exact[seed].shape:
                raise SystemExit(
                    f"video shapes differ (seed {seed}): {label} {tuple(video.shape)} vs "
                    f"diffusers {tuple(exact[seed].shape)}"
                )

    per_seed = {}
    for seed in seeds:
        entry_variants = {}
        for name in args.timestep_dtypes:
            entry_variants[name] = {
                "metrics": image_metrics(videos[name][seed], exact[seed]),
                "timings": timings_d[name][seed],
            }
            print(
                f"seed {seed}: difflet (timestep_dtype={name}) vs diffusers (whole video, "
                f"pixels in [0,1]): {entry_variants[name]['metrics']}",
                flush=True,
            )
        floor = None
        if seed in perturbed:
            floor = {
                "metrics": image_metrics(perturbed[seed], exact[seed]),
                "timings": perturbed_t[seed],
            }
            print(
                f"seed {seed}: noise floor, diffusers (bf16-rounded timestep) vs diffusers: "
                f"{floor['metrics']}",
                flush=True,
            )
        per_seed[str(seed)] = {
            "variants": entry_variants,
            "noise_floor": floor,
            "diffusers": {"timings": exact_t[seed]},
            "images": {
                "difflet": {
                    name: str(variant_png(images, name, seed=seed)) for name in args.timestep_dtypes
                },
                "diffusers": str(diffusers_png(images, seed=seed)),
                "diffusers_bf16ts": (
                    str(diffusers_png(images, seed=seed, bf16ts=True)) if floor else None
                ),
            },
        }
    entries = [per_seed[str(seed)] for seed in seeds]
    aggregate = {
        "variants": {
            name: aggregate_metrics([e["variants"][name] for e in entries])
            for name in args.timestep_dtypes
        },
        "noise_floor": (
            aggregate_metrics([e["noise_floor"] for e in entries]) if args.noise_floor else None
        ),
    }

    first_entry = per_seed[str(first_seed)]
    variants = {
        name: {**first_entry["variants"][name], "image": str(variant_png(images, name))}
        for name in args.timestep_dtypes
    }
    first = variants[args.timestep_dtypes[0]]
    payload = {
        "model": "wan",
        "model_id": args.model_id,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "seed": first_seed,
        "seeds": seeds,
        "shape": [args.height, args.width, args.num_frames],
        "latent_shape": list(noises[first_seed].shape),
        "video_shape": list(exact[first_seed].shape),
        "device": _device_name(),
        "torch": torch.__version__,
        "timestep_dtypes": list(args.timestep_dtypes),
        # The single-seed keys: the first seed's, the first variant's on top.
        "metrics": first["metrics"],
        "timings": {"difflet": first["timings"], "diffusers": exact_t[first_seed]},
        "images": {"difflet": first["image"], "diffusers": str(diffusers_png(images))},
        "variants": variants,
        "noise_floor": first_entry["noise_floor"],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    for name, agg in aggregate["variants"].items():
        print(f"aggregate over {len(seeds)} seed(s), timestep_dtype={name}: {agg}", flush=True)
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
