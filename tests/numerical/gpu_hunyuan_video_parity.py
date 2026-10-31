"""HunyuanVideo reference controls and clean GPU generation measurements.

Generated evidence must be written outside the checkout. Missing calibrated limits
or an unqualified Omni mode stay explicit; this tool never substitutes HunyuanVideo 1.5.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

import torch


def tensor_metrics(reference, actual, *, data_range=None):
    if reference.shape != actual.shape:
        raise ValueError("reference and actual tensor shapes differ")
    ref, got = reference.detach().float().cpu().flatten(), actual.detach().float().cpu().flatten()
    counts = dict(
        reference_nonfinite=int((~torch.isfinite(ref)).sum()),
        actual_nonfinite=int((~torch.isfinite(got)).sum()),
    )
    if counts["reference_nonfinite"] or counts["actual_nonfinite"]:
        return {"finite": False, **counts}
    error = got - ref
    mse = float(error.square().mean())
    return dict(
        finite=True,
        **counts,
        max_abs=float(error.abs().max()),
        mean_abs=float(error.abs().mean()),
        relative_l2=float(error.norm() / ref.norm().clamp_min(1e-12)),
        relative_l1=float(error.abs().sum() / ref.abs().sum().clamp_min(1e-12)),
        reference_l2=float(ref.norm()),
        rmse=math.sqrt(mse),
        cosine=float(torch.nn.functional.cosine_similarity(ref[None], got[None])),
        psnr_db=None if mse == 0 or data_range is None else 10 * math.log10(data_range**2 / mse),
    )


def quality_verdict(metrics, limits):
    if not metrics.get("finite", False):
        return "fail"
    if not limits:
        return "uncalibrated"
    if any(
        k not in {"max_abs", "mean_abs", "relative_l1", "relative_l2", "rmse"}
        or not math.isfinite(float(v))
        or float(v) < 0
        for k, v in limits.items()
    ):
        raise ValueError("calibration limits must name metrics and be finite and nonnegative")
    return "pass" if all(metrics[k] <= v for k, v in limits.items()) else "fail"


def checkpoint_revision(model_path, revision):
    """Verify the pinned Hugging Face snapshot rather than trusting a report label."""
    snapshot = Path(model_path).resolve()
    if snapshot.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        raise ValueError("model-path must resolve to a pinned Hugging Face snapshot")
    if revision != snapshot.name:
        raise ValueError("checkpoint revision does not match the resolved snapshot")
    return snapshot.name


def compare_outputs(reference, actual, ref_trajectory, trajectory, *, steps, limits):
    if len(ref_trajectory) != steps or len(trajectory) != steps:
        raise ValueError("observed trajectory length differs from requested steps")
    video = tensor_metrics(reference, actual, data_range=2.0)
    latent_metrics = [tensor_metrics(a, b) for a, b in zip(ref_trajectory, trajectory)]
    video_quality = quality_verdict(video, limits.get("video") if limits else None)
    trajectory_quality = [
        quality_verdict(m, limits.get("trajectory") if limits else None) for m in latent_metrics
    ]
    statuses = [video_quality, *trajectory_quality]
    quality = (
        "fail"
        if "fail" in statuses
        else ("pass" if all(s == "pass" for s in statuses) else "uncalibrated")
    )
    return dict(
        video=video,
        video_quality=video_quality,
        trajectory=latent_metrics,
        trajectory_quality=trajectory_quality,
        quality=quality,
    )


def require_matching_capture(reference, actual):
    required = {
        "model",
        "checkpoint",
        "shape",
        "steps",
        "guidance",
        "prompt",
        "noise_hash",
        "timestep_hash",
        "dit_timestep_hash",
        "guidance_hash",
        "vae_tiling",
        "output_boundary",
    }
    for row in (reference, actual):
        if not required.issubset(row.get("settings", {})):
            raise ValueError("quality capture contract is incomplete")
    if reference["seed"] != actual["seed"] or reference["settings"] != actual["settings"]:
        raise ValueError("quality capture contract differs between reference and candidate")


def paired_performance(difflet, omni, *, settings, mode, ratio_limit=1.05):
    if not omni:
        return {"status": "baseline_unavailable", "mode": mode}
    if len(difflet) != len(omni):
        raise ValueError("paired row counts differ")
    ratios, denoise = [], []
    for d, o in zip(difflet, omni):
        if d.get("mode") != mode or o.get("mode") != mode:
            raise ValueError("execution mode mismatch")
        if d.get("settings") != o.get("settings") or any(
            d.get("settings", {}).get(k) != v for k, v in settings.items()
        ):
            raise ValueError("generation settings mismatch")
        if not d.get("execution_verified") or not o.get("execution_verified"):
            return {"status": "inconclusive", "reason": "execution mode unverified", "mode": mode}
        for row in (d, o):
            if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ("latency_s", "denoise_s")):
                raise ValueError("latencies must be positive and finite")
        ratios.append(d["latency_s"] / o["latency_s"])
        denoise.append(d["denoise_s"] / o["denoise_s"])
    if not 6 <= len(ratios) <= 12:
        return {"status": "inconclusive", "reason": "requires 6 to 12 matched pairs", "mode": mode}
    g = random.Random(0)

    def interval(values):
        samples = sorted(statistics.median(g.choices(values, k=len(values))) for _ in range(2000))
        return [samples[49], samples[1949]]

    generation_ci, denoise_ci = interval(ratios), interval(denoise)
    status = "inconclusive"
    if max(generation_ci[1], denoise_ci[1]) <= ratio_limit:
        status = "pass"
    elif generation_ci[0] > ratio_limit or denoise_ci[0] > ratio_limit:
        status = "fail"
    return dict(
        status=status,
        mode=mode,
        generation_ratio=statistics.median(ratios),
        denoise_ratio=statistics.median(denoise),
        generation_ci95=generation_ci,
        denoise_ci95=denoise_ci,
        proposed_ratio_limit=ratio_limit,
        pairs=len(ratios),
        paired_ratios=ratios,
        interval_method="paired bootstrap median; 2000 draws; seed0; small sample",
    )


def _hash(tensor):
    # Size-one dimensions can be contiguous with a zero/non-unit stride. A fresh
    # flat layout also handles scalars before viewing their raw bytes.
    flat = tensor.detach().cpu().reshape(-1).clone(memory_format=torch.contiguous_format)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def _load(engine, mode, args):
    from difflet.backends.gpu.application import compile_blocks

    if engine == "diffusers":
        from diffusers import HunyuanVideoPipeline

        pipe = HunyuanVideoPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        transformer = pipe.transformer
        if mode == "compiled":
            compile_blocks(
                [transformer.transformer_blocks, transformer.single_transformer_blocks],
                dynamic=False,
            )
        app = None
    else:
        from difflet.backends.cuda.hunyuan_video.application import CudaHunyuanVideoApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        app = CudaHunyuanVideoApplication(
            model_path=args.model_path,
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="bf16",
            shape=dict(height=args.height, width=args.width, num_frames=args.num_frames),
            torch_compile=mode == "compiled",
        )
        artifact = str(args.output_dir / f"manifest-{mode}")
        app.compile(artifact)
        app.load(artifact)
        pipe, transformer = app.pipe, app.transformer.module
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    return app, pipe, transformer


def _request(app, pipe, transformer, args, *, seed, capture=False):
    shape = (1, 16, (args.num_frames - 1) // 4 + 1, args.height // 8, args.width // 8)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    steps, trajectory, dit_times, guidance_values = [], [], [], []

    def record_forward(module, inputs, kwargs):
        if not dit_times:
            begin.record()
        timestep = kwargs["timestep"] if "timestep" in kwargs else inputs[1]
        guidance = kwargs["guidance"] if "guidance" in kwargs else inputs[5]
        dit_times.append(timestep.detach().reshape(-1))
        guidance_values.append(guidance.detach().reshape(-1))

    handle = transformer.register_forward_pre_hook(record_forward, with_kwargs=True)
    step = pipe.scheduler.step

    def record_step(*a, **kw):
        out = step(*a, **kw)
        # Keep metadata on its original device until the timer has stopped.
        # A per-step .cpu() introduces a synchronization into the denoise loop.
        steps.append(a[1].detach())
        if len(steps) == args.steps:
            end.record()
        if capture:
            trajectory.append(out[0].detach().cpu().clone())
        return out

    pipe.scheduler.step = record_step
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            if app is not None:
                frames = app.generate(
                    args.prompt, latents=noise, num_inference_steps=args.steps, guidance_scale=6.0
                )
            else:
                out = pipe(
                    prompt=args.prompt,
                    latents=noise,
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    guidance_scale=6.0,
                    true_cfg_scale=1.0,
                    max_sequence_length=256,
                    output_type="pt",
                ).frames
                frames = out.permute(0, 2, 1, 3, 4).float().cpu() * 2 - 1
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        if len(steps) != args.steps or len(dit_times) != args.steps:
            raise RuntimeError("observed transformer/scheduler execution differs from request")
        row = dict(
            latency_s=latency,
            denoise_s=begin.elapsed_time(end) / 1000,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            seed=seed,
            dit_timestep_dtypes=[str(t.dtype) for t in dit_times],
            settings=dict(
                model="hunyuan_video",
                checkpoint=args.checkpoint_revision,
                shape=[args.height, args.width, args.num_frames],
                steps=args.steps,
                guidance=6.0,
                prompt=args.prompt,
                noise_hash=_hash(noise),
                timestep_hash=_hash(torch.stack(steps).float()),
                dit_timestep_hash=_hash(torch.cat([t.float() for t in dit_times])),
                guidance_hash=_hash(torch.cat([g.float() for g in guidance_values])),
                vae_tiling=args.vae_tiling,
                output_boundary="cpu-fp32-video",
            ),
        )
        return frames, trajectory, row
    finally:
        pipe.scheduler.step = step
        handle.remove()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--checkpoint-revision", required=True)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=61)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--prompt", default="a cat walking across a sunlit garden")
    p.add_argument("--mode", choices=("eager", "compiled", "both"), default="both")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--calibration-limits", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def profile_request_args(args):
    if args.profile_steps <= 0:
        raise ValueError("profile steps must be positive")
    return argparse.Namespace(**{**vars(args), "steps": min(args.steps, args.profile_steps)})


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.checkpoint_revision = checkpoint_revision(args.model_path, args.checkpoint_revision)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU qualification requires CUDA; no model or parity gate was run")
    if args.steps <= 0 or args.repeats <= 0:
        raise ValueError("steps and repeats must be positive")
    modes = ("eager", "compiled") if args.mode == "both" else (args.mode,)
    traced_args = profile_request_args(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    limits = json.loads(args.calibration_limits.read_text()) if args.calibration_limits else None
    report = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        checkpoint=args.checkpoint_revision,
        modes={},
        omni={
            m: {
                "status": "baseline_unavailable",
                "reason": "actual original-model route/mode not qualified by this tool",
            }
            for m in modes
        },
        runner_completed=False,
    )
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    for mode in modes:
        samples, controls = {}, {}
        for engine in ("diffusers", "difflet"):
            report["modes"].setdefault(mode, {})[engine] = {"status": "incomplete"}
            report_path.write_text(json.dumps(report, indent=2))
            app, pipe, model = _load(engine, mode, args)
            _request(app, pipe, model, args, seed=seeds[0])  # Explicit untimed warm-up.
            clean, captures = [], {}
            for seed in seeds:
                frames, trajectory, capture_row = _request(
                    app, pipe, model, args, seed=seed, capture=True
                )
                capture_row = {**capture_row, "engine": engine, "mode": mode, "capture": True}
                capture_path = args.output_dir / f"{engine}-{mode}-seed{seed}.pt"
                torch.save(
                    dict(frames=frames, trajectory=trajectory, capture_metadata=capture_row),
                    capture_path,
                )
                digest = hashlib.sha256()
                with capture_path.open("rb") as source:
                    for block in iter(lambda: source.read(8388608), b""):
                        digest.update(block)
                captures[seed] = {
                    "artifact": capture_path.name,
                    "sha256": digest.hexdigest(),
                    "capture_metadata": capture_row,
                }
                if engine == "diffusers":
                    controls[seed] = (frames, trajectory, capture_row)
                else:
                    reference, ref_trajectory, reference_row = controls[seed]
                    require_matching_capture(reference_row, capture_row)
                    samples[seed] = compare_outputs(
                        reference,
                        frames,
                        ref_trajectory,
                        trajectory,
                        steps=args.steps,
                        limits=limits,
                    )
                    samples[seed]["capture_contract"] = capture_row["settings"]
                for _ in range(args.repeats):
                    _, _, row = _request(app, pipe, model, args, seed=seed)
                    row["mode"] = mode
                    clean.append(row)
            # Separate execution trace: clean timing above includes no profiler.
            scopes = []

            def before(module, inputs):
                scope = torch.profiler.record_function("phase/DiT")
                scope.__enter__()
                scopes.append(scope)

            def after(module, inputs, output):
                scopes.pop().__exit__(None, None, None)

            handles = [
                model.register_forward_pre_hook(before),
                model.register_forward_hook(after, always_call=True),
            ]
            try:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    _request(app, pipe, model, traced_args, seed=seeds[0])
            finally:
                for handle in handles:
                    handle.remove()
            events = profile.events()
            names = sorted({e.name for e in events})

            def under_dit(event):
                parent = event.cpu_parent
                while parent is not None:
                    if parent.name == "phase/DiT":
                        return True
                    parent = parent.cpu_parent
                return False

            compiled_seen = any(
                "compiled region" in e.name.lower() and under_dit(e) for e in events
            )
            cuda_seen = any(e.device_type == torch.autograd.DeviceType.CUDA for e in events)
            verified = (
                cuda_seen
                and "phase/DiT" in names
                and (compiled_seen if mode == "compiled" else not compiled_seen)
            )
            for row in clean:
                row["execution_verified"] = verified
            report["modes"].setdefault(mode, {})[engine] = dict(
                status="completed" if verified else "execution_unverified",
                clean=clean,
                captures=captures,
                execution_verified=verified,
                trace_events=names,
                profile_steps=traced_args.steps,
            )
            profile.export_chrome_trace(str(args.output_dir / f"{engine}-{mode}.json"))
            del profile, app, pipe, model
            gc.collect()
            torch.cuda.empty_cache()
        report["modes"][mode]["correctness"] = samples
        report_path.write_text(json.dumps(report, indent=2))
    # Calibrated quality and Omni parity are independent, never inferred from runner completion.
    metrics_pass = all(
        r["quality"] == "pass" for m in modes for r in report["modes"][m]["correctness"].values()
    )
    execution_pass = all(
        report["modes"][m][engine]["execution_verified"]
        for m in modes
        for engine in ("diffusers", "difflet")
    )
    accepted = metrics_pass and execution_pass
    report["numerical_quality_metrics_passing"] = metrics_pass
    report["requested_modes_execution_verified"] = execution_pass
    report["runner_completed"] = True
    report["reference_quality_calibrated_and_passing"] = accepted
    report_path.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "reference_quality_calibrated_and_passing": accepted,
                "omni_parity": "unverified",
                "report": str(args.output_dir / "report.json"),
            }
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
