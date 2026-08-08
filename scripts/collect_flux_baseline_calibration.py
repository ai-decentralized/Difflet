#!/usr/bin/env python3
"""Collect full-DiT baseline PNGs for one registered FLUX resolution bucket."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.offline.cache_profile.collector import (
    build_baseline_adapter,
    extract_image,
    load_pipeline,
    prepare_output_directory,
    sample_matrix,
    utc_now,
    write_json,
)
from scripts.flux_cache_protocol import (
    build_experiment_protocol,
    load_prompt_suite,
    python_source_sha256,
)
from scripts.multires_quality_contract import (
    BASELINE_MANIFEST_SCHEMA,
    SCHEMA_REVISION,
    bucket_for,
    load_protocol,
    sha256_file,
    validate_observed_generation,
)

BASELINE_FOREGROUND_ACK = "I am running FLUX multires baseline calibration"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _run_baseline_sample(
    pipe: Any,
    *,
    sample: dict[str, Any],
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    output_root: Path,
) -> dict[str, Any]:
    import torch

    generator = torch.Generator().manual_seed(int(sample["seed"]))
    started = time.perf_counter()
    result = pipe(
        prompt=sample["prompt"],
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
        generator=generator,
    )
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("pipeline returned an invalid wall-clock duration")
    image = extract_image(result)
    image_path = output_root / "artifacts" / "baseline" / f"{sample['sample_id']}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return {
        **sample,
        "elapsed_s": float(elapsed),
        "artifacts": {
            "image": _relative(image_path, output_root),
            "image_sha256": sha256_file(image_path),
        },
    }


def collect(args: argparse.Namespace) -> Path:
    if not args.allow_hardware:
        raise RuntimeError("FLUX baseline collection requires --allow-hardware")
    if args.foreground_ack != BASELINE_FOREGROUND_ACK:
        raise RuntimeError(f"--foreground-ack must equal {BASELINE_FOREGROUND_ACK!r}")

    registration_path = Path(args.registration).expanduser().resolve()
    registration = load_protocol(registration_path)
    bucket = bucket_for(registration, args.bucket_id)
    registered_source = registration["source_registration"]["python_source_sha256"]
    observed_source = python_source_sha256()
    if observed_source != registered_source:
        raise RuntimeError("current Python source differs from the prospectively registered source")
    os.environ["DIFFLET_ALLOW_DIRTY_PYTHON_SOURCE_SHA256"] = registered_source

    prompt_binding = bucket["prompt_suite"]
    prompt_path = (ROOT / prompt_binding["path"]).resolve()
    prompt_selection = load_prompt_suite(prompt_path, prompt_binding["split"])
    samples = sample_matrix(prompt_selection.prompts, prompt_binding["seeds"])
    output_root = Path(args.out_dir).expanduser().resolve()
    prepare_output_directory(output_root)
    started_at = utc_now()

    controlled = registration["controlled_generation"]
    pipeline_args = argparse.Namespace(
        dtype=controlled["dtype"],
        model_id=controlled["model_id"],
        model_revision=controlled["model_revision"],
        tp_degree=controlled["tp_degree"],
        compile_cache_dir=args.compile_cache_dir,
        height=bucket["height"],
        width=bucket["width"],
        force_compile=bool(args.force_compile),
        skip_warmup=bool(args.skip_warmup),
        collect_online_signals=False,
    )
    pipe = load_pipeline(pipeline_args)
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False
    flux_pipeline._tc_output_dynamics_record = False
    adapter = build_baseline_adapter(controlled["num_steps"])
    flux_pipeline.teacache_controller = adapter
    experiment = build_experiment_protocol(
        pipe=pipe,
        scheduler=flux_pipeline.scheduler,
        prompt_selection=prompt_selection,
        seeds=prompt_binding["seeds"],
        num_steps=controlled["num_steps"],
        height=bucket["height"],
        width=bucket["width"],
        guidance_scale=controlled["guidance_scale"],
        dtype=controlled["dtype"],
        tp_degree=controlled["tp_degree"],
        requested_model_revision=controlled["model_revision"],
        cache_coordinate="index",
        pipeline_warmup_enabled=not bool(args.skip_warmup),
    )
    validate_observed_generation(experiment, registration, bucket)
    print(
        f"[flux-baseline-calibration] protocol={experiment['sha256']} "
        f"bucket={args.bucket_id} samples={len(samples)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for sample in samples:
        row = _run_baseline_sample(
            pipe,
            sample=sample,
            height=bucket["height"],
            width=bucket["width"],
            num_steps=controlled["num_steps"],
            guidance_scale=controlled["guidance_scale"],
            output_root=output_root,
        )
        stats = adapter.stats()
        if stats["full_steps"] != controlled["num_steps"] or stats["skipped_steps"] != 0:
            raise RuntimeError("baseline adapter did not execute every denoise step")
        rows.append(row)
        print(
            f"[flux-baseline-calibration] {args.bucket_id} "
            f"{sample['sample_id']} {row['elapsed_s']:.3f}s",
            flush=True,
        )

    manifest = {
        "schema": BASELINE_MANIFEST_SCHEMA,
        "schema_revision": SCHEMA_REVISION,
        "registration": {
            "path": str(registration_path),
            "file_sha256": sha256_file(registration_path),
            "content_sha256": registration["sha256"],
        },
        "bucket_id": args.bucket_id,
        "python_source_sha256": registered_source,
        "started_at": started_at,
        "completed_at": utc_now(),
        "protocol": experiment,
        "baseline_samples": rows,
    }
    manifest_path = output_root / "baseline-calibration-manifest-v1.json"
    write_json(manifest_path, manifest)
    print(f"[flux-baseline-calibration] manifest -> {manifest_path}", flush=True)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--bucket-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--compile-cache-dir", default=None)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        collect(args)
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
