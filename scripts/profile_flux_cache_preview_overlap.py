#!/usr/bin/env python3
"""Measure whether a resident FLUX VAE preview fits inside cache-step bubbles.

The experiment launches the already-loaded VAE decoder from the step-20 x0
estimate in a Python thread while the aggressive cache profile continues.  It
does not score the preview and makes no quality or serving claim; it measures
only readiness at the terminal@29 deadline and contention at the next real
Transformer anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flux_cache_brake_intervention import _git_identity, _write_json  # noqa: E402


HARDWARE_ACK = "I am profiling resident VAE overlap in cache bubbles"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
CANDIDATE = ROOT / "benchmark/flux_cache/adaptive-vqa-stress-i32-k40-candidate.json"
SOURCE_QUALITY = Path(
    "/home/ubuntu/difflet-artifacts/flux-cache-warmup-vqa-stress-extreme-20260804/quality-input-v2.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _tensor_digest(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    if str(value.dtype) == "torch.bfloat16":
        value = value.view(dtype=__import__("torch").uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _extract_latent(result: Any) -> Any:
    images = getattr(result, "images", None)
    if images is None and isinstance(result, (tuple, list)) and result:
        images = result[0]
    if isinstance(images, (tuple, list)):
        images = images[0]
    if images is None:
        raise RuntimeError("latent pipeline run returned no output")
    return images


def _sample() -> dict[str, Any]:
    source = _load_json(SOURCE_QUALITY)
    for comparison in source["comparisons"]:
        if str(comparison["sample_id"]) == "p000-s2":
            return {
                "sample_id": "p000-s2",
                "prompt": str(comparison["prompt"]),
                "seed": int(comparison["seed"]),
            }
    raise ValueError("source quality report has no p000-s2 sample")


def _run_once(pipe: Any, arm: Any, sample: dict[str, Any], *, preview: bool) -> dict[str, Any]:
    import torch

    flux = pipe.app.pipe
    adapter = arm.build_pipeline_adapter(50)
    flux.teacache_controller = adapter
    flux._tc_record = True
    flux._tc_output_dynamics_record = False
    callback_rows: list[dict[str, Any]] = []
    previous_latent = None
    preview_thread: threading.Thread | None = None
    preview_state: dict[str, Any] = {
        "launched": False,
        "started_s": None,
        "completed_s": None,
        "error": None,
        "output_digest": None,
    }
    started = time.perf_counter()

    def decode_preview(x0: Any) -> None:
        preview_state["started_s"] = time.perf_counter() - started
        try:
            unpacked = flux._unpack_latents(x0, 1024, 1024, flux.vae_scale_factor)
            vae_input = (
                unpacked / float(flux.vae.config.scaling_factor)
                + float(flux.vae.config.shift_factor)
            ).to(dtype=torch.bfloat16)
            with torch.no_grad():
                decoded = flux.vae.decode(vae_input, return_dict=False)[0]
            preview_state["output_digest"] = _tensor_digest(decoded)
        except BaseException as error:  # thread errors must be surfaced on the main rank
            preview_state["error"] = f"{type(error).__name__}: {error}"
        finally:
            preview_state["completed_s"] = time.perf_counter() - started

    def callback(_pipeline: Any, step_index: int, _timestep: Any, values: dict[str, Any]):
        nonlocal previous_latent, preview_thread
        now = time.perf_counter() - started
        callback_rows.append(
            {
                "step_index": int(step_index),
                "callback_s": now,
                "preview_complete": preview_state["completed_s"] is not None,
            }
        )
        latents = values["latents"]
        if preview and step_index == 18:
            previous_latent = latents.detach().clone()
        if preview and step_index == 19:
            if previous_latent is None:
                raise RuntimeError("step-20 preview has no previous latent")
            sigmas = flux.scheduler.sigmas.detach().float().cpu()
            previous_sigma = float(sigmas[19])
            current_sigma = float(sigmas[20])
            current = latents.detach().clone()
            x0 = current - current_sigma * (current - previous_latent) / (
                current_sigma - previous_sigma
            )
            preview_state["launched"] = True
            preview_thread = threading.Thread(
                target=decode_preview,
                args=(x0,),
                name="difflet-preview-vae",
                daemon=False,
            )
            preview_thread.start()
        return values

    generator = torch.Generator().manual_seed(int(sample["seed"]))
    result = pipe(
        prompt=sample["prompt"],
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        output_type="latent",
        generator=generator,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    denoise_completed_s = time.perf_counter() - started
    if preview_thread is not None:
        preview_thread.join(timeout=30.0)
        if preview_thread.is_alive():
            raise TimeoutError("resident VAE preview did not complete within 30 seconds")
    completed_s = time.perf_counter() - started
    if preview_state["error"] is not None:
        raise RuntimeError(str(preview_state["error"]))
    by_step = {int(row["step_index"]): row for row in callback_rows}
    if set(by_step) != set(range(50)):
        raise RuntimeError("callback did not observe all 50 denoising steps")
    terminal29_deadline_s = float(by_step[27]["callback_s"])
    next_anchor_start_bound_s = float(by_step[38]["callback_s"])
    preview_completed_s = preview_state["completed_s"]
    return {
        "preview": preview,
        "denoise_completed_s": denoise_completed_s,
        "thread_join_completed_s": completed_s,
        "terminal29_deadline_s": terminal29_deadline_s,
        "next_anchor_step39_start_bound_s": next_anchor_start_bound_s,
        "preview_state": preview_state,
        "ready_by_terminal29": bool(
            preview_completed_s is not None and preview_completed_s <= terminal29_deadline_s
        ),
        "ready_before_step39_anchor": bool(
            preview_completed_s is not None and preview_completed_s <= next_anchor_start_bound_s
        ),
        "step20_to_terminal29_s": terminal29_deadline_s - float(by_step[19]["callback_s"]),
        "step20_to_step39_anchor_s": next_anchor_start_bound_s
        - float(by_step[19]["callback_s"]),
        "final_latent_sha256": _tensor_digest(_extract_latent(result)),
        "runner_stats": adapter.stats(),
        "callback_rows": callback_rows,
    }


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def run(args: argparse.Namespace) -> Path:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "4")
    os.environ.setdefault("NEURON_RT_NUM_CORES", "4")
    from scripts.collect_flux_cache_ab import _load_pipeline, load_adaptive_candidate

    pipe_args = SimpleNamespace(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tp_degree=4,
        dtype="bfloat16",
        compile_cache_dir=args.compile_cache_dir,
        height=1024,
        width=1024,
        force_compile=False,
        skip_warmup=bool(args.skip_warmup),
        collect_online_signals=True,
    )
    load_started = time.perf_counter()
    pipe = _load_pipeline(pipe_args)
    load_seconds = time.perf_counter() - load_started
    arm = load_adaptive_candidate(CANDIDATE)
    sample = _sample()
    rank = _rank()
    if rank == 0:
        print(f"[preview-overlap] loaded in {load_seconds:.3f}s", flush=True)

    # One unrecorded request removes first-request effects after skip_warmup loads.
    _run_once(pipe, arm, sample, preview=False)
    sequence = (False, True, False, True)
    rows = []
    for index, preview in enumerate(sequence, start=1):
        row = _run_once(pipe, arm, sample, preview=preview)
        rows.append(row)
        if rank == 0:
            print(
                f"[preview-overlap] {index}/{len(sequence)} preview={preview} "
                f"denoise={row['denoise_completed_s']:.3f}s "
                f"gap29={row['step20_to_terminal29_s']:.4f}s "
                f"ready29={row['ready_by_terminal29']}",
                flush=True,
            )
    _barrier()
    out = args.out.resolve()
    if rank == 0:
        baselines = [row for row in rows if not row["preview"]]
        previews = [row for row in rows if row["preview"]]
        result = {
            "schema": "difflet-flux-cache-preview-overlap-profile",
            "schema_revision": 1,
            "serving_claim": False,
            "execution": {
                **_git_identity(),
                "instance_type": "trn2.3xlarge",
                "neuron_cores": 4,
                "transformer_tp_degree": 4,
                "decoder_tp_degree": 1,
                "compiled_async_mode": False,
                "load_seconds": load_seconds,
                "rank_count": 1,
                "runtime_layout": "single Python process managing four NeuronCores",
            },
            "signal": {
                "launch": "after zero-based denoising step 19 (step-20 x0)",
                "deadline": "after zero-based step 27, before terminal@29 action",
                "decoder": "already-loaded Neuron FLUX VAE",
                "extra_transformer_calls": 0,
                "quality_scoring": False,
            },
            "sample": sample,
            "candidate": str(CANDIDATE),
            "runs": rows,
            "summary": {
                "all_preview_latents_match_baseline": all(
                    row["final_latent_sha256"] == baselines[0]["final_latent_sha256"]
                    for row in rows
                ),
                "preview_ready_by_terminal29_count": sum(
                    bool(row["ready_by_terminal29"]) for row in previews
                ),
                "preview_ready_by_terminal29_total": len(previews),
                "preview_ready_before_step39_count": sum(
                    bool(row["ready_before_step39_anchor"]) for row in previews
                ),
                "preview_ready_before_step39_total": len(previews),
                "baseline_mean_denoise_s": sum(
                    float(row["denoise_completed_s"]) for row in baselines
                )
                / len(baselines),
                "preview_mean_denoise_s": sum(
                    float(row["denoise_completed_s"]) for row in previews
                )
                / len(previews),
                "preview_mean_added_denoise_s": (
                    sum(float(row["denoise_completed_s"]) for row in previews)
                    / len(previews)
                    - sum(float(row["denoise_completed_s"]) for row in baselines)
                    / len(baselines)
                ),
                "baseline_mean_step20_to_terminal29_s": sum(
                    float(row["step20_to_terminal29_s"]) for row in baselines
                )
                / len(baselines),
                "mean_step20_to_terminal29_s": sum(
                    float(row["step20_to_terminal29_s"]) for row in previews
                )
                / len(previews),
                "mean_terminal29_deadline_inflation_s": (
                    sum(float(row["step20_to_terminal29_s"]) for row in previews)
                    / len(previews)
                    - sum(float(row["step20_to_terminal29_s"]) for row in baselines)
                    / len(baselines)
                ),
                "mean_preview_decode_s": sum(
                    float(row["preview_state"]["completed_s"])
                    - float(row["preview_state"]["started_s"])
                    for row in previews
                )
                / len(previews),
                "interpretation": (
                    "preview completion before the observed terminal29 callback is not "
                    "hidden overlap when the preview itself inflates that callback deadline"
                ),
            },
        }
        _write_json(out, result, add_digest=True)
        print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
        print(f"[preview-overlap] result={out}", flush=True)
    _barrier()
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-cache-dir")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benchmark/flux_cache/preview-overlap-profile.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
