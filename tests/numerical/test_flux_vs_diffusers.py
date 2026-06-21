"""Flux trajectory numerical alignment against Hugging Face diffusers.

This test is intentionally opt-in. It loads FLUX reference weights and Difflet's
compiled Trainium artifacts, so it is too slow and hardware-heavy for default
test runs.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from difflet import DiffletParallelConfig, DiffletPipeline


pytestmark = [
    pytest.mark.numerical,
    pytest.mark.neuron,
    pytest.mark.slow,
]


def test_flux_full_trajectory_cosine_matches_diffusers():
    if os.environ.get("DIFFLET_RUN_FLUX_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_FLUX_NUMERICAL=1 to run the Flux numerical gate")

    from diffusers import FluxPipeline

    model_id = os.environ.get("DIFFLET_FLUX_NUMERICAL_MODEL", "black-forest-labs/FLUX.1-dev")
    prompt = os.environ.get("DIFFLET_FLUX_NUMERICAL_PROMPT", "a cat")
    height = int(os.environ.get("DIFFLET_FLUX_NUMERICAL_HEIGHT", "1024"))
    width = int(os.environ.get("DIFFLET_FLUX_NUMERICAL_WIDTH", "1024"))
    steps = int(os.environ.get("DIFFLET_FLUX_NUMERICAL_STEPS", "28"))
    seed = int(os.environ.get("DIFFLET_FLUX_NUMERICAL_SEED", "42"))
    guidance_scale = float(os.environ.get("DIFFLET_FLUX_NUMERICAL_GUIDANCE_SCALE", "3.5"))
    max_sequence_length = int(os.environ.get("DIFFLET_FLUX_NUMERICAL_MAX_SEQUENCE_LENGTH", "512"))
    threshold = float(os.environ.get("DIFFLET_FLUX_NUMERICAL_MIN_COSINE", "0.95"))
    tp_degree = int(os.environ.get("DIFFLET_FLUX_NUMERICAL_TP_DEGREE", "4"))
    local_files_only = _env_flag("DIFFLET_FLUX_NUMERICAL_LOCAL_FILES_ONLY", default=False)
    cache_dir = os.environ.get("DIFFLET_FLUX_NUMERICAL_CACHE_DIR")

    common_call_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "output_type": "latent",
    }

    ref_steps = _run_diffusers_reference(
        FluxPipeline=FluxPipeline,
        model_id=model_id,
        seed=seed,
        local_files_only=local_files_only,
        call_kwargs=common_call_kwargs,
    )
    difflet_steps = _run_difflet_flux(
        model_id=model_id,
        seed=seed,
        tp_degree=tp_degree,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        call_kwargs=common_call_kwargs,
    )

    assert len(ref_steps) == len(difflet_steps) == steps
    metrics = _trajectory_cosines(reference=ref_steps, actual=difflet_steps)
    _write_metrics_if_requested(metrics)

    min_cosine = min(item["cosine"] for item in metrics)
    assert min_cosine >= threshold, _format_failure(metrics, threshold)


def _run_diffusers_reference(
    *,
    FluxPipeline: type,
    model_id: str,
    seed: int,
    local_files_only: bool,
    call_kwargs: dict[str, Any],
) -> list[torch.Tensor]:
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    step_latents: list[torch.Tensor] = []

    with torch.inference_mode():
        pipe(
            **call_kwargs,
            generator=torch.Generator().manual_seed(seed),
            callback_on_step_end=_capture_latents(step_latents),
            callback_on_step_end_tensor_inputs=["latents"],
        )

    del pipe
    gc.collect()
    return step_latents


def _run_difflet_flux(
    *,
    model_id: str,
    seed: int,
    tp_degree: int,
    cache_dir: str | None,
    local_files_only: bool,
    call_kwargs: dict[str, Any],
) -> list[torch.Tensor]:
    pipe = DiffletPipeline.from_pretrained(
        model_id,
        model_type="flux",
        parallel=DiffletParallelConfig(tp_degree=tp_degree),
        dtype=torch.bfloat16,
        height=call_kwargs["height"],
        width=call_kwargs["width"],
        compile_cache_dir=cache_dir,
        local_files_only=local_files_only,
        skip_warmup=True,
    )
    pipe.app.pipe.set_progress_bar_config(disable=True)
    step_latents: list[torch.Tensor] = []

    with torch.inference_mode():
        pipe(
            **call_kwargs,
            generator=torch.Generator().manual_seed(seed),
            callback_on_step_end=_capture_latents(step_latents),
            callback_on_step_end_tensor_inputs=["latents"],
        )

    return step_latents


def _capture_latents(target: list[torch.Tensor]):
    def callback(_pipe, _step, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"].detach().to("cpu", dtype=torch.float32).clone()
        target.append(latents)
        return {}

    return callback


def _trajectory_cosines(
    *,
    reference: list[torch.Tensor],
    actual: list[torch.Tensor],
) -> list[dict[str, float | int | list[int]]]:
    metrics: list[dict[str, float | int | list[int]]] = []
    for index, (ref, out) in enumerate(zip(reference, actual)):
        assert list(ref.shape) == list(out.shape), (
            f"step {index} shape mismatch: diffusers={tuple(ref.shape)} difflet={tuple(out.shape)}"
        )
        cosine = F.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()
        max_abs = (ref - out).abs().max().item()
        mean_abs = (ref - out).abs().mean().item()
        metrics.append(
            {
                "step": index,
                "cosine": cosine,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "shape": list(ref.shape),
            }
        )
    return metrics


def _write_metrics_if_requested(metrics: list[dict[str, float | int | list[int]]]) -> None:
    path = os.environ.get("DIFFLET_FLUX_NUMERICAL_METRICS")
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_failure(metrics: list[dict[str, float | int | list[int]]], threshold: float) -> str:
    rows = ", ".join(
        f"step={item['step']} cosine={item['cosine']:.6f} "
        f"max_abs={item['max_abs']:.6f} mean_abs={item['mean_abs']:.6f}"
        for item in metrics
    )
    return f"Flux trajectory cosine below {threshold}: {rows}"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
