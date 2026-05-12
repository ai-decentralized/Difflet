"""HunyuanVideo trajectory numerical alignment against Hugging Face diffusers.

This test is intentionally opt-in. It loads real HunyuanVideo transformer
weights and Nova's compiled Trainium artifacts, so it is too slow and
hardware-heavy for default test runs.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


pytestmark = [
    pytest.mark.numerical,
    pytest.mark.neuron,
    pytest.mark.slow,
]


def test_hunyuan_video_cached_trajectory_cosine_matches_diffusers():
    if os.environ.get("NOVA_RUN_HUNYUAN_VIDEO_NUMERICAL") != "1":
        pytest.skip(
            "set NOVA_RUN_HUNYUAN_VIDEO_NUMERICAL=1 to run the HunyuanVideo numerical gate"
        )

    source_dir = Path(
        os.environ.get(
            "NOVA_HUNYUAN_VIDEO_SOURCE_DIR",
            "/home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real",
        )
    )
    compiled_dir = Path(
        os.environ.get(
            "NOVA_HUNYUAN_VIDEO_COMPILED_DIR",
            "/home/ubuntu/nova/.nova-cache/hunyuan_n4_20d40s2r/compiled",
        )
    )
    bundle_path = Path(
        os.environ.get(
            "NOVA_HUNYUAN_VIDEO_BUNDLE",
            "/home/ubuntu/nova/.nova-cache/hunyuan_dit_inputs/cat_walking_4step.safetensors",
        )
    )
    threshold = float(os.environ.get("NOVA_HUNYUAN_VIDEO_MIN_COSINE", "0.999"))
    tp_degree = int(os.environ.get("NOVA_HUNYUAN_VIDEO_TP_DEGREE", "4"))
    reference_dtype = _parse_dtype(
        os.environ.get("NOVA_HUNYUAN_VIDEO_REFERENCE_DTYPE", "bfloat16")
    )
    num_threads = int(os.environ.get("NOVA_HUNYUAN_VIDEO_NUM_THREADS", "0"))
    if num_threads > 0:
        torch.set_num_threads(num_threads)

    scheduler_config = source_dir / "scheduler" / "scheduler_config.json"
    assert scheduler_config.exists(), (
        "HunyuanVideo trajectory parity requires a real HF scheduler at "
        f"{scheduler_config}; otherwise Nova would use its fallback scheduler."
    )

    meta, tensors = _load_artifact(bundle_path)
    timesteps = tensors["timesteps"]

    ref_steps = _run_diffusers_reference(
        source_dir=source_dir,
        tensors=tensors,
        timesteps=timesteps,
        dtype=reference_dtype,
    )
    nova_steps = _run_nova_trainium(
        source_dir=source_dir,
        compiled_dir=compiled_dir,
        meta=meta,
        tensors=tensors,
        timesteps=timesteps,
        tp_degree=tp_degree,
    )

    assert len(ref_steps) == len(nova_steps) == int(meta["num_inference_steps"])
    metrics = _trajectory_cosines(reference=ref_steps, actual=nova_steps)
    _write_metrics_if_requested(metrics)

    min_cosine = min(item["cosine"] for item in metrics)
    assert min_cosine >= threshold, _format_failure(metrics, threshold)


def _load_artifact(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    meta_path = Path(str(path) + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, load_file(str(path))


def _run_diffusers_reference(
    *,
    source_dir: Path,
    tensors: dict[str, torch.Tensor],
    timesteps: torch.Tensor,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel as DiffusersHunyuanVideoTransformer3DModel,
    )

    device = torch.device("cpu")
    transformer = DiffusersHunyuanVideoTransformer3DModel.from_pretrained(
        source_dir / "transformer",
        torch_dtype=dtype,
    ).eval()
    transformer.to(device=device, dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(source_dir / "scheduler")
    timesteps = _set_scheduler_timesteps(scheduler, timesteps, device=device)

    latents = tensors["latents_init"].to(device=device, dtype=dtype)
    trajectory: list[torch.Tensor] = []
    with torch.inference_mode():
        for timestep in timesteps:
            timestep_batch = timestep.to(device=device, dtype=dtype).expand(latents.shape[0])
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep_batch,
                encoder_hidden_states=tensors["encoder_hidden_states"].to(
                    device=device,
                    dtype=dtype,
                ),
                encoder_attention_mask=tensors["encoder_attention_mask"].to(device=device),
                pooled_projections=tensors["pooled_projections"].to(device=device, dtype=dtype),
                guidance=tensors["guidance"].to(device=device, dtype=dtype),
                return_dict=False,
            )[0]
            latents = scheduler.step(
                noise_pred.to(dtype=latents.dtype),
                timestep,
                latents,
                return_dict=False,
            )[0]
            trajectory.append(latents.detach().to("cpu", dtype=torch.float32).clone())

    del transformer, scheduler
    gc.collect()
    return trajectory


def _run_nova_trainium(
    *,
    source_dir: Path,
    compiled_dir: Path,
    meta: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    timesteps: torch.Tensor,
    tp_degree: int,
) -> list[torch.Tensor]:
    os.environ.setdefault("NOVA_BACKEND", "trainium")

    from nova.models.hunyuan_video.application import (
        HunyuanVideoDiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.parallel_config import NovaParallelConfig

    app = NeuronHunyuanVideoApplication(
        model_path=str(source_dir),
        parallel=NovaParallelConfig(tp_degree=tp_degree, cp_enabled=False),
        dtype=torch.bfloat16,
        shape={
            "height": meta["height"],
            "width": meta["width"],
            "num_frames": meta["num_frames"],
        },
        text_seq_len=meta["text_seq_len"],
    )
    assert app.pipeline.scheduler is not None
    app.load(str(compiled_dir), skip_warmup=True)

    bundle = HunyuanVideoDiTInputBundle(
        hidden_states=tensors["latents_init"],
        timestep=timesteps[:1].clone(),
        encoder_hidden_states=tensors["encoder_hidden_states"],
        encoder_attention_mask=tensors["encoder_attention_mask"],
        pooled_projections=tensors["pooled_projections"],
        guidance=tensors["guidance"],
    )
    with torch.inference_mode():
        output = app.pipeline(
            bundle=bundle,
            timesteps=timesteps,
            output_type="latent",
            return_trajectory=True,
        )
    assert output.trajectory is not None
    return [step.to(dtype=torch.float32).clone() for step in output.trajectory[1:]]


def _set_scheduler_timesteps(
    scheduler,
    expected_timesteps: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    sigmas = np.linspace(1.0, 0.0, int(expected_timesteps.numel()) + 1)[:-1]
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    actual = scheduler.timesteps.to(device=device)
    expected = expected_timesteps.to(device=device)
    assert actual.shape == expected.shape
    assert torch.allclose(
        actual.to(dtype=expected.dtype).float(),
        expected.float(),
        atol=1e-3,
        rtol=1e-4,
    )
    return actual


def _trajectory_cosines(
    *,
    reference: list[torch.Tensor],
    actual: list[torch.Tensor],
) -> list[dict[str, float | int | list[int]]]:
    metrics: list[dict[str, float | int | list[int]]] = []
    for index, (ref, out) in enumerate(zip(reference, actual)):
        assert list(ref.shape) == list(out.shape), (
            f"step {index} shape mismatch: diffusers={tuple(ref.shape)} nova={tuple(out.shape)}"
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
    path = os.environ.get("NOVA_HUNYUAN_VIDEO_NUMERICAL_METRICS")
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
    return f"HunyuanVideo trajectory cosine below {threshold}: {rows}"


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")
