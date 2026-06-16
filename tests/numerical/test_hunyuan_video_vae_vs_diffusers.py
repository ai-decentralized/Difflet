"""HunyuanVideo VAE numerical alignment against Hugging Face diffusers.

This is the regression gate for cclog 38 §15.5. It decodes the same latent
through the segmented Trainium VAE (`NeuronHunyuanVideoApplication`,
`enable_vae_decoder=True`) and through `diffusers.AutoencoderKLHunyuanVideo`
on CPU and asserts cosine >= `NOVA_HUNYUAN_VAE_MIN_COSINE` (default 0.999,
matching cclog 38 §14 closure).

Opt-in: set `NOVA_RUN_HUNYUAN_VAE_NUMERICAL=1` to enable. The test loads
the compiled segmented VAE artifact and runs the ~150 s HF CPU reference,
so it is too slow for default test runs.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


pytestmark = [
    pytest.mark.numerical,
    pytest.mark.neuron,
    pytest.mark.slow,
]


def test_hunyuan_video_vae_segmented_cosine_matches_diffusers():
    if os.environ.get("NOVA_RUN_HUNYUAN_VAE_NUMERICAL") != "1":
        pytest.skip(
            "set NOVA_RUN_HUNYUAN_VAE_NUMERICAL=1 to run the HunyuanVideo VAE numerical gate"
        )

    source_dir = Path(
        os.environ.get(
            "NOVA_HUNYUAN_VAE_SOURCE_DIR",
            "/home/ubuntu/.cache/huggingface/hub/hunyuanvideo-real",
        )
    )
    compiled_dir = Path(
        os.environ.get(
            "NOVA_HUNYUAN_VAE_COMPILED_DIR",
            "/home/ubuntu/nova/.nova-cache/hunyuan_n4_20d40s2r/compiled",
        )
    )
    latents_path = Path(
        os.environ.get(
            "NOVA_HUNYUAN_VAE_LATENTS",
            "/home/ubuntu/nova/.nova-cache/hunyuan_dit_inputs/cat_walking_4step_nova_latents.pt",
        )
    )
    threshold = float(os.environ.get("NOVA_HUNYUAN_VAE_MIN_COSINE", "0.999"))
    tp_degree = int(os.environ.get("NOVA_HUNYUAN_VAE_TP_DEGREE", "1"))
    height = int(os.environ.get("NOVA_HUNYUAN_VAE_HEIGHT", "320"))
    width = int(os.environ.get("NOVA_HUNYUAN_VAE_WIDTH", "512"))
    num_frames = int(os.environ.get("NOVA_HUNYUAN_VAE_FRAMES", "61"))

    assert source_dir.exists(), f"HF source dir missing: {source_dir}"
    assert (source_dir / "vae" / "config.json").exists(), (
        f"HF VAE config missing under {source_dir}/vae"
    )
    assert compiled_dir.exists(), f"Compiled VAE artifact dir missing: {compiled_dir}"
    assert latents_path.exists(), f"Latent tensor missing: {latents_path}"

    latents = torch.load(latents_path, map_location="cpu")

    trainium_out, scaling = _run_trainium_vae(
        source_dir=source_dir,
        compiled_dir=compiled_dir,
        latents=latents,
        tp_degree=tp_degree,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    hf_out = _run_hf_vae(
        source_dir=source_dir,
        latents=latents,
        scaling=scaling,
    )

    assert trainium_out.shape == hf_out.shape, (
        f"shape mismatch: trainium {tuple(trainium_out.shape)} "
        f"vs HF {tuple(hf_out.shape)}"
    )

    metrics = _stats(trainium_out, hf_out)
    metrics["threshold"] = threshold
    _write_metrics_if_requested(metrics)

    assert metrics["cosine"] >= threshold, (
        f"HunyuanVideo VAE cosine {metrics['cosine']:.10f} below threshold "
        f"{threshold}: mean_abs={metrics['mean_abs']:.6e} "
        f"max_abs={metrics['max_abs']:.6e}"
    )


def _run_trainium_vae(
    *,
    source_dir: Path,
    compiled_dir: Path,
    latents: torch.Tensor,
    tp_degree: int,
    height: int,
    width: int,
    num_frames: int,
) -> tuple[torch.Tensor, float]:
    os.environ.setdefault("NOVA_BACKEND", "trainium")

    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from nova.pipeline.parallel_config import NovaParallelConfig

    app = NeuronHunyuanVideoApplication(
        model_path=str(source_dir),
        parallel=NovaParallelConfig(tp_degree=tp_degree),
        dtype=torch.bfloat16,
        shape={"height": height, "width": width, "num_frames": num_frames},
        enable_transformer=False,
        enable_vae_decoder=True,
    )
    assert app.vae_decoder is not None, "vae_decoder failed to initialize"
    app.load(str(compiled_dir), skip_warmup=True)

    scaling = float(app.vae_decoder.config.scaling_factor)
    pre_scaled = latents.to(dtype=torch.bfloat16) / scaling

    with torch.inference_mode():
        decoded = app.vae_decoder.decode(pre_scaled, return_dict=False)[0]
    decoded_cpu = decoded.detach().to(device="cpu", dtype=torch.float32).clone()

    del app
    gc.collect()
    return decoded_cpu, scaling


def _run_hf_vae(
    *,
    source_dir: Path,
    latents: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    from diffusers import AutoencoderKLHunyuanVideo

    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        source_dir / "vae", torch_dtype=torch.bfloat16
    ).eval()
    vae.enable_tiling()
    pre_scaled = latents.to(dtype=torch.bfloat16) / scaling

    with torch.no_grad():
        decoded = vae.decode(pre_scaled, return_dict=False)[0]
    decoded_cpu = decoded.detach().to(device="cpu", dtype=torch.float32).clone()

    del vae
    gc.collect()
    return decoded_cpu


def _stats(trainium: torch.Tensor, hf: torch.Tensor) -> dict[str, float | list[int]]:
    a = trainium.reshape(-1)
    b = hf.reshape(-1)
    diff = a - b
    cosine = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()
    return {
        "cosine": float(cosine),
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "trainium_mean": float(a.mean().item()),
        "trainium_std": float(a.std().item()),
        "hf_mean": float(b.mean().item()),
        "hf_std": float(b.std().item()),
        "shape": list(trainium.shape),
    }


def _write_metrics_if_requested(metrics: dict[str, float | list[int]]) -> None:
    path = os.environ.get("NOVA_HUNYUAN_VAE_NUMERICAL_METRICS")
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
