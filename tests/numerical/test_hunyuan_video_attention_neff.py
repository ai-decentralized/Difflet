"""HunyuanVideo dual-stream masked attention NEFF-vs-CPU alignment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


pytestmark = [
    pytest.mark.numerical,
    pytest.mark.neuron,
    pytest.mark.slow,
]


class HunyuanDualStreamAttentionProbe(nn.Module):
    def forward(
        self,
        latent_q,
        latent_k,
        latent_v,
        context_q,
        context_k,
        context_v,
        attention_mask,
    ):
        from difflet.models.hunyuan_video.modeling_hunyuan_video import dual_stream_attention

        latent, context = dual_stream_attention(
            latent_q,
            latent_k,
            latent_v,
            context_q,
            context_k,
            context_v,
            attention_mask=attention_mask,
        )
        return torch.cat([latent, context], dim=1)


def test_hunyuan_video_masked_dual_stream_attention_neff_matches_cpu():
    if os.environ.get("DIFFLET_RUN_HUNYUAN_ATTENTION_NEFF") != "1":
        pytest.skip("set DIFFLET_RUN_HUNYUAN_ATTENTION_NEFF=1 to run the Hunyuan attention NEFF gate")

    import torch_neuronx

    batch = int(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_BATCH", "1"))
    latent_seq = int(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_LATENT_SEQ", "3840"))
    context_seq = int(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_CONTEXT_SEQ", "256"))
    heads = int(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_HEADS", "24"))
    head_dim = int(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_HEAD_DIM", "128"))
    masked_tail = int(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_MASKED_TAIL", "64"))
    cosine_min = float(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_COSINE_MIN", "0.999"))
    mean_abs_max = float(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_MEAN_ABS_MAX", "0.005"))
    work_dir = Path(os.environ.get("DIFFLET_HUNYUAN_ATTENTION_WORK_DIR", "/tmp/difflet_hunyuan_attention_neff"))
    compiler_args = os.environ.get(
        "DIFFLET_HUNYUAN_ATTENTION_COMPILER_ARGS",
        "--model-type=transformer -O1 --auto-cast=none "
        "--internal-hlo2tensorizer-options='--verify-hlo=true'",
    )

    inputs = _make_inputs(
        batch=batch,
        latent_seq=latent_seq,
        context_seq=context_seq,
        heads=heads,
        head_dim=head_dim,
        masked_tail=masked_tail,
    )
    model = HunyuanDualStreamAttentionProbe().eval()

    os.environ["DIFFLET_BACKEND"] = "cpu"
    with torch.no_grad():
        ref = model(*inputs).detach().cpu()

    os.environ["DIFFLET_BACKEND"] = "trainium"
    traced = torch_neuronx.trace(
        model,
        inputs,
        compiler_workdir=str(work_dir / "compiler"),
        compiler_args=compiler_args,
    )
    with torch.no_grad():
        out = traced(*inputs).detach().cpu()

    metrics = _compare(ref, out)
    metrics.update(
        {
            "batch": batch,
            "latent_seq": latent_seq,
            "context_seq": context_seq,
            "heads": heads,
            "head_dim": head_dim,
            "masked_tail": masked_tail,
            "cosine_min": cosine_min,
            "mean_abs_max": mean_abs_max,
        }
    )
    _write_metrics_if_requested(metrics)
    assert metrics["cosine"] >= cosine_min, metrics
    assert metrics["mean_abs"] <= mean_abs_max, metrics


def _make_inputs(*, batch, latent_seq, context_seq, heads, head_dim, masked_tail):
    if masked_tail < 0 or masked_tail >= context_seq:
        raise ValueError("masked_tail must be >= 0 and < context_seq")
    generator = torch.Generator(device="cpu").manual_seed(20260510)
    shape_latent = (batch, latent_seq, heads, head_dim)
    shape_context = (batch, context_seq, heads, head_dim)
    tensors = [
        torch.randn(shape_latent, generator=generator, dtype=torch.bfloat16),
        torch.randn(shape_latent, generator=generator, dtype=torch.bfloat16),
        torch.randn(shape_latent, generator=generator, dtype=torch.bfloat16),
        torch.randn(shape_context, generator=generator, dtype=torch.bfloat16),
        torch.randn(shape_context, generator=generator, dtype=torch.bfloat16),
        torch.randn(shape_context, generator=generator, dtype=torch.bfloat16),
    ]
    mask = torch.ones((batch, 1, 1, latent_seq + context_seq), dtype=torch.bool)
    if masked_tail:
        mask[:, :, :, -masked_tail:] = False
    return (*tensors, mask)


def _compare(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float | list[int] | str]:
    if tuple(ref.shape) != tuple(out.shape):
        raise AssertionError(f"shape mismatch: ref={tuple(ref.shape)} out={tuple(out.shape)}")
    diff = (ref.float() - out.float()).abs()
    return {
        "shape": list(ref.shape),
        "dtype_ref": str(ref.dtype),
        "dtype_neff": str(out.dtype),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "cosine": float(F.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0)),
    }


def _write_metrics_if_requested(metrics: dict) -> None:
    path = os.environ.get("DIFFLET_HUNYUAN_ATTENTION_METRICS")
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
