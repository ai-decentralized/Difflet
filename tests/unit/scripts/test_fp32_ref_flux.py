"""Phase G's FLUX.1-dev CPU fp32 reference (``scripts/fp32_ref/flux.py``) against the device runs
it scores (``flux_d_tp4`` A1 bf16 and B1 MXFP8; FLUX's object in
``/home/ubuntu/fp8-bench/phase_g_research.json``): the pipeline settings and the snapshot, the
initial-noise law at the device shape (a CPU randn, no model load) and the guard that keeps every
component fp32 (the fp32-math guard is the helper's: test_fp32_ref_common.py)."""

from __future__ import annotations

import pytest
import torch
from diffusers import FluxPipeline

from scripts.fp32_ref import flux as ref


def test_settings_are_the_device_runs():
    """run_steps_2_5.sh's prompt and shape, the orchestrator's embedded guidance 3.5 with no true
    CFG, the application's 512 T5 tokens; 50 steps (diffusers' own default is 28), seed 42."""
    assert ref.CALL == {"prompt": "a cat sitting on a bench", "height": 1024, "width": 1024,
                        "guidance_scale": 3.5, "negative_prompt": None, "true_cfg_scale": 1.0,
                        "max_sequence_length": 512}
    assert (ref.STEPS, ref.SEED) == (50, 42)
    assert ref.SNAPSHOT.parts[-3:] == ("models--black-forest-labs--FLUX.1-dev", "snapshots",
                                       "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")


def _device_draw(dtype: torch.dtype) -> torch.Tensor:
    """diffusers' FluxPipeline.prepare_latents at ``dtype`` from a CPU generator seeded 42: the
    device's path to its noise at its bf16 application dtype, a plain fp32 pipeline's at fp32."""
    pipe = FluxPipeline.__new__(FluxPipeline)
    pipe.vae_scale_factor = 8
    latents, _ = pipe.prepare_latents(1, 16, 1024, 1024, dtype, torch.device("cpu"),
                                      torch.Generator().manual_seed(42))
    return latents


def test_initial_noise_is_the_devices():
    """The research's md5 of the device's packed bf16 draw, equal to the device's own draw
    upcast and not to the fp32 draw diffusers would make (rel L2 1.7e-3 away)."""
    latents, md5 = ref.initial_latents()
    assert latents.dtype == torch.float32 and latents.shape == (1, 4096, 64)
    assert md5 == ref.NOISE_MD5 == "be1ad0a63781cbc9b93ec9007d44b4ad"
    assert torch.equal(latents, _device_draw(torch.bfloat16).float())
    assert not torch.equal(latents, _device_draw(torch.float32))


def test_refuses_a_component_not_in_fp32():
    ref.require_fp32({"vae": torch.nn.Linear(2, 2), "scheduler": object(), "tokenizer": None})
    half = torch.nn.Linear(2, 2)
    half.register_buffer("scale", torch.ones(2, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="text_encoder_2"):
        ref.require_fp32({"vae": torch.nn.Linear(2, 2), "text_encoder_2": half})
