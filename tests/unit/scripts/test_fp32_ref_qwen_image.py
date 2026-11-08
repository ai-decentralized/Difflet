"""Phase G's Qwen-Image CPU fp32 reference (``scripts/fp32_ref/qwen_image.py``) against the device
runs it scores (``qwen_bf16_tp4_ref2`` bf16 and ``qwen_mx_hm0_tp4`` MXFP8; Qwen-Image's object in
``/home/ubuntu/fp8-bench/phase_g_research.json``): the pipeline settings and the snapshot, the
initial-noise law at the device shape (a CPU randn, no model load) and the guard that keeps every
component fp32 (the fp32-math guard is the helper's: test_fp32_ref_common.py)."""

from __future__ import annotations

import pytest
import torch
from diffusers import QwenImagePipeline

from scripts.fp32_ref import qwen_image as ref


def test_settings_are_the_device_runs():
    """The device command's prompt, shape, 50 steps and seed 42; one forward per step (no true
    CFG, where diffusers' own default is 4.0) and no guidance (the checkpoint has
    guidance_embeds=false, so the command's --guidance-scale 7.5 is dropped); the prompt's 11
    rows after the template's drop."""
    assert ref.PROMPT == "a cat sitting on a bench"
    assert ref.CALL == {"height": 1024, "width": 1024, "negative_prompt": None,
                        "true_cfg_scale": 1.0, "guidance_scale": None}
    assert (ref.STEPS, ref.SEED, ref.TEXT_ROWS) == (50, 42, 11)
    assert ref.SNAPSHOT.parts[-3:] == ("models--Qwen--Qwen-Image", "snapshots",
                                       "75e0b4be04f60ec59a75f475837eced720f823b6")


def _fp32_draw() -> torch.Tensor:
    """diffusers' own seed-42 draw in an fp32 QwenImagePipeline, packed."""
    noise = torch.randn((1, 1, 16, 128, 128), generator=torch.Generator().manual_seed(42))
    return QwenImagePipeline._pack_latents(noise, 1, 16, 128, 128)


def test_initial_noise_is_the_devices():
    """initial_latents asserts its draw against difflet's own (the orchestrator's
    prepare_latents in bf16 from a CPU generator); its md5 is FLUX's research md5, the same
    draw at the same shape and packing, and it is not the fp32 draw diffusers would make."""
    latents, md5 = ref.initial_latents()
    assert latents.dtype == torch.float32 and latents.shape == (1, 4096, 64)
    assert md5 == "be1ad0a63781cbc9b93ec9007d44b4ad"
    fp32 = _fp32_draw()
    assert torch.equal(latents, fp32.to(torch.bfloat16).float())
    assert not torch.equal(latents, fp32)


def test_refuses_a_component_not_in_fp32():
    ref.require_fp32({"transformer": torch.nn.Linear(2, 2), "scheduler": object()})
    with pytest.raises(RuntimeError, match="text_encoder"):
        ref.require_fp32({"text_encoder": torch.nn.Linear(2, 2).to(torch.bfloat16)})


class _Encoder:
    """A pipeline whose encode_prompt returns ``rows`` prompt rows and ``mask``."""

    def __init__(self, rows: int, mask=None) -> None:
        self.rows, self.mask = rows, mask

    def encode_prompt(self, *, prompt, device):
        return torch.zeros(1, self.rows, 8), self.mask


def test_encode_refuses_a_prompt_the_device_encoded_differently():
    """The device kept 11 rows, unmasked; any other count means another prompt or template."""
    embeds, mask = ref.encode(_Encoder(11))
    assert embeds.shape == (1, 11, 8) and mask is None
    with pytest.raises(RuntimeError, match="11"):
        ref.encode(_Encoder(12))
    with pytest.raises(RuntimeError, match="mask"):
        ref.encode(_Encoder(11, torch.ones(1, 11)))
