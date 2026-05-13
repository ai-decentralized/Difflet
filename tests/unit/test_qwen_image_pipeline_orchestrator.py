from types import SimpleNamespace

import pytest
import torch

from nova.models.qwen_image.application import QwenImageDiTInputBundle
from nova.models.qwen_image.pipeline import (
    QwenImageOrchestrator,
    QwenImagePipelineOutput,
    pack_qwen_image_latents,
    unpack_qwen_image_latents,
)


class FakeTransformer:
    def __init__(self, *, value: float = 0.25):
        self.dtype = torch.float32
        self.value = value
        self.calls = []

    def __call__(self, bundle: QwenImageDiTInputBundle):
        self.calls.append(bundle)
        return {"sample": torch.ones_like(bundle.hidden_states) * self.value}


class FakeVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(latents_mean=[0.5] * 16, latents_std=[2.0] * 16)
        self.inputs = []

    def decode(self, latents, return_dict):
        assert return_dict is False
        self.inputs.append(latents.detach().clone())
        return (latents[:, :3],)


def _bundle(latents: torch.Tensor | None = None) -> QwenImageDiTInputBundle:
    latents = torch.zeros((1, 16, 64), dtype=torch.float32) if latents is None else latents
    return QwenImageDiTInputBundle(
        hidden_states=latents,
        timestep=torch.zeros([1], dtype=torch.float32),
        encoder_hidden_states=torch.ones((1, 4, 8), dtype=torch.float32),
        encoder_hidden_states_mask=torch.ones((1, 4), dtype=torch.bool),
        guidance=torch.zeros([1], dtype=torch.float32),
    )


def test_pack_and_unpack_qwen_image_latents_round_trip():
    latents = torch.arange(1 * 1 * 16 * 8 * 8, dtype=torch.float32).reshape(1, 1, 16, 8, 8)

    packed = pack_qwen_image_latents(latents)
    unpacked = unpack_qwen_image_latents(packed, height=64, width=64, vae_scale_factor=8)

    assert packed.shape == (1, 16, 64)
    assert torch.equal(unpacked, latents.permute(0, 2, 1, 3, 4))


def test_qwen_orchestrator_prepares_packed_latents(tmp_path):
    pipeline = QwenImageOrchestrator(
        model_path=str(tmp_path),
        dtype=torch.float32,
        height=64,
        width=64,
    )

    latents = pipeline.prepare_latents(batch_size=2)

    assert latents.shape == (2, 16, 64)
    assert latents.dtype == torch.float32


def test_qwen_orchestrator_runs_bundle_denoise_with_fallback_scheduler(tmp_path):
    transformer = FakeTransformer(value=0.5)
    with pytest.warns(RuntimeWarning, match="scheduler_config.json"):
        pipeline = QwenImageOrchestrator(
            model_path=str(tmp_path),
            transformer=transformer,
            dtype=torch.float32,
            height=64,
            width=64,
        )

    output = pipeline(
        bundle=_bundle(),
        timesteps=torch.tensor([1.0, 0.5]),
        return_trajectory=True,
    )

    assert isinstance(output, QwenImagePipelineOutput)
    assert torch.allclose(output.latents, torch.full((1, 16, 64), -0.5))
    assert len(transformer.calls) == 2
    assert transformer.calls[0].timestep.shape == (1,)
    assert transformer.calls[0].encoder_hidden_states_mask.dtype == torch.bool
    assert output.trajectory is not None
    assert len(output.trajectory) == 3


def test_qwen_orchestrator_builds_bundle_from_named_tensors(tmp_path):
    transformer = FakeTransformer(value=1.0)
    pipeline = QwenImageOrchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        dtype=torch.float32,
        height=64,
        width=64,
        scheduler=None,
    )

    pipeline(
        latents=torch.zeros((1, 16, 64), dtype=torch.float32),
        timesteps=torch.tensor([1.0]),
        encoder_hidden_states=torch.ones((1, 4, 8), dtype=torch.float32),
    )

    assert len(transformer.calls) == 1
    assert transformer.calls[0].encoder_hidden_states_mask.shape == (1, 4)


def test_qwen_orchestrator_decodes_pt_output_with_vae_latent_stats(tmp_path):
    vae = FakeVAE()
    pipeline = QwenImageOrchestrator(
        model_path=str(tmp_path),
        vae=vae,
        dtype=torch.float32,
        height=64,
        width=64,
    )
    packed = torch.ones((1, 16, 64), dtype=torch.float32)

    output = pipeline(
        bundle=_bundle(packed),
        output_type="pt",
        return_dict=False,
    )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 3, 1, 8, 8)
    assert torch.allclose(vae.inputs[0], torch.full((1, 16, 1, 8, 8), 2.5))


def test_qwen_orchestrator_requires_encoder_hidden_states(tmp_path):
    pipeline = QwenImageOrchestrator(model_path=str(tmp_path), dtype=torch.float32)

    with pytest.raises(ValueError, match="encoder_hidden_states"):
        pipeline()
