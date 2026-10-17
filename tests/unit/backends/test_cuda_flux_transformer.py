"""Flux DiT as a CUDA component, driven like ``FluxPipeline`` drives a transformer (F5 task 3).

Runs on the CPU (``device="cpu"``): the application's load path — meta build,
materialize, diffusers safetensors → ``convert_flux_hf_state_dict`` → strict load — and
the ``FluxDiTAdapter`` (diffusers' call signature: ids instead of rope, optional
guidance, ``return_dict``) are device-independent. The diffusers
``FluxTransformer2DModel`` is the oracle; the bf16 GPU numbers come from
``tests/numerical/cuda_parity_flux_gpu.py``.
"""

import pytest
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from safetensors.torch import save_file

from difflet.backends.cuda.flux.transformer import CudaFluxTransformerApplication, FluxDiTAdapter
from difflet.models.flux.dit_config import FluxDiTConfig

# Rope axes must be even (cos/sin pairs) and sum to attention_head_dim (8).
AXES_DIMS_ROPE = (2, 2, 4)
TINY = dict(
    patch_size=1,
    in_channels=4,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=8,
    axes_dims_rope=AXES_DIMS_ROPE,
)


def _reference(guidance_embeds: bool) -> FluxTransformer2DModel:
    torch.manual_seed(0)
    return FluxTransformer2DModel(**TINY, guidance_embeds=guidance_embeds).eval()


def _config(guidance_embeds: bool) -> FluxDiTConfig:
    return FluxDiTConfig(
        **TINY,
        out_channels=None,
        guidance_embeds=guidance_embeds,
        height=64,
        width=64,
        torch_dtype=torch.float32,
    )


def _adapter(tmp_path, guidance_embeds=True, *, warmup=False):
    """Write the diffusers weights as one unindexed shard, then compile + load the app."""
    reference = _reference(guidance_embeds)
    transformer = tmp_path / "transformer"
    transformer.mkdir(parents=True)
    state = {k: v.detach().clone().contiguous() for k, v in reference.state_dict().items()}
    save_file(state, str(transformer / "diffusion_pytorch_model.safetensors"))
    app = CudaFluxTransformerApplication(
        model_path=str(transformer), config=_config(guidance_embeds), device="cpu"
    )
    cache = tmp_path / "cache"
    app.compile(str(cache))
    app.load(str(cache), skip_warmup=not warmup)
    return reference, FluxDiTAdapter(app)


def _inputs():
    torch.manual_seed(1)
    # 64x64 pixels -> 8x8 VAE latent -> 4x4 packed tokens = 16 image tokens.
    return dict(
        hidden_states=torch.randn(1, 16, 4),
        encoder_hidden_states=torch.randn(1, 6, 16),
        pooled_projections=torch.randn(1, 8),
        timestep=torch.tensor([0.5]),
        img_ids=FluxPipeline._prepare_latent_image_ids(1, 4, 4, "cpu", torch.float32),
        txt_ids=torch.zeros(6, 3),
    )


def test_cuda_flux_transformer_loads_diffusers_weights_and_matches(tmp_path):
    reference, adapter = _adapter(tmp_path, warmup=True)
    inputs = _inputs()
    guidance = torch.tensor([3.5])
    with torch.no_grad():
        expected = reference(**inputs, guidance=guidance, return_dict=False)[0]
        (actual,) = adapter(**inputs, guidance=guidance, return_dict=False)
        sample = adapter(**inputs, guidance=guidance).sample
    assert actual.shape == expected.shape == (1, 16, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(sample, actual, rtol=0, atol=0)


def test_adapter_handles_guidance_none_and_tensor(tmp_path):
    # schnell: no guidance embedding, FluxPipeline passes guidance=None.
    reference, schnell = _adapter(tmp_path / "schnell", guidance_embeds=False)
    inputs = _inputs()
    with torch.no_grad():
        expected = reference(**inputs, guidance=None, return_dict=False)[0]
        actual = schnell(**inputs, guidance=None, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    # dev: a float and a tensor guidance feed the DiT the same thing.
    _, dev = _adapter(tmp_path / "dev", guidance_embeds=True)
    with torch.no_grad():
        from_float = dev(**inputs, guidance=3.5, return_dict=False)[0]
        from_tensor = dev(**inputs, guidance=torch.tensor([3.5]), return_dict=False)[0]
    assert torch.equal(from_float, from_tensor)


def test_adapter_exposes_what_flux_pipeline_reads(tmp_path):
    _, adapter = _adapter(tmp_path)
    assert adapter.config.in_channels == 4
    assert adapter.config.guidance_embeds is True
    assert adapter.dtype == torch.float32
    with adapter.cache_context("cond"):
        pass


def test_rope_is_cached_per_ids_shape(tmp_path):
    _, adapter = _adapter(tmp_path)
    inputs = _inputs()
    txt_ids, img_ids = inputs["txt_ids"], inputs["img_ids"]
    first = adapter.rotary_embedding(txt_ids, img_ids)
    assert first.shape == (6 + 16, 8, 2) and first.dtype == torch.float32
    # FluxPipeline's deprecated 3-D ids (batch dim first) resolve to the same cache entry.
    assert adapter.rotary_embedding(txt_ids.clone(), img_ids[None].clone()) is first

    # Same shapes, different positions (e.g. 2x8 vs 8x2 latent grids): never a stale rope.
    other_ids = FluxPipeline._prepare_latent_image_ids(1, 2, 8, "cpu", torch.float32)
    assert other_ids.shape == img_ids.shape and not torch.equal(other_ids, img_ids)
    other = adapter.rotary_embedding(txt_ids, other_ids)
    assert other is not first and not torch.equal(other, first)


def test_forward_before_load_is_an_explicit_error(tmp_path):
    app = CudaFluxTransformerApplication(
        model_path=str(tmp_path), config=_config(True), device="cpu"
    )
    with pytest.raises(RuntimeError, match="not loaded"):
        FluxDiTAdapter(app)(**_inputs(), guidance=3.5)
