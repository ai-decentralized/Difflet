"""Unit tests for difflet.models.wan.vae.modeling_vae (W3c)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

import pytest
import torch


def test_modeling_vae_imports_only_from_allowed_modules():
    src = (_REPO_ROOT / "difflet/models/wan/vae/modeling_vae.py").read_text()
    tree = ast.parse(src)
    forbidden_roots = {"neuronx_distributed", "nkilib", "torch_neuronx"}
    forbidden_prefixes = ("difflet.core",)
    offending: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden_roots or alias.name.startswith(forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if root in forbidden_roots or mod.startswith(forbidden_prefixes):
                offending.append(mod)

    assert not offending, f"forbidden imports in modeling_vae.py: {offending}"


def test_wan_vae_decoder_config_defaults_match_wan22():
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

    cfg = WanVAEDecoderConfig()
    assert cfg.base_dim == 96
    assert cfg.decoder_base_dim == 96
    assert cfg.z_dim == 16
    assert cfg.dim_mult == [1, 2, 4, 4]
    assert cfg.num_res_blocks == 2
    assert cfg.attn_scales == []
    assert cfg.temperal_downsample == [False, True, True]
    assert cfg.temperal_upsample == [True, True, False]
    assert cfg.scale_factor_temporal == 4
    assert cfg.scale_factor_spatial == 8


def test_wan_vae_decoder_config_from_diffusers_dict_filters_unknown_keys():
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

    raw = {
        "base_dim": 96,
        "z_dim": 16,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "_class_name": "AutoencoderKLWan",
        "_diffusers_version": "0.38.0",
    }
    cfg = WanVAEDecoderConfig.from_diffusers_dict(raw)
    assert cfg.base_dim == 96
    assert cfg.z_dim == 16
    assert cfg.dim_mult == [1, 2, 4, 4]


def test_wan_vae_decoder_config_rejects_unsupported_modes():
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

    with pytest.raises(NotImplementedError, match="is_residual"):
        WanVAEDecoderConfig(is_residual=True)
    with pytest.raises(NotImplementedError, match="patchified"):
        WanVAEDecoderConfig(patch_size=2)
    with pytest.raises(ValueError, match="temperal_downsample"):
        WanVAEDecoderConfig(dim_mult=[1, 2, 4], temperal_downsample=[True])


def test_modeling_vae_classes_are_importable():
    from difflet.models.wan.vae import modeling_vae

    expected = [
        "WanCausalConv3d",
        "WanDecoder3d",
        "WanRMSNorm",
        "WanVAEDecoderConfig",
        "WanVAEDecoderModel",
    ]
    for name in expected:
        cls = getattr(modeling_vae, name)
        assert inspect.isclass(cls), f"{name} should be a class"


def test_wan_vae_decoder_state_dict_key_layout_matches_hf_decoder():
    from difflet.models.wan.vae.modeling_vae import (
        WanVAEDecoderConfig,
        WanVAEDecoderModel,
    )

    cfg = WanVAEDecoderConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
    )
    keys = set(WanVAEDecoderModel(cfg).state_dict())

    assert "post_quant_conv.weight" in keys
    assert "decoder.conv_in.weight" in keys
    assert "decoder.mid_block.resnets.0.conv1.weight" in keys
    assert "decoder.up_blocks.0.resnets.0.conv1.weight" in keys
    assert "decoder.norm_out.gamma" in keys
    assert "decoder.conv_out.weight" in keys
    assert not any(key.startswith("encoder.") for key in keys)
    assert not any(key.startswith("quant_conv.") for key in keys)


def test_wan_vae_decoder_tiny_forward_uses_causal_temporal_decode():
    from difflet.models.wan.vae.modeling_vae import (
        WanVAEDecoderConfig,
        WanVAEDecoderModel,
    )

    cfg = WanVAEDecoderConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
    )
    model = WanVAEDecoderModel(cfg).eval()
    latents = torch.randn(1, 4, 2, 4, 4)

    with torch.no_grad():
        out = model(latents)

    # Two latent frames with one temporal upsample stage decode to 1 + 2 frames.
    assert out.shape == (1, 3, 3, 8, 8)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_wan_vae_decoder_inference_config_shapes():
    from difflet.backends.trainium.wan.vae import (
        ModelWrapperWanVAEDecoder,
        WanVAEDecoderInferenceConfig,
    )
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.utils.diffusers_adapter import load_diffusers_config

    snap = (
        "/home/ubuntu/.cache/huggingface/hub/"
        "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
        "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
    )
    if not Path(f"{snap}/vae/config.json").exists():
        pytest.skip("Wan2.2 VAE config snapshot not present")

    nc = NeuronConfig(
        batch_size=1,
        tp_degree=1,
        world_size=1,
        torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    cfg = WanVAEDecoderInferenceConfig(
        neuron_config=nc,
        load_config=load_diffusers_config(f"{snap}/vae"),
        height=480,
        width=832,
        num_frames=9,
    )
    assert cfg.latent_frames == 3
    assert cfg.latent_height == 60
    assert cfg.latent_width == 104

    mw = ModelWrapperWanVAEDecoder(
        config=cfg, model_cls=object, tag="WanVAEDecoderModel"
    )
    inputs = mw.input_generator()
    assert tuple(inputs[0][0].shape) == (1, 16, 3, 60, 104)
    assert inputs[0][0].dtype == torch.bfloat16


def test_attention_block_uses_explicit_attention_on_neuron_only():
    """The neuron backend's fused SDPA (nkilib attention_cte) fails this block's one head of
    d=384 at >= 8192 key tokens (device 2026-10-18, Qwen-Image 1024x1024: 16384 tokens), so the
    neuron device gets the written-out attention; every other device keeps the fused call."""
    import torch.nn.functional as F

    from difflet.models.wan.vae import modeling_vae as m

    assert m._spatial_attention_impl("neuron") is m._explicit_attention
    for device_type in ("cpu", "xla", "cuda"):
        assert m._spatial_attention_impl(device_type) is F.scaled_dot_product_attention


def test_explicit_attention_matches_sdpa():
    import torch.nn.functional as F

    from difflet.models.wan.vae import modeling_vae as m

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 1, 77, 24, dtype=torch.float64) for _ in range(3))
    torch.testing.assert_close(
        m._explicit_attention(q, k, v), F.scaled_dot_product_attention(q, k, v),
        rtol=1e-12, atol=1e-12,
    )


def test_attention_block_output_is_the_same_with_either_attention(monkeypatch):
    """The whole block (norm, qkv conv, attention, proj, residual) on a non-square frame: the
    explicit attention the neuron device runs gives the fused call's output."""
    from difflet.models.wan.vae import modeling_vae as m

    torch.manual_seed(0)
    block = m.WanAttentionBlock(16).double().eval()
    x = torch.randn(1, 16, 2, 6, 10, dtype=torch.float64)
    with torch.no_grad():
        fused = block(x)
        monkeypatch.setattr(m, "_spatial_attention_impl", lambda device_type: m._explicit_attention)
        explicit = block(x)
    torch.testing.assert_close(explicit, fused, rtol=1e-10, atol=1e-10)


# ---- reduced precision (phase 3: the neuron decoder's bf16 opt-in) ---------------------------

#: The tiny decoder with the real stage layout: the latent (1, 4, 2, 4, 6) decodes to 5 frames
#: of 32 x 48, through the mid block's attention, two temporal and three spatial upsamples.
_TINY_BF16_CFG = dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                      temperal_downsample=[False, True, True])


def _psnr(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """PSNR of decoded frames in [-1, 1] (peak-to-peak 2)."""
    import math

    mse = float(((actual.double() - reference.double()) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(4.0 / mse)


def _tiny_decoders(seed: int, *, fp32_output_head: bool):
    """``(fp32 decoder, bf16 decoder)`` with the same seeded weights."""
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel

    config = WanVAEDecoderConfig(**_TINY_BF16_CFG)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fp32 = WanVAEDecoderModel(config, fp32_output_head=fp32_output_head).eval()
    bf16 = WanVAEDecoderModel(config, fp32_output_head=fp32_output_head).eval()
    bf16.load_state_dict(fp32.state_dict())
    return fp32, bf16.to(torch.bfloat16)


def test_fp32_output_head_changes_nothing_in_fp32():
    """The neuron decoder always builds the fp32 head: in fp32 (its default) the decode must be
    the one it was, bit for bit."""
    plain, _ = _tiny_decoders(0, fp32_output_head=False)
    head, _ = _tiny_decoders(0, fp32_output_head=True)
    assert head.decoder.conv_out.fp32_compute and not plain.decoder.conv_out.fp32_compute
    z = torch.randn(1, 4, 2, 4, 6, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(head(z), plain(z))


def test_the_default_model_keeps_bf16_end_to_end():
    """Without the head (the Trainium path's model) a bf16 decode stays bf16, as before."""
    _, bf16 = _tiny_decoders(0, fp32_output_head=False)
    z = torch.randn(1, 4, 2, 4, 6, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert bf16(z.to(torch.bfloat16)).dtype == torch.bfloat16


@pytest.mark.parametrize("attention", ["sdpa", "explicit"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bf16_decode_with_the_fp32_head_is_close_to_fp32(monkeypatch, attention, seed):
    """bf16 weights and activations with the fp32 output head: fp32 frames (the head's weights
    stay bf16, upcast per call), within bf16's error of the fp32 decode of the same latents. On
    these random decoders bf16 lands at 48.9-55.3 dB; 40 dB catches a broken path (an fp32 op
    fed bf16 garbage, a dropped upcast) without tracking rounding noise. The device gate is 50
    dB on the real decoder (the plan's phase-3 runbook). ``explicit`` is the attention the
    neuron device runs."""
    from difflet.models.wan.vae import modeling_vae as m

    if attention == "explicit":
        monkeypatch.setattr(m, "_spatial_attention_impl", lambda device_type: m._explicit_attention)
    fp32, bf16 = _tiny_decoders(seed, fp32_output_head=True)
    assert bf16.decoder.conv_out.weight.dtype == torch.bfloat16
    z = torch.randn(1, 4, 2, 4, 6, generator=torch.Generator().manual_seed(seed + 10))
    with torch.no_grad():
        reference = fp32(z)
        frames = bf16(z.to(torch.bfloat16))
    assert frames.dtype == torch.float32 and frames.shape == reference.shape == (1, 3, 5, 32, 48)
    assert frames.abs().max() <= 1.0
    assert _psnr(frames, reference) >= 40.0
    assert float((frames - reference).abs().max()) < 0.15


def test_explicit_attention_scores_in_fp32_for_bf16_inputs():
    """bf16 q, k, v with large logits: the scores and the softmax in fp32 keep the written-out
    attention near the fp64 result (here 0.48 max abs, against 1.81 for bf16 scores); the output
    stays in v's dtype."""
    import torch.nn.functional as F

    from difflet.models.wan.vae import modeling_vae as m

    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 1, 300, 32, dtype=torch.float64, generator=gen) * 4
               for _ in range(3))
    reference = F.scaled_dot_product_attention(q, k, v)
    qb, kb, vb = (t.to(torch.bfloat16) for t in (q, k, v))
    out = m._explicit_attention(qb, kb, vb)
    bf16_scores = torch.matmul(
        torch.softmax(torch.matmul(qb, kb.transpose(-1, -2)) * (32 ** -0.5), dim=-1), vb)
    assert out.dtype == torch.bfloat16
    error = float((out.double() - reference).abs().max())
    assert error < 0.5 * float((bf16_scores.double() - reference).abs().max())
