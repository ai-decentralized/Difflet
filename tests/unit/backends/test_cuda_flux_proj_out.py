"""The Flux single block's ``proj_out`` as one gemm at tp=1, as a CUDA load-time swap.

Diffusers runs one ``Linear(dim + mlp, dim)`` over ``cat([attn, mlp])``; the shared modeling
keeps the Megatron split (``proj_out_attn`` + ``proj_out_mlp``). ``fuse_proj_out`` rebuilds
diffusers' layout from the split weights on a loaded block and binds the fused forward to it;
the model file, the converter and the Trainium/TPU graphs keep the split."""

import os

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

# With no backend set, bind difflet.ops to the CPU ops for the modules imported below (a
# Trainium host would bind the Neuron ops).
_PINNED = "DIFFLET_BACKEND" not in os.environ
os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.backends.cuda.flux.proj_out import fuse_proj_out  # noqa: E402
from difflet.layers.embeddings import FluxPosEmbed  # noqa: E402
from difflet.models.flux import modeling_flux as mf  # noqa: E402
from difflet.models.flux.dit_config import FluxDiTConfig  # noqa: E402
from difflet.models.flux.modeling_flux import (  # noqa: E402
    NeuronFluxTransformer2DModel,
    convert_flux_hf_state_dict,
)

if _PINNED:
    del os.environ["DIFFLET_BACKEND"]

AXES = (2, 2, 4)
TINY = dict(
    patch_size=1,
    in_channels=4,
    num_layers=1,
    num_single_layers=2,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=8,
    guidance_embeds=True,
    axes_dims_rope=AXES,
)


def _models():
    torch.manual_seed(0)
    reference = FluxTransformer2DModel(**TINY).eval()
    config = FluxDiTConfig(
        **TINY, out_channels=None, height=64, width=64, torch_dtype=torch.float32
    )
    model = NeuronFluxTransformer2DModel(config).eval()
    model.load_state_dict(
        convert_flux_hf_state_dict(
            reference.state_dict(), num_single_layers=2, inner_dim=config.inner_dim
        ),
        strict=True,
    )
    return reference, model, config


def _inputs():
    torch.manual_seed(1)
    img_ids = FluxPipeline._prepare_latent_image_ids(1, 4, 4, "cpu", torch.float32)
    txt_ids = torch.zeros(6, 3)
    rope = torch.stack(FluxPosEmbed(10000, AXES)(torch.cat((txt_ids, img_ids), dim=0)), dim=2)
    return dict(
        hidden=torch.randn(1, 16, 4),
        text=torch.randn(1, 6, 16),
        pooled=torch.randn(1, 8),
        timestep=torch.tensor([0.5]),
        guidance=torch.tensor([3.5]),
        img_ids=img_ids,
        txt_ids=txt_ids,
        rope=rope,
    )


def _run(model, i):
    with torch.no_grad():
        return model(
            hidden_states=i["hidden"],
            encoder_hidden_states=i["text"],
            pooled_projections=i["pooled"],
            timestep=i["timestep"],
            guidance=i["guidance"],
            image_rotary_emb=i["rope"],
        )


def test_the_shared_model_has_no_fused_projection():
    _, model, _ = _models()
    for block in model.single_transformer_blocks:
        assert not hasattr(block, "proj_out")
        assert block.proj_out_attn is not None and block.proj_out_mlp is not None


def test_fuse_proj_out_builds_diffusers_layout_and_keeps_the_output():
    reference, model, config = _models()
    i = _inputs()
    split_out = _run(model, i)
    for n, block in enumerate(model.single_transformer_blocks):
        attn_w = block.proj_out_attn.weight.detach().clone()
        mlp_w = block.proj_out_mlp.weight.detach().clone()
        bias = block.proj_out_attn.bias.detach().clone()
        assert fuse_proj_out(block) is True
        fused = block.proj_out
        assert isinstance(fused, torch.nn.Linear)
        assert tuple(fused.weight.shape) == (
            config.inner_dim,
            config.inner_dim + block.mlp_hidden_dim,
        )
        assert torch.equal(fused.weight, torch.cat([attn_w, mlp_w], dim=1))
        assert torch.equal(fused.bias, bias)
        # Diffusers' own weight, to the bit.
        assert torch.equal(fused.weight, reference.single_transformer_blocks[n].proj_out.weight)
        assert block.proj_out_attn is None and block.proj_out_mlp is None
        assert fuse_proj_out(block) is False  # idempotent
    fused_out = _run(model, i)
    torch.testing.assert_close(fused_out, split_out, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        expected = reference(
            hidden_states=i["hidden"],
            encoder_hidden_states=i["text"],
            pooled_projections=i["pooled"],
            timestep=i["timestep"],
            img_ids=i["img_ids"],
            txt_ids=i["txt_ids"],
            guidance=i["guidance"],
            return_dict=False,
        )[0]
    torch.testing.assert_close(fused_out, expected, rtol=1e-4, atol=1e-4)


def test_fuse_proj_out_is_refused_off_tp1(monkeypatch):
    _, model, _ = _models()
    monkeypatch.setattr(mf, "_safe_tp_size", lambda: 2)
    block = model.single_transformer_blocks[0]
    assert fuse_proj_out(block) is False
    assert getattr(block, "proj_out", None) is None and block.proj_out_attn is not None


def test_split_state_dict_still_loads_strict_and_the_fused_block_has_no_split_keys():
    _, model, _ = _models()
    for block in model.single_transformer_blocks:
        fuse_proj_out(block)
    keys = set(model.state_dict())
    assert not any("proj_out_attn" in k or "proj_out_mlp" in k for k in keys)
    assert "single_transformer_blocks.0.proj_out.weight" in keys
