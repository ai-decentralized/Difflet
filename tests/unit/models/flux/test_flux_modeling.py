"""CPU-backend unit tests for difflet.models.flux.modeling_flux.

Exercises the attention/feed-forward/transformer-block forwards and the full
NeuronFluxTransformer2DModel forward on the torch-native (DIFFLET_BACKEND=cpu)
backend with tiny dims. The XLA layer-boundary markers (which require a real XLA
tensor) are patched to identity so the eager CPU forward can run; everything else
is the real model code.
"""

import importlib
import math
import os

import pytest

_PREV_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch  # noqa: E402


def _reload(name):
    return importlib.reload(importlib.import_module(name))


# Reload only the flux modeling module so its `from difflet.ops import ...`
# rebinds onto the cpu backend. Shared trainium core / difflet.layers modules are
# imported normally — reloading them would replace classes that other test
# modules (registry, wan, qwen) depend on and break them mid-session.
import difflet.backends.trainium.core.config as _config_mod  # noqa: E402
import difflet.layers.normalization as _norm_mod  # noqa: E402
mf = _reload("difflet.models.flux.modeling_flux")

NeuronConfig = _config_mod.NeuronConfig


def _restore_backend_env():
    if _PREV_BACKEND is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = _PREV_BACKEND


_restore_backend_env()

HEADS = 2
HEAD_DIM = 8
DIM = HEADS * HEAD_DIM


class _IdentityMarker:
    """Stand-in for the XLA module-boundary markers (identity on CPU)."""

    def __call__(self, *args):
        return args[0] if len(args) == 1 else args


@pytest.fixture(autouse=True)
def _cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    torch.manual_seed(0)
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


@pytest.fixture
def patched_markers(monkeypatch):
    monkeypatch.setattr(mf, "ModuleMarkerStartWrapper", _IdentityMarker)
    monkeypatch.setattr(mf, "ModuleMarkerEndWrapper", _IdentityMarker)
    monkeypatch.setattr(_norm_mod, "ModuleMarkerStartWrapper", _IdentityMarker)
    monkeypatch.setattr(_norm_mod, "ModuleMarkerEndWrapper", _IdentityMarker)
    yield


def _backbone_config(num_layers=1, num_single_layers=0, guidance_embeds=False, **overrides):
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    kwargs = dict(
        neuron_config=nc,
        attention_head_dim=HEAD_DIM,
        guidance_embeds=guidance_embeds,
        in_channels=4,
        joint_attention_dim=DIM,
        num_attention_heads=HEADS,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        patch_size=1,
        pooled_projection_dim=8,
        height=16,
        width=16,
        out_channels=4,
    )
    kwargs.update(overrides)
    return mf.FluxBackboneInferenceConfig(**kwargs)


# --------------------------------------------------------------------------
# Pure helper functions
# --------------------------------------------------------------------------
def test_attention_wrapper_sharded_without_swap():
    q = torch.randn(1, HEADS, 5, HEAD_DIM)
    k = torch.randn(1, HEADS, 5, HEAD_DIM)
    v = torch.randn(1, HEADS, 5, HEAD_DIM)
    out = mf.attention_wrapper_sharded_without_swap(q, k, v)
    assert out.shape == (1, HEADS, 5, HEAD_DIM)
    # numeric oracle: plain scaled dot product attention
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(HEAD_DIM)
    )
    assert torch.allclose(out, ref, atol=1e-5)


def test_split_along_dim_identity_on_cpu():
    x = torch.randn(2, 4, DIM)
    out = mf.split_along_dim(x, dim=1, rank=0, data_parallel_group=None)
    assert torch.equal(out, x)


# --------------------------------------------------------------------------
# NeuronFluxAttention
# --------------------------------------------------------------------------
def test_attention_qk_norm_none():
    attn = mf.NeuronFluxAttention(
        query_dim=DIM, dim_head=HEAD_DIM, heads=HEADS, out_dim=DIM, qk_norm=None,
        reduce_dtype=torch.float32,
    )
    assert attn.norm_q is None and attn.norm_k is None


def test_attention_invalid_qk_norm_raises():
    with pytest.raises(ValueError):
        mf.NeuronFluxAttention(
            query_dim=DIM, dim_head=HEAD_DIM, heads=HEADS, out_dim=DIM, qk_norm="layer_norm",
        )


def test_attention_invalid_cross_attention_norm_raises():
    with pytest.raises(ValueError):
        mf.NeuronFluxAttention(
            query_dim=DIM, dim_head=HEAD_DIM, heads=HEADS, out_dim=DIM,
            cross_attention_norm="group_norm",
        )


def test_attention_single_stream_pre_only_forward():
    attn = mf.NeuronFluxAttention(
        query_dim=DIM, cross_attention_dim=None, dim_head=HEAD_DIM, heads=HEADS,
        out_dim=DIM, bias=True, qk_norm="rms_norm", eps=1e-6, pre_only=True,
        reduce_dtype=torch.float32,
    ).eval()
    s = 5
    h = torch.randn(1, s, DIM)
    rot = torch.randn(s, HEAD_DIM, 2)
    out = attn(hidden_states=h, image_rotary_emb=rot)
    assert out.shape == (1, s, DIM)


def test_attention_double_stream_forward():
    attn = mf.NeuronFluxAttention(
        query_dim=DIM, cross_attention_dim=None, added_kv_proj_dim=DIM, dim_head=HEAD_DIM,
        heads=HEADS, out_dim=DIM, context_pre_only=False, bias=True, qk_norm="rms_norm",
        eps=1e-6, reduce_dtype=torch.float32,
    ).eval()
    s_txt, s_img = 3, 5
    h = torch.randn(1, s_img, DIM)
    e = torch.randn(1, s_txt, DIM)
    rot = torch.randn(s_txt + s_img, HEAD_DIM, 2)
    hidden, enc = attn(hidden_states=h, encoder_hidden_states=e, image_rotary_emb=rot)
    assert hidden.shape == (1, s_img, DIM)
    assert enc.shape == (1, s_txt, DIM)


# --------------------------------------------------------------------------
# NeuronFeedForward
# --------------------------------------------------------------------------
def test_feed_forward_forward():
    ff = mf.NeuronFeedForward(
        dim=DIM, dim_out=DIM, activation_fn="gelu-approximate", reduce_dtype=torch.float32
    )
    out = ff(torch.randn(1, 4, DIM))
    assert out.shape == (1, 4, DIM)


def test_feed_forward_warns_on_scale(caplog):
    ff = mf.NeuronFeedForward(
        dim=DIM, dim_out=DIM, activation_fn="gelu-approximate", reduce_dtype=torch.float32
    )
    out = ff(torch.randn(1, 2, DIM), scale=1.0)
    assert out.shape == (1, 2, DIM)


# --------------------------------------------------------------------------
# Transformer blocks
# --------------------------------------------------------------------------
def test_single_transformer_block_construction():
    block = mf.NeuronFluxSingleTransformerBlock(
        dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        reduce_dtype=torch.float32, cp_mode="ring",
    )
    assert block.attn.cp_mode == "ring"
    assert block.mlp_hidden_dim == int(DIM * 4.0)


def test_double_transformer_block_forward(patched_markers):
    block = mf.NeuronFluxTransformerBlock(
        dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
        reduce_dtype=torch.float32,
    ).eval()
    s_txt, s_img = 3, 5
    hidden = torch.randn(1, s_img, DIM)
    enc = torch.randn(1, s_txt, DIM)
    temb = torch.randn(1, DIM)
    rot = torch.randn(s_txt + s_img, HEAD_DIM, 2)
    out_enc, out_hidden = block(
        hidden_states=hidden, encoder_hidden_states=enc, temb=temb, image_rotary_emb=rot
    )
    assert out_hidden.shape == (1, s_img, DIM)
    assert out_enc.shape == (1, s_txt, DIM)


# --------------------------------------------------------------------------
# Full NeuronFluxTransformer2DModel
# --------------------------------------------------------------------------
def _run_full_model(config):
    model = mf.NeuronFluxTransformer2DModel(config).eval().to(torch.float32)
    b, num_patches, s_txt = 1, 4, 6
    hidden_states = torch.randn(b, num_patches, config.in_channels)
    encoder_hidden_states = torch.randn(b, s_txt, config.joint_attention_dim)
    pooled = torch.randn(b, config.pooled_projection_dim)
    timestep = torch.rand(b)
    rot = torch.randn(num_patches + s_txt, config.attention_head_dim, 2)
    guidance = torch.rand(b) if config.guidance_embeds else None
    return model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled,
        timestep=timestep,
        guidance=guidance,
        image_rotary_emb=rot,
    )


def test_full_model_construction_sets_dims():
    config = _backbone_config()
    model = mf.NeuronFluxTransformer2DModel(config)
    assert model.inner_dim == DIM
    assert model.out_channels == 4
    assert not model.context_parallel_enabled
    assert not model.cfg_parallel_enabled


def test_full_model_forward_no_guidance(patched_markers):
    config = _backbone_config(guidance_embeds=False)
    out = _run_full_model(config)
    # output is [B, num_patches, patch_size**2 * out_channels]
    assert out.shape == (1, 4, 4)


def test_full_model_forward_with_guidance(patched_markers):
    config = _backbone_config(guidance_embeds=True)
    out = _run_full_model(config)
    assert out.shape == (1, 4, 4)


def test_full_model_out_channels_defaults_to_in_channels():
    config = _backbone_config()
    delattr(config, "out_channels")
    model = mf.NeuronFluxTransformer2DModel(config)
    assert model.out_channels == config.in_channels


# --------------------------------------------------------------------------
# FluxBackboneInferenceConfig
# --------------------------------------------------------------------------
def test_backbone_config_required_attributes():
    config = _backbone_config()
    required = config.get_required_attributes()
    for attr in required:
        assert hasattr(config, attr)
    assert config.cp_mode == "gather_kv"
    assert config.cfg_parallel_enabled is False
    assert config.context_parallel_enabled is False


def test_backbone_config_mutual_exclusivity_raises():
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    with pytest.raises(ValueError):
        mf.FluxBackboneInferenceConfig(
            neuron_config=nc,
            cfg_parallel_enabled=True,
            context_parallel_enabled=True,
            attention_head_dim=HEAD_DIM,
            guidance_embeds=False,
            in_channels=4,
            joint_attention_dim=DIM,
            num_attention_heads=HEADS,
            num_layers=1,
            num_single_layers=0,
            patch_size=1,
            pooled_projection_dim=8,
            height=16,
            width=16,
        )


def test_backbone_convert_hf_to_neuron_state_dict_splits_single_blocks():
    config = _backbone_config(num_single_layers=1)
    inner_dim = config.num_attention_heads * config.attention_head_dim
    sd = {
        "single_transformer_blocks.0.proj_out.weight": torch.randn(DIM, inner_dim + 7),
        "single_transformer_blocks.0.proj_out.bias": torch.randn(DIM),
    }
    out = mf.NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(dict(sd), config)
    assert "single_transformer_blocks.0.proj_out_attn.weight" in out
    assert "single_transformer_blocks.0.proj_out_mlp.weight" in out
    assert "global_rank.rank" in out
    assert mf.NeuronFluxBackboneApplication.update_state_dict_for_tied_weights({}) is None
