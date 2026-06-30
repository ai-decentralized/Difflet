"""Unit tests for difflet.models.wan.modeling_wan (W3a)."""

from __future__ import annotations

import inspect
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

import pytest
import torch


def test_modeling_wan_imports_only_from_allowed_modules():
    """Rule 1 of cclogs/09 §2: Wan modeling imports only torch / stdlib /
    diffusers / difflet.ops. No direct neuronx_distributed / nkilib /
    torch_neuronx imports.

    Walks the AST so prose mentions of forbidden module names (in docstrings
    or comments describing the rule itself) don't trip the gate.
    """
    import ast

    src = (_REPO_ROOT / "difflet/models/wan/modeling_wan.py").read_text()
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

    assert not offending, f"forbidden imports in modeling_wan.py: {offending}"


def test_wan_transformer_config_defaults_match_a14b():
    from difflet.models.wan.modeling_wan import WanTransformerConfig

    cfg = WanTransformerConfig()
    assert cfg.patch_size == (1, 2, 2)
    assert cfg.num_attention_heads == 40
    assert cfg.attention_head_dim == 128
    assert cfg.in_channels == 16
    assert cfg.out_channels == 16
    assert cfg.text_dim == 4096
    assert cfg.ffn_dim == 13824
    assert cfg.num_layers == 40
    assert cfg.cross_attn_norm is True
    assert cfg.qk_norm == "rms_norm_across_heads"
    assert cfg.rope_max_seq_len == 1024
    assert cfg.inner_dim == 5120


def test_wan_transformer_config_from_diffusers_dict_filters_unknown_keys():
    from difflet.models.wan.modeling_wan import WanTransformerConfig

    raw = {
        "patch_size": [1, 2, 2],
        "num_attention_heads": 40,
        "ffn_dim": 13824,
        "_class_name": "WanTransformer3DModel",  # noise key, must be ignored
        "_diffusers_version": "0.38.0",  # noise key
    }
    cfg = WanTransformerConfig.from_diffusers_dict(raw)
    assert cfg.patch_size == (1, 2, 2)
    assert cfg.num_attention_heads == 40
    assert cfg.ffn_dim == 13824


def test_wan_transformer_config_rejects_unsupported_i2v_fields():
    from difflet.models.wan.modeling_wan import WanTransformerConfig

    with pytest.raises(NotImplementedError, match="image_dim"):
        WanTransformerConfig(image_dim=1280)

    with pytest.raises(NotImplementedError, match="added_kv_proj_dim"):
        WanTransformerConfig(added_kv_proj_dim=1280)

    with pytest.raises(NotImplementedError, match="pos_embed_seq_len"):
        WanTransformerConfig(pos_embed_seq_len=257)


def test_wan_transformer_config_rejects_unsupported_qk_norm():
    from difflet.models.wan.modeling_wan import WanTransformerConfig

    with pytest.raises(NotImplementedError, match="qk_norm"):
        WanTransformerConfig(qk_norm=None)


def test_wan_rotary_pos_embed_shapes_match_diffusers_reference():
    """WanRotaryPosEmbed should produce (1, S, 1, head_dim) freq tensors
    matching the diffusers WanRotaryPosEmbed for the same config.
    """
    from diffusers.models.transformers.transformer_wan import (
        WanRotaryPosEmbed as DiffWanRotaryPosEmbed,
    )

    from difflet.models.wan.modeling_wan import WanRotaryPosEmbed

    head_dim = 128
    patch = (1, 2, 2)
    max_seq_len = 1024

    difflet_rope = WanRotaryPosEmbed(head_dim, patch, max_seq_len)
    diff_rope = DiffWanRotaryPosEmbed(head_dim, patch, max_seq_len)

    # tiny tensor: B=1, C=16, T=2, H=8, W=8 — patch (1,2,2) → ppf=2, pph=4, ppw=4
    x = torch.randn(1, 16, 2, 8, 8)
    difflet_cos, difflet_sin = difflet_rope(x)
    diff_cos, diff_sin = diff_rope(x)

    s_tokens = 2 * 4 * 4
    assert difflet_cos.shape == (1, s_tokens, 1, head_dim)
    assert difflet_sin.shape == (1, s_tokens, 1, head_dim)
    assert torch.allclose(difflet_cos, diff_cos, atol=1e-6, rtol=1e-6)
    assert torch.allclose(difflet_sin, diff_sin, atol=1e-6, rtol=1e-6)


def test_wan_rotary_axis_split_uses_canonical_t_h_w_dims():
    """Wan splits attention_head_dim into (t_dim, h_dim, w_dim) where
    h_dim = w_dim = 2 * (head_dim // 6). For head_dim=128: h=w=2*21=42, t=44.
    """
    from difflet.models.wan.modeling_wan import WanRotaryPosEmbed

    rope = WanRotaryPosEmbed(attention_head_dim=128, patch_size=(1, 2, 2), max_seq_len=64)
    assert rope.t_dim == 44
    assert rope.h_dim == 42
    assert rope.w_dim == 42
    assert rope.t_dim + rope.h_dim + rope.w_dim == 128


def test_modeling_wan_classes_are_importable():
    """All public classes/functions are exported from the module."""
    from difflet.models.wan import modeling_wan

    expected = [
        "WanAttention",
        "WanFeedForward",
        "WanRotaryPosEmbed",
        "WanTimeTextEmbedding",
        "WanTransformer3DModel",
        "WanTransformerBlock",
        "WanTransformerConfig",
    ]
    for name in expected:
        cls = getattr(modeling_wan, name)
        assert inspect.isclass(cls), f"{name} should be a class"


def test_wan_transformer_3d_model_top_level_signature_matches_plan():
    """Forward must accept (hidden_states, timestep, encoder_hidden_states,
    timestep_seq_len=None). Locked by cclogs/09 W3 plan and survey §3.
    """
    from difflet.models.wan.modeling_wan import WanTransformer3DModel

    sig = inspect.signature(WanTransformer3DModel.forward)
    params = list(sig.parameters)
    assert params[:4] == ["self", "hidden_states", "timestep", "encoder_hidden_states"]
    assert "timestep_seq_len" in sig.parameters


def test_wan_time_text_embedding_drops_image_branch():
    """T2V-only spike: WanTimeTextEmbedding must NOT accept image conditioning
    args (cclogs/09 §1 non-goals: no I2V).
    """
    from difflet.models.wan.modeling_wan import WanTimeTextEmbedding

    sig = inspect.signature(WanTimeTextEmbedding.__init__)
    assert "image_embed_dim" not in sig.parameters
    assert "pos_embed_seq_len" not in sig.parameters
    sig_fwd = inspect.signature(WanTimeTextEmbedding.forward)
    assert "encoder_hidden_states_image" not in sig_fwd.parameters


@pytest.mark.parametrize(
    "shape",
    [(1, 16, 1, 4, 4), (2, 16, 3, 8, 8)],
)
def test_wan_rotary_pos_embed_handles_varying_input_shapes(shape):
    """Rotary should produce S = (T/p_t)*(H/p_h)*(W/p_w) tokens for any
    input shape that respects patch_size divisibility.
    """
    from difflet.models.wan.modeling_wan import WanRotaryPosEmbed

    head_dim = 128
    rope = WanRotaryPosEmbed(head_dim, (1, 2, 2), max_seq_len=128)
    x = torch.randn(*shape)
    cos, sin = rope(x)

    _, _, T, H, W = shape
    s_expected = T * (H // 2) * (W // 2)
    assert cos.shape == (1, s_expected, 1, head_dim)
    assert sin.shape == (1, s_expected, 1, head_dim)
