"""CPU-backend forward-pass coverage for difflet.models.wan.umt5.modeling_umt5.

Exercises the gated FFN, attention (with/without relative bias, masked, reused
position bias), relative-position bucketing, blocks, encoder stack and the
top-level encoder model on the ``cpu`` op backend with a tiny config.
"""

from __future__ import annotations

import os
import importlib

import pytest
import torch

_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"

import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.wan.umt5.modeling_umt5 as umt5  # noqa: E402

importlib.reload(umt5)

# The cpu op symbols are now bound into the reloaded module; restore the original
# backend env so importing this module doesn't leak cpu to later suite modules.
if _ORIG_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _ORIG_BACKEND


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DIFFLET_BACKEND", None)
        else:
            os.environ["DIFFLET_BACKEND"] = prev


def _tiny_config(**overrides):
    base = dict(
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_heads=4,
        num_layers=2,
        vocab_size=128,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
    )
    base.update(overrides)
    return umt5.WanUmT5Config(**base)


# ---------------------------------------------------------------------------
# Config


def test_config_inner_dim_and_from_diffusers_dict():
    cfg = _tiny_config()
    assert cfg.inner_dim == 4 * 8
    built = umt5.WanUmT5Config.from_diffusers_dict(
        {"d_model": 16, "num_heads": 2, "junk": 1}
    )
    assert built.d_model == 16
    assert built.num_heads == 2


def test_config_rejects_non_gated_act():
    with pytest.raises(NotImplementedError, match="gated"):
        _tiny_config(is_gated_act=False)


# ---------------------------------------------------------------------------
# Feed-forward


def test_dense_gated_act_dense_forward():
    cfg = _tiny_config()
    ff = umt5.WanUmT5DenseGatedActDense(cfg)
    out = ff(torch.randn(1, 5, cfg.d_model))
    assert out.shape == (1, 5, cfg.d_model)


def test_layer_ff_residual_forward():
    cfg = _tiny_config()
    layer = umt5.WanUmT5LayerFF(cfg)
    x = torch.randn(1, 5, cfg.d_model)
    out = layer(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Attention


def test_relative_position_bucket_bidirectional_and_unidirectional():
    pos = torch.arange(-4, 5).view(1, -1)
    bi = umt5.WanUmT5Attention._relative_position_bucket(
        pos, bidirectional=True, num_buckets=8, max_distance=16
    )
    uni = umt5.WanUmT5Attention._relative_position_bucket(
        pos, bidirectional=False, num_buckets=8, max_distance=16
    )
    assert bi.shape == pos.shape
    assert uni.shape == pos.shape
    # Bidirectional encodes sign in the high bucket half; unidirectional clamps
    # positive offsets to bucket 0.
    assert int(uni[0, -1]) == 0


def test_attention_with_relative_bias_forward():
    cfg = _tiny_config()
    attn = umt5.WanUmT5Attention(cfg, has_relative_attention_bias=True)
    hidden = torch.randn(1, 6, cfg.d_model)
    out, position_bias = attn(hidden)
    assert out.shape == hidden.shape
    assert position_bias.shape == (1, cfg.num_heads, 6, 6)


def test_attention_without_relative_bias_uses_zero_bias():
    cfg = _tiny_config()
    attn = umt5.WanUmT5Attention(cfg, has_relative_attention_bias=False)
    hidden = torch.randn(1, 6, cfg.d_model)
    out, position_bias = attn(hidden)
    assert out.shape == hidden.shape
    assert torch.count_nonzero(position_bias) == 0


def test_attention_applies_mask_into_position_bias():
    cfg = _tiny_config()
    attn = umt5.WanUmT5Attention(cfg, has_relative_attention_bias=False)
    hidden = torch.randn(1, 4, cfg.d_model)
    mask = torch.zeros(1, 1, 1, 4)
    mask[..., -1] = torch.finfo(hidden.dtype).min
    out, position_bias = attn(hidden, mask=mask)
    assert out.shape == hidden.shape
    # The mask was folded into the returned position bias.
    assert float(position_bias[0, 0, 0, -1]) < -1e30


def test_attention_reuses_passed_position_bias():
    cfg = _tiny_config()
    attn = umt5.WanUmT5Attention(cfg, has_relative_attention_bias=True)
    hidden = torch.randn(1, 5, cfg.d_model)
    bias = torch.zeros(1, cfg.num_heads, 5, 5)
    out, returned = attn(hidden, position_bias=bias)
    assert out.shape == hidden.shape
    assert returned is bias


def test_layer_self_attention_and_block_forward():
    cfg = _tiny_config()
    block = umt5.WanUmT5Block(cfg)
    hidden = torch.randn(1, 5, cfg.d_model)
    out, position_bias = block(hidden)
    assert out.shape == hidden.shape
    assert position_bias.shape == (1, cfg.num_heads, 5, 5)


# ---------------------------------------------------------------------------
# Encoder stack / model


def test_encoder_model_forward_no_mask():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = umt5.WanUmT5EncoderModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model(input_ids)
    assert out.shape == (1, 5, cfg.d_model)


def test_encoder_model_forward_with_attention_mask():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = umt5.WanUmT5EncoderModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.int64)
    mask[0, -2:] = 0
    out = model(input_ids, attention_mask=mask)
    assert out.shape == (2, 6, cfg.d_model)
