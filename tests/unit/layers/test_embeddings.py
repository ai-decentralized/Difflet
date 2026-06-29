"""Unit tests for difflet.layers.embeddings on the CPU backend."""

import importlib
import math

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    import difflet.ops as ops

    importlib.reload(ops)
    import difflet.layers.activations as acts

    importlib.reload(acts)
    import difflet.layers.embeddings as emb

    importlib.reload(emb)
    yield


def _emb():
    import difflet.layers.embeddings as emb

    return emb


def test_apply_rotary_emb_use_real_unbind_minus1():
    emb = _emb()
    torch.manual_seed(0)
    x = torch.randn(1, 2, 3, 4)
    # freqs_cis shape [S, D, 2]
    s, d = 3, 4
    cos = torch.randn(s, d)
    sin = torch.randn(s, d)
    freqs = torch.stack([cos, sin], dim=-1)
    out = emb.apply_rotary_emb(x, freqs, use_real=True, use_real_unbind_dim=-1)
    assert out.shape == x.shape

    # reference
    c = cos[None, None]
    s_ = sin[None, None]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    expected = (x.float() * c + x_rot.float() * s_).to(x.dtype)
    assert torch.allclose(out, expected, atol=1e-5)


def test_apply_rotary_emb_use_real_unbind_minus2():
    emb = _emb()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 2, 4)
    cos = torch.randn(2, 4)
    sin = torch.randn(2, 4)
    freqs = torch.stack([cos, sin], dim=-1)
    out = emb.apply_rotary_emb(x, freqs, use_real=True, use_real_unbind_dim=-2)
    assert out.shape == x.shape


def test_apply_rotary_emb_invalid_unbind_dim():
    emb = _emb()
    x = torch.randn(1, 1, 2, 4)
    freqs = torch.randn(2, 4, 2)
    with pytest.raises(ValueError, match="use_real_unbind_dim"):
        emb.apply_rotary_emb(x, freqs, use_real=True, use_real_unbind_dim=0)


def test_apply_rotary_emb_complex_path():
    emb = _emb()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 2, 4)
    # complex freqs_cis broadcastable to [.., S, D//2]
    freqs_cis = torch.view_as_complex(torch.randn(2, 2, 2))
    out = emb.apply_rotary_emb(x, freqs_cis, use_real=False)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_get_timestep_embedding_shape_and_flip():
    emb = _emb()
    ts = torch.arange(4, dtype=torch.float32)
    out = emb.get_timestep_embedding(ts, 16, flip_sin_to_cos=True)
    assert out.shape == (4, 16)
    out2 = emb.get_timestep_embedding(ts, 16, flip_sin_to_cos=False)
    assert out2.shape == (4, 16)


def test_get_timestep_embedding_odd_dim_zero_pads():
    emb = _emb()
    ts = torch.arange(3, dtype=torch.float32)
    out = emb.get_timestep_embedding(ts, 7)
    assert out.shape == (3, 7)


def test_get_1d_rotary_pos_embed_repeat_interleave():
    emb = _emb()
    cos, sin = emb.get_1d_rotary_pos_embed(4, 5, use_real=True, repeat_interleave_real=True)
    assert cos.shape == (5, 4)
    assert sin.shape == (5, 4)


def test_get_1d_rotary_pos_embed_concat():
    emb = _emb()
    cos, sin = emb.get_1d_rotary_pos_embed(
        4, np.arange(5), use_real=True, repeat_interleave_real=False
    )
    assert cos.shape == (5, 4)


def test_get_1d_rotary_pos_embed_complex():
    emb = _emb()
    out = emb.get_1d_rotary_pos_embed(4, 5, use_real=False)
    assert out.shape == (5, 2)
    assert torch.is_complex(out)


def test_timesteps_module_forward():
    emb = _emb()
    layer = emb.Timesteps(num_channels=8, flip_sin_to_cos=True, downscale_freq_shift=0)
    out = layer(torch.arange(3, dtype=torch.float32))
    assert out.shape == (3, 8)


def test_neuron_timestep_embedding_forward():
    emb = _emb()
    torch.manual_seed(0)
    layer = emb.NeuronTimestepEmbedding(
        in_channels=8, time_embed_dim=16, reduce_dtype=torch.float32
    )
    out = layer(torch.randn(2, 8))
    assert out.shape == (2, 16)


def test_neuron_timestep_embedding_with_out_dim_and_post_act():
    emb = _emb()
    torch.manual_seed(0)
    layer = emb.NeuronTimestepEmbedding(
        in_channels=8,
        time_embed_dim=16,
        out_dim=4,
        post_act_fn="silu",
        reduce_dtype=torch.float32,
    )
    out = layer(torch.randn(1, 8))
    assert out.shape == (1, 4)


def test_neuron_label_embedding_forward_and_dropout():
    emb = _emb()
    torch.manual_seed(0)
    layer = emb.NeuronLabelEmbedding(num_classes=10, hidden_size=8, dropout_prob=0.1)
    labels = torch.tensor([0, 3, 9])
    out = layer(labels)
    assert out.shape == (3, 8)
    # force_drop_ids path
    dropped = layer.token_drop(labels, force_drop_ids=torch.tensor([1, 0, 1]))
    assert dropped[0].item() == 10  # mapped to num_classes
    out2 = layer(labels, force_drop_ids=torch.tensor([1, 0, 0]))
    assert out2.shape == (3, 8)


def test_neuron_label_embedding_training_dropout_path():
    emb = _emb()
    torch.manual_seed(0)
    layer = emb.NeuronLabelEmbedding(num_classes=10, hidden_size=8, dropout_prob=0.5)
    layer.train()
    out = layer(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 8)


def test_neuron_combined_timestep_label_embeddings_forward():
    emb = _emb()
    torch.manual_seed(0)
    layer = emb.NeuronCombinedTimestepLabelEmbeddings(
        num_classes=10, embedding_dim=16, reduce_dtype=torch.float32
    )
    out = layer(
        torch.arange(2, dtype=torch.float32),
        torch.tensor([1, 2]),
        hidden_dtype=torch.float32,
    )
    assert out.shape == (2, 16)


def test_neuron_pixart_alpha_text_projection_variants():
    emb = _emb()
    torch.manual_seed(0)
    for act_fn in ("gelu_tanh", "silu", "silu_fp32"):
        layer = emb.NeuronPixArtAlphaTextProjection(
            in_features=8, hidden_size=16, act_fn=act_fn, reduce_dtype=torch.float32
        )
        out = layer(torch.randn(2, 8))
        assert out.shape == (2, 16)


def test_neuron_pixart_alpha_text_projection_bad_act():
    emb = _emb()
    with pytest.raises(ValueError, match="Unknown activation"):
        emb.NeuronPixArtAlphaTextProjection(8, 16, act_fn="bogus")


def test_neuron_combined_timestep_text_proj_embeddings():
    emb = _emb()
    torch.manual_seed(0)
    layer = emb.NeuronCombinedTimestepTextProjEmbeddings(
        embedding_dim=16, pooled_projection_dim=8, reduce_dtype=torch.float32
    )
    out = layer(torch.arange(2, dtype=torch.float32), torch.randn(2, 8))
    assert out.shape == (2, 16)


def test_neuron_combined_timestep_guidance_text_proj_embeddings():
    emb = _emb()
    torch.manual_seed(0)
    layer = emb.NeuronCombinedTimestepGuidanceTextProjEmbeddings(
        embedding_dim=16, pooled_projection_dim=8, reduce_dtype=torch.float32
    )
    out = layer(
        torch.arange(2, dtype=torch.float32),
        torch.arange(2, dtype=torch.float32),
        torch.randn(2, 8),
    )
    assert out.shape == (2, 16)


def test_flux_pos_embed_forward():
    emb = _emb()
    layer = emb.FluxPosEmbed(theta=10000, axes_dim=[4, 4])
    ids = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    cos, sin = layer(ids)
    assert cos.shape[-1] == 8
    assert sin.shape == cos.shape
