"""Unit tests for difflet.layers.normalization on the CPU backend."""

import importlib

import pytest
import torch


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    import difflet.ops as ops

    importlib.reload(ops)
    import difflet.layers.embeddings as emb

    importlib.reload(emb)
    import difflet.layers.normalization as norm

    importlib.reload(norm)
    yield


def _norm():
    import difflet.layers.normalization as norm

    return norm


def test_ada_layer_norm_zero_single_forward():
    norm = _norm()
    torch.manual_seed(0)
    layer = norm.NeuronAdaLayerNormZeroSingle(8, reduce_dtype=torch.float32)
    x = torch.randn(2, 3, 8)
    emb = torch.randn(2, 8)
    out, gate = layer(x, emb)
    assert out.shape == (2, 3, 8)
    assert gate.shape == (2, 8)


def test_ada_layer_norm_zero_single_invalid_norm_type():
    norm = _norm()
    with pytest.raises(ValueError, match="Unsupported"):
        norm.NeuronAdaLayerNormZeroSingle(8, norm_type="rms")


def test_ada_layer_norm_zero_single_nonparallel_linear():
    norm = _norm()
    torch.manual_seed(0)
    layer = norm.NeuronAdaLayerNormZeroSingle(8, use_parallel_layer=False)
    assert isinstance(layer.linear, torch.nn.Linear)
    out, gate = layer(torch.randn(1, 2, 8), torch.randn(1, 8))
    assert out.shape == (1, 2, 8)


def test_ada_layer_norm_zero_forward_with_emb():
    norm = _norm()
    torch.manual_seed(0)
    layer = norm.NeuronAdaLayerNormZero(8, reduce_dtype=torch.float32)
    x = torch.randn(2, 3, 8)
    emb = torch.randn(2, 8)
    out = layer(x, emb=emb)
    assert len(out) == 5
    x_out, gate_msa, shift_mlp, scale_mlp, gate_mlp = out
    assert x_out.shape == (2, 3, 8)
    for t in (gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert t.shape == (2, 8)


def test_ada_layer_norm_zero_forward_with_class_embeddings():
    norm = _norm()
    torch.manual_seed(0)
    layer = norm.NeuronAdaLayerNormZero(
        8, num_embeddings=10, reduce_dtype=torch.float32
    )
    assert layer.emb is not None
    x = torch.randn(2, 3, 8)
    out = layer(
        x,
        timestep=torch.arange(2, dtype=torch.float32),
        class_labels=torch.tensor([1, 2]),
        hidden_dtype=torch.float32,
    )
    assert out[0].shape == (2, 3, 8)


def test_ada_layer_norm_zero_invalid_norm_type():
    norm = _norm()
    with pytest.raises(ValueError, match="Unsupported"):
        norm.NeuronAdaLayerNormZero(8, norm_type="rms")


def test_ada_layer_norm_zero_nonparallel_linear():
    norm = _norm()
    layer = norm.NeuronAdaLayerNormZero(8, use_parallel_layer=False)
    assert isinstance(layer.linear, torch.nn.Linear)


def test_ada_layer_norm_continuous_forward():
    norm = _norm()
    torch.manual_seed(0)
    layer = norm.NeuronAdaLayerNormContinuous(
        embedding_dim=8, conditioning_embedding_dim=4, reduce_dtype=torch.float32
    )
    x = torch.randn(2, 3, 8)
    cond = torch.randn(2, 4)
    out = layer(x, cond)
    assert out.shape == (2, 3, 8)


def test_ada_layer_norm_continuous_invalid_norm_type():
    norm = _norm()
    with pytest.raises(ValueError, match="unknown norm_type"):
        norm.NeuronAdaLayerNormContinuous(8, 4, norm_type="rms")


def test_ada_layer_norm_continuous_nonparallel():
    norm = _norm()
    layer = norm.NeuronAdaLayerNormContinuous(8, 4, use_parallel_layer=False)
    assert isinstance(layer.linear, torch.nn.Linear)
