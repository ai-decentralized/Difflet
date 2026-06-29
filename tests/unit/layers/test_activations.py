"""Unit tests for difflet.layers.activations on the CPU backend."""

import importlib

import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    import difflet.ops as ops

    importlib.reload(ops)
    import difflet.layers.activations as acts

    importlib.reload(acts)
    yield


def _acts():
    import difflet.layers.activations as acts

    return acts


def test_neuron_gelu_forward_shapes_and_values():
    torch.manual_seed(0)
    acts = _acts()
    layer = acts.NeuronGELU(8, 16, approximate="none", reduce_dtype=torch.float32)
    x = torch.randn(2, 3, 8)
    out = layer(x)
    assert out.shape == (2, 3, 16)
    # forward = gelu(proj(x))
    expected = torch.nn.functional.gelu(layer.proj(x), approximate="none")
    assert torch.allclose(out, expected, atol=1e-5)


def test_neuron_gelu_tanh_approximate():
    torch.manual_seed(0)
    acts = _acts()
    layer = acts.NeuronGELU(8, 8, approximate="tanh")
    x = torch.randn(1, 8)
    out = layer(x)
    expected = torch.nn.functional.gelu(layer.proj(x), approximate="tanh")
    assert torch.allclose(out, expected, atol=1e-5)


def test_fp32_silu_upcasts_and_restores_dtype():
    acts = _acts()
    layer = acts.FP32SiLU()
    x = torch.randn(4, dtype=torch.bfloat16)
    out = layer(x)
    assert out.dtype == torch.bfloat16
    expected = torch.nn.functional.silu(x.float()).to(torch.bfloat16)
    assert torch.equal(out, expected)


def test_activation_functions_table_keys():
    acts = _acts()
    assert set(acts.ACTIVATION_FUNCTIONS) == {"swish", "silu", "mish", "gelu", "relu"}
    assert isinstance(acts.ACTIVATION_FUNCTIONS["silu"], nn.SiLU)


@pytest.mark.parametrize(
    "name,cls",
    [
        ("swish", nn.SiLU),
        ("SILU", nn.SiLU),
        ("Mish", nn.Mish),
        ("gelu", nn.GELU),
        ("RELU", nn.ReLU),
    ],
)
def test_get_activation_case_insensitive(name, cls):
    acts = _acts()
    assert isinstance(acts.get_activation(name), cls)


def test_get_activation_unsupported_raises():
    acts = _acts()
    with pytest.raises(ValueError, match="Unsupported activation"):
        acts.get_activation("not_an_activation")
