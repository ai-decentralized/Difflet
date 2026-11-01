"""``shard_modulation_linears``: the AdaLN modulation linears made column-parallel.

On the CPU backend ``ColumnParallelLinear`` is an ``nn.Linear`` (tp=1), so a swapped model with the
original weights must give the original output, under the original checkpoint keys. Each rank's
rows at tp=4 are covered per backend (``tests/unit/backends/test_neuron_hv1_dit.py``,
``tests/unit/backends/test_trainium_hv1_modulation.py``).
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"

import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.hunyuan_video.modeling_hunyuan_video as hv  # noqa: E402

importlib.reload(hv)

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


_TINY = dict(
    in_channels=2, out_channels=2, num_attention_heads=2, attention_head_dim=4,
    num_layers=2, num_single_layers=2, num_refiner_layers=1, mlp_ratio=2.0,
    patch_size=2, patch_size_t=1, guidance_embeds=True, text_embed_dim=6,
    pooled_projection_dim=5, rope_axes_dim=(2, 2, 0),
)


def _tiny_model():
    torch.manual_seed(0)
    model = hv.HunyuanVideoTransformer3DModel(**_TINY)
    model.eval()
    return model


def _tiny_inputs():
    torch.manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, 2, 2, 4, 4),
        timestep=torch.tensor([7], dtype=torch.long),
        encoder_hidden_states=torch.randn(1, 4, 6),
        encoder_attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        pooled_projections=torch.randn(1, 5),
        guidance=torch.tensor([3], dtype=torch.long),
    )


def _modulation_layers(model):
    for block in model.transformer_blocks:
        yield block.norm1.linear
        yield block.norm1_context.linear
    for block in model.single_transformer_blocks:
        yield block.norm.linear


def test_every_modulation_linear_becomes_column_parallel():
    model = _tiny_model()
    shapes = [(layer.in_features, layer.out_features) for layer in _modulation_layers(model)]

    hv.shard_modulation_linears(model)

    layers = list(_modulation_layers(model))
    assert len(layers) == 2 * _TINY["num_layers"] + _TINY["num_single_layers"]
    assert all(isinstance(layer, hv.ColumnParallelLinear) for layer in layers)
    assert [(layer.in_features, layer.out_features) for layer in layers] == shapes
    assert all(layer.bias is not None for layer in layers)


def test_swap_keeps_the_checkpoint_keys_and_the_output():
    reference = _tiny_model()
    swapped = _tiny_model()
    hv.shard_modulation_linears(swapped)

    assert list(swapped.state_dict()) == list(reference.state_dict())
    swapped.load_state_dict(reference.state_dict())

    inputs = _tiny_inputs()
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]
        actual = swapped(**inputs, return_dict=False)[0]
    assert torch.equal(actual, expected)
