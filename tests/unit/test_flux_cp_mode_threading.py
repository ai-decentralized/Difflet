"""Unit test: cp_mode is threaded from NeuronFluxAttention through NeuronFluxTransformerBlock.

CPU construction only (tp=1). Skipped when the unit-test conftest mocks torch
(set DIFFLET_BACKEND=cpu + NEURON_PLATFORM_TARGET_OVERRIDE=cpu to run).

The Flux module imports from difflet.backends.trainium.core and difflet.layers, which
pull in torch_neuronx / neuronx_distributed. We pre-stub those before any difflet
import so the module loads cleanly on a CPU host. This exactly mirrors the approach used
in test_wan_cp_mode_threading.py (which stubs torch_xla for the Wan case).
"""

import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock

import pytest
import torch

os.environ.setdefault("DIFFLET_BACKEND", "cpu")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "cpu")

# tests/conftest.py mocks torch for the default logic-only unit run; this is a
# real-torch (CPU) construction test, so skip it when torch is the MagicMock stand-in.
pytestmark = pytest.mark.skipif(
    isinstance(torch, MagicMock),
    reason="requires real torch (unit-test conftest mocks torch)",
)


def _mock_pkg(name: str) -> types.ModuleType:
    """Insert a thin types.ModuleType stub into sys.modules."""
    m = types.ModuleType(name)
    m.__path__ = []  # type: ignore[attr-defined]
    m.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None, origin="mock")
    sys.modules[name] = m
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], attr, m)
    return m


if not isinstance(torch, MagicMock):
    # ---- torch_xla stub (diffusers imports torch_xla at module level) ----
    if "torch_xla" not in sys.modules:
        for _xla_name in [
            "torch_xla",
            "torch_xla.core",
            "torch_xla.core.xla_model",
            "torch_xla.core.xla_builder",
            "torch_xla.core.xla_op_registry",
            "torch_xla.experimental",
            "torch_xla.experimental.custom_kernel",
            "torch_xla.runtime",
        ]:
            _mock_pkg(_xla_name)
        sys.modules["torch_xla.experimental.custom_kernel"].flash_attention = MagicMock()  # type: ignore[attr-defined]
        sys.modules["torch_xla.runtime"].is_spmd = MagicMock(return_value=False)  # type: ignore[attr-defined]

    # ---- difflet.backends.trainium stubs (modeling_flux imports these directly) ----
    # We stub only the difflet-internal trainium modules so the heavy
    # neuronx_distributed / torch_neuronx chains are never executed on a CPU host.

    class _InferenceConfig:  # noqa: E303
        pass

    class _NeuronApplicationBase:
        pass

    class _ModelWrapper:
        pass

    class _BaseModelInstance:
        pass

    class _ModuleMarkerEndWrapper:
        def __call__(self, *args):
            return args

    class _ModuleMarkerStartWrapper:
        def __call__(self, *args):
            return args

    _trainium_stubs: list[tuple[str, dict]] = [
        ("difflet.backends.trainium", {}),
        ("difflet.backends.trainium.core", {}),
        (
            "difflet.backends.trainium.core.application_base",
            {"NeuronApplicationBase": _NeuronApplicationBase},
        ),
        (
            "difflet.backends.trainium.core.config",
            {"InferenceConfig": _InferenceConfig, "NeuronConfig": MagicMock()},
        ),
        (
            "difflet.backends.trainium.core.model_wrapper",
            {"ModelWrapper": _ModelWrapper, "BaseModelInstance": _BaseModelInstance},
        ),
        (
            "difflet.backends.trainium.core.layer_boundary_marker",
            {
                "ModuleMarkerEndWrapper": _ModuleMarkerEndWrapper,
                "ModuleMarkerStartWrapper": _ModuleMarkerStartWrapper,
            },
        ),
    ]
    for _mod_name, _attrs in _trainium_stubs:
        if _mod_name not in sys.modules:
            _m = _mock_pkg(_mod_name)
            for _attr, _val in _attrs.items():
                setattr(_m, _attr, _val)

    from difflet.models.flux.modeling_flux import (  # noqa: E402
        NeuronFluxAttention,
        NeuronFluxTransformerBlock,
        NeuronFluxSingleTransformerBlock,
    )
else:
    # Conftest mocked run — tests are skipped; define stubs so function bodies
    # don't raise NameError at collection time.
    NeuronFluxAttention = None  # type: ignore[assignment,misc]
    NeuronFluxTransformerBlock = None  # type: ignore[assignment,misc]
    NeuronFluxSingleTransformerBlock = None  # type: ignore[assignment,misc]


def test_flux_attention_stores_cp_mode_default():
    attn = NeuronFluxAttention(query_dim=64, heads=2, dim_head=32)
    assert attn.cp_mode == "gather_kv"


def test_flux_attention_stores_cp_mode_ring():
    attn = NeuronFluxAttention(query_dim=64, heads=2, dim_head=32, cp_mode="ring")
    assert attn.cp_mode == "ring"


def test_flux_double_stream_block_threads_cp_mode():
    """NeuronFluxTransformerBlock passes cp_mode down to its NeuronFluxAttention."""
    block = NeuronFluxTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        cp_mode="ring",
    )
    assert block.attn.cp_mode == "ring"


def test_flux_single_stream_block_threads_cp_mode():
    """NeuronFluxSingleTransformerBlock passes cp_mode down to its NeuronFluxAttention."""
    block = NeuronFluxSingleTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        cp_mode="ring",
    )
    assert block.attn.cp_mode == "ring"
