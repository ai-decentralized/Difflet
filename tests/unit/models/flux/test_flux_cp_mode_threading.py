"""Unit test: cp_mode is threaded from NeuronFluxAttention through NeuronFluxTransformerBlock.

CPU construction only (tp=1). Skipped when the unit-test conftest mocks torch
(set DIFFLET_BACKEND=cpu + NEURON_PLATFORM_TARGET_OVERRIDE=cpu to run).

The Flux module imports from difflet.backends.trainium.core and difflet.layers, which
pull in torch_neuronx / neuronx_distributed. The ``flux`` fixture stubs those, imports the
classes under test, and restores ``sys.modules`` (with the parent packages' attributes) and
the environment exactly afterwards. Nothing happens at import time: a module-level stubbed
import would leave a half-stubbed ``modeling_flux`` cached for every other test in the session.
"""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

# tests/conftest.py mocks torch for the default logic-only unit run; this is a
# real-torch (CPU) construction test, so skip it when torch is the MagicMock stand-in.
pytestmark = pytest.mark.skipif(
    isinstance(torch, MagicMock),
    reason="requires real torch (unit-test conftest mocks torch)",
)

_ENV_KEYS = ("DIFFLET_BACKEND", "NEURON_PLATFORM_TARGET_OVERRIDE")
#: Imported afresh under the stubs (any cached real copies are set aside, then restored).
_FRESH_PREFIXES = ("difflet.models.flux", "difflet.backends.trainium", "difflet.layers")


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


def _install_stubs() -> None:
    # ---- torch_xla stub (diffusers imports torch_xla at module level) ----
    if "torch_xla" not in sys.modules:
        for name in [
            "torch_xla",
            "torch_xla.core",
            "torch_xla.core.xla_model",
            "torch_xla.core.xla_builder",
            "torch_xla.core.xla_op_registry",
            "torch_xla.experimental",
            "torch_xla.experimental.custom_kernel",
            "torch_xla.runtime",
        ]:
            _mock_pkg(name)
        sys.modules["torch_xla.experimental.custom_kernel"].flash_attention = MagicMock()  # type: ignore[attr-defined]
        sys.modules["torch_xla.runtime"].is_spmd = MagicMock(return_value=False)  # type: ignore[attr-defined]

    # ---- difflet.backends.trainium stubs (modeling_flux imports these directly) ----
    class _InferenceConfig:
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

    stubs: list[tuple[str, dict]] = [
        ("difflet.backends.trainium", {}),
        ("difflet.backends.trainium.core", {}),
        ("difflet.backends.trainium.core.application_base", {"NeuronApplicationBase": _NeuronApplicationBase}),
        ("difflet.backends.trainium.core.config", {"InferenceConfig": _InferenceConfig, "NeuronConfig": MagicMock()}),
        ("difflet.backends.trainium.core.model_wrapper",
         {"ModelWrapper": _ModelWrapper, "BaseModelInstance": _BaseModelInstance}),
        ("difflet.backends.trainium.core.layer_boundary_marker",
         {"ModuleMarkerEndWrapper": _ModuleMarkerEndWrapper, "ModuleMarkerStartWrapper": _ModuleMarkerStartWrapper}),
    ]
    for name, attrs in stubs:
        if name not in sys.modules:
            m = _mock_pkg(name)
            for attr, val in attrs.items():
                setattr(m, attr, val)


def _restore_modules(saved: dict) -> None:
    """Put sys.modules, and each affected parent package's attribute, back as they were."""
    changed = {n for n in sys.modules if n not in saved or sys.modules[n] is not saved[n]}
    changed |= {n for n in saved if n not in sys.modules}
    for name in changed:
        sys.modules.pop(name, None)
    sys.modules.update({n: saved[n] for n in changed if n in saved})
    for name in changed:
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name) if parent_name else None
        if parent is None:
            continue
        if name in saved:
            setattr(parent, child, saved[name])
        elif getattr(parent, child, None) is not None and child in vars(parent):
            delattr(parent, child)


@pytest.fixture(scope="module")
def flux():
    saved_modules = dict(sys.modules)
    saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ.setdefault("DIFFLET_BACKEND", "cpu")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "cpu")
    try:
        for name in [n for n in sys.modules if n.startswith(_FRESH_PREFIXES)]:
            del sys.modules[name]
        _install_stubs()
        from difflet.models.flux.modeling_flux import (
            NeuronFluxAttention,
            NeuronFluxSingleTransformerBlock,
            NeuronFluxTransformerBlock,
        )

        yield types.SimpleNamespace(
            Attention=NeuronFluxAttention,
            Block=NeuronFluxTransformerBlock,
            SingleBlock=NeuronFluxSingleTransformerBlock,
        )
    finally:
        _restore_modules(saved_modules)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_flux_attention_stores_cp_mode_default(flux):
    attn = flux.Attention(query_dim=64, heads=2, dim_head=32)
    assert attn.cp_mode == "gather_kv"


def test_flux_attention_stores_cp_mode_ring(flux):
    attn = flux.Attention(query_dim=64, heads=2, dim_head=32, cp_mode="ring")
    assert attn.cp_mode == "ring"


def test_flux_double_stream_block_threads_cp_mode(flux):
    """NeuronFluxTransformerBlock passes cp_mode down to its NeuronFluxAttention."""
    block = flux.Block(dim=64, num_attention_heads=2, attention_head_dim=32, cp_mode="ring")
    assert block.attn.cp_mode == "ring"


def test_flux_single_stream_block_threads_cp_mode(flux):
    """NeuronFluxSingleTransformerBlock passes cp_mode down to its NeuronFluxAttention."""
    block = flux.SingleBlock(dim=64, num_attention_heads=2, attention_head_dim=32, cp_mode="ring")
    assert block.attn.cp_mode == "ring"
