"""Unit test: cp_mode is threaded into _QwenImageTrainiumAttnProcessor.

CPU construction only (no Neuron hardware). Uses --noconftest to avoid the
MagicMock torch stand-in so real nn.Module construction works; the pytestmark
guard keeps normal unit-test runs clean when conftest mocks torch.

_QwenImageTrainiumAttnProcessor is a plain Python class (not nn.Module), but
its module (difflet.backends.trainium.qwen_image.transformer) imports torch and
diffusers at the top level, so this test requires real torch (--noconftest).
"""

import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock

import pytest
import torch

_ORIG_ENV = {k: os.environ.get(k) for k in ("DIFFLET_BACKEND", "NEURON_PLATFORM_TARGET_OVERRIDE")}
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
    _pre_modules = set(sys.modules)
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

    # ---- difflet.backends.trainium stubs (transformer.py imports these) ----
    # We stub only the difflet-internal trainium core modules so the heavy
    # neuronx_distributed chain is never executed on a CPU host.

    class _InferenceConfig:  # noqa: E303
        pass

    class _NeuronApplicationBase:
        pass

    class _ModelWrapper:
        pass

    class _BaseModelInstance:
        pass

    # Stub ONLY the heavy leaf modules that import neuronx_distributed /
    # torch_neuronx.  Do NOT stub 'difflet.backends.trainium' or
    # 'difflet.backends.trainium.core' — those are real packages whose __path__
    # must stay intact so Python can resolve qwen_image.transformer below.
    _trainium_stubs: list[tuple[str, dict]] = [
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
    ]
    for _mod_name, _attrs in _trainium_stubs:
        if _mod_name not in sys.modules:
            _m = _mock_pkg(_mod_name)
            for _attr, _val in _attrs.items():
                setattr(_m, _attr, _val)

    from difflet.backends.trainium.qwen_image.transformer import (  # noqa: E402
        _QwenImageTrainiumAttnProcessor,
    )

    # Suite-safety: purge everything imported in this stubbed context — the stub
    # modules themselves (origin="mock") AND any real difflet.backends.trainium
    # modules whose classes got bound to the stub bases above. Leaving the latter
    # cached makes a later real import reuse a half-real module (its base class is
    # the stub) and crash. The processor class we need is already bound locally.
    for _stub_name in [
        _n
        for _n in set(sys.modules) - _pre_modules
        if getattr(getattr(sys.modules.get(_n), "__spec__", None), "origin", None) == "mock"
        or _n.startswith("difflet.backends.trainium")
    ]:
        del sys.modules[_stub_name]

    # Suite-safety: restore env vars so we do not force later tests onto the cpu
    # backend (construction above only needs them at import time).
    for _k, _v in _ORIG_ENV.items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v
else:
    # In the conftest mocked run the tests are skipped; define a stub so the
    # function bodies don't raise NameError at collection time.
    _QwenImageTrainiumAttnProcessor = None  # type: ignore[assignment,misc]


def test_qwen_processor_stores_cp_mode_default():
    proc = _QwenImageTrainiumAttnProcessor()
    assert proc.cp_mode == "gather_kv"


def test_qwen_processor_stores_cp_mode_ring():
    proc = _QwenImageTrainiumAttnProcessor(context_parallel_enabled=True, cp_mode="ring")
    assert proc.cp_mode == "ring"
