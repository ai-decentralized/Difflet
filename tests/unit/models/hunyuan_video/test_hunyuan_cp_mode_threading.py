"""Unit test: cp_mode is threaded from HunyuanVideoAttention through HunyuanVideoTransformerBlock.

CPU construction only (tp=1). Uses --noconftest to avoid the MagicMock torch stand-in
so real nn.Module construction works; the pytestmark guard keeps normal unit-test runs
clean when conftest mocks torch.
"""

import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock

import pytest
import torch

_ORIG_DIFFLET_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"

# tests/conftest.py mocks torch for the default logic-only unit run; this is a
# real-torch (CPU) construction test, so skip it when torch is the MagicMock stand-in.
pytestmark = pytest.mark.skipif(
    isinstance(torch, MagicMock),
    reason="requires real torch (unit-test conftest mocks torch)",
)

if not isinstance(torch, MagicMock):
    # Pre-mock torch_xla before any difflet/diffusers imports so that the
    # Neuron-bundled diffusers (which imports torch_xla at module level) doesn't
    # try to initialise XLA hardware on a box without Neuron drivers.  This block
    # is safe: difflet.models.hunyuan_video.modeling_hunyuan_video is backend-neutral
    # and never calls torch_xla itself — torch_xla only leaks in via
    # diffusers.models.embeddings → diffusers.models.attention_processor.
    # Putting the stub in sys.modules before diffusers is first imported prevents
    # the real __init__ from running.
    if "torch_xla" not in sys.modules:
        def _mock_pkg(name: str) -> types.ModuleType:
            m = types.ModuleType(name)
            m.__path__ = []  # type: ignore[attr-defined]
            m.__package__ = name
            m.__spec__ = importlib.util.spec_from_loader(name, loader=None, origin="mock")
            return m

        _xla = _mock_pkg("torch_xla")
        sys.modules["torch_xla"] = _xla
        for _sub in [
            "core",
            "core.xla_model",
            "experimental",
            "experimental.custom_kernel",
            "runtime",
        ]:
            _full = f"torch_xla.{_sub}"
            _m = _mock_pkg(_full)
            sys.modules[_full] = _m
            _parent = _full.rsplit(".", 1)[0]
            setattr(sys.modules[_parent], _sub.split(".")[-1], _m)

        # Populate attributes that diffusers imports by name
        sys.modules["torch_xla.experimental.custom_kernel"].flash_attention = MagicMock()  # type: ignore[attr-defined]
        sys.modules["torch_xla.runtime"].is_spmd = MagicMock(return_value=False)  # type: ignore[attr-defined]

    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoAttention,
        HunyuanVideoTransformerBlock,
    )

    # Suite-safety: drop the import-time stubs we inserted (origin="mock") so they
    # do not shadow the real torch_xla / difflet.backends.trainium packages when
    # later test modules are collected in the same pytest session.
    for _stub_name in [
        _n
        for _n, _mod in list(sys.modules.items())
        if getattr(getattr(_mod, "__spec__", None), "origin", None) == "mock"
    ]:
        del sys.modules[_stub_name]

    # Suite-safety: restore DIFFLET_BACKEND so we do not force later tests onto the
    # cpu backend (construction above only needs it at import time).
    if _ORIG_DIFFLET_BACKEND is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = _ORIG_DIFFLET_BACKEND
else:
    # In the conftest mocked run the tests are skipped; define stubs so the
    # function bodies don't raise NameError at collection time.
    HunyuanVideoAttention = HunyuanVideoTransformerBlock = None  # type: ignore[assignment,misc]


def test_hunyuan_attention_stores_cp_mode_default():
    attn = HunyuanVideoAttention(
        hidden_size=128, num_attention_heads=4, attention_head_dim=32
    )
    assert attn.cp_mode == "gather_kv"


def test_hunyuan_block_threads_cp_mode_to_attention():
    block = HunyuanVideoTransformerBlock(
        num_attention_heads=4, attention_head_dim=32, mlp_ratio=4.0, cp_mode="ring"
    )
    assert block.attn.cp_mode == "ring"
