"""Unit test: cp_mode is threaded from WanAttention through WanTransformerBlock.

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
    # is safe: difflet.models.wan.modeling_wan is backend-neutral and never calls
    # torch_xla itself — torch_xla only leaks in via diffusers.models.embeddings
    # → diffusers.models.attention_processor.  Putting the stub in sys.modules
    # before diffusers is first imported prevents the real __init__ from running.
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

    from difflet.models.wan.modeling_wan import WanAttention, WanTransformerBlock
else:
    # In the conftest mocked run the tests are skipped; define stubs so the
    # function bodies don't raise NameError at collection time.
    WanAttention = WanTransformerBlock = None  # type: ignore[assignment,misc]


def test_wan_attention_stores_cp_mode_default():
    attn = WanAttention(dim=128, heads=4, head_dim=32)
    assert attn.cp_mode == "gather_kv"


def test_wan_attention_stores_cp_mode_ring():
    attn = WanAttention(
        dim=128, heads=4, head_dim=32, context_parallel_enabled=False, cp_mode="ring"
    )
    assert attn.cp_mode == "ring"


def test_wan_block_threads_cp_mode_to_attentions():
    block = WanTransformerBlock(dim=128, ffn_dim=256, num_heads=4, cp_mode="ring")
    assert block.attn1.cp_mode == "ring"
    assert block.attn2.cp_mode == "ring"
