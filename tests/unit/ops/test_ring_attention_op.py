import math
from unittest.mock import MagicMock

import pytest
import torch


# tests/conftest.py mocks torch for the default logic-only unit run; this is a
# real-torch (CPU) numerical test, so skip it when torch is the MagicMock stand-in.
pytestmark = pytest.mark.skipif(
    isinstance(torch, MagicMock),
    reason="requires real torch (unit-test conftest mocks torch)",
)


def _set_cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


def test_ring_attention_cpu_matches_plain_attention_cp1(monkeypatch):
    _set_cpu_backend(monkeypatch)
    from difflet.ops import attention, ring_attention

    torch.manual_seed(0)
    b, h, s, d = 1, 2, 128, 64
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / math.sqrt(d)

    ref = attention(
        q.reshape(b * h, s, d), k.reshape(b * h, s, d), v.reshape(b * h, s, d),
        scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False,
    ).reshape(b, h, s, d)
    out = ring_attention(q, k, v, scale=scale, causal=False)

    assert out.shape == (b, h, s, d)
    assert torch.allclose(ref.float(), out.float(), atol=1e-4, rtol=1e-4)
