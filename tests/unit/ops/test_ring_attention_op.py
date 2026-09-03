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


def test_trainium_ring_attention_rejects_non_128_multiple_seqlen():
    # Wan 480x832x9 under cp=2 yields a 2340-token per-rank shard; the nkilib
    # ring kernel then dies deep in neuronx-cc with INTERNAL_ERROR NCC_INKI016
    # ("seqlen must be divisible by 128"). The trainium op must fail fast with
    # an actionable message instead (found on device, 2026-08-29 matrix run).
    pytest.importorskip("nkilib")
    from difflet.backends.trainium.ops_impl.attention import ring_attention

    q = torch.randn(1, 2, 2340, 64)
    with pytest.raises(ValueError, match="multiple of 128"):
        ring_attention(q, q, q, scale=0.125, causal=False)
