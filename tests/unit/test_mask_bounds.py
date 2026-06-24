"""Unit test: contiguous-key-pad attention_mask -> attention_cte bound_min/bound_max.

Pure CPU (no hardware): the helper decides, at trace time, whether an additive
attention_mask is a per-query CONTIGUOUS KV range (expressible losslessly as
attention_cte's bound_min/bound_max). Contiguous key-padding and packed/block-diagonal
masks -> bounds; arbitrary/sparse masks -> None (caller falls back to SDPA).
Bounds are returned per (batch*head) query with shape (B*H, S_q, 1) int32, matching the
kernel layout proven in tests/numerical/test_attention_cte_bound_mask.py.
"""

from unittest.mock import MagicMock

import pytest
import torch

from difflet.backends.trainium.ops_impl.mask_bounds import mask_to_contiguous_bounds

# tests/conftest.py mocks torch for the default logic-only unit run; this is a
# real-torch (CPU) numerical test, so skip it when torch is the MagicMock stand-in.
pytestmark = pytest.mark.skipif(
    isinstance(torch, MagicMock),
    reason="requires real torch (unit-test conftest mocks torch)",
)

NEG = float("-inf")


def _keypad_additive(B, Sq, Skv, valid):
    """[B,1,1,Skv] additive mask: keys [0,valid) attend, [valid,Skv) masked."""
    m = torch.zeros((B, 1, 1, Skv))
    m[..., valid:] = NEG
    return m


def test_contiguous_keypad_returns_bounds():
    B, H, Sq, Skv, valid = 2, 4, 8, 8, 5
    m = _keypad_additive(B, Sq, Skv, valid)
    out = mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq)
    assert out is not None
    bmin, bmax = out
    assert bmin.shape == (B * H, Sq, 1) and bmax.shape == (B * H, Sq, 1)
    assert bmin.dtype == torch.int32 and bmax.dtype == torch.int32
    assert torch.equal(bmin, torch.zeros((B * H, Sq, 1), dtype=torch.int32))
    assert torch.equal(bmax, torch.full((B * H, Sq, 1), valid, dtype=torch.int32))


def test_sparse_mask_returns_none():
    # alternating valid/invalid keys -> non-contiguous -> None
    B, H, Sq, Skv = 1, 2, 4, 8
    m = torch.zeros((B, 1, Sq, Skv))
    m[..., 1::2] = NEG  # mask odd keys -> {0,2,4,6} not contiguous
    assert mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq) is None


def test_per_query_packed_bounds():
    # two packed sequences: queries 0-1 attend keys [0,4), queries 2-3 attend [4,8)
    B, H, Sq, Skv = 1, 1, 4, 8
    m = torch.full((B, 1, Sq, Skv), NEG)
    m[0, 0, 0:2, 0:4] = 0.0
    m[0, 0, 2:4, 4:8] = 0.0
    out = mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq)
    assert out is not None
    bmin, bmax = out
    assert torch.equal(bmin[:, :, 0], torch.tensor([[0, 0, 4, 4]], dtype=torch.int32))
    assert torch.equal(bmax[:, :, 0], torch.tensor([[4, 4, 8, 8]], dtype=torch.int32))


def test_all_valid_returns_full_range():
    # no masking -> [0, Skv) for every query (harmless; equals unmasked attention)
    B, H, Sq, Skv = 1, 2, 4, 6
    m = torch.zeros((B, 1, Sq, Skv))
    out = mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq)
    assert out is not None
    bmin, bmax = out
    assert torch.equal(bmin, torch.zeros((B * H, Sq, 1), dtype=torch.int32))
    assert torch.equal(bmax, torch.full((B * H, Sq, 1), Skv, dtype=torch.int32))


def test_bool_mask_true_is_attend():
    # boolean mask convention: True = attend (diffusers key_padding style)
    B, H, Sq, Skv, valid = 1, 1, 4, 6, 3
    m = torch.zeros((B, 1, 1, Skv), dtype=torch.bool)
    m[..., :valid] = True
    out = mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq)
    assert out is not None
    bmin, bmax = out
    assert torch.equal(bmax, torch.full((B * H, Sq, 1), valid, dtype=torch.int32))


def test_int_validity_mask_attends_nonzero():
    # integer validity mask: 1 = attend, 0 = pad (contiguous [0, valid))
    B, H, Sq, Skv, valid = 1, 2, 4, 8, 5
    m = torch.zeros((B, 1, 1, Skv), dtype=torch.int64)
    m[..., :valid] = 1
    out = mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq)
    assert out is not None, "integer 1/0 validity mask must be read as validity, not misread"
    _, bmax = out
    assert torch.equal(bmax, torch.full((B * H, Sq, 1), valid, dtype=torch.int32))


def test_float_validity_mask_returns_none():
    # AMBIGUOUS: a float {0,1} mask looks like validity, NOT a 0/-inf additive mask.
    # Rather than misread it, bail to None (-> SDPA). Positive values disqualify it.
    B, H, Sq, Skv = 1, 2, 4, 8
    m = torch.zeros((B, 1, 1, Skv), dtype=torch.float32)
    m[..., :5] = 1.0
    assert mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq) is None


def test_soft_bias_float_returns_none():
    # mildly-negative additive bias (not a hard -inf mask): cannot be expressed as a
    # hard [lo,hi) bound -> None (do not silently treat -2.5 as "attend").
    B, H, Sq, Skv = 1, 2, 4, 8
    m = torch.zeros((B, 1, 1, Skv), dtype=torch.float32)
    m[..., 5:] = -2.5
    assert mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq) is None


def test_bf16_minus10000_sentinel_resolves():
    # diffusers/HF -10000 additive sentinel rounds to ~-9984 in bf16; it must still be
    # read as a HARD mask (the LTX-2 / HV convention), not a rejected soft bias.
    B, H, Sq, Skv, valid = 1, 4, 8, 16, 9
    v = torch.zeros((B, 1, 1, Skv))
    v[..., :valid] = 1.0
    m = ((1.0 - v) * -10000.0).to(torch.bfloat16)
    out = mask_to_contiguous_bounds(m, num_heads=H, seq_q=Sq)
    assert out is not None, "bf16 -10000 sentinel must resolve as a hard mask"
    _, bmax = out
    assert torch.equal(bmax, torch.full((B * H, Sq, 1), valid, dtype=torch.int32))
