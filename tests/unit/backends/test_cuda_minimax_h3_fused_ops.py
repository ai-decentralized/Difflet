"""MiniMax-H3's fused Triton kernels against their plain FP32 formulas (CUDA only)."""

from __future__ import annotations

import pytest
import torch

from tests.unit.backends import minimax_h3_reference_ops as reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

BF16 = torch.bfloat16


def _ulp_close(got, want, *, share=1e-3):
    """Equal up to one-BF16-rounding differences in a small share of values."""
    diff = (got.float() - want.float()).abs()
    tolerance = want.float().abs().clamp_min(1e-2) * 2**-7
    assert bool((diff <= tolerance).all())
    assert (diff > 0).float().mean().item() <= share


@pytest.fixture
def ops():
    import difflet.backends.cuda.minimax_h3.fused_ops  # noqa: F401

    torch.manual_seed(0)
    return torch.ops.difflet_h3


def test_rms_norm_modulate_matches_the_formula(ops):
    rows, n, entries = 257, 5376, 9
    x = torch.randn(1, rows, n, device="cuda", dtype=BF16)
    weight = (1 + 0.1 * torch.randn(n, device="cuda")).to(BF16)
    table = (0.1 * torch.randn(entries, 6 * n, device="cuda")).to(BF16)
    shift, scale = table[:, :n], table[:, n : 2 * n]
    index = torch.randint(0, entries, (rows,), device="cuda")
    want = reference.rms_norm_modulate(x, weight, 1e-5, shift, scale, index)
    _ulp_close(ops.rms_norm_modulate(x, weight, 1e-5, shift, scale, index), want)


def test_gated_residual_matches_the_formula_exactly(ops):
    rows, n, entries = 129, 5376, 9
    residual, update = (torch.randn(1, rows, n, device="cuda", dtype=BF16) for _ in range(2))
    gate = (0.1 * torch.randn(entries, 6 * n, device="cuda")).to(BF16)[:, 2 * n : 3 * n]
    index = torch.randint(0, entries, (rows,), device="cuda")
    want = reference.gated_residual(residual, gate, index, update)
    assert torch.equal(ops.gated_residual(residual, gate, index, update), want)


def test_head_norm_rotary_matches_norm_then_rotation(ops):
    rows, heads, dim, rotary = 97, 56, 128, 96
    fused_qkv = torch.randn(1, rows, 3 * heads * dim, device="cuda", dtype=BF16)
    query = fused_qkv[..., : heads * dim].unflatten(-1, (heads, dim))  # strided rows
    weight = (1 + 0.1 * torch.randn(dim, device="cuda")).to(BF16)
    angles = torch.rand(rows, rotary // 2, device="cuda") * 50
    angles = torch.cat([angles, angles], -1)
    cos, sin = angles.cos(), angles.sin()
    got = ops.head_norm_rotary(query, weight, 1e-5, cos, sin)
    assert got.is_contiguous() and got.shape == query.shape
    _ulp_close(got, reference.head_norm_rotary(query, weight, 1e-5, cos, sin))


def test_swiglu_rounds_once(ops):
    x = torch.randn(1, 65, 2 * 14336, device="cuda", dtype=BF16)
    _ulp_close(ops.swiglu(x), reference.swiglu(x))
