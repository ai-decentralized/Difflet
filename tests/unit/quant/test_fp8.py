"""FP8 e4m3 absmax math: scales, rounding bounds, activations, reference linear."""

from __future__ import annotations

import pytest
import torch

from difflet.quant import fp8


def test_fp8_range_is_the_trainium_e4m3_range():
    """Pins the all-NaN weight-only probe of 2026-10-01 on trn2: e4m3fn encodings
    above 240 decode as inf/NaN under the e4m3fn-as-e4m3 compiler flag, so the
    absmax law must saturate at 240, never at torch's 448."""
    assert fp8.FP8_MAX == 240.0
    assert fp8.FP8_MAX < torch.finfo(torch.float8_e4m3fn).max
    w = torch.randn(32, 32) * 5.0
    for granularity in ("tensor", "channel"):
        q, _ = fp8.quantize_weight(w, granularity)
        assert q.float().abs().max() <= 240.0
    q, _ = fp8.quantize_activation(torch.randn(2, 9, 16) * 50.0)
    assert q.float().abs().max() <= 240.0


def test_weight_scale_shapes_and_absmax_law():
    w = torch.randn(6, 8) * 3.0
    per_tensor = fp8.weight_scale(w, "tensor")
    per_channel = fp8.weight_scale(w, "channel")
    assert per_tensor.shape == (1,) and per_tensor.dtype == torch.float32
    assert per_channel.shape == (6, 1) and per_channel.dtype == torch.float32
    assert torch.allclose(per_tensor, w.abs().amax().reshape(1) / fp8.FP8_MAX)
    assert torch.allclose(per_channel, w.abs().amax(dim=1, keepdim=True) / fp8.FP8_MAX)


def test_quantize_weight_rounds_within_e4m3_precision():
    torch.manual_seed(0)
    # Values spread over ~4 binades so no element falls into the fp8 subnormal range.
    w = torch.empty(64, 128).uniform_(0.25, 4.0) * torch.randint(0, 2, (64, 128)).mul(2).sub(1)
    for granularity in ("tensor", "channel"):
        q, scale = fp8.quantize_weight(w, granularity)
        assert q.dtype == torch.float8_e4m3fn and q.shape == w.shape
        back = fp8.dequantize(q, scale)
        rel = ((back - w).abs() / w.abs()).max().item()
        assert rel <= 2**-4 + 1e-6, rel  # 3 mantissa bits -> half-ulp relative error 2^-4
        # The absmax element maps exactly onto +-FP8_MAX * scale.
        assert torch.isclose(back.abs().max(), w.abs().max(), rtol=1e-6, atol=0)


def test_zero_weight_uses_scale_floor_and_stays_finite():
    q, scale = fp8.quantize_weight(torch.zeros(4, 4), "tensor")
    assert scale.item() == pytest.approx(fp8.FP8_MIN_SCALE)
    assert torch.isfinite(q.float()).all() and (q.float() == 0).all()
    _, per_channel = fp8.quantize_weight(torch.zeros(4, 4), "channel")
    assert torch.allclose(per_channel, torch.full((4, 1), fp8.FP8_MIN_SCALE))


def test_dynamic_activation_quantization_is_per_tensor_absmax():
    x = torch.randn(3, 5, 7)
    q, scale = fp8.quantize_activation(x)
    assert q.dtype == torch.float8_e4m3fn and scale.shape == (1,)
    assert torch.isclose(scale, x.abs().amax().reshape(1) / fp8.FP8_MAX)
    back = fp8.fake_quant_activation(x)
    assert back.dtype == x.dtype and back.shape == x.shape
    assert (back - x).abs().max() <= x.abs().max() * 2**-4 + 1e-6


def test_reference_linear_tracks_bf16_and_weight_only_is_more_accurate():
    torch.manual_seed(1)
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    weight = torch.randn(24, 32, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(24, dtype=torch.bfloat16)
    exact = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    q, scale = fp8.quantize_weight(weight, "tensor")

    w8a8 = fp8.fp8_linear_reference(x, q, scale, bias, activation="dynamic")
    w8 = fp8.fp8_linear_reference(x, q, scale, bias, activation="none")
    assert w8a8.dtype == torch.bfloat16 and w8a8.shape == (16, 24)

    cos = torch.nn.functional.cosine_similarity
    assert cos(w8a8.float().flatten(), exact.flatten(), dim=0) > 0.995
    err_w8a8 = (w8a8.float() - exact).norm()
    err_w8 = (w8.float() - exact).norm()
    assert err_w8 < err_w8a8  # quantizing activations too can only add error
    assert err_w8a8 > 0  # and quantization actually happened
