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
    """Lean law (2026-10-03): scale = absmax / 240 * (1 + 2^-7); the margin lets the
    device skip the clamp — bf16 rounding of x * (1/scale) can then never reach the
    next fp8 value above 240 (256, which is inf/NaN on Trainium)."""
    x = torch.randn(3, 5, 7)
    q, scale = fp8.quantize_activation(x)
    assert q.dtype == torch.float8_e4m3fn and scale.shape == (1,)
    assert torch.isclose(scale, x.abs().amax().reshape(1) / fp8.FP8_MAX * fp8.ACT_SCALE_MARGIN)
    assert fp8.ACT_SCALE_MARGIN == 1 + 2**-7
    back = fp8.fake_quant_activation(x)
    assert back.dtype == x.dtype and back.shape == x.shape
    assert (back - x).abs().max() <= x.abs().max() * 2**-4 * fp8.ACT_SCALE_MARGIN + 1e-6


def test_activation_quantization_in_bf16_never_exceeds_240_without_a_clamp():
    """The device quantizes in the activation's dtype (bf16) with a multiply by the
    bf16 reciprocal scale and no clamp; the worst-case product must round to <= 240.
    Adversarial inputs: the absmax element, values just below it, large magnitudes."""
    torch.manual_seed(3)
    for scale_pow in (-10, 0, 7, 15):
        base = torch.randn(64, 256, dtype=torch.bfloat16) * (2.0**scale_pow)
        base[0, 0] = base.abs().max() * 1.0  # exact absmax, positive
        base[1, 1] = -base.abs().max()  # and negative
        q, scale = fp8.quantize_activation(base)
        assert q.float().abs().max() <= 240.0, (scale_pow, q.float().abs().max())
        assert torch.isfinite(q.float()).all()
        # Round trip stays within fp8 precision of the (margin-scaled) range.
        back = fp8.dequantize(q, scale, torch.float32)
        assert (back - base.float()).abs().max() <= base.float().abs().max() * 2**-4 * 1.02 + 1e-30


def test_reference_linear_tracks_bf16_and_quantizes_both_operands():
    """W8A8 only (weight-only was removed 2026-10-03): the reference quantizes the
    activation dynamically per call and the weight from its stored scale."""
    torch.manual_seed(1)
    x = torch.randn(16, 32, dtype=torch.bfloat16)
    weight = torch.randn(24, 32, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(24, dtype=torch.bfloat16)
    exact = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    q, scale = fp8.quantize_weight(weight, "tensor")

    w8a8 = fp8.fp8_linear_reference(x, q, scale, bias)
    assert w8a8.dtype == torch.bfloat16 and w8a8.shape == (16, 24)
    with pytest.raises(TypeError):
        fp8.fp8_linear_reference(x, q, scale, bias, activation="none")  # type: ignore[call-arg]

    cos = torch.nn.functional.cosine_similarity
    assert cos(w8a8.float().flatten(), exact.flatten(), dim=0) > 0.995
    assert (w8a8.float() - exact).norm() > 0  # quantization actually happened
    # Matches the explicit composition (the device path's contract): the activation is
    # quantized in its own dtype and its fp8 values are used exactly.
    x_q, x_scale = fp8.quantize_activation(x)
    composed = (x_q.float() * x_scale) @ fp8.dequantize(q, scale).t() + bias.float()
    assert torch.allclose(w8a8.float(), composed.to(torch.bfloat16).float())



def test_reference_linear_static_input_scale_matches_the_static_law():
    """Static mode of the CPU reference: quantize with the given constant input scale
    (multiply by its reciprocal, clamp to ±240, cast) instead of the dynamic absmax."""
    torch.manual_seed(7)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16) * 0.1
    q, scale = fp8.quantize_weight(weight, "tensor")
    input_scale = torch.tensor([0.02], dtype=torch.float32)
    out = fp8.fp8_linear_reference(x, q, scale, None, input_scale=input_scale)
    x_q = (x.float() * (1.0 / input_scale)).clamp(-fp8.FP8_MAX, fp8.FP8_MAX).to(torch.float8_e4m3fn)
    expected = ((x_q.float() * input_scale) @ fp8.dequantize(q, scale).t()).to(torch.bfloat16)
    assert torch.equal(out, expected)
    assert fp8.STATIC_ACT_MARGIN >= 1.0
