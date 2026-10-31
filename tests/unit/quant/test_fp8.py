"""The fp8 law: scales, clamping, round trips."""

from __future__ import annotations

import pytest
import torch

from difflet.quant import fp8
from difflet.quant.fp8 import FP8_DTYPE, FP8_MIN_SCALE


def test_weight_scale_is_amax_over_max_as_fp32_vector():
    w = torch.tensor([[1.0, -3.0], [0.5, 2.0]], dtype=torch.bfloat16)
    s = fp8.weight_scale(w)
    assert s.dtype == torch.float32 and s.shape == (1,)
    assert s.item() == pytest.approx(3.0 / 448.0)


def test_weight_scale_floor_protects_all_zero_weights():
    s = fp8.weight_scale(torch.zeros(4, 4))
    assert s.item() == pytest.approx(FP8_MIN_SCALE, rel=1e-6)  # fp32 holds the floor to 7 digits
    q, _ = fp8.quantize_weight(torch.zeros(4, 4))
    assert q.dtype == FP8_DTYPE and torch.equal(q.float(), torch.zeros(4, 4))


def test_quantize_clamps_before_the_cast():
    scale = torch.tensor([1.0])
    q = fp8.quantize(torch.tensor([1000.0, -1000.0, 0.0, 1.5]), scale)
    assert q.dtype == FP8_DTYPE
    assert torch.equal(q.float(), torch.tensor([448.0, -448.0, 0.0, 1.5]))
    assert not torch.isnan(q.float()).any()


def test_quantize_honours_fp8_max_240_the_trn2_legacy_range():
    q = fp8.quantize(torch.tensor([400.0]), torch.tensor([1.0]), fp8_max=240.0)
    assert q.float().item() == 240.0


def test_round_trip_error_is_fp8_rounding_only():
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    q, s = fp8.quantize_weight(w)
    back = fp8.dequantize(q, s)
    rel = ((back - w).norm() / w.norm()).item()
    # e4m3 has 3 mantissa bits: ~6% relative error per element, a few % in norm.
    assert 0.01 < rel < 0.08
    assert torch.equal(fp8.fake_quantize(w, s), back)


def test_activation_scale_uses_margin_and_floor():
    assert fp8.activation_scale(448.0, margin=1.25).item() == pytest.approx(1.25)
    assert fp8.activation_scale(0.0, margin=1.25).item() == pytest.approx(FP8_MIN_SCALE, rel=1e-6)


def test_quantize_takes_bf16_inputs_and_keeps_device():
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    q = fp8.quantize(x, fp8.weight_scale(x))
    assert q.dtype == FP8_DTYPE and q.device == x.device
