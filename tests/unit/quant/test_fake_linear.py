"""CPU fake-quant linear and the module swap on a tiny Wan backbone."""

from __future__ import annotations

import importlib
import os

import pytest
import torch
import torch.nn as nn

from difflet.quant.fake_linear import FakeQuantLinear, quantize_module_
from difflet.quant.spec import QuantSpec

# Bind the CPU op backend into modeling_wan (same recipe as the Wan forward tests).
_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.wan.modeling_wan as wan  # noqa: E402

importlib.reload(wan)
if _ORIG_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _ORIG_BACKEND


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DIFFLET_BACKEND", None)
        else:
            os.environ["DIFFLET_BACKEND"] = prev


def test_fake_quant_linear_matches_reference_and_keeps_bias():
    torch.manual_seed(0)
    lin = nn.Linear(32, 24)
    x = torch.randn(5, 32)
    exact = lin(x)
    quant = FakeQuantLinear(lin, QuantSpec())
    assert quant.weight.dtype == torch.float8_e4m3fn
    assert quant.weight_scale.shape == (1,)
    assert quant.bias is not None and torch.equal(quant.bias, lin.bias)
    out = quant(x)
    assert out.shape == exact.shape and out.dtype == exact.dtype
    cos = torch.nn.functional.cosine_similarity(out.flatten(), exact.flatten(), dim=0)
    assert cos > 0.995
    assert (out - exact).abs().max() > 0
    assert "fp8-tensor" in repr(quant)

    no_bias = FakeQuantLinear(nn.Linear(4, 4, bias=False), QuantSpec(weight_granularity="channel"))
    assert no_bias.bias is None and no_bias.weight_scale.shape == (4, 1)


def _tiny_wan():
    cfg = wan.WanTransformerConfig(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=24,
        freq_dim=32,
        ffn_dim=48,
        num_layers=2,
    )
    torch.manual_seed(0)
    return wan.WanTransformer3DModel(cfg).eval()


def test_quantize_module_swaps_exactly_the_target_linears_of_a_wan_backbone():
    model = _tiny_wan()
    report = quantize_module_(model, QuantSpec())
    expected = {
        f"blocks.{b}.{name}"
        for b in range(2)
        for name in (
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
            "ffn.net_in", "ffn.net_out",
        )
    }
    assert set(report["quantized"]) == expected
    assert report["num_quantized"] == 20
    for name in expected:
        assert isinstance(model.get_submodule(name), FakeQuantLinear)
    # Embedders, modulation and the output head stay plain (bf16-equivalent) layers.
    assert isinstance(model.proj_out, nn.Linear) and not isinstance(model.proj_out, FakeQuantLinear)
    assert isinstance(model.condition_embedder.time_embedder.linear_1, nn.Linear)
    assert not isinstance(model.condition_embedder.time_embedder.linear_1, FakeQuantLinear)
    # Idempotent: a second pass finds nothing left to swap.
    assert quantize_module_(model, QuantSpec())["num_quantized"] == 0


def test_quantized_wan_forward_stays_close_to_the_bf16_reference():
    reference = _tiny_wan()
    quantized = _tiny_wan()  # same seed -> same weights
    quantize_module_(quantized, QuantSpec())
    torch.manual_seed(1)
    latents = torch.randn(1, 4, 2, 8, 8)
    timestep = torch.tensor([500.0])
    text = torch.randn(1, 6, 24)
    with torch.no_grad():
        ref = reference(latents, timestep, text)
        out = quantized(latents, timestep, text)
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert out.shape == ref.shape and torch.isfinite(out).all()
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos > 0.95
    assert not torch.equal(out, ref)


def test_fake_quant_linear_uses_the_calibrated_static_input_scale(tmp_path):
    # With spec.calibration the CPU emulation quantizes the activation with the layer's
    # calibrated input_scale (absmax * STATIC_ACT_MARGIN / 240, clamped to +-240), exactly
    # like the device's static path, instead of the dynamic per-call absmax.
    import json

    from difflet.quant.fp8 import FP8_MAX, STATIC_ACT_MARGIN, fp8_linear_reference

    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    x = torch.randn(3, 16)
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"layers": {"blocks.0.ffn.net_in": {"amax": 2.0}}}))
    spec = QuantSpec(calibration=str(calib))
    # the HF name of the target maps onto the Difflet calibration name via calibrated_amax
    quant = FakeQuantLinear(lin, spec, name="blocks.0.ffn.net.0.proj")
    expected_scale = torch.tensor(2.0 * STATIC_ACT_MARGIN / FP8_MAX)
    assert torch.allclose(quant.input_scale, expected_scale)
    out = quant(x)
    ref = fp8_linear_reference(x, quant.weight, quant.weight_scale, quant.bias, input_scale=expected_scale)
    assert torch.equal(out, ref)
    dyn = fp8_linear_reference(x, quant.weight, quant.weight_scale, quant.bias)
    assert not torch.equal(out, dyn)
    assert "static" in repr(quant)

    # quantize_module_ hands every swapped layer its own qualified name
    model = nn.Sequential()
    model.add_module("blocks", nn.ModuleList([nn.ModuleDict({"ffn": nn.ModuleDict({"net_in": nn.Linear(16, 8)})})]))
    quantize_module_(model, spec)
    swapped = model.blocks[0]["ffn"]["net_in"]
    assert isinstance(swapped, FakeQuantLinear) and torch.allclose(swapped.input_scale, expected_scale)
