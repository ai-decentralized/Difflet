"""FakeQuantLinear: the CPU reference arithmetic, and the in-place swap."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.quant import fp8
from difflet.quant.calibration import read_calibration, write_calibration
from difflet.quant.fake_linear import FakeQuantLinear, fake_quantize_model_
from difflet.quant.spec import QuantSpec


def _calibration(tmp_path, amax):
    return read_calibration(write_calibration(
        tmp_path / "cal.json", amax=amax, model="toy", checkpoint={}, prompts=[], steps=1,
        shape={}))


def test_fake_quant_linear_matches_the_law_by_hand():
    torch.manual_seed(0)
    linear = nn.Linear(16, 8).to(torch.bfloat16)
    x = torch.randn(4, 16, dtype=torch.bfloat16) * 2
    input_scale = fp8.activation_scale(x.abs().amax().item(), margin=1.25)
    layer = FakeQuantLinear.from_linear(linear, input_scale, fp8_max=448.0)
    assert layer.weight.dtype == fp8.FP8_DTYPE
    out = layer(x)
    assert out.dtype == torch.bfloat16 and out.shape == (4, 8)

    w_q, w_s = fp8.quantize_weight(linear.weight.detach())
    expect = F.linear(fp8.fake_quantize(x, input_scale), fp8.dequantize(w_q, w_s),
                      linear.bias.float()).to(torch.bfloat16)
    assert torch.equal(out, expect)
    # Close to the bf16 layer, but not the bf16 layer: the quantization is visible.
    plain = linear(x)
    rel = ((out.float() - plain.float()).norm() / plain.float().norm()).item()
    assert 0.0 < rel < 0.15


def test_fake_quant_linear_without_bias():
    linear = nn.Linear(8, 4, bias=False)
    layer = FakeQuantLinear.from_linear(linear, torch.tensor([0.01]), fp8_max=448.0)
    assert layer.bias is None
    assert layer(torch.zeros(2, 8)).abs().sum() == 0


def test_fake_quant_linear_refuses_non_fp8_weights():
    with pytest.raises(TypeError, match="float8"):
        FakeQuantLinear(torch.zeros(4, 4), torch.ones(1), torch.ones(1), None, fp8_max=448.0)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.to_q = nn.Linear(8, 8)
            block.attn.to_out = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.0))
            block.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(8, 48))
            self.transformer_blocks.append(block)
        self.proj_out = nn.Linear(8, 4)


def test_swap_replaces_exactly_the_targets(tmp_path):
    spec = QuantSpec(targets=(r"transformer_blocks\.\d+\.attn\.(to_q|to_out\.0)",))
    names = ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_out.0",
             "transformer_blocks.1.attn.to_q", "transformer_blocks.1.attn.to_out.0"]
    cal = _calibration(tmp_path, {n: 4.0 for n in names})
    model = _Toy()
    replaced = fake_quantize_model_(model, spec, cal)
    assert replaced == names
    assert isinstance(model.transformer_blocks[0].attn.to_q, FakeQuantLinear)
    assert isinstance(model.transformer_blocks[1].attn.to_out[0], FakeQuantLinear)
    assert isinstance(model.transformer_blocks[0].img_mod[1], nn.Linear)
    assert isinstance(model.proj_out, nn.Linear)
    assert model.transformer_blocks[0].attn.to_q.input_scale.item() == pytest.approx(
        4.0 * 1.25 / 448.0)


def test_swap_refuses_an_incomplete_calibration(tmp_path):
    spec = QuantSpec(targets=(r"transformer_blocks\.\d+\.attn\.to_q",))
    cal = _calibration(tmp_path, {"transformer_blocks.0.attn.to_q": 1.0})
    with pytest.raises(ValueError, match="calibration lacks 1 targets"):
        fake_quantize_model_(_Toy(), spec, cal)
