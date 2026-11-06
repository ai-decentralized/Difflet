"""FakeQuantMXLinear: the CPU reference arithmetic, and the in-place swap."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.quant import mx
from difflet.quant.fake_linear import FakeQuantMXLinear, fake_quantize_model_
from difflet.quant.spec import QuantSpec


def test_forward_is_the_block_law_in_fp32():
    torch.manual_seed(0)
    linear = nn.Linear(256, 16)
    x = torch.randn(5, 256).to(torch.bfloat16)
    layer = FakeQuantMXLinear.from_linear(linear)
    w_q, w_s = mx.quantize_mx(linear.weight.detach())
    expect = F.linear(mx.fake_quantize_mx(x), mx.dequantize_mx(w_q, w_s),
                      linear.bias.float()).to(torch.bfloat16)
    assert torch.equal(layer(x), expect)
    assert layer.weight.dtype == mx.FP8_DTYPE and layer.weight_scale.shape == (16, 8)
    assert layer.weight_scale.dtype == torch.uint8
    # Close to, but not equal to, the unquantized layer.
    rel = ((layer(x).float() - linear(x.float())).norm() / linear(x.float()).norm()).item()
    assert 0.0 < rel < 0.1


def test_bias_free_and_input_shapes():
    torch.manual_seed(0)
    linear = nn.Linear(128, 8, bias=False)
    layer = FakeQuantMXLinear.from_linear(linear)
    assert layer.bias is None
    x = torch.randn(2, 3, 128)
    assert layer(x).shape == (2, 3, 8) and layer(x).dtype == torch.float32


def test_refuses_the_wrong_dtypes():
    with pytest.raises(TypeError, match="fp8 weight and uint8"):
        FakeQuantMXLinear(torch.zeros(4, 128), torch.ones(4, 4, dtype=torch.uint8), None)
    with pytest.raises(TypeError, match="fp8 weight and uint8"):
        FakeQuantMXLinear(torch.zeros(4, 128, dtype=mx.FP8_DTYPE), torch.ones(4, 4), None)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.to_q = nn.Linear(256, 128)
            block.attn.to_out = nn.ModuleList([nn.Linear(128, 256)])
            block.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(256, 768))
            self.transformer_blocks.append(block)


def test_swap_replaces_exactly_the_targets_with_their_k_shards():
    spec = QuantSpec(targets=(r"transformer_blocks\.\d+\.attn\.(to_q|to_out\.0)",),
                     row_parallel=(r"transformer_blocks\.\d+\.attn\.to_out\.0",), tp_degree=1)
    model = _Toy()
    names = fake_quantize_model_(model, spec)
    assert names == ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_out.0",
                     "transformer_blocks.1.attn.to_q", "transformer_blocks.1.attn.to_out.0"]
    assert isinstance(model.transformer_blocks[0].attn.to_q, FakeQuantMXLinear)
    assert isinstance(model.transformer_blocks[1].attn.to_out[0], FakeQuantMXLinear)
    assert isinstance(model.transformer_blocks[0].img_mod[1], nn.Linear)
    assert model.transformer_blocks[0].attn.to_out[0].k_shards == 1
    with pytest.raises(ValueError, match="match no nn.Linear"):
        fake_quantize_model_(_Toy(), QuantSpec(targets=(r"nothing",)))
