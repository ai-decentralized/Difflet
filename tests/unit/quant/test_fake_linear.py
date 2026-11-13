"""FakeQuantMXLinear: the CPU reference arithmetic, and the in-place swap."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.quant import mx
from difflet.quant.fake_linear import (
    FakeQuantMXLinear,
    SplitLinear,
    fake_quantize_model_,
    split_linear_columns,
)
from difflet.quant.spec import QuantSpec
from difflet.quant.targets import SINGLE_BLOCK_SPLIT_LINEARS


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


SPLIT = SINGLE_BLOCK_SPLIT_LINEARS
HALVES = (r"single_transformer_blocks\.\d+\.proj_out_(attn|mlp)",)
P = "single_transformer_blocks.0"


class _SplitToy(nn.Module):
    """One single-stream block whose fused ``proj_out`` is ``[256, 768]``: halves of K 256 and
    512, which tp2 splits into multiples of 128."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        block = nn.Module()
        block.proj_mlp = nn.Linear(256, 512)
        block.proj_out = nn.Linear(768, 256)
        self.single_transformer_blocks = nn.ModuleList([block])


def test_a_split_linear_runs_as_its_two_column_ranges_each_quantized_on_its_own():
    model = _SplitToy()
    fused = model.single_transformer_blocks[0].proj_out
    attn_ref, mlp_ref = split_linear_columns(fused)
    spec = QuantSpec(targets=HALVES, row_parallel=HALVES, split_linears=SPLIT, tp_degree=2)
    assert fake_quantize_model_(model, spec) == [f"{P}.proj_out_attn", f"{P}.proj_out_mlp"]
    m = model.single_transformer_blocks[0].proj_out
    assert isinstance(m, SplitLinear)
    assert m.attn.k_shards == m.mlp.k_shards == 2  # the halves' own names are row-parallel
    assert m.mlp.bias is None and torch.equal(m.attn.bias, fused.bias)
    x = torch.randn(3, 768)
    expect = (FakeQuantMXLinear.from_linear(attn_ref, k_shards=2)(x[:, :256])
              + FakeQuantMXLinear.from_linear(mlp_ref, k_shards=2)(x[:, 256:]))
    assert torch.equal(m(x), expect)
    # only the MLP half targeted: the attention half stays the plain column range
    model = _SplitToy()
    spec = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_out_mlp",),
                     split_linears=SPLIT)
    assert fake_quantize_model_(model, spec) == [f"{P}.proj_out_mlp"]
    m = model.single_transformer_blocks[0].proj_out
    assert type(m.attn) is nn.Linear and isinstance(m.mlp, FakeQuantMXLinear)
    assert torch.equal(m.attn.weight, fused.weight[:, :256])
    # a halves target on the fused model without the split, and a dead pattern beside live ones
    with pytest.raises(ValueError, match="match no nn.Linear"):
        fake_quantize_model_(_SplitToy(), QuantSpec(targets=HALVES))
    with pytest.raises(ValueError, match=r"match no nn.Linear in the model: nothing$"):
        fake_quantize_model_(_SplitToy(),
                             QuantSpec(targets=(*HALVES, "nothing"), split_linears=SPLIT))
    with pytest.raises(ValueError, match="needs K > N"):
        split_linear_columns(nn.Linear(128, 256))


def test_split_linear_columns_are_clones_in_the_source_dtype():
    fused = nn.Linear(384, 128).to(torch.bfloat16)
    attn, mlp = split_linear_columns(fused)
    assert (attn.in_features, attn.out_features, mlp.in_features, mlp.out_features) == (
        128, 128, 256, 128)
    assert attn.weight.dtype == mlp.weight.dtype == torch.bfloat16 and mlp.bias is None
    assert torch.equal(attn.weight, fused.weight[:, :128])
    assert torch.equal(mlp.weight, fused.weight[:, 128:])
    assert torch.equal(attn.bias, fused.bias)
    assert attn.weight.data_ptr() != fused.weight.data_ptr()
    fused = fused.float()
    x = torch.randn(2, 384)
    torch.testing.assert_close(SplitLinear(*split_linear_columns(fused), split=128)(x), fused(x))
