"""The mxfp8 format through the stack on CPU: spec, the CPU reference layer, the quantized
checkpoint (row-parallel weights per rank K range), the neuron FP8 linears in mx mode, the swap
and the kernel-layout preparation."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl import fp8_linear as Q
from difflet.backends.neuron.ops_impl import parallel_mesh as pm
from difflet.backends.tpu.core.weights import shard_dim
from difflet.pipeline.parallel_mesh import MeshSpec
from difflet.quant import mx
from difflet.quant.fake_linear import FakeQuantMXLinear, fake_quantize_model_
from difflet.quant.fp8 import FP8_DTYPE
from difflet.quant.spec import QuantSpec, quant_cache_fields
from difflet.quant.targets import default_spec


@pytest.fixture(autouse=True)
def _fresh_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def test_default_mx_spec():
    spec = default_spec("qwen_image", format="mxfp8", tp_degree=4)
    assert (spec.weight_granularity, spec.activation) == ("block32", "mx")
    assert spec.is_mx and not spec.needs_calibration and spec.calibration_sha256 is None
    assert spec.k_shards("transformer_blocks.3.attn.to_out.0") == 4  # row-parallel even when not a target
    assert spec.k_shards("transformer.transformer_blocks.3.img_mlp.net.2") == 4
    assert spec.k_shards("transformer_blocks.3.attn.to_q") == 1
    assert spec.k_shards("transformer_blocks.3.img_mlp.net.0.proj") == 1
    assert default_spec("qwen_image", tp_degree=4).k_shards("transformer_blocks.3.attn.to_out.0") == 1
    assert QuantSpec.from_dict(spec.to_dict()) == spec
    fields = quant_cache_fields(spec)
    assert fields["quant"] == "mxfp8" and fields["quant_tp_degree"] == 4
    assert fields != quant_cache_fields(dataclasses.replace(spec, tp_degree=2))
    with pytest.raises(ValueError, match="quantizes weights per block32"):
        QuantSpec(targets=spec.targets, format="mxfp8")
    with pytest.raises(ValueError, match="quant format"):
        default_spec("qwen_image", format="int8")


def test_fake_mx_linear_is_the_block_law_in_fp32():
    torch.manual_seed(0)
    linear = nn.Linear(1536, 64)
    x = torch.randn(5, 1536).to(torch.bfloat16)
    layer = FakeQuantMXLinear.from_linear(linear, k_shards=2)
    expect = F.linear(mx.fake_quantize_mx_sharded(x, 2),
                      mx.dequantize_mx_sharded(*mx.quantize_mx_sharded(linear.weight.detach(), 2), 2),
                      linear.bias.float()).to(torch.bfloat16)
    assert torch.equal(layer(x), expect)
    assert layer.weight.dtype == FP8_DTYPE and layer.weight_scale.shape == (64, 48)
    rel = ((layer(x).float() - linear(x.float())).norm() / linear(x.float()).norm()).item()
    assert 0.0 < rel < 0.1


def test_sharded_quantization_equals_per_shard_quantization():
    torch.manual_seed(0)
    w = torch.randn(8, 1536)
    q, s = mx.quantize_mx_sharded(w, 2)
    q0, s0 = mx.quantize_mx(w[:, :768])
    q1, s1 = mx.quantize_mx(w[:, 768:])
    assert torch.equal(q[:, :768], q0) and torch.equal(q[:, 768:], q1)
    assert torch.equal(s[:, :24], s0) and torch.equal(s[:, 24:], s1)
    assert not torch.equal(s, mx.quantize_mx(w)[1])  # another block layout than the whole


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.to_q = nn.Linear(256, 128)
            block.attn.to_out = nn.ModuleList([nn.Linear(128, 256)])
            self.transformer_blocks.append(block)


def test_swap_without_calibration_and_with_row_parallel_shards():
    spec = QuantSpec(targets=(r"transformer_blocks\.\d+\.attn\.(to_q|to_out\.0)",),
                     row_parallel=(r"transformer_blocks\.\d+\.attn\.to_out\.0",), format="mxfp8",
                     weight_granularity="block32", activation="mx", tp_degree=1)
    model = _Toy()
    names = fake_quantize_model_(model, spec)
    assert len(names) == 4
    assert isinstance(model.transformer_blocks[0].attn.to_q, FakeQuantMXLinear)
    assert model.transformer_blocks[0].attn.to_out[0].k_shards == 1
    with pytest.raises(ValueError, match="needs a calibration"):
        fake_quantize_model_(_Toy(), default_spec("qwen_image"))


def test_neuron_mx_linears_hold_block_scales_sharded_with_the_weight():
    pm.init_parallel_mesh(MeshSpec(tp=4))
    col = Q.FP8ColumnParallelLinear(256, 512, compute="nki", mx=True)
    row = Q.FP8RowParallelLinear(512, 256, compute="nki", mx=True)
    assert col.weight.shape == (128, 256) and col.weight_scale.shape == (128, 8)
    assert col.weight_scale.dtype == torch.uint8 and col.input_scale is None
    assert shard_dim(col, "weight_scale") == 0
    assert row.weight.shape == (256, 128) and row.weight_scale.shape == (256, 4)
    assert shard_dim(row, "weight_scale") == 1
    assert "input_scale" not in col.state_dict()


def _fill_mx(layer, weight):
    q, s = mx.quantize_mx(weight)
    with torch.no_grad():
        layer.weight.copy_(q)
        layer.weight_scale.copy_(s)


def test_neuron_mx_dequant_matches_the_cpu_reference_exactly():
    torch.manual_seed(0)
    weight = torch.randn(64, 256) * 0.1
    layer = Q.FP8Linear(256, 64, bias=False, compute="dequant", mx=True)
    _fill_mx(layer, weight)
    x = torch.randn(3, 256)
    expect = F.linear(x, mx.dequantize_mx(*mx.quantize_mx(weight)))
    assert torch.equal(layer(x), expect)
    layer.prepare_kernel_()  # dequant mode keeps a bf16 copy and drops the fp8 weight
    assert layer.weight is None and layer.weight_bf16.dtype == torch.bfloat16
    assert torch.allclose(layer(x), expect, atol=2e-2, rtol=2e-2)


def test_prepare_mx_kernel_builds_the_swizzled_layout():
    torch.manual_seed(0)
    weight = torch.randn(64, 512) * 0.1
    layer = Q.FP8Linear(512, 64, bias=True, compute="nki", mx=True)
    _fill_mx(layer, weight)
    q, s = mx.quantize_mx(weight)
    data, scales = mx.kernel_layout(q, s)
    Q.prepare_fp8_kernels_(layer.__class__.__mro__[0] and nn.Sequential(layer), lnc=2)
    assert torch.equal(layer.weight_packed, data) and torch.equal(layer.weight_scales_hw, scales)
    assert layer.weight is None and layer.weight_scale is not None
    with pytest.raises(RuntimeError, match="mxfp8 kernel"):
        Q.FP8Linear(512, 64, compute="nki", mx=True)(torch.randn(2, 512))


def test_quantized_dir_mx(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    from difflet.backends._torch_megatron.checkpoint import build_weight_map
    from difflet.quant.checkpoint import check_quantized_dir, prepare_quantized_checkpoint, read_manifest

    torch.manual_seed(0)
    src = tmp_path / "transformer"
    src.mkdir()
    tensors = {
        "transformer_blocks.0.attn.to_q.weight": torch.randn(128, 256),
        "transformer_blocks.0.attn.to_q.bias": torch.randn(128),
        "transformer_blocks.0.attn.to_out.0.weight": torch.randn(256, 1536),
        "transformer_blocks.0.img_mod.1.weight": torch.randn(64, 256),
    }
    save_file(tensors, str(src / "model.safetensors"))
    (src / "config.json").write_text("{}")
    spec = QuantSpec(targets=(r"transformer_blocks\.\d+\.attn\.(to_q|to_out\.0)",),
                     row_parallel=(r"transformer_blocks\.\d+\.attn\.to_out\.0",), format="mxfp8",
                     weight_granularity="block32", activation="mx", tp_degree=2)
    out, spec2 = prepare_quantized_checkpoint(src, spec, None, create=True, cache_root=tmp_path / "cache")
    assert spec2 == spec and read_manifest(out)["calibration"] is None
    weight_map = build_weight_map(out)
    assert set(weight_map) == {*tensors, "transformer_blocks.0.attn.to_q.weight_scale",
                               "transformer_blocks.0.attn.to_out.0.weight_scale"}
    with safe_open(weight_map["transformer_blocks.0.attn.to_out.0.weight_scale"], framework="pt") as h:
        s = h.get_tensor("transformer_blocks.0.attn.to_out.0.weight_scale")
        q = h.get_tensor("transformer_blocks.0.attn.to_out.0.weight")
        mod = h.get_tensor("transformer_blocks.0.img_mod.1.weight")
    assert s.dtype == torch.uint8 and s.shape == (256, 48) and q.dtype == FP8_DTYPE
    expect_q, expect_s = mx.quantize_mx_sharded(tensors["transformer_blocks.0.attn.to_out.0.weight"], 2)
    assert torch.equal(q, expect_q) and torch.equal(s, expect_s)
    assert torch.equal(mod, tensors["transformer_blocks.0.img_mod.1.weight"])
    check_quantized_dir(out, src, spec, None)
    with pytest.raises(ValueError, match="another quantization spec"):
        check_quantized_dir(out, src, dataclasses.replace(spec, tp_degree=4), None)
    with pytest.raises(ValueError, match="needs a calibration"):
        prepare_quantized_checkpoint(src, default_spec("qwen_image"), None, create=True,
                                     cache_root=tmp_path / "cache")


def test_fused_projections_concatenate_q_k_v():
    torch.manual_seed(0)
    attn = nn.Module()
    layers = []
    for name in ("to_q", "to_k", "to_v"):
        layer = Q.FP8Linear(256, 64, compute="nki", mx=True)
        _fill_mx(layer, torch.randn(64, 256) * 0.1)
        with torch.no_grad():
            layer.bias.normal_()
        setattr(attn, name, layer)
        layers.append(layer)
    attn.to_out = nn.ModuleList([Q.FP8Linear(256, 256, compute="nki", mx=True)])  # K: 128-multiple
    _fill_mx(attn.to_out[0], torch.randn(256, 256) * 0.1)
    expect_w = torch.cat([l.weight for l in layers], 0)
    expect_s = torch.cat([l.weight_scale for l in layers], 0)
    expect_bias = torch.cat([l.bias for l in layers], 0)
    data, scales = mx.kernel_layout(expect_w, expect_s)
    root = nn.Module()
    root.attn = attn
    assert Q.fuse_projections_(root) == ["attn.fused_qkv"]
    fused = root.attn.fused_qkv
    assert isinstance(fused, Q.FusedFP8Projections) and fused.splits == [64, 64, 64]
    assert torch.equal(fused.weight_packed, data) and torch.equal(fused.weight_scales_hw, scales)
    assert torch.equal(fused.bias, expect_bias)
    assert all(l.weight is None for l in layers) and attn.to_out[0].weight is not None
    assert Q.fuse_projections_(root) == []  # idempotent: the weights are gone
    # prepare_fp8_kernels_ fuses first, then prepares what is left.
    assert Q.prepare_fp8_kernels_(root, lnc=2, fuse=True) == 4
    assert attn.to_out[0].weight_packed is not None


def test_gathered_or_partial_groups_do_not_fuse():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    attn = nn.Module()
    for name in ("to_q", "to_k", "to_v"):
        layer = Q.FP8ColumnParallelLinear(256, 64, gather_output=True, compute="nki", mx=True)
        _fill_mx(layer, torch.randn(32, 256))
        setattr(attn, name, layer)
    root = nn.Module()
    root.attn = attn
    assert Q.fuse_projections_(root) == []
    attn2 = nn.Module()
    attn2.to_q = Q.FP8Linear(256, 64, compute="nki", mx=True)
    attn2.to_k = nn.Linear(256, 64)
    root2 = nn.Module()
    root2.attn = attn2
    assert Q.fuse_projections_(root2) == []
