"""The neuron backend's MXFP8 linears (``ops_impl/fp8_linear.py``): construction, sharding,
the dequant compute path against the CPU reference, the in-place swap, meta materialisation,
the checkpoint load of a quantized directory, the weight store, the kernel-layout preparation
and TP4 on gloo.

CPU only; the ``nki`` compute path needs the device and is covered by the device checks.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends._torch_megatron.weights import is_float8, materialize_meta_
from difflet.backends.neuron.core.application_base import rank_weight_bytes
from difflet.backends.neuron.ops_impl import fp8_linear as Q
from difflet.backends.neuron.ops_impl import linear as L
from difflet.backends.neuron.ops_impl import parallel_mesh as pm
from difflet.backends.tpu.core.weights import load_sharded_state_dict, shard_dim
from difflet.pipeline.parallel_mesh import MeshSpec
from difflet.quant import mx
from difflet.quant.fake_linear import FakeQuantMXLinear
from difflet.quant.spec import QuantSpec
from tests.unit.backends._neuron_fp8_workers import (
    DIM,
    HIDDEN,
    ToyFP8MLP,
    fp8_toy_full_weights,
    fp8_toy_reference,
    fp8_tp_mlp_worker,
)
from tests.unit.backends._neuron_gloo import run_ranks


@pytest.fixture(autouse=True)
def _fresh_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def _spec_only(tp: int) -> None:
    pm.init_parallel_mesh(MeshSpec(tp=tp))


def _fill(layer, weight: torch.Tensor) -> None:
    q, s = mx.quantize_mx(weight)
    with torch.no_grad():
        layer.weight.copy_(q)
        layer.weight_scale.copy_(s)


# ------------------------------------------------------------------- construction


def test_fp8_layers_hold_an_fp8_weight_and_uint8_block_scales():
    for layer in (Q.FP8Linear(128, 16, compute="dequant"),
                  Q.FP8ColumnParallelLinear(128, 16, compute="dequant"),
                  Q.FP8RowParallelLinear(128, 16, compute="dequant")):
        assert layer.weight.dtype == torch.float8_e4m3fn and not layer.weight.requires_grad
        assert layer.weight_scale.dtype == torch.uint8
        assert layer.weight_scale.shape == (layer.weight.shape[0], layer.weight.shape[1] // 32)
        assert layer.bias.dtype == torch.float32
        assert "weight_scale" in layer.state_dict() and "weight_packed" not in layer.state_dict()


def test_dtype_names_the_bias_not_the_weight():
    layer = Q.FP8Linear(128, 16, dtype=torch.bfloat16, compute="dequant")
    assert layer.bias.dtype == torch.bfloat16 and layer.weight.dtype == torch.float8_e4m3fn


def test_unknown_compute_and_odd_k_are_refused():
    with pytest.raises(ValueError, match="fp8 compute"):
        Q.FP8Linear(128, 16, compute="torchao")
    with pytest.raises(ValueError, match="multiple of 128"):
        Q.FP8Linear(40, 16, compute="dequant")
    with pytest.raises(ValueError, match="multiple of 128"):
        Q.FP8Linear(64, 8, compute="dequant")  # a multiple of 32 but not of 128


# --------------------------------------------------------------- dequant at tp == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dequant_matches_the_weight_only_reference_exactly(dtype):
    torch.manual_seed(0)
    weight = torch.randn(16, 128) * 0.1
    layer = Q.FP8Linear(128, 16, dtype=dtype, compute="dequant")
    _fill(layer, weight)
    x = torch.randn(3, 128).to(dtype)
    dequant = mx.dequantize_mx(layer.weight, layer.weight_scale).to(dtype)
    expect = F.linear(x, dequant, layer.bias)
    assert torch.equal(layer(x), expect)
    row = Q.FP8RowParallelLinear(128, 16, dtype=dtype, compute="dequant")
    _fill(row, weight)
    expect = F.linear(x, dequant) + row.bias
    assert torch.equal(row(x), expect)
    # After prepare_kernel_ the dequantized bf16 copy stands in and the fp8 weight is gone.
    layer.prepare_kernel_()
    assert layer.weight is None and layer.weight_bf16.dtype == torch.bfloat16
    assert torch.allclose(layer(x), F.linear(x, dequant, layer.bias), atol=2e-2, rtol=2e-2)


def test_dequant_is_the_quantized_weight_not_the_bf16_one():
    torch.manual_seed(0)
    plain = nn.Linear(128, 32)
    layer = Q.FP8Linear(128, 32, compute="dequant")
    _fill(layer, plain.weight.detach())
    with torch.no_grad():
        layer.bias.copy_(plain.bias)
    x = torch.randn(4, 128)
    out, ref = layer(x), plain(x)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert 0.0 < rel < 0.1
    # And it is within the activation's block rounding of the W8A8 CPU reference.
    fake = FakeQuantMXLinear.from_linear(plain)
    rel_fake = ((out - fake(x)).norm() / fake(x).norm()).item()
    assert rel_fake < 0.1


def test_nki_compute_needs_the_packed_weight():
    layer = Q.FP8Linear(128, 16, compute="nki")
    with pytest.raises(RuntimeError, match="prepare_fp8_kernels_"):
        layer(torch.randn(2, 128))


# ----------------------------------------------------------- sharding (tp = 4)


def test_column_shards_the_output_and_its_block_scales():
    _spec_only(4)
    layer = Q.FP8ColumnParallelLinear(128, 16, compute="dequant")
    assert layer.weight.shape == (4, 128) and layer.bias.shape == (4,)
    assert layer.weight_scale.shape == (4, 4)
    assert layer._difflet_shard == {"weight": 0, "bias": 0, "weight_scale": 0}
    assert shard_dim(layer, "weight") == 0 and shard_dim(layer, "weight_scale") == 0


def test_row_shards_the_input_and_its_block_scales_and_replicates_the_bias():
    _spec_only(4)
    layer = Q.FP8RowParallelLinear(512, 16, compute="dequant")
    assert layer.weight.shape == (16, 128) and layer.bias.shape == (16,)
    assert layer.weight_scale.shape == (16, 4)
    assert layer._difflet_shard == {"weight": 1, "bias": None, "weight_scale": 1}


def test_a_row_layers_per_rank_k_is_checked_at_the_tp_it_is_built_at():
    """K=256 is whole in a column layer and has a layout; split over tp=4 it is 64 per rank and
    has none, and the refusal says so with the full K and the tp, not the per-rank K alone."""
    _spec_only(4)
    Q.FP8ColumnParallelLinear(256, 16, compute="dequant")
    with pytest.raises(ValueError, match=r"multiple of 128 .*got 64 \(256 split over tp=4\)"):
        Q.FP8RowParallelLinear(256, 16, compute="dequant")


def test_sharded_layers_load_their_slice_of_the_full_fp8_weights():
    _spec_only(4)
    model = ToyFP8MLP(DIM, HIDDEN, dtype=torch.float32)
    full = fp8_toy_full_weights(DIM, HIDDEN, tp=4)
    load_sharded_state_dict(model, full, tp_size=4, tp_rank=2)
    assert torch.equal(model.up.weight, full["up.weight"][256:384])
    assert torch.equal(model.up.weight_scale, full["up.weight_scale"][256:384])
    assert torch.equal(model.down.weight, full["down.weight"][:, 256:384])
    assert torch.equal(model.down.weight_scale, full["down.weight_scale"][:, 8:12])
    assert torch.equal(model.down.bias, full["down.bias"])


# --------------------------------------------------------------------- the swap


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = L.ColumnParallelLinear(128, 128, gather_output=False)
        self.attn.to_out = nn.ModuleList([L.RowParallelLinear(512, 128, input_is_parallel=True,
                                                               reduce_output=False)])
        self.img_mod = nn.Sequential(nn.SiLU(), L.ColumnParallelLinear(128, 768, gather_output=True))
        self.plain = nn.Linear(128, 128)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.transformer_blocks = nn.ModuleList([_Block(), _Block()])
        self.transformer.proj_out = nn.Linear(128, 4)


SPEC = QuantSpec(targets=(r"transformer_blocks\.\d+\.attn\.(to_q|to_out\.0)",
                          r"transformer_blocks\.\d+\.plain"), compute="dequant")


def test_quantize_module_replaces_exactly_the_targets_and_keeps_their_flags():
    _spec_only(4)
    with torch.device("meta"):
        model = _Model()
    names = Q.quantize_module_(model, SPEC)
    assert names == ["transformer.transformer_blocks.0.attn.to_q",
                     "transformer.transformer_blocks.0.attn.to_out.0",
                     "transformer.transformer_blocks.0.plain",
                     "transformer.transformer_blocks.1.attn.to_q",
                     "transformer.transformer_blocks.1.attn.to_out.0",
                     "transformer.transformer_blocks.1.plain"]
    block = model.transformer.transformer_blocks[0]
    assert isinstance(block.attn.to_q, Q.FP8ColumnParallelLinear)
    assert block.attn.to_q.gather_output is False and block.attn.to_q.weight.shape == (32, 128)
    assert isinstance(block.attn.to_out[0], Q.FP8RowParallelLinear)
    assert block.attn.to_out[0].input_is_parallel and block.attn.to_out[0].reduce_output is False
    assert isinstance(block.plain, Q.FP8Linear)
    assert isinstance(block.img_mod[1], L.ColumnParallelLinear)
    assert isinstance(model.transformer.proj_out, nn.Linear)
    assert all(m.weight.is_meta for m in (block.attn.to_q, block.plain))
    assert Q.fp8_linears(model) == names
    # Idempotent: already-quantized layers are not targets again.
    with pytest.raises(ValueError, match="match no linear"):
        Q.quantize_module_(model, SPEC)


def test_another_backends_parallel_linears_are_recognised_by_name():
    """A model module bound difflet.ops' linears from the CPU backend (a CPU-only session):
    plain nn.Linear subclasses without the TP attributes, still swapped by their kind."""
    from difflet.backends.cpu.ops_impl import linear as cpu_linear

    col = cpu_linear.ColumnParallelLinear(128, 64, gather_output=False)
    row = cpu_linear.RowParallelLinear(128, 64, input_is_parallel=True, skip_bias_add=True)
    twin_col = Q.fp8_twin(col, compute="dequant")
    twin_row = Q.fp8_twin(row, compute="dequant")
    assert isinstance(twin_col, Q.FP8ColumnParallelLinear) and twin_col.weight.shape == (64, 128)
    assert twin_col.gather_output is True  # the CPU class drops the flag: the tp=1 default
    assert isinstance(twin_row, Q.FP8RowParallelLinear) and twin_row.skip_bias_add is True
    with pytest.raises(TypeError, match="cannot quantize"):
        Q.fp8_twin(nn.Conv1d(4, 4, 1), compute="dequant")


def test_quantize_module_refuses_a_spec_matching_nothing():
    with pytest.raises(ValueError, match="match no linear"):
        Q.quantize_module_(_Model(), QuantSpec(targets=(r"nothing",)))


def _wan_ffn_toy() -> nn.Module:
    """``blocks.0.ffn.net_in``: the module name Wan's application gives the first FFN
    projection, whose checkpoint key is ``blocks.0.ffn.net.0.proj.weight``."""
    block = nn.Module()
    block.ffn = nn.Module()
    block.ffn.net_in = nn.Linear(128, 16)
    module = nn.Module()
    module.blocks = nn.ModuleList([block])
    return module


def test_quantize_module_matches_through_a_key_function():
    """A model whose module names differ from its checkpoint keys matches the spec through the
    application's ``checkpoint_key`` (the real Wan one, which renames parameter keys: the
    module name is mapped through its weight's key, as the writer derives its prefix); the
    returned names are still the module's. Once the key function is in force a pattern written
    against the module name matches nothing: the device would otherwise quantize a layer the
    checkpoint holds in bf16."""
    from difflet.backends.neuron.wan.transformer import checkpoint_key

    module = _wan_ffn_toy()
    spec = QuantSpec(targets=(r"blocks\.\d+\.ffn\.net\.0\.proj",), compute="dequant")
    assert Q.quantize_module_(module, spec, key_fn=checkpoint_key) == ["blocks.0.ffn.net_in"]
    assert isinstance(module.blocks[0].ffn.net_in, Q.FP8Linear)
    module = _wan_ffn_toy()
    by_module_name = QuantSpec(targets=(r"blocks\.\d+\.ffn\.net_in",), compute="dequant")
    with pytest.raises(ValueError, match=r"match no linear.*net_in"):
        Q.quantize_module_(module, by_module_name, key_fn=checkpoint_key)
    assert type(module.blocks[0].ffn.net_in) is nn.Linear


def test_quantize_module_names_every_dead_pattern():
    """One live pattern does not excuse a dead one: the refusal names the patterns that match
    no linear, and the module is left as it was."""
    module = nn.Module()
    module.a = nn.Linear(128, 16)
    module.b = nn.Linear(128, 16)
    spec = QuantSpec(targets=("a", "zzz", r"b\.weight"), compute="dequant")
    with pytest.raises(ValueError, match=r"match no linear.*\['zzz', 'b\\\\.weight'\]"):
        Q.quantize_module_(module, spec)
    assert type(module.a) is nn.Linear and type(module.b) is nn.Linear


def test_quantize_module_names_the_layer_whose_k_is_wrong():
    """The refusal carries the layer's name and the layer's own text (a plain linear has no
    tp to blame; a row layer's text says ``split over tp=``), and no twin is swapped in before
    every target has been built, so a caller that catches the error holds the module it gave."""
    module = nn.Module()
    module.ok = nn.Linear(128, 16)
    module.proj = nn.Linear(64, 16)
    spec = QuantSpec(targets=("ok", "proj"), compute="dequant")
    with pytest.raises(ValueError, match=r"proj: mxfp8 needs in_features a multiple of 128") as e:
        Q.quantize_module_(module, spec)
    assert "tp" not in str(e.value)
    assert type(module.ok) is nn.Linear and type(module.proj) is nn.Linear


# ------------------------------------------------ meta materialisation and bytes


def test_materialize_keeps_fp8_and_casts_the_rest():
    _spec_only(4)
    with torch.device("meta"):
        model = _Model()
    Q.quantize_module_(model, SPEC)
    block = model.transformer.transformer_blocks[0]
    # fp8 weights count one byte, the meta bf16 parameters two, real buffers their own.
    expect = sum(p.numel() * (1 if is_float8(p.dtype) else 2) for p in model.parameters())
    expect += sum(b.numel() * b.element_size() for b in model.buffers())
    assert rank_weight_bytes(model, torch.bfloat16) == expect
    materialize_meta_(model, device="cpu", dtype=torch.bfloat16)
    assert block.attn.to_q.weight.dtype == torch.float8_e4m3fn
    assert block.attn.to_q.bias.dtype == torch.bfloat16
    assert block.img_mod[1].weight.dtype == torch.bfloat16
    assert block.attn.to_q.weight_scale.dtype == torch.uint8
    assert not any(p.is_meta for p in model.parameters())


def test_load_sharded_checkpoint_fills_fp8_weights_from_a_quantized_dir(tmp_path):
    """A quantized directory (``difflet/quant/checkpoint.py`` output form) loaded through the
    native lazy loader into a meta-built fp8 module at tp=1."""
    from safetensors.torch import save_file

    from difflet.backends.neuron.core.checkpoint import build_on_meta, load_sharded_checkpoint

    full = fp8_toy_full_weights(DIM, HIDDEN)
    save_file(full, str(tmp_path / "model.safetensors"))
    module = build_on_meta(lambda: ToyFP8MLP(DIM, HIDDEN, dtype=torch.bfloat16))
    assert module.up.weight.is_meta and not module.up.weight_scale.is_meta
    report = load_sharded_checkpoint(module, tmp_path, device="cpu", dtype=torch.bfloat16,
                                     tp_size=1, tp_rank=0)
    assert report == {"missing": [], "unexpected": []}
    assert torch.equal(module.up.weight, full["up.weight"])
    assert torch.equal(module.down.weight_scale, full["down.weight_scale"])
    assert module.up.bias.dtype == torch.bfloat16
    x = torch.randn(2, DIM).to(torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(module(x), fp8_toy_reference(full, x))


def test_weight_store_round_trips_fp8_tensors(tmp_path, monkeypatch):
    from difflet.backends.neuron.core import weight_store
    from difflet.backends.neuron.core.checkpoint import build_on_meta, load_sharded_checkpoint
    from safetensors.torch import save_file

    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "1")
    full = fp8_toy_full_weights(DIM, HIDDEN)
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    save_file(full, str(model_dir / "model.safetensors"))
    root = tmp_path / "store"

    def load(**kw):
        module = build_on_meta(lambda: ToyFP8MLP(DIM, HIDDEN, dtype=torch.bfloat16))
        report, result = weight_store.load_with_store(
            module, model_dir, device=torch.device("cpu"), dtype=torch.bfloat16, rename=None,
            load_checkpoint=load_sharded_checkpoint, root=root, **kw)
        return module, result

    written, first = load()
    assert first.status == "written", first
    read, second = load()
    assert second.status == "hit", second
    assert torch.equal(read.up.weight, written.up.weight)
    assert torch.equal(read.up.weight_scale, written.up.weight_scale)
    assert read.up.weight.dtype == torch.float8_e4m3fn
    manifest = next(root.rglob("*.manifest.json"))
    import json

    tensors = json.loads(manifest.read_text())["tensors"]
    assert ["up.weight", "F8_E4M3", [HIDDEN, DIM]] in tensors


# ---------------------------------------------------------------- tp4 on gloo


@pytest.mark.multirank
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_tp4_fp8_mlp_matches_the_unsharded_reference(mode):
    torch.manual_seed(0)
    x = torch.randn(2, 3, DIM)
    full = fp8_toy_full_weights(DIM, HIDDEN, tp=4)
    expect = fp8_toy_reference(full, x, tp=4)
    results = run_ranks(fp8_tp_mlp_worker, DIM, HIDDEN, x.tolist(), mode, world_size=4)
    for rank, out in enumerate(results):
        out = torch.tensor(out)
        assert out.shape == expect.shape
        # The all-reduce sums four fp32 partials in another order than one matmul, and the
        # prepared bf16 weight copy rounds once: tolerance.
        assert torch.allclose(out, expect, atol=2e-2, rtol=2e-2), rank


# ------------------------------------------------------------ the kernel's host side


def test_prepare_kernel_lays_the_weight_out_and_drops_the_unpacked_one():
    torch.manual_seed(0)
    layer = Q.FP8Linear(512, 64, compute="nki")
    weight = torch.randn(64, 512) * 0.1
    _fill(layer, weight)
    data, scales = mx.kernel_layout(layer.weight.detach(), layer.weight_scale)
    layer.prepare_kernel_()
    assert layer.weight is None
    assert torch.equal(layer.weight_packed, data) and torch.equal(layer.weight_scales_hw, scales)
    assert "weight_packed" not in layer.state_dict()
    layer.prepare_kernel_()  # idempotent
    assert Q.prepare_fp8_kernels_(nn.Sequential(layer)) == 1
