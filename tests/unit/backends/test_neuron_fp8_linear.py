"""The neuron backend's FP8 linears (``ops_impl/fp8_linear.py``): construction, sharding, the
dequant compute path against the CPU reference, the in-place swap, meta materialisation, the
checkpoint load of a quantized directory, the weight store, and TP4 on gloo.

CPU only; the ``nki`` compute path needs the device and is covered by the device checks.
"""

from __future__ import annotations

import dataclasses

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
from difflet.quant import fp8 as law
from difflet.quant.fake_linear import FakeQuantLinear
from difflet.quant.spec import QuantSpec
from tests.unit.backends._neuron_fp8_workers import (
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


def _fill(layer, weight: torch.Tensor, input_amax: float = 4.0) -> None:
    q, s = law.quantize_weight(weight)
    with torch.no_grad():
        layer.weight.copy_(q)
        layer.weight_scale.copy_(s)
        layer.input_scale.copy_(law.activation_scale(input_amax, margin=1.25))


# ------------------------------------------------------------------- construction


def test_fp8_layers_hold_an_fp8_weight_and_fp32_scales():
    for layer in (Q.FP8Linear(8, 16, compute="dequant"),
                  Q.FP8ColumnParallelLinear(8, 16, compute="dequant"),
                  Q.FP8RowParallelLinear(16, 8, compute="dequant")):
        assert layer.weight.dtype == torch.float8_e4m3fn and not layer.weight.requires_grad
        assert layer.weight_scale.dtype == torch.float32 and layer.weight_scale.shape == (1,)
        assert layer.input_scale.dtype == torch.float32 and layer.input_scale.shape == (1,)
        assert layer.bias.dtype == torch.float32
        assert "weight_scale" in layer.state_dict() and "input_scale" in layer.state_dict()
        assert layer.fp8_max == 448.0


def test_dtype_names_the_bias_not_the_weight():
    layer = Q.FP8Linear(8, 16, dtype=torch.bfloat16, compute="dequant")
    assert layer.bias.dtype == torch.bfloat16 and layer.weight.dtype == torch.float8_e4m3fn


def test_unknown_compute_is_refused():
    with pytest.raises(ValueError, match="fp8 compute"):
        Q.FP8Linear(8, 16, compute="torchao")


# --------------------------------------------------------------- dequant at tp == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dequant_matches_the_weight_only_reference_exactly(dtype):
    torch.manual_seed(0)
    weight = torch.randn(16, 8) * 0.1
    layer = Q.FP8Linear(8, 16, dtype=dtype, compute="dequant")
    _fill(layer, weight)
    x = torch.randn(3, 8).to(dtype)
    expect = F.linear(x, (layer.weight.float() * layer.weight_scale).to(dtype), layer.bias)
    assert torch.equal(layer(x), expect)
    row = Q.FP8RowParallelLinear(8, 16, dtype=dtype, compute="dequant")
    _fill(row, weight)
    expect = F.linear(x, (row.weight.float() * row.weight_scale).to(dtype)) + row.bias
    assert torch.equal(row(x), expect)


def test_dequant_is_the_quantized_weight_not_the_bf16_one():
    torch.manual_seed(0)
    plain = nn.Linear(32, 32)
    layer = Q.FP8Linear(32, 32, compute="dequant")
    _fill(layer, plain.weight.detach())
    with torch.no_grad():
        layer.bias.copy_(plain.bias)
    x = torch.randn(4, 32)
    out, ref = layer(x), plain(x)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert 0.0 < rel < 0.1
    # And it is within fp8 activation rounding of the W8A8 CPU reference.
    fake = FakeQuantLinear.from_linear(plain, layer.input_scale, fp8_max=448.0)
    rel_fake = ((out - fake(x)).norm() / fake(x).norm()).item()
    assert rel_fake < 0.1


def test_nki_compute_needs_the_packed_weight():
    layer = Q.FP8Linear(8, 16, compute="nki")
    with pytest.raises(RuntimeError, match="prepare_fp8_kernels_"):
        layer(torch.randn(2, 8))


# ----------------------------------------------------------- sharding (tp = 4)


def test_column_shards_the_output_and_replicates_the_scales():
    _spec_only(4)
    layer = Q.FP8ColumnParallelLinear(8, 16, compute="dequant")
    assert layer.weight.shape == (4, 8) and layer.bias.shape == (4,)
    assert layer._difflet_shard == {"weight": 0, "bias": 0, "weight_scale": None,
                                    "input_scale": None}
    assert shard_dim(layer, "weight") == 0 and shard_dim(layer, "weight_scale") is None


def test_row_shards_the_input_and_replicates_bias_and_scales():
    _spec_only(4)
    layer = Q.FP8RowParallelLinear(8, 16, compute="dequant")
    assert layer.weight.shape == (16, 2) and layer.bias.shape == (16,)
    assert layer._difflet_shard == {"weight": 1, "bias": None, "weight_scale": None,
                                    "input_scale": None}


def test_sharded_layers_load_their_slice_of_the_full_fp8_weights():
    _spec_only(4)
    dim, hidden = 8, 16
    model = ToyFP8MLP(dim, hidden, dtype=torch.float32)
    full = fp8_toy_full_weights(dim, hidden)
    load_sharded_state_dict(model, full, tp_size=4, tp_rank=2)
    assert torch.equal(model.up.weight, full["up.weight"][8:12])
    assert torch.equal(model.down.weight, full["down.weight"][:, 8:12])
    assert torch.equal(model.up.weight_scale, full["up.weight_scale"])
    assert torch.equal(model.down.bias, full["down.bias"])


# --------------------------------------------------------------------- the swap


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = L.ColumnParallelLinear(8, 8, gather_output=False)
        self.attn.to_out = nn.ModuleList([L.RowParallelLinear(8, 8, input_is_parallel=True,
                                                               reduce_output=False)])
        self.img_mod = nn.Sequential(nn.SiLU(), L.ColumnParallelLinear(8, 48, gather_output=True))
        self.plain = nn.Linear(8, 8)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.transformer_blocks = nn.ModuleList([_Block(), _Block()])
        self.transformer.proj_out = nn.Linear(8, 4)


SPEC = QuantSpec(targets=(r"transformer_blocks\.\d+\.attn\.(to_q|to_out\.0)",
                          r"transformer_blocks\.\d+\.plain"), compute="dequant")


def test_quantize_module_replaces_exactly_the_targets_and_keeps_their_flags():
    _spec_only(4)
    model = _Model()
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
    assert block.attn.to_q.gather_output is False and block.attn.to_q.weight.shape == (2, 8)
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


def test_quantize_module_refuses_a_spec_matching_nothing():
    with pytest.raises(ValueError, match="match no linear"):
        Q.quantize_module_(_Model(), QuantSpec(targets=(r"nothing",)))


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
    assert block.attn.to_q.weight_scale.dtype == torch.float32
    assert not any(p.is_meta for p in model.parameters())


def test_load_sharded_checkpoint_fills_fp8_weights_from_a_quantized_dir(tmp_path):
    """A quantized directory (``difflet/quant/checkpoint.py`` output form) loaded through the
    native lazy loader into a meta-built fp8 module at tp=1."""
    from safetensors.torch import save_file

    from difflet.backends.neuron.core.checkpoint import build_on_meta, load_sharded_checkpoint

    full = fp8_toy_full_weights(8, 16)
    save_file(full, str(tmp_path / "model.safetensors"))
    module = build_on_meta(lambda: ToyFP8MLP(8, 16, dtype=torch.bfloat16))
    assert module.up.weight.is_meta and not module.up.weight_scale.is_meta
    report = load_sharded_checkpoint(module, tmp_path, device="cpu", dtype=torch.bfloat16,
                                     tp_size=1, tp_rank=0)
    assert report == {"missing": [], "unexpected": []}
    assert torch.equal(module.up.weight, full["up.weight"])
    assert torch.equal(module.down.input_scale, full["down.input_scale"])
    assert module.up.bias.dtype == torch.bfloat16
    x = torch.randn(2, 8).to(torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(module(x), fp8_toy_reference(full, x))


def test_weight_store_round_trips_fp8_tensors(tmp_path, monkeypatch):
    from difflet.backends.neuron.core import weight_store
    from difflet.backends.neuron.core.checkpoint import build_on_meta, load_sharded_checkpoint
    from safetensors.torch import save_file

    monkeypatch.setenv("DIFFLET_NEURON_WEIGHT_STORE_ALL", "1")
    full = fp8_toy_full_weights(8, 16)
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    save_file(full, str(model_dir / "model.safetensors"))
    root = tmp_path / "store"

    def load(**kw):
        module = build_on_meta(lambda: ToyFP8MLP(8, 16, dtype=torch.bfloat16))
        report, result = weight_store.load_with_store(
            module, model_dir, device=torch.device("cpu"), dtype=torch.bfloat16, rename=None,
            load_checkpoint=load_sharded_checkpoint, root=root, **kw)
        return module, result

    written, first = load()
    assert first.status == "written", first
    read, second = load()
    assert second.status == "hit", second
    assert torch.equal(read.up.weight, written.up.weight)
    assert read.up.weight.dtype == torch.float8_e4m3fn
    manifest = next(root.rglob("*.manifest.json"))
    import json

    tensors = json.loads(manifest.read_text())["tensors"]
    assert ["up.weight", "F8_E4M3", [16, 8]] in tensors


# ---------------------------------------------------------------- tp4 on gloo


@pytest.mark.multirank
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_tp4_fp8_mlp_matches_the_unsharded_reference(mode):
    dim, hidden = 8, 16
    torch.manual_seed(0)
    x = torch.randn(2, 3, dim)
    full = fp8_toy_full_weights(dim, hidden)
    expect = fp8_toy_reference(full, x)
    results = run_ranks(fp8_tp_mlp_worker, dim, hidden, x.tolist(), mode, world_size=4)
    for rank, out in enumerate(results):
        out = torch.tensor(out)
        assert out.shape == expect.shape
        # The all-reduce sums four fp32 partials in another order than one matmul: tolerance.
        assert torch.allclose(out, expect, atol=1e-5, rtol=1e-5), rank


# ------------------------------------------------------------ the kernel's host side


def test_pack_x4_round_trips_and_puts_the_lowest_k_in_byte_zero():
    from difflet.backends.neuron.ops_impl import fp8_kernels as K

    t = (torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8) - 7).to(torch.float8_e4m3fn)
    packed = K.pack_x4(t)
    assert packed.dtype == torch.int32 and packed.shape == (2, 2)
    assert torch.equal(K.unpack_x4(packed).float(), t.float())
    # Element (p, row) packs t[row, 4p .. 4p+3], byte 0 (the low byte) is k = 4p.
    low_byte = packed[0, 0].item() & 0xFF
    assert low_byte == t[0, 0].view(torch.uint8).item()


def test_k_tiles_and_weight_chunks_cover_the_qwen_shard_shapes():
    from difflet.backends.neuron.ops_impl import fp8_kernels as K

    assert K.k_tiles(3072) == (6, 0)
    assert K.k_tiles(768) == (1, 64)       # tp4 row-parallel attention out
    assert K.k_tiles(12288) == (24, 0)     # tp1 MLP down
    with pytest.raises(ValueError, match="32, 64 or 128"):
        K.k_tiles(3072 + 16)
    assert K.weight_chunk(3072, 3072) == 3072       # 72 KiB per partition: resident
    assert K.weight_chunk(768, 768) == 768
    chunk = K.weight_chunk(12288, 3072)             # tp1 MLP up: 295 KiB would not fit
    assert chunk % K.BANK == 0 and 12288 % chunk == 0 and chunk < 12288
    chunk = K.weight_chunk(3072, 12288)             # tp1 MLP down: 24 K tiles
    assert chunk % K.BANK == 0 and 3072 % chunk == 0 and 24 * 4 * chunk <= K.WEIGHT_SBUF_BYTES
    assert K.padded_rows(5120, 2) == 5120 and K.padded_rows(5000, 2) == 5120
    assert K.padded_rows(4097, 2) == 4352


def test_prepare_kernel_packs_the_weight_and_drops_the_unpacked_one():
    from difflet.backends.neuron.ops_impl import fp8_kernels as K

    torch.manual_seed(0)
    layer = Q.FP8Linear(768, 512, compute="nki")
    weight = torch.randn(512, 768) * 0.1
    _fill(layer, weight)
    expect_packed = K.pack_x4(layer.weight.detach())
    expect_scale = (layer.input_scale * layer.weight_scale).item()
    layer.prepare_kernel_(lnc=2)
    assert layer.weight is None
    assert torch.equal(layer.weight_packed, expect_packed)
    assert layer.out_scale.shape == (128, 1) and layer.out_scale[5, 0].item() == pytest.approx(expect_scale)
    assert (layer.kernel_n_chunk, layer.kernel_k_full, layer.kernel_k_rem, layer.kernel_lnc) == (512, 1, 64, 2)
    assert "weight_packed" not in layer.state_dict()
    layer.prepare_kernel_(lnc=2)  # idempotent
    dequant = Q.FP8Linear(8, 8, compute="dequant")
    dequant.prepare_kernel_()
    assert dequant.weight is not None and dequant.weight_packed is None


def test_kernel_reference_is_the_fake_quant_arithmetic():
    from difflet.backends.neuron.ops_impl import fp8_kernels as K

    torch.manual_seed(0)
    x = torch.randn(4, 16).to(torch.bfloat16)
    linear = nn.Linear(16, 8, bias=False)
    wq, ws = law.quantize_weight(linear.weight.detach())
    xs = law.activation_scale(4.0, margin=1.25)
    fake = FakeQuantLinear(wq, ws, xs, None, fp8_max=448.0)
    assert torch.equal(K.reference(x, wq, ws, xs, fp8_max=448.0), fake(x))


def test_pack_x4_equals_the_byte_wise_permute():
    from difflet.backends.neuron.ops_impl import fp8_kernels as K

    torch.manual_seed(0)
    t = (torch.randn(6, 16) * 3).to(torch.float8_e4m3fn)
    byte_wise = t.reshape(6, 4, 4).permute(1, 0, 2).contiguous().view(torch.int32).squeeze(-1)
    assert torch.equal(K.pack_x4(t), byte_wise)
