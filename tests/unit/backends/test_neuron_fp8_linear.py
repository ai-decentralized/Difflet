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
from difflet.backends.neuron.core.application_base import (
    TorchNeuronApplicationBase,
    rank_weight_bytes,
)
from difflet.backends.neuron.ops_impl import fp8_kernels
from difflet.backends.neuron.ops_impl import fp8_linear as Q
from difflet.backends.neuron.ops_impl import linear as L
from difflet.backends.neuron.ops_impl import parallel_mesh as pm
from difflet.backends.tpu.core.weights import load_sharded_state_dict, shard_dim
from difflet.pipeline.parallel_config import DiffletParallelConfig
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
        # The layers allocate the bias with torch.empty (a checkpoint fills it). Left as
        # is, it holds whatever the allocator hands back, NaN included, and NaN != NaN
        # fails the exact comparisons. A fixed ramp keeps the bias add tested.
        if layer.bias is not None:
            layer.bias.copy_(torch.linspace(-0.5, 0.5, layer.bias.numel()))


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


def test_padded_rows_is_the_next_multiple_of_8():
    assert fp8_kernels.M_MULTIPLE == 8
    assert fp8_kernels.padded_rows(126) == 128  # LTX-2's audio tokens at the default shape
    assert fp8_kernels.padded_rows(2 * 126) == 256
    assert fp8_kernels.padded_rows(6144) == 6144  # its video tokens: already a multiple
    assert fp8_kernels.padded_rows(1) == 8
    assert fp8_kernels.padded_rows(8) == 8  # the smallest count the kernel takes
    # HunyuanVideo-1.5's text and Wan+CP2's token counts: the padded M is sharded M // 2 per
    # physical core under LNC2 (996 and 1172, not multiples of 8), probed correct on the device
    # (fp8-bench/ltx2_probe/probe_mx_shapes_lnc2.json), so 8 is the whole rule.
    assert fp8_kernels.padded_rows(1985) == 1992
    assert fp8_kernels.padded_rows(2340) == 2344


def test_mx_linear_pads_the_rows_to_a_multiple_of_8(monkeypatch):
    # The kernel refuses a row count that is not a multiple of 8 (its F-dim rule); the op is
    # faked with any row-wise function of the activation so the slice-back is checked exactly.
    seen = []

    def fake_op(x2, packed, scales):
        seen.append(tuple(x2.shape))
        assert x2.shape[0] % 8 == 0
        return x2[:, :packed.shape[1]]

    monkeypatch.setattr(fp8_kernels, "_MX_OP", fake_op)
    packed = torch.zeros(64, 32, dtype=torch.int32)
    scales = torch.zeros(64, 32, dtype=torch.uint8)
    x = torch.randn(2, 126, 256).to(torch.bfloat16)
    out = fp8_kernels.mx_linear(x, packed, scales)
    assert out.shape == (2, 126, 32) and seen == [(256, 256)]  # 252 rows -> 256
    assert torch.equal(out.reshape(-1, 32), x.reshape(-1, 256)[:, :32])
    fp8_kernels.mx_linear(torch.randn(8, 256).to(torch.bfloat16), packed, scales)
    assert seen[-1] == (8, 256)  # already a multiple: no pad
    # A float32 activation is cast to bf16 before the pad and the output back after it: the
    # padded kernel call is bf16 and the sliced result carries the activation's dtype.
    x32 = torch.randn(126, 256)
    out32 = fp8_kernels.mx_linear(x32, packed, scales)
    assert out32.dtype == torch.float32 and seen[-1] == (128, 256)
    assert torch.equal(out32, x32.to(torch.bfloat16)[:, :32].float())


# ------------------------------------------------------------ the application base


class _TwoLinears(nn.Module):
    """``fc1`` (K=128, the spec's target) and ``fc2`` (K=16: no MX layout, stays as it is)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _ToyApp(TorchNeuronApplicationBase):
    """The base's lifecycle over ``_TwoLinears`` on the CPU at tp=1, no block stack to compile:
    the subclass declares its target table's key and builds the bf16 module, nothing else."""

    quant_model = "toy"
    block_attrs = ()

    def build_module(self):
        return _TwoLinears(128, 16)

    def get_example_inputs(self):
        return (torch.randn(2, 128),)


_SPEC = QuantSpec(targets=("fc1",), compute="dequant")


def test_base_refuses_quant_without_a_target_table():
    """A subclass without ``quant_model`` has no FP8 support: ``quant=`` is refused at
    construction, not run as the bf16 module against a quantized checkpoint."""
    class NoQuant(_ToyApp):
        quant_model = None

    with pytest.raises(NotImplementedError, match="NoQuant has no MXFP8 target layers"):
        NoQuant(model_path="/x", device="cpu", exec_mode="compile", quant=_SPEC)
    assert NoQuant(model_path="/x", device="cpu", exec_mode="compile").quant is None


def test_base_refuses_eager_with_quant_under_the_models_label():
    """The D43 refusal names the model by ``quant_label``, or by its target-table key."""
    class Labelled(_ToyApp):
        quant_label = "Toy DiT"

    with pytest.raises(ValueError, match="the Toy DiT FP8 DiT needs exec_mode='compile'"):
        Labelled(model_path="/x", device="cpu", exec_mode="eager", quant=_SPEC)
    with pytest.raises(ValueError, match="the toy FP8 DiT needs exec_mode='compile'"):
        _ToyApp(model_path="/x", device="cpu", exec_mode="eager", quant=_SPEC.to_dict())


def test_base_refuses_a_spec_quantized_for_another_tp():
    """A row-parallel weight is quantized per rank K range, so a spec made for one tp holds
    scales grouped for that tp's shards: at another tp every rank would load a shape-compatible
    slice with the wrong block membership. Refused at construction, naming both degrees; the
    pipeline's ``"mxfp8"`` spelling builds its spec at the launch's tp, a spec given directly
    is the caller's."""
    with pytest.raises(ValueError, match=r"the toy FP8 DiT spec was quantized for tp_degree=1 "
                                         r"but this launch has tp_degree=2"):
        _ToyApp(model_path="/x", device="cpu", exec_mode="compile",
                parallel=DiffletParallelConfig(tp_degree=2), quant=_SPEC)


def test_base_refuses_a_quant_that_is_not_a_spec():
    """``"mxfp8"`` is the pipeline application's spelling, which resolves it to a spec; the
    transformer base takes the spec or its dict and refuses anything else at construction,
    not in ``load``'s collective phase with an AttributeError that names no cause."""
    with pytest.raises(TypeError, match="_ToyApp: quant must be a QuantSpec or its dict, got str"):
        _ToyApp(model_path="/x", device="cpu", exec_mode="compile", quant="mxfp8")


def test_base_quantizes_after_build_and_prepares_after_load(tmp_path):
    """``load`` swaps the spec's targets on the meta module ``build_module`` returned, fills
    them from the quantized directory and prepares them in ``_after_load``, so the forward is
    the dequant one on the checkpoint's fp8 weight."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    reference = _TwoLinears(128, 16)
    q, s = mx.quantize_mx(reference.fc1.weight.detach())
    save_file({"fc1.weight": q, "fc1.weight_scale": s, "fc1.bias": reference.fc1.bias.detach(),
               "fc2.weight": reference.fc2.weight.detach(),
               "fc2.bias": reference.fc2.bias.detach()},
              str(tmp_path / "model.safetensors"))
    app = _ToyApp(model_path=str(tmp_path), parallel=DiffletParallelConfig(), dtype=torch.float32,
                  exec_mode="compile", device="cpu", quant=_SPEC.to_dict())
    assert app.quant == _SPEC and app.quantized_linears == []
    app.load()
    assert app.quantized_linears == ["fc1"]
    assert app.load_report == {"missing": [], "unexpected": []}
    assert isinstance(app.module.fc1, Q.FP8Linear) and type(app.module.fc2) is nn.Linear
    assert app.module.fc1.weight is None and app.module.fc1.weight_bf16 is not None  # prepared
    x = torch.randn(2, 128)
    with torch.no_grad():
        reference.fc1.weight.copy_(mx.dequantize_mx(q, s))
        expect = reference(x)
    # The prepared bf16 weight copy rounds the dequantized weight once more than the reference.
    assert torch.allclose(app(x), expect, atol=2e-2, rtol=2e-2)


class _Attention(nn.Module):
    """One attention-shaped module: ``attn.to_q``/``to_k``/``to_v`` over K=128, the group
    ``fuse_projections_`` reads. No forward: the kernel path runs on the device only."""

    def __init__(self, dim: int, inner: int):
        super().__init__()
        self.attn = nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(self.attn, name, nn.Linear(dim, inner))


_ATTN_SPEC = QuantSpec(targets=(r"attn\.to_[qkv]",), compute="nki")


@pytest.mark.parametrize("fuse", [True, False])
def test_base_fuses_q_k_v_after_load_only_when_the_model_says_so(tmp_path, fuse):
    """``_after_load`` prepares the kernel-path twins under ``quant_fuse_qkv``: a model whose
    attention reads ``fused_qkv`` gets its three projections fused into one kernel call (the
    members drop their weights), any other keeps three prepared layers. The kernel layout is
    laid out on the host; the forward needs the device, so the warm-up is skipped."""
    from safetensors.torch import save_file

    class Fusing(_ToyApp):
        quant_fuse_qkv = fuse

        def build_module(self):
            return _Attention(128, 16)

    torch.manual_seed(0)
    reference = _Attention(128, 16)
    tensors = {}
    for name in ("to_q", "to_k", "to_v"):
        linear = getattr(reference.attn, name)
        q, s = mx.quantize_mx(linear.weight.detach())
        tensors[f"attn.{name}.weight"], tensors[f"attn.{name}.weight_scale"] = q, s
        tensors[f"attn.{name}.bias"] = linear.bias.detach()
    save_file(tensors, str(tmp_path / "model.safetensors"))
    app = Fusing(model_path=str(tmp_path), dtype=torch.float32, exec_mode="compile",
                 device="cpu", quant=_ATTN_SPEC)
    app.load(skip_warmup=True)
    assert app.quantized_linears == ["attn.to_q", "attn.to_k", "attn.to_v"]
    assert app.load_report == {"missing": [], "unexpected": []}
    members = [getattr(app.module.attn, n) for n in ("to_q", "to_k", "to_v")]
    assert all(isinstance(l, Q.FP8Linear) and l.compute == "nki" and l.weight is None
               for l in members)
    if fuse:
        assert isinstance(app.module.attn.fused_qkv, Q.FusedFP8Projections)
        assert app.module.attn.fused_qkv.splits == [16, 16, 16]
        assert all(l.weight_packed is None for l in members)
    else:
        assert not hasattr(app.module.attn, "fused_qkv")
        assert all(l.weight_packed is not None for l in members)


# ------------------------------------------------------ the pipeline side's resolver


@pytest.fixture(scope="module")
def tiny_quantized_toy(tmp_path_factory):
    """``_TwoLinears`` saved in bf16 as a one-file checkpoint, and its ``fc1`` quantized for a
    tp2 spec under a cache root: what a pipeline application resolves from ``"mxfp8"``."""
    from safetensors.torch import save_file

    from difflet.quant.checkpoint import prepare_quantized_checkpoint

    root = tmp_path_factory.mktemp("tiny_quantized")
    transformer_dir = root / "transformer"
    transformer_dir.mkdir()
    torch.manual_seed(0)
    save_file({k: v.to(torch.bfloat16) for k, v in _TwoLinears(128, 16).state_dict().items()},
              str(transformer_dir / "model.safetensors"))
    spec = QuantSpec(targets=("fc1",), tp_degree=2)
    quantized = prepare_quantized_checkpoint(transformer_dir, spec, create=True,
                                             cache_root=root / "cache")
    return {"cache": root / "cache", "transformer_dir": str(transformer_dir), "spec": spec,
            "quantized": str(quantized)}


def test_resolve_quant_default_spec_and_checkpoint(tiny_quantized_toy, monkeypatch):
    """One resolver for every pipeline application: ``"mxfp8"`` is the model's default spec at
    the launch's tp and the quantized directory under the compile cache (never written here:
    every rank resolves, ``difflet compile`` writes on the driver); a spec or its dict is taken
    as given; an explicit directory is checked against the spec; a directory without a spec,
    another spelling than ``"mxfp8"`` and a quant of another type are refused by name."""
    from difflet.backends.neuron.core.quant import resolve_quant
    from difflet.quant import targets

    toy = tiny_quantized_toy
    monkeypatch.setitem(targets.TARGETS_BY_MODEL, "toy", ("fc1",))
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(toy["cache"]))
    spec, path = resolve_quant("toy", "mxfp8", None, toy["transformer_dir"], tp_degree=2)
    assert spec == targets.default_spec("toy", tp_degree=2) and path == toy["quantized"]
    resolved = (toy["spec"], toy["quantized"])
    for given in (toy["spec"], toy["spec"].to_dict()):
        assert resolve_quant("toy", given, None, toy["transformer_dir"], tp_degree=2) == resolved
    assert resolve_quant("toy", toy["spec"], toy["quantized"], toy["transformer_dir"],
                         tp_degree=2) == resolved
    assert resolve_quant("toy", None, None, toy["transformer_dir"], tp_degree=2) \
        == (None, toy["transformer_dir"])
    with pytest.raises(ValueError, match="difflet compile"):  # no tp1 directory was written
        resolve_quant("toy", "mxfp8", None, toy["transformer_dir"], tp_degree=1)
    with pytest.raises(ValueError, match="another quantization spec"):
        resolve_quant("toy", QuantSpec(targets=("fc2",)), toy["quantized"],
                      toy["transformer_dir"], tp_degree=2)
    with pytest.raises(ValueError, match="quant_checkpoint needs quant"):
        resolve_quant("toy", None, "/q", toy["transformer_dir"], tp_degree=1)
    with pytest.raises(ValueError, match="quant must be 'mxfp8'"):
        resolve_quant("toy", "fp8", None, toy["transformer_dir"], tp_degree=1)
    with pytest.raises(TypeError, match="quant must be 'mxfp8'"):
        resolve_quant("toy", 8, None, toy["transformer_dir"], tp_degree=1)
