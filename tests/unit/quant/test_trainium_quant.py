"""Trainium FP8 hook: NeuronConfig kwargs, compiler flag, q-config, convert call.

The NxD imports are stubbed so the wiring is checked without a Neuron install.
"""

from __future__ import annotations

import enum
import sys
import types
from types import SimpleNamespace

import pytest

from difflet.backends.trainium.core import quant as tq
from difflet.quant.spec import QuantSpec


def test_neuron_config_kwargs_map_the_spec_onto_nxdi_fields(tmp_path):
    kwargs = tq.neuron_config_kwargs(QuantSpec(), tmp_path / "q")
    assert kwargs == {
        "quantized": True,
        "quantized_checkpoints_path": str(tmp_path / "q"),
        "quantization_type": "per_tensor_symmetric",
        "quantization_dtype": "f8e4m3",
        "activation_quantization_type": "dynamic",
        "quant_targets": list(QuantSpec().targets),
    }
    per_channel = tq.neuron_config_kwargs(QuantSpec(weight_granularity="channel"), "/q")
    assert per_channel["quantization_type"] == "per_channel_symmetric"
    assert per_channel["activation_quantization_type"] == "dynamic"  # always W8A8


def test_neuron_config_kwargs_pass_nxd_validation():
    """The raw strings must be the NxD enum *values* — NeuronConfig validates them.

    Pins the Phase-0 probe failure of 2026-10-01 on trn2: ``"DYNAMIC"`` (the
    member name) raised ``AssertionError: Unsupported activation quantization
    type: DYNAMIC`` from ``validate_activation_quantization_type`` before the
    fp8 arm could compile. Runs where neuronx_distributed is installed.
    """
    pytest.importorskip("neuronx_distributed")
    from neuronx_distributed.quantization.quantization_config import (
        ActivationQuantizationType,
        QuantizationType,
    )

    from difflet.backends.trainium.core.config import validate_activation_quantization_type

    for spec in (QuantSpec(), QuantSpec(weight_granularity="channel")):
        kwargs = tq.neuron_config_kwargs(spec, "/q")
        validate_activation_quantization_type(kwargs["activation_quantization_type"])
        assert ActivationQuantizationType(kwargs["activation_quantization_type"]) \
            is ActivationQuantizationType.DYNAMIC
        assert QuantizationType(kwargs["quantization_type"])


def test_neuron_config_keeps_the_spec_weight_granularity_with_dynamic_activations():
    """NxDI's NeuronConfig rewrote quantization_type to per_channel_symmetric for any
    activation quantization (its quantized-MLP-kernel scheme); on trn2 (2026-10-01)
    that made the layers expect [out, 1] scales for a per-tensor [1] checkpoint
    ("expected shape torch.Size([128, 1]) for blocks.0.attn1.to_q.scale but found
    torch.Size([1])"). The override now applies only with the quantized MLP kernel."""
    pytest.importorskip("neuronx_distributed")
    import torch

    from difflet.backends.trainium.core.config import NeuronConfig

    for granularity in ("tensor", "channel"):
        kwargs = tq.neuron_config_kwargs(QuantSpec(weight_granularity=granularity), "/q")
        config = NeuronConfig(tp_degree=1, world_size=1, batch_size=1, torch_dtype=torch.bfloat16, **kwargs)
        assert config.quantization_type == kwargs["quantization_type"]
        assert config.activation_quantization_type == "dynamic"
        assert config.quantization_dtype == "f8e4m3"


def test_fp8_compiler_flag_only_for_fp8_quantized_configs():
    assert tq.fp8_hlo2tensorizer_options(SimpleNamespace(quantized=False)) == ""
    assert tq.fp8_hlo2tensorizer_options(SimpleNamespace(quantized=True, quantization_dtype="int8")) == ""
    assert (
        tq.fp8_hlo2tensorizer_options(SimpleNamespace(quantized=True, quantization_dtype="f8e4m3"))
        == "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 "
    )


def test_quantize_traced_model_is_a_no_op_without_quantization():
    model = object()
    assert tq.quantize_traced_model_(model, SimpleNamespace(quantized=False)) is model
    assert tq.quantize_traced_model_(model, SimpleNamespace()) is model


@pytest.fixture
def fake_nxd(monkeypatch):
    """Minimal stand-in for neuronx_distributed.quantization.{quantization_config,quantize}."""

    class QuantizationType(enum.Enum):
        PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
        PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"

    class QuantizedDtype(enum.Enum):
        INT8 = "int8"
        F8E4M3 = "f8e4m3"

        @classmethod
        def get_dtype(cls, name):
            return cls(name)

    class ActivationQuantizationType(enum.Enum):  # values as in NxD: lowercase
        NONE = None
        DYNAMIC = "dynamic"
        STATIC = "static"

    calls = []

    def convert(model, q_config, inplace, mapping, modules_to_not_convert=None, include=None):
        calls.append(dict(model=model, q_config=q_config, inplace=inplace, mapping=mapping,
                          modules_to_not_convert=modules_to_not_convert, include=include))
        return model

    cfg = types.ModuleType("neuronx_distributed.quantization.quantization_config")
    cfg.QuantizationType = QuantizationType
    cfg.QuantizedDtype = QuantizedDtype
    cfg.ActivationQuantizationType = ActivationQuantizationType
    cfg.get_default_custom_qconfig_dict = lambda: {"quantization_type": "per_tensor", "default": True}
    cfg.get_default_per_channel_custom_qconfig_dict = lambda: {"quantization_type": "per_channel", "default": True}
    quantize = types.ModuleType("neuronx_distributed.quantization.quantize")
    quantize.convert = convert

    import torch

    class ColumnParallelLinear(torch.nn.Module):  # the float layers convert() maps from
        pass

    class RowParallelLinear(torch.nn.Module):
        pass

    class QuantizedColumnParallel(torch.nn.Module):  # NxD's quantized forms Difflet subclasses
        activation_quantization_type = ActivationQuantizationType.NONE

        @classmethod
        def from_float(cls, mod, q_config=None):
            # NxD builds the quantized layer from mod.dtype (construction-time dtype)
            # and never looks at mod.skip_bias_add.
            new = QuantizedColumnParallel()
            new.dtype = mod.dtype
            new.dequantized_dtype = mod.dtype
            new.bias = torch.nn.Parameter(torch.ones(4, dtype=mod.dtype), requires_grad=False)
            return new

        def forward(self, input, *args, **kwargs):  # NxD: bias added unconditionally
            return (input + self.bias) if self.bias is not None else input

    class QuantizedRowParallel(QuantizedColumnParallel):
        @classmethod
        def from_float(cls, mod, q_config=None):
            new = QuantizedRowParallel()
            new.dtype = mod.dtype
            new.dequantized_dtype = mod.dtype
            new.bias = torch.nn.Parameter(torch.ones(4, dtype=mod.dtype), requires_grad=False)
            return new

    layers = types.ModuleType("neuronx_distributed.parallel_layers.layers")
    layers.ColumnParallelLinear = ColumnParallelLinear
    layers.RowParallelLinear = RowParallelLinear
    q_layers = types.ModuleType("neuronx_distributed.quantization.quantization_layers")
    q_layers.QuantizedColumnParallel = QuantizedColumnParallel
    q_layers.QuantizedRowParallel = QuantizedRowParallel
    q_layers._DEFAULT_CUSTOM_QCONFIG_DICT = {}
    root = types.ModuleType("neuronx_distributed")
    pkg = types.ModuleType("neuronx_distributed.quantization")
    pkg.quantization_layers = q_layers
    parallel = types.ModuleType("neuronx_distributed.parallel_layers")
    for name, module in {
        "neuronx_distributed": root,
        "neuronx_distributed.quantization": pkg,
        "neuronx_distributed.quantization.quantization_config": cfg,
        "neuronx_distributed.quantization.quantize": quantize,
        "neuronx_distributed.quantization.quantization_layers": q_layers,
        "neuronx_distributed.parallel_layers": parallel,
        "neuronx_distributed.parallel_layers.layers": layers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(tq, "_MAPPING", None)  # never leak stub-derived classes
    return SimpleNamespace(calls=calls, types=(QuantizationType, QuantizedDtype, ActivationQuantizationType),
                           layers=(ColumnParallelLinear, RowParallelLinear,
                                   QuantizedColumnParallel, QuantizedRowParallel))


def test_build_q_config_mirrors_nxdi_for_each_granularity(fake_nxd):
    _, QuantizedDtype, ActivationQuantizationType = fake_nxd.types
    per_tensor = tq.build_q_config(SimpleNamespace(
        quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type="dynamic", quantize_clamp_bound=float("inf"),
    ))
    assert per_tensor["quantization_type"] == "per_tensor"
    assert per_tensor["quantized_dtype"] is QuantizedDtype.F8E4M3
    assert per_tensor["activation_quantization_type"] is ActivationQuantizationType.DYNAMIC
    assert per_tensor["clamp_bound"] == float("inf")

    per_channel = tq.build_q_config(SimpleNamespace(
        quantization_type="per_channel_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type="dynamic", quantize_clamp_bound=100.0,
    ))
    assert per_channel["quantization_type"] == "per_channel"
    assert per_channel["activation_quantization_type"] is ActivationQuantizationType.DYNAMIC
    assert per_channel["clamp_bound"] == 100.0

    with pytest.raises(ValueError):
        tq.build_q_config(SimpleNamespace(quantization_type="blockwise_symmetric",
                                          quantization_dtype="f8e4m3",
                                          activation_quantization_type=None))


def _tiny_wan_module(fake_nxd):
    """The Wan device layer set, as stand-in parallel linears (no HF spellings)."""
    import torch

    column, row = fake_nxd.layers[0], fake_nxd.layers[1]
    block = torch.nn.Module()
    block.to_q, block.to_k, block.to_v = column(), column(), column()
    block.to_out = torch.nn.ModuleList([row()])
    block.ffn = torch.nn.Module()
    block.ffn.net_in, block.ffn.net_out = column(), row()
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    model.proj_out = torch.nn.Linear(4, 4)
    return model


def test_quantize_traced_model_calls_convert_with_difflet_layers(fake_nxd):
    """convert() must map NxD's float layers onto Difflet's subclasses of the
    NxD quantized layers (the ones with the per-tensor DYNAMIC forward)."""
    model = _tiny_wan_module(fake_nxd)
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type="dynamic", quantize_clamp_bound=float("inf"),
        modules_to_not_convert=["proj_out"],   # NxDI field; Difflet ignores it (scoped by include)
    )
    assert tq.quantize_traced_model_(model, neuron_config) is model
    (call,) = fake_nxd.calls
    assert call["model"] is model and call["inplace"] is True
    assert call["modules_to_not_convert"] is None  # scoped by include, never by exclusion
    assert call["q_config"]["quantized_dtype"].value == "f8e4m3"
    column, row, q_column, q_row = fake_nxd.layers
    assert set(call["mapping"]) == {column, row}
    assert issubclass(call["mapping"][column], q_column) and call["mapping"][column] is not q_column
    assert issubclass(call["mapping"][row], q_row) and call["mapping"][row] is not q_row
    assert tq.quant_module_mapping() is call["mapping"]  # built once


def test_neuron_config_kwargs_carry_the_targets(tmp_path):
    kwargs = tq.neuron_config_kwargs(QuantSpec.for_model("flux"), tmp_path / "q")
    assert kwargs["quant_targets"] == list(QuantSpec.for_model("flux").targets)
    assert tq.neuron_config_kwargs(QuantSpec(), "/q")["quant_targets"] == list(QuantSpec().targets)


def test_include_patterns_cover_suffixes_and_globs():
    assert tq.include_patterns(("to_q", "single_transformer_blocks.*.proj_out")) == [
        "to_q", "*.to_q",
        "single_transformer_blocks.*.proj_out", "*.single_transformer_blocks.*.proj_out",
    ]


def _model_with(fake_nxd, **children):
    import torch

    column = fake_nxd.layers[0]
    model = torch.nn.Module()
    for name in children:
        setattr(model, name, column())
    return model


def test_quantize_traced_model_scopes_convert_to_the_targets(fake_nxd):
    model = _model_with(fake_nxd, to_q=True)
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type="dynamic", quantize_clamp_bound=float("inf"),
        quant_targets=["to_q"],
    )
    tq.quantize_traced_model_(model, neuron_config)
    (call,) = fake_nxd.calls
    assert call["include"] == ["to_q", "*.to_q"]
    assert call["modules_to_not_convert"] is None


def test_scoped_convert_raises_on_unmatched_target(fake_nxd):
    """A renamed layer must not silently stay bf16 (Review Focus 1)."""
    model = _model_with(fake_nxd, to_q=True)
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type=None, quantize_clamp_bound=float("inf"),
        quant_targets=["to_q", "ffn.net_in"],
    )
    with pytest.raises(ValueError, match="ffn.net_in"):
        tq.quantize_traced_model_(model, neuron_config)
    assert fake_nxd.calls == []


def test_hf_only_proj_out_glob_is_satisfied_by_the_device_halves(fake_nxd):
    """FLUX / HunyuanVideo: the HF fused proj_out exists on device only as its halves."""
    import torch

    column = fake_nxd.layers[0]
    model = torch.nn.Module()
    model.single_transformer_blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.single_transformer_blocks[0].proj_out_attn = column()
    model.single_transformer_blocks[0].proj_out_mlp = column()
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type=None, quantize_clamp_bound=float("inf"),
        quant_targets=["single_transformer_blocks.*.proj_out", "single_transformer_blocks.*.proj_out_attn",
                       "single_transformer_blocks.*.proj_out_mlp"],
    )
    tq.quantize_traced_model_(model, neuron_config)   # must not raise on the HF-only glob
    (call,) = fake_nxd.calls
    assert "single_transformer_blocks.*.proj_out_attn" in call["include"]
    assert "single_transformer_blocks.*.proj_out" not in call["include"]  # HF-only, not a device module


def test_wan_default_targets_without_quant_targets_field_still_convert(fake_nxd):
    """A NeuronConfig from before quant_targets existed: Wan's device set applies."""
    model = _tiny_wan_module(fake_nxd)
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type=None, quantize_clamp_bound=float("inf"),
    )
    tq.quantize_traced_model_(model, neuron_config)
    (call,) = fake_nxd.calls
    assert call["include"] == tq.include_patterns(
        ["to_q", "to_k", "to_v", "to_out.0", "ffn.net_in", "ffn.net_out"])


def test_from_float_types_the_quantized_layer_from_the_live_weight_dtype(fake_nxd):
    """Pins the fp32 promotion seen on trn2 (2026-10-01): the model is built in fp32
    and cast to bf16, NxD's from_float reads the construction-time mod.dtype, so the
    bias / dequantized dtype were fp32 and 316 of 400 weight-only dots ran as F32."""
    import torch

    column, _, q_column, _ = fake_nxd.layers
    mod = SimpleNamespace(dtype=torch.float32, weight=torch.zeros(4, 4, dtype=torch.bfloat16))
    new = tq.quant_module_mapping()[column].from_float(mod, {})
    assert isinstance(new, q_column) and type(new) is not q_column
    assert new.dtype is torch.bfloat16 and new.dequantized_dtype is torch.bfloat16
    assert new.bias.dtype is torch.bfloat16


def test_quantize_activation_per_tensor_matches_the_cpu_reference():
    """Same law as difflet.quant.fp8.quantize_activation, 0-D scale, clamp bound honoured.

    Pins the trace failure of 2026-10-01 on trn2 ("input_sizes.size() <=
    output_sizes.size() (4 vs. 3)"): NxD's DYNAMIC path makes a per-channel
    (axis 1) scale that cannot dequantize a 3-D output; a per-tensor scale can.
    """
    import torch

    from difflet.quant import fp8

    torch.manual_seed(0)
    x = torch.randn(2, 7, 16, dtype=torch.bfloat16) * 3
    q, scale = tq.quantize_activation_per_tensor(x)
    ref_q, ref_scale = fp8.quantize_activation(x)
    assert q.dtype == torch.float8_e4m3fn and scale.shape == () and scale.dtype == torch.float32
    assert torch.equal(q.float(), ref_q.float()) and torch.equal(scale.reshape(1), ref_scale)
    assert q.float().abs().max() <= fp8.FP8_MAX

    clamped_q, clamped_scale = tq.quantize_activation_per_tensor(x, clamp_bound=1.0)
    assert clamped_scale.item() == pytest.approx(1.0 / fp8.FP8_MAX * fp8.ACT_SCALE_MARGIN)
    assert clamped_q.float().abs().max() == fp8.FP8_MAX  # saturated at the clamp
    # Lean law: no fp32 up-cast of the activation, no clamp, bf16 multiply (x * 1/scale).
    assert (q.float().abs().max() <= fp8.FP8_MAX) and q.float().abs().max() < 248


@pytest.mark.parametrize("granularity", ["tensor", "channel"])
def test_dynamic_fp8_linear_matches_the_reference_on_3d_inputs(granularity):
    import torch

    from difflet.quant import fp8

    torch.manual_seed(1)
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    weight = torch.randn(24, 32, dtype=torch.bfloat16) * 0.1
    q_weight, scale = fp8.quantize_weight(weight, granularity)  # scale [1] or [out, 1]

    def forward_impl(input, weight, bias, **kwargs):  # the NxD matmul, emulated in fp32
        assert input.dtype == torch.float8_e4m3fn and weight.dtype == torch.float8_e4m3fn
        assert kwargs["process_group"] == "tp"
        return torch.nn.functional.linear(input.float(), weight.float())

    layer = SimpleNamespace(weight=q_weight, scale=scale, clamp_bound=float("inf"),
                            _forward_impl=forward_impl)
    out = tq.dynamic_fp8_linear(layer, x, process_group="tp")
    ref = fp8.fp8_linear_reference(x, q_weight, scale, None)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 5, 24)
    assert torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=1e-3)


def test_wan_application_resolves_and_ensures_quantized_checkpoints(tmp_path):
    """The application's checkpoint plumbing, without constructing a NeuronConfig.

    Importing the application module needs the Neuron stack (its config module
    imports neuronx_distributed at import time), so this runs in the reference
    environment only.
    """
    pytest.importorskip("neuronx_distributed")
    import json

    from safetensors.torch import save_file
    import torch

    from difflet.models.wan.application import NeuronWanApplication

    source = tmp_path / "model" / "transformer"
    source.mkdir(parents=True)
    save_file({"blocks.0.attn1.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
               "proj_out.weight": torch.randn(4, 8, dtype=torch.bfloat16)},
              str(source / "diffusion_pytorch_model.safetensors"))
    (source / "config.json").write_text(json.dumps({}))

    app = NeuronWanApplication.__new__(NeuronWanApplication)
    app.model_path = str(tmp_path / "model")
    app.quant_spec = QuantSpec()
    app._quant_cache_dir = str(tmp_path / "cache")
    app.quant_checkpoint_dirs = {}

    dest = app._quant_checkpoint_dir("transformer")
    assert dest.startswith(str(tmp_path / "cache" / "quantized"))
    assert app._quant_checkpoint_dir("transformer") == dest  # memoized
    with pytest.raises(FileNotFoundError):
        app.ensure_quantized_checkpoints(create=False)
    assert app.ensure_quantized_checkpoints(create=True) == {"transformer": dest}
    assert (tmp_path / "cache" / "quantized").exists()
    assert app.ensure_quantized_checkpoints(create=False) == {"transformer": dest}

    bf16 = NeuronWanApplication.__new__(NeuronWanApplication)
    bf16.quant_spec = None
    bf16.quant_checkpoint_dirs = {}
    assert bf16._quant_checkpoint_dir("transformer") is None
    assert bf16.ensure_quantized_checkpoints(create=True) == {}


def test_from_float_carries_skip_bias_add(fake_nxd):
    """NxD's from_float drops RowParallelLinear.skip_bias_add; FLUX / HunyuanVideo single
    blocks rely on it (proj_out_attn returns (out, bias), the bias is added once after
    the merged all-reduce)."""
    import torch

    _, row, _, q_row = fake_nxd.layers
    weight = torch.zeros(4, 4, dtype=torch.bfloat16)
    plain = tq.quant_module_mapping()[row].from_float(SimpleNamespace(dtype=torch.float32, weight=weight), {})
    assert isinstance(plain, q_row) and plain.skip_bias_add is False
    skip = tq.quant_module_mapping()[row].from_float(
        SimpleNamespace(dtype=torch.float32, weight=weight, skip_bias_add=True), {})
    assert skip.skip_bias_add is True


@pytest.mark.parametrize("layer_index", [0, 1], ids=["column", "row"])
def test_skip_bias_add_returns_output_and_bias(fake_nxd, layer_index, monkeypatch):
    """Pins the trn2 failure of 2026-10-02 (FLUX fp8 compile): ``out_attn, bias =
    self.proj_out_attn(attn_output)`` → ``ValueError: not enough values to unpack``
    because NxD's quantized forward adds the bias and returns one tensor. With
    reduce_output=False that would also add the bias on every TP rank before the
    model's own all-reduce. Difflet's dynamic forward (the only path now) returns
    ``(out, bias)`` when the float layer had skip_bias_add."""
    import torch

    ActivationQuantizationType = fake_nxd.types[2]
    float_layer = fake_nxd.layers[layer_index]
    weight = torch.zeros(4, 4, dtype=torch.bfloat16)
    monkeypatch.setattr(tq, "dynamic_fp8_linear", lambda mod, inp, **kw: inp * 3)

    def make(**extra):
        layer = tq.quant_module_mapping()[float_layer].from_float(
            SimpleNamespace(dtype=torch.float32, weight=weight, **extra), {})
        layer.activation_quantization_type = ActivationQuantizationType.DYNAMIC
        layer.async_tensor_model_parallel_allreduce = True  # column: input already parallel
        layer.gather_output = False
        layer.input_is_parallel = True  # row
        layer.reduce_output = False
        layer.sequence_parallel_enabled = False
        layer.sequence_dimension = None
        layer.autograd_func_class = None
        layer.tensor_parallel_group = None
        return layer

    x = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    out, bias = make(skip_bias_add=True)(x)
    assert torch.equal(out, x * 3)  # bias NOT added
    assert torch.equal(bias, torch.ones(4, dtype=torch.bfloat16))
    assert torch.equal(make()(x), x * 3 + 1)


def test_dynamic_row_forward_honours_skip_bias_add_toggle(fake_nxd, monkeypatch):
    import torch

    ActivationQuantizationType = fake_nxd.types[2]
    _, row, _, _ = fake_nxd.layers
    weight = torch.zeros(4, 4, dtype=torch.bfloat16)
    layer = tq.quant_module_mapping()[row].from_float(
        SimpleNamespace(dtype=torch.float32, weight=weight, skip_bias_add=True), {})
    layer.activation_quantization_type = ActivationQuantizationType.DYNAMIC
    layer.input_is_parallel = True
    layer.reduce_output = False
    layer.sequence_parallel_enabled = False
    layer.sequence_dimension = None
    layer.autograd_func_class = None
    layer.tensor_parallel_group = None
    monkeypatch.setattr(tq, "dynamic_fp8_linear", lambda mod, inp, **kw: inp * 3)
    x = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    out, bias = layer(x)
    assert torch.equal(out, x * 3) and bias is layer.bias
    layer.skip_bias_add = False
    assert torch.equal(layer(x), x * 3 + 1)



def test_neuron_config_kwargs_select_static_activation_quantization(tmp_path):
    calib = tmp_path / "calib.json"
    calib.write_text('{"layers": {}}')
    kwargs = tq.neuron_config_kwargs(QuantSpec(calibration=str(calib)), "/q")
    assert kwargs["activation_quantization_type"] == "static"
    assert tq.neuron_config_kwargs(QuantSpec(), "/q")["activation_quantization_type"] == "dynamic"


def test_static_input_scale_path_skips_the_reductions_and_clamps(monkeypatch):
    """A layer carrying NxD's static ``input_scale`` quantizes with that constant: no
    absmax reductions, one fused multiply + clamp + cast; outputs match the CPU
    reference; values beyond the calibrated range saturate at ±240 (finite)."""
    import torch

    from difflet.quant import fp8

    torch.manual_seed(5)
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    q_weight, w_scale = fp8.quantize_weight(torch.randn(24, 32, dtype=torch.bfloat16) * 0.1, "tensor")
    input_scale = torch.tensor([float(x.abs().max()) * 1.25 / fp8.FP8_MAX], dtype=torch.float32)

    def forward_impl(input, weight, bias, **kwargs):
        assert input.dtype == torch.float8_e4m3fn
        assert float(input.float().abs().max()) <= fp8.FP8_MAX
        return torch.nn.functional.linear(input.float(), weight.float())

    def no_reductions(*args, **kwargs):
        raise AssertionError("static path must not take the dynamic absmax")

    monkeypatch.setattr(tq, "quantize_activation_per_tensor", no_reductions)
    layer = SimpleNamespace(weight=q_weight, scale=w_scale, input_scale=input_scale,
                            clamp_bound=float("inf"), _forward_impl=forward_impl)
    out = tq.dynamic_fp8_linear(layer, x)
    ref = fp8.fp8_linear_reference(x, q_weight, w_scale, None, input_scale=input_scale)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=1e-3)
    # Beyond the calibrated range: saturates instead of overflowing to 256 / NaN.
    big = x * 8
    out_big = tq.dynamic_fp8_linear(layer, big)
    assert torch.isfinite(out_big.float()).all()


def _static_layer():
    import torch

    from difflet.quant import fp8

    torch.manual_seed(2)
    weight = torch.randn(24, 32, dtype=torch.bfloat16) * 0.1
    q_weight, scale = fp8.quantize_weight(weight, "tensor")

    def forward_impl(input, weight, bias, **kwargs):
        return torch.nn.functional.linear(input.float(), weight.float())

    return SimpleNamespace(weight=q_weight, scale=scale, clamp_bound=float("inf"),
                           input_scale=torch.tensor([3.0 / 240.0]), _forward_impl=forward_impl)


def test_dynamic_fp8_linear_folds_the_bias_into_the_dequantize():
    import torch

    layer = _static_layer()
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    bias = torch.randn(24, dtype=torch.bfloat16)
    fused = tq.dynamic_fp8_linear(layer, x, bias=bias)
    separate = tq.dynamic_fp8_linear(layer, x) + bias
    assert fused.dtype == torch.bfloat16
    assert torch.allclose(fused.float(), separate.float(), rtol=2e-2, atol=2e-2)


def test_dynamic_fp8_linear_defers_the_scale_for_a_post_reduce_epilogue():
    import torch

    layer = _static_layer()
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    raw, combined = tq.dynamic_fp8_linear(layer, x, defer_scale=True)
    assert combined.numel() == 1
    # The deferred path casts the raw dot output to bf16 before the caller's all-reduce
    # (the 2026-10-05 fused-bias NaN fix), so it differs from the in-line dequant by one
    # bf16 rounding of the unscaled output.
    assert raw.dtype == torch.bfloat16
    assert torch.allclose((raw.float() * combined).to(torch.bfloat16).float(),
                          tq.dynamic_fp8_linear(layer, x).float(), rtol=1.6e-2, atol=1e-3)


def test_fp8_targets_env_override(monkeypatch):
    from difflet.quant.targets import WAN_TARGETS, targets_for

    assert targets_for("wan") == WAN_TARGETS
    monkeypatch.setenv("DIFFLET_FP8_TARGETS", "ffn.net_in, ffn.net_out")
    assert targets_for("wan") == ("ffn.net_in", "ffn.net_out")


def test_dynamic_fp8_linear_row_padding_is_exact(monkeypatch):
    import torch

    layer = _static_layer()
    x = torch.randn(1, 300, 32, dtype=torch.bfloat16)
    plain = tq.dynamic_fp8_linear(layer, x)
    monkeypatch.setattr(tq, "_PAD_ROWS", 512)
    padded = tq.dynamic_fp8_linear(layer, x)
    assert padded.shape == plain.shape == (1, 300, 24)
    assert torch.equal(padded, plain)


def test_dynamic_fp8_linear_bf16_dequant_matches_the_fp32_dequant(monkeypatch):
    """``DIFFLET_FP8_BF16_DEQUANT=1`` casts the dot output to the activation dtype and
    applies the combined scale as one bf16 multiply (2026-10-05 speed switch: the fp32
    dequant ACTIVATE was 48.5 ms of the Wan 2.1 DiT step). Same values as the fp32
    dequant up to one bf16 rounding; the switch is read at import time, so flip the
    module constant directly."""
    import torch

    layer = _static_layer()
    x = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    reference = tq.dynamic_fp8_linear(layer, x)
    monkeypatch.setattr(tq, "_BF16_DEQUANT", True)
    out = tq.dynamic_fp8_linear(layer, x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    # One extra bf16 rounding of the raw dot output (|rel| <= 2^-8) plus the scale cast.
    assert torch.allclose(out.float(), reference.float(), rtol=1.6e-2, atol=1e-3)
    # The fused-bias and deferred-scale epilogues are unaffected by the switch.
    bias = torch.randn(24, dtype=torch.bfloat16)
    fused = tq.dynamic_fp8_linear(layer, x, bias=bias)
    assert torch.allclose(fused.float(), (reference + bias).float(), rtol=2e-2, atol=2e-2)
    raw, combined = tq.dynamic_fp8_linear(layer, x, defer_scale=True)
    assert raw.dtype == torch.bfloat16 and combined.numel() == 1
