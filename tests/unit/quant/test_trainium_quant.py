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
    }
    weight_only = tq.neuron_config_kwargs(
        QuantSpec(weight_granularity="channel", activation="none"), "/q"
    )
    assert weight_only["quantization_type"] == "per_channel_symmetric"
    assert "activation_quantization_type" not in weight_only


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

    def convert(model, q_config, inplace, mapping, modules_to_not_convert):
        calls.append(dict(model=model, q_config=q_config, inplace=inplace,
                          mapping=mapping, modules_to_not_convert=modules_to_not_convert))
        return model

    cfg = types.ModuleType("neuronx_distributed.quantization.quantization_config")
    cfg.QuantizationType = QuantizationType
    cfg.QuantizedDtype = QuantizedDtype
    cfg.ActivationQuantizationType = ActivationQuantizationType
    cfg.get_default_custom_qconfig_dict = lambda: {"quantization_type": "per_tensor", "default": True}
    cfg.get_default_per_channel_custom_qconfig_dict = lambda: {"quantization_type": "per_channel", "default": True}
    quantize = types.ModuleType("neuronx_distributed.quantization.quantize")
    quantize.convert = convert

    class ColumnParallelLinear:  # the float layers convert() maps from
        pass

    class RowParallelLinear:
        pass

    import torch

    class QuantizedColumnParallel:  # NxD's quantized forms Difflet subclasses
        @classmethod
        def from_float(cls, mod, q_config=None):
            # NxD builds the quantized layer from mod.dtype (construction-time dtype)
            new = QuantizedColumnParallel()
            new.dtype = mod.dtype
            new.dequantized_dtype = mod.dtype
            new.bias = torch.nn.Parameter(torch.zeros(4, dtype=mod.dtype), requires_grad=False)
            return new

    class QuantizedRowParallel(QuantizedColumnParallel):
        pass

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

    weight_only = tq.build_q_config(SimpleNamespace(
        quantization_type="per_channel_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type=None, quantize_clamp_bound=100.0,
    ))
    assert weight_only["quantization_type"] == "per_channel"
    assert weight_only["activation_quantization_type"] is ActivationQuantizationType.NONE
    assert weight_only["clamp_bound"] == 100.0

    with pytest.raises(ValueError):
        tq.build_q_config(SimpleNamespace(quantization_type="blockwise_symmetric",
                                          quantization_dtype="f8e4m3",
                                          activation_quantization_type=None))


def test_quantize_traced_model_calls_convert_with_difflet_layers(fake_nxd):
    """convert() must map NxD's float layers onto Difflet's subclasses of the
    NxD quantized layers (the ones with the per-tensor DYNAMIC forward)."""
    model = SimpleNamespace(name="wan")
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type="dynamic", quantize_clamp_bound=float("inf"),
        modules_to_not_convert=["proj_out"],
    )
    assert tq.quantize_traced_model_(model, neuron_config) is model
    (call,) = fake_nxd.calls
    assert call["model"] is model and call["inplace"] is True
    assert call["modules_to_not_convert"] == ["proj_out"]
    assert call["q_config"]["quantized_dtype"].value == "f8e4m3"
    column, row, q_column, q_row = fake_nxd.layers
    assert set(call["mapping"]) == {column, row}
    assert issubclass(call["mapping"][column], q_column) and call["mapping"][column] is not q_column
    assert issubclass(call["mapping"][row], q_row) and call["mapping"][row] is not q_row
    assert tq.quant_module_mapping() is call["mapping"]  # built once


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
    assert clamped_scale.item() == pytest.approx(1.0 / fp8.FP8_MAX)
    assert clamped_q.float().abs().max() == fp8.FP8_MAX  # saturated at the clamp


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
    ref = fp8.fp8_linear_reference(x, q_weight, scale, None, activation="dynamic")
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
