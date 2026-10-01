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
    root = types.ModuleType("neuronx_distributed")
    pkg = types.ModuleType("neuronx_distributed.quantization")
    for name, module in {
        "neuronx_distributed": root,
        "neuronx_distributed.quantization": pkg,
        "neuronx_distributed.quantization.quantization_config": cfg,
        "neuronx_distributed.quantization.quantize": quantize,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(calls=calls, types=(QuantizationType, QuantizedDtype, ActivationQuantizationType))


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


def test_quantize_traced_model_calls_convert_in_place(fake_nxd):
    model = SimpleNamespace(name="wan")
    neuron_config = SimpleNamespace(
        quantized=True, quantization_type="per_tensor_symmetric", quantization_dtype="f8e4m3",
        activation_quantization_type="dynamic", quantize_clamp_bound=float("inf"),
        modules_to_not_convert=["proj_out"],
    )
    assert tq.quantize_traced_model_(model, neuron_config) is model
    (call,) = fake_nxd.calls
    assert call["model"] is model and call["inplace"] is True and call["mapping"] is None
    assert call["modules_to_not_convert"] == ["proj_out"]
    assert call["q_config"]["quantized_dtype"].value == "f8e4m3"


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
