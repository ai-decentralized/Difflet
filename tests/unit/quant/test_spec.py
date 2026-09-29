"""QuantSpec: validation, matching, serialization, CLI round trip, identity."""

from __future__ import annotations

import argparse

import pytest

from difflet.quant.spec import DEFAULT_TARGETS, QuantSpec


def test_defaults_mirror_fastvideo_fp8_config():
    spec = QuantSpec()
    assert spec.format == "fp8_e4m3"
    assert spec.weight_granularity == "tensor"
    assert spec.activation == "dynamic"
    assert spec.targets == DEFAULT_TARGETS
    assert spec.label() == "fp8-tensor-dyn"
    assert QuantSpec(weight_granularity="channel", activation="none").label() == "fp8-channel-wo"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int8"},
        {"weight_granularity": "token"},
        {"activation": "static"},
        {"targets": ()},
        {"targets": ("",)},
    ],
)
def test_rejects_unknown_values(kwargs):
    with pytest.raises(ValueError):
        QuantSpec(**kwargs)


def test_matches_targets_by_dotted_suffix_only():
    spec = QuantSpec()
    assert spec.matches("blocks.0.attn1.to_q")
    assert spec.matches("blocks.3.attn2.to_out.0")
    assert spec.matches("blocks.1.ffn.net_in")
    assert spec.matches("blocks.1.ffn.net.0.proj")  # diffusers spelling (HF checkpoint)
    assert spec.matches("blocks.1.ffn.net.2")
    assert spec.matches("to_q")
    # Not targets: embedders, modulation, output head, and look-alike names.
    assert not spec.matches("condition_embedder.time_embedder.linear_1")
    assert not spec.matches("proj_out")
    assert not spec.matches("blocks.0.attn1.to_out.1")
    assert not spec.matches("blocks.0.attn1.my_to_q")


def test_dict_round_trip_and_coerce():
    spec = QuantSpec(weight_granularity="channel", activation="none", targets=("to_q",))
    data = spec.to_dict()
    assert data == {
        "format": "fp8_e4m3",
        "weight_granularity": "channel",
        "activation": "none",
        "targets": ["to_q"],
    }
    assert QuantSpec.from_dict(data) == spec
    assert QuantSpec.coerce(None) is None
    assert QuantSpec.coerce(spec) is spec
    assert QuantSpec.coerce(data) == spec
    with pytest.raises(TypeError):
        QuantSpec.coerce("fp8")  # type: ignore[arg-type]


def test_cli_args_round_trip():
    args = argparse.Namespace(quant="fp8", quant_granularity="channel", quant_act="none")
    spec = QuantSpec.from_args(args)
    assert spec == QuantSpec(weight_granularity="channel", activation="none")
    assert spec.cli_args() == [
        "--quant", "fp8", "--quant-granularity", "channel", "--quant-act", "none",
    ]
    assert QuantSpec.from_args(argparse.Namespace(quant=None)) is None
    assert QuantSpec.from_args(argparse.Namespace()) is None
    with pytest.raises(ValueError):
        QuantSpec.from_args(argparse.Namespace(quant="int4"))


def test_checkpoint_identity_ignores_activation_mode():
    dyn = QuantSpec(activation="dynamic")
    wo = QuantSpec(activation="none")
    assert dyn.checkpoint_identity() == wo.checkpoint_identity()
    assert dyn.checkpoint_label() == "fp8-tensor"
    assert dyn.checkpoint_hash("/a") == wo.checkpoint_hash("/a")
    assert dyn.checkpoint_hash("/a") != dyn.checkpoint_hash("/b")
    assert dyn.checkpoint_hash("/a") != QuantSpec(weight_granularity="channel").checkpoint_hash("/a")
    assert len(dyn.checkpoint_hash()) == 8
