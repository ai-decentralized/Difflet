"""QuantSpec: validation, matching, serialization, CLI round trip, identity."""

from __future__ import annotations

import argparse

import pytest

from difflet.quant.spec import DEFAULT_TARGETS, QuantSpec


def test_defaults_mirror_fastvideo_fp8_config():
    spec = QuantSpec()
    assert spec.format == "fp8_e4m3"
    assert spec.weight_granularity == "tensor"
    assert spec.targets == DEFAULT_TARGETS
    assert spec.label() == "fp8-tensor"
    assert QuantSpec(weight_granularity="channel").label() == "fp8-channel"


def test_activations_are_always_dynamic():
    """Weight-only (``activation="none"``) was removed on 2026-10-03: FP8 PTQ is
    FastVideo's W8A8 scheme only. The field is gone, and legacy dicts / CLI
    namespaces that ask for weight-only fail loudly instead of silently running
    dynamic."""
    with pytest.raises(TypeError):
        QuantSpec(activation="none")  # type: ignore[call-arg]
    assert QuantSpec.from_dict({"format": "fp8_e4m3", "activation": "dynamic"}) == QuantSpec()
    with pytest.raises(ValueError, match="weight-only"):
        QuantSpec.from_dict({"format": "fp8_e4m3", "activation": "none"})
    with pytest.raises(ValueError, match="weight-only"):
        QuantSpec.from_args(argparse.Namespace(quant="fp8", quant_act="none"))
    assert "activation" not in QuantSpec().to_dict()
    assert "--quant-act" not in QuantSpec().cli_args()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "int8"},
        {"weight_granularity": "token"},
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
    spec = QuantSpec(weight_granularity="channel", targets=("to_q",))
    data = spec.to_dict()
    assert data == {
        "format": "fp8_e4m3",
        "weight_granularity": "channel",
        "targets": ["to_q"],
    }
    assert QuantSpec.from_dict(data) == spec
    assert QuantSpec.coerce(None) is None
    assert QuantSpec.coerce(spec) is spec
    assert QuantSpec.coerce(data) == spec
    with pytest.raises(TypeError):
        QuantSpec.coerce("fp8")  # type: ignore[arg-type]


def test_cli_args_round_trip():
    args = argparse.Namespace(quant="fp8", quant_granularity="channel")
    spec = QuantSpec.from_args(args)
    assert spec == QuantSpec(weight_granularity="channel")
    assert spec.cli_args() == ["--quant", "fp8", "--quant-granularity", "channel"]
    assert QuantSpec.from_args(argparse.Namespace(quant=None)) is None
    assert QuantSpec.from_args(argparse.Namespace()) is None
    with pytest.raises(ValueError):
        QuantSpec.from_args(argparse.Namespace(quant="int4"))


def test_checkpoint_identity_is_the_weight_recipe():
    spec = QuantSpec()
    assert spec.checkpoint_identity()["fp8_max"] == 240.0  # Trainium e4m3 range, in the hash
    assert spec.checkpoint_label() == "fp8-tensor"
    assert spec.checkpoint_hash("/a") != spec.checkpoint_hash("/b")
    assert spec.checkpoint_hash("/a") != QuantSpec(weight_granularity="channel").checkpoint_hash("/a")
    assert len(spec.checkpoint_hash()) == 8
