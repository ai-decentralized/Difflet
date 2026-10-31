"""QuantSpec: validation, matching, serialisation and the cache fields (torch-free)."""

from __future__ import annotations

import dataclasses

import pytest

from difflet.quant.spec import FP8_MAX, QUANT_SCHEMA, QuantSpec, quant_cache_fields
from difflet.quant.targets import QWEN_IMAGE_KEPT_BF16, QWEN_IMAGE_TARGETS, default_spec, targets_for


def test_default_qwen_spec_is_fp8_static_per_tensor():
    spec = default_spec("qwen_image")
    assert spec.format == "fp8" and spec.fp8_max == FP8_MAX == 448.0
    assert spec.weight_granularity == "tensor" and spec.activation == "static"
    assert spec.compute == "nki" and spec.schema == QUANT_SCHEMA
    assert spec.targets == QWEN_IMAGE_TARGETS


def test_unknown_model_has_no_targets():
    with pytest.raises(ValueError, match="no FP8 target layers"):
        targets_for("minimax_h3")


@pytest.mark.parametrize("name", [
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.59.attn.add_k_proj",
    "transformer_blocks.7.img_mlp.net.0.proj",
    "transformer_blocks.7.txt_mlp.net.2",
    "transformer.transformer_blocks.7.txt_mlp.net.2",  # the trace-module prefix is stripped
])
def test_qwen_targets_match_the_block_projections(name):
    assert default_spec("qwen_image").matches(name)


@pytest.mark.parametrize("name", [
    "transformer_blocks.0.img_mod.1",
    "transformer_blocks.0.txt_mod.1",
    "time_text_embed.timestep_embedder.linear_1",
    "img_in",
    "txt_in",
    "norm_out.linear",
    "proj_out",
    "transformer_blocks.3.attn.to_out.0",  # measured: faster and closer to bf16 when kept bf16
    "transformer_blocks.3.attn.to_add_out",
    "transformer_blocks.0.attn.to_out",  # the Sequential, not its linear
    "transformer_blocks.0.attn.to_q.weight",  # a parameter, not a module
])
def test_qwen_targets_keep_conditioning_and_io_in_bf16(name):
    import re

    spec = default_spec("qwen_image")
    assert not spec.matches(name)
    if name in ("transformer_blocks.0.attn.to_out", "transformer_blocks.0.attn.to_q.weight"):
        return
    assert any(re.fullmatch(p, name) for p in QWEN_IMAGE_KEPT_BF16), name


def test_dict_round_trip_and_frozen():
    spec = default_spec("qwen_image", calibration_sha256="ab" * 32, compute="dequant")
    assert QuantSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.compute = "nki"  # type: ignore[misc]


@pytest.mark.parametrize("field, value, message", [
    ("format", "int8", "quant format"),
    ("weight_granularity", "channel", "weight_granularity"),
    ("activation", "dynamic", "activation must"),
    ("compute", "torchao", "compute must"),
    ("fp8_max", 0.0, "fp8_max"),
    ("activation_margin", 0.5, "activation_margin"),
    ("targets", (), "at least one"),
    ("targets", ("(",), None),  # an invalid regex raises re.error
])
def test_invalid_specs_are_refused(field, value, message):
    kwargs = {"targets": QWEN_IMAGE_TARGETS, field: value}
    with pytest.raises(Exception, match=message) if message else pytest.raises(Exception):
        QuantSpec(**kwargs)


def test_cache_fields_absent_when_off_and_complete_when_on():
    assert quant_cache_fields(None) == {}
    spec = default_spec("qwen_image", calibration_sha256="cd" * 32)
    fields = quant_cache_fields(spec)
    assert fields["quant"] == "fp8" and fields["quant_schema"] == QUANT_SCHEMA
    assert fields["quant_compute"] == "nki" and fields["quant_fp8_max"] == 448.0
    assert fields["quant_calibration_sha256"] == "cd" * 32
    assert all(k.startswith("quant") for k in fields)
    # Every field is JSON-able (the keys are hashed as JSON).
    import json

    json.dumps(fields, sort_keys=True)


def test_cache_fields_change_with_what_changes_the_artifact():
    base = default_spec("qwen_image", calibration_sha256="a" * 64)
    variants = [
        dataclasses.replace(base, compute="dequant"),
        dataclasses.replace(base, calibration_sha256="b" * 64),
        dataclasses.replace(base, fp8_max=240.0),
        dataclasses.replace(base, activation_margin=1.5),
        dataclasses.replace(base, targets=base.targets[:1]),
    ]
    keys = {str(sorted(v.cache_fields().items())) for v in [base, *variants]}
    assert len(keys) == len(variants) + 1


def test_weights_identity_ignores_the_compute_path():
    a = default_spec("qwen_image", compute="nki")
    b = default_spec("qwen_image", compute="dequant")
    assert a.weights_identity() == b.weights_identity()
    assert a.cache_fields() != b.cache_fields()
