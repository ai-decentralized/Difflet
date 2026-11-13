"""QuantSpec: validation, matching, serialisation and the cache fields (torch-free)."""

from __future__ import annotations

import dataclasses
import re

import pytest

from difflet.quant.spec import FORMAT, QUANT_SCHEMA, QuantSpec, quant_cache_fields
from difflet.quant.targets import (
    QWEN_IMAGE_KEPT_BF16,
    QWEN_IMAGE_TARGETS,
    SINGLE_BLOCK_SPLIT_LINEARS,
    default_spec,
    targets_for,
)


def test_default_qwen_spec_is_mxfp8_on_the_kernel():
    spec = default_spec("qwen_image")
    assert spec.format == FORMAT == "mxfp8"
    assert spec.compute == "nki" and spec.schema == QUANT_SCHEMA and spec.tp_degree == 1
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


def test_unmatched_targets_names_the_dead_patterns():
    """A pattern that full-matches none of the names is reported by itself, so a spec with
    one live and one dead pattern is refused naming the dead one instead of quantizing less
    than it claims; the trace-module prefix is stripped from the names as in ``matches``."""
    spec = QuantSpec(
        targets=(r"blocks\.\d+\.ffn\.net\.0\.proj", r"blocks\.\d+\.attn\.to_q", "zzz"))
    names = ["transformer.blocks.0.ffn.net.0.proj", "blocks.1.ffn.net.0.proj",
             "blocks.0.proj_out"]
    assert spec.unmatched_targets(names) == [r"blocks\.\d+\.attn\.to_q", "zzz"]
    assert spec.unmatched_targets(iter(names + ["blocks.0.attn.to_q"])) == ["zzz"]
    assert spec.unmatched_targets([]) == list(spec.targets)


def test_row_parallel_targets_are_quantized_per_rank_k_range():
    spec = default_spec("qwen_image", tp_degree=4)
    assert spec.k_shards("transformer_blocks.3.attn.to_out.0") == 4  # row-parallel even when not a target
    assert spec.k_shards("transformer.transformer_blocks.3.img_mlp.net.2") == 4
    assert spec.k_shards("transformer_blocks.3.attn.to_q") == 1
    assert spec.k_shards("transformer_blocks.3.img_mlp.net.0.proj") == 1
    assert default_spec("qwen_image").k_shards("transformer_blocks.3.img_mlp.net.2") == 1


def test_dict_round_trip_and_frozen():
    spec = default_spec("qwen_image", compute="dequant", tp_degree=2)
    body = spec.to_dict()
    assert body["format"] == "mxfp8"
    assert QuantSpec.from_dict(body) == spec
    assert QuantSpec.from_dict({k: v for k, v in body.items() if k != "format"}) == spec
    with pytest.raises(ValueError, match="quant format"):
        QuantSpec.from_dict({**body, "format": "fp8"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.compute = "nki"  # type: ignore[misc]


@pytest.mark.parametrize("field, value, message", [
    ("compute", "torchao", "compute must"),
    ("tp_degree", 0, "tp_degree"),
    ("targets", (), "at least one"),
    ("targets", ("(",), None),  # an invalid regex raises re.error
])
def test_invalid_specs_are_refused(field, value, message):
    kwargs = {"targets": QWEN_IMAGE_TARGETS, field: value}
    with pytest.raises(Exception, match=message) if message else pytest.raises(Exception):
        QuantSpec(**kwargs)


def test_cache_fields_absent_when_off_and_complete_when_on():
    assert quant_cache_fields(None) == {}
    spec = default_spec("qwen_image", tp_degree=4)
    fields = quant_cache_fields(spec)
    assert fields == {"quant": "mxfp8", "quant_schema": QUANT_SCHEMA, "quant_compute": "nki",
                      "quant_targets_sha256": spec.targets_sha256(), "quant_tp_degree": 4}
    # Every field is JSON-able (the keys are hashed as JSON).
    import json

    json.dumps(fields, sort_keys=True)


def test_cache_fields_change_with_what_changes_the_artifact():
    base = default_spec("qwen_image", tp_degree=4)
    variants = [
        dataclasses.replace(base, compute="dequant"),
        dataclasses.replace(base, tp_degree=2),
        dataclasses.replace(base, targets=base.targets[:1]),
    ]
    keys = {str(sorted(v.cache_fields().items())) for v in [base, *variants]}
    assert len(keys) == len(variants) + 1


def test_weights_identity_ignores_the_compute_path():
    a = default_spec("qwen_image", compute="nki")
    b = default_spec("qwen_image", compute="dequant")
    assert a.weights_identity() == b.weights_identity()
    assert a.cache_fields() != b.cache_fields()
    assert a.weights_identity() != dataclasses.replace(a, tp_degree=2).weights_identity()


def test_split_linears_stay_out_of_every_unsplit_spec():
    """A spec without splits serialises and identifies its weights as before the field existed,
    so the Qwen-Image, LTX-2 and Wan directories keep their digests; a split changes the bytes,
    not the compiled graph, so it enters the weights identity and no cache field."""
    for model in ("qwen_image", "ltx_2", "wan"):
        spec = default_spec(model, tp_degree=4)
        assert spec.split_linears == ()
        assert "split_linears" not in spec.to_dict()
        assert "quant_split_linears" not in spec.weights_identity()
    spec = QuantSpec(targets=(r"single_transformer_blocks\.\d+\.proj_mlp",),
                     split_linears=SINGLE_BLOCK_SPLIT_LINEARS)
    assert spec.to_dict()["split_linears"] == list(SINGLE_BLOCK_SPLIT_LINEARS)
    assert QuantSpec.from_dict(spec.to_dict()) == spec
    assert spec.is_split("transformer.single_transformer_blocks.3.proj_out")
    assert not spec.is_split("single_transformer_blocks.3.proj_out_attn")
    unsplit = dataclasses.replace(spec, split_linears=())
    assert spec.weights_identity() != unsplit.weights_identity()
    assert spec.cache_fields() == unsplit.cache_fields()
    assert spec.unmatched_splits(["single_transformer_blocks.0.proj_out"]) == []
    assert spec.unmatched_splits(["single_transformer_blocks.0.proj_out_attn"]) == list(
        SINGLE_BLOCK_SPLIT_LINEARS)
    with pytest.raises(re.error):
        QuantSpec(targets=QWEN_IMAGE_TARGETS, split_linears=("(",))


@pytest.mark.parametrize("name", [
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.18.attn.add_v_proj",
    "transformer_blocks.3.ff.net.0.proj",
    "transformer_blocks.3.ff_context.net.2",
    "single_transformer_blocks.37.attn.to_k",
    "single_transformer_blocks.0.proj_mlp",
])
def test_flux_targets_match_the_block_projections(name):
    assert default_spec("flux").matches(name)


@pytest.mark.parametrize("name", [
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.0.attn.to_add_out",
    "single_transformer_blocks.0.proj_out",
    "single_transformer_blocks.0.proj_out_attn",
    "single_transformer_blocks.0.proj_out_mlp",
    "transformer_blocks.0.norm1.linear",
    "transformer_blocks.0.norm1_context.linear",
    "single_transformer_blocks.0.norm.linear",
    "time_text_embed.timestep_embedder.linear_1",
    "context_embedder",
    "x_embedder",
    "norm_out.linear",
    "proj_out",
])
def test_flux_targets_keep_outputs_conditioning_and_io_in_bf16(name):
    import re

    from difflet.quant.targets import FLUX_KEPT_BF16

    spec = default_spec("flux", tp_degree=4)
    assert not spec.matches(name)
    assert any(re.fullmatch(p, name) for p in FLUX_KEPT_BF16), name
    assert spec.k_shards("transformer_blocks.0.ff.net.2") == 4
    assert spec.k_shards("transformer_blocks.0.ff.net.0.proj") == 1


def test_flux_splits_its_proj_out_and_the_halves_are_row_parallel_override_targets():
    """FLUX's spec stores the single blocks' ``proj_out`` as its halves from now on, but the
    default set does not quantize them until the device picks it; an override that names them
    gets them per rank K range of each half."""
    from difflet.quant.targets import FLUX_SPLIT_TARGETS, FLUX_TARGETS

    spec = default_spec("flux", tp_degree=4)
    assert spec.split_linears == SINGLE_BLOCK_SPLIT_LINEARS
    assert spec.is_split("single_transformer_blocks.0.proj_out")
    assert not spec.matches("single_transformer_blocks.0.proj_out_attn")
    assert spec.k_shards("single_transformer_blocks.0.proj_out_attn") == 4
    assert spec.k_shards("single_transformer_blocks.37.proj_out_mlp") == 4
    d = default_spec("flux", tp_degree=4, targets=FLUX_TARGETS + FLUX_SPLIT_TARGETS)
    assert d.matches("single_transformer_blocks.0.proj_out_attn")
    assert d.matches("single_transformer_blocks.37.proj_out_mlp")
    assert not d.matches("single_transformer_blocks.0.proj_out")
    assert default_spec("qwen_image").split_linears == ()
