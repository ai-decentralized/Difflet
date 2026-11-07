"""The LTX-2 MXFP8 target table against diffusers' own module names: the video- and
text-input linears are quantized, the audio stream and every output projection stay bf16,
and the row-parallel set is what the TP layer shards along K."""

from __future__ import annotations

import argparse

import pytest
import torch

from difflet.cli.quant import QUANT_MODELS, quant_spec
from difflet.quant.targets import (
    LTX_2_KEPT_BF16,
    LTX_2_ROW_PARALLEL,
    LTX_2_TARGETS,
    ROW_PARALLEL_BY_MODEL,
    TARGETS_BY_MODEL,
    default_spec,
)

pytest.importorskip("diffusers")

from tests.unit.backends._neuron_ltx2_toy import TINY_FP8_LTX2  # noqa: E402

#: Per block: 3 q/k/v for the two video attentions, the a2v query, the v2a key and value,
#: and the two video MLP linears.
TARGETS_PER_BLOCK = 11


@pytest.fixture(scope="module")
def model():
    from diffusers import LTX2VideoTransformer3DModel

    return LTX2VideoTransformer3DModel(**TINY_FP8_LTX2)


def _linears(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def test_default_targets_cover_the_video_and_text_input_linears_only(model):
    spec = default_spec("ltx_2")
    names = [n for n in _linears(model) if spec.matches(n)]
    per_block = {n.split(".", 2)[2] for n in names}
    assert per_block == {"attn1.to_q", "attn1.to_k", "attn1.to_v",
                         "attn2.to_q", "attn2.to_k", "attn2.to_v",
                         "audio_to_video_attn.to_q",
                         "video_to_audio_attn.to_k", "video_to_audio_attn.to_v",
                         "ff.net.0.proj", "ff.net.2"}
    assert len(names) == TARGETS_PER_BLOCK * TINY_FP8_LTX2["num_layers"]
    assert all(spec.is_row_parallel(n) == n.endswith(("to_out.0", "ff.net.2"))
               for n, _ in model.named_modules())
    assert not any(spec.matches(n) for n in (
        "proj_out", "caption_projection.linear_1", "transformer_blocks.0.audio_ff.net.2"))


def test_every_linear_is_a_target_or_documented_bf16(model):
    """The kept-bf16 table is documentation: with the targets it covers every linear of the
    model exactly once, so a reader can tell each layer's precision from the table alone."""
    target = default_spec("ltx_2")
    kept = default_spec("ltx_2", targets=LTX_2_KEPT_BF16)
    for name in _linears(model):
        assert target.matches(name) != kept.matches(name), name


def test_the_trace_prefix_is_stripped():
    spec = default_spec("ltx_2", tp_degree=4)
    assert spec.matches("transformer.transformer_blocks.47.attn1.to_q")
    assert spec.k_shards("transformer.transformer_blocks.47.ff.net.2") == 4
    assert spec.k_shards("transformer.transformer_blocks.47.audio_attn1.to_out.0") == 4
    assert spec.k_shards("transformer.transformer_blocks.47.attn1.to_q") == 1


def test_the_registries_and_the_cli_id_name_the_table():
    assert TARGETS_BY_MODEL["ltx_2"] is LTX_2_TARGETS
    assert ROW_PARALLEL_BY_MODEL["ltx_2"] is LTX_2_ROW_PARALLEL
    assert QUANT_MODELS["Lightricks/LTX-2"] == "ltx_2"
    args = argparse.Namespace(quant="mxfp8", quant_compute=None, model_id="Lightricks/LTX-2",
                              tp_degree=None)
    assert quant_spec(args) == default_spec("ltx_2", compute="nki", tp_degree=4)
