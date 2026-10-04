"""Per-model FP8 target sets and the glob-aware target matching."""

from __future__ import annotations

import argparse

import pytest

from difflet.quant.spec import QuantSpec
from difflet.quant.targets import TARGETS_BY_MODEL, targets_for


def test_every_wired_model_has_a_target_set():
    assert set(TARGETS_BY_MODEL) == {"wan", "flux", "qwen_image", "hunyuan_video", "ltx_2"}
    for model_type, targets in TARGETS_BY_MODEL.items():
        assert targets and all(isinstance(t, str) and t for t in targets), model_type


def test_unknown_model_type_lists_the_wired_ones():
    with pytest.raises(ValueError, match="hunyuan_video_15.*wan, flux"):
        targets_for("hunyuan_video_15")


def test_glob_targets_exclude_root_and_refiner():
    flux = QuantSpec.for_model("flux")
    assert flux.matches("transformer_blocks.3.attn.to_q")
    assert flux.matches("transformer_blocks.3.attn.add_q_proj")
    assert flux.matches("transformer_blocks.3.ff_context.net.2")
    assert flux.matches("single_transformer_blocks.7.proj_mlp")
    assert flux.matches("single_transformer_blocks.7.proj_out")
    assert not flux.matches("proj_out")  # root projection stays bf16
    assert not flux.matches("norm_out.linear")
    assert not flux.matches("time_text_embed.timestep_embedder.linear_1")

    hv = QuantSpec.for_model("hunyuan_video")
    assert hv.matches("transformer_blocks.0.attn.to_q")
    assert hv.matches("single_transformer_blocks.0.proj_out")
    assert not hv.matches("context_embedder.token_refiner.refiner_blocks.0.attn.to_q")
    assert not hv.matches("proj_out")
    # The text-stream q / k projections stay bf16: with them in fp8 the tp4 device graph
    # returns all-NaN (trn2, 2026-10-04, real weights); v and the other text layers are fine.
    assert not hv.matches("transformer_blocks.0.attn.add_q_proj")
    assert not hv.matches("transformer_blocks.0.attn.add_k_proj")
    assert hv.matches("transformer_blocks.0.attn.add_v_proj")
    assert hv.matches("transformer_blocks.0.attn.to_add_out")
    assert hv.matches("transformer_blocks.0.ff_context.net.0.proj")
    assert flux.matches("transformer_blocks.0.attn.add_k_proj")  # FLUX is unaffected

    ltx = QuantSpec.for_model("ltx_2")
    assert ltx.matches("transformer.transformer_blocks.1.audio_to_video_attn.to_k")
    assert ltx.matches("transformer.transformer_blocks.1.audio_ff.net.2")
    assert not ltx.matches("transformer.proj_out")

    qwen = QuantSpec.for_model("qwen_image")
    assert qwen.matches("transformer.transformer_blocks.2.img_mlp.net.0.proj")
    assert not qwen.matches("transformer.transformer_blocks.2.img_mod.1")


def test_wan_targets_are_the_default_and_still_suffix_matched():
    wan = QuantSpec.for_model("wan")
    assert wan.targets == QuantSpec().targets
    assert wan.matches("blocks.3.attn1.to_q") and wan.matches("blocks.3.ffn.net_in")
    assert not wan.matches("proj_out")


def test_per_model_checkpoint_identities_follow_the_target_sets():
    # Every distinct target set hashes differently (HunyuanVideo = FLUX's layout minus the
    # text q / k, so its own set), and the checkpoint dir is keyed by the source model path
    # on top, so no model can pick up another's checkpoint.
    ids = {m: QuantSpec.for_model(m).checkpoint_hash("/src") for m in TARGETS_BY_MODEL}
    distinct_sets = {tuple(t) for t in TARGETS_BY_MODEL.values()}
    assert len(set(ids.values())) == len(distinct_sets) == 5
    assert len(set(ids.values())) == len(ids)


def test_from_args_uses_the_model_targets():
    args = argparse.Namespace(quant="fp8", quant_granularity="tensor")
    assert QuantSpec.from_args(args, model_type="flux").targets == targets_for("flux")
    assert QuantSpec.from_args(args).targets == targets_for("wan")  # default unchanged
