"""Benchmark harness: FP8 PTQ knobs and the Wan per-step log parser (pure functions)."""

from __future__ import annotations

from benchmark.adapters.trainium import parse_dit_step_seconds, spec_slug
from benchmark.models import MATRIX, BenchConfig


def test_quant_flags_and_slug_suffix():
    bf16 = BenchConfig(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", model_type="wan")
    assert bf16.quant_flags() == [] and bf16.quant_dict() is None and bf16.slug_suffix() == ""
    assert spec_slug(bf16) == "wan2_1_t2v_14b_diffusers"

    fp8 = BenchConfig(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", model_type="wan",
                      quant="fp8", quant_granularity="channel", quant_act="none")
    assert fp8.quant_flags() == ["--quant", "fp8", "--quant-granularity", "channel", "--quant-act", "none"]
    assert fp8.quant_dict() == {"format": "fp8", "weight_granularity": "channel", "activation": "none"}
    assert spec_slug(fp8) == "wan2_1_t2v_14b_diffusers_fp8_channel_wo"


def test_fp8_partners_mirror_their_bf16_entry():
    for slug in ("flux_1_dev", "qwen_image", "hunyuan_video", "ltx_2", "wan_2_1", "wan_2_2"):
        base = MATRIX[slug]
        for suffix, act in (("_fp8", "dynamic"), ("_fp8_wo", "none")):
            fp8 = MATRIX[slug + suffix]
            assert fp8.quant == "fp8" and fp8.quant_granularity == "tensor" and fp8.quant_act == act
            assert spec_slug(fp8) != spec_slug(base)
            for field in ("model_id", "revision", "model_type", "tp", "cp", "sp", "height", "width",
                          "num_frames", "steps", "guidance_scale", "seed", "prompt", "output_kind"):
                assert getattr(base, field) == getattr(fp8, field), (slug, suffix, field)
            assert "FP8 PTQ" in fp8.config_label


def test_matrix_has_the_fp8_partners_of_wan_2_1():
    base, fp8, wo = MATRIX["wan_2_1"], MATRIX["wan_2_1_fp8"], MATRIX["wan_2_1_fp8_wo"]
    assert fp8.quant == "fp8" and wo.quant == "fp8" and base.quant is None
    assert fp8.quant_act == "dynamic" and wo.quant_act == "none"
    assert spec_slug(fp8) != spec_slug(wo)  # separate report files
    for partner in (fp8, wo):
        for field in ("model_id", "revision", "tp", "height", "width", "num_frames", "steps", "seed", "prompt"):
            assert getattr(base, field) == getattr(partner, field)


def test_parse_dit_step_seconds_drops_step_zero_and_takes_the_last_loop():
    log = (
        "noise\n[wan] dit-step-seconds: [0.90, 0.55, 0.56]\n[wan] dit-step ms: n=2 ...\n"
        "[wan] dit-step-seconds: [0.80, 0.50, 0.51, 0.52]\n"
    )
    assert parse_dit_step_seconds(log) == [0.50, 0.51, 0.52]
    assert parse_dit_step_seconds("no timing here") == []
    assert parse_dit_step_seconds("[wan] dit-step-seconds: []") == []
