"""Benchmark harness: FP8 PTQ knobs and the Wan per-step log parser (pure functions)."""

from __future__ import annotations

from benchmark.adapters.trainium import parse_dit_step_seconds, spec_slug
from benchmark.models import MATRIX, BenchConfig


def test_quant_flags_and_slug_suffix():
    bf16 = BenchConfig(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", model_type="wan")
    assert bf16.quant_flags() == [] and bf16.quant_dict() is None and bf16.slug_suffix() == ""
    assert spec_slug(bf16) == "wan2_1_t2v_14b_diffusers"

    fp8 = BenchConfig(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", model_type="wan",
                      quant="fp8", quant_granularity="channel")
    assert fp8.quant_flags() == ["--quant", "fp8", "--quant-granularity", "channel"]
    assert fp8.quant_dict() == {"format": "fp8", "weight_granularity": "channel"}
    assert spec_slug(fp8) == "wan2_1_t2v_14b_diffusers_fp8_channel"
    assert not hasattr(fp8, "quant_act")  # weight-only removed 2026-10-03


def test_fp8_partners_mirror_their_bf16_entry():
    for slug in ("flux_1_dev", "qwen_image", "hunyuan_video", "ltx_2", "wan_2_1", "wan_2_2"):
        base = MATRIX[slug]
        fp8 = MATRIX[slug + "_fp8"]
        assert slug + "_fp8_wo" not in MATRIX
        assert fp8.quant == "fp8" and fp8.quant_granularity == "tensor"
        assert spec_slug(fp8) != spec_slug(base)
        for field in ("model_id", "revision", "model_type", "tp", "cp", "sp", "height", "width",
                      "num_frames", "steps", "guidance_scale", "seed", "prompt", "output_kind"):
            assert getattr(base, field) == getattr(fp8, field), (slug, field)
        assert "FP8 PTQ" in fp8.config_label


def test_parse_dit_step_seconds_drops_step_zero_and_takes_the_last_loop():
    log = (
        "noise\n[wan] dit-step-seconds: [0.90, 0.55, 0.56]\n[wan] dit-step ms: n=2 ...\n"
        "[wan] dit-step-seconds: [0.80, 0.50, 0.51, 0.52]\n"
    )
    assert parse_dit_step_seconds(log) == [0.50, 0.51, 0.52]
    assert parse_dit_step_seconds("no timing here") == []
    assert parse_dit_step_seconds("[wan] dit-step-seconds: []") == []


def test_fp8_static_partners_carry_their_calibration_file():
    # The static arms (calibrated per-tensor activation scales) mirror the bf16 entry
    # like the dynamic ones and point at the model's calibration JSON in the evidence tree.
    from benchmark.models import CALIBRATION_FILES

    for slug in ("flux_1_dev", "qwen_image", "hunyuan_video", "ltx_2", "wan_2_1", "wan_2_2"):
        base, static = MATRIX[slug], MATRIX[slug + "_fp8_static"]
        assert static.quant == "fp8" and static.quant_granularity == "tensor"
        assert static.quant_calibration == CALIBRATION_FILES[slug]
        assert static.quant_calibration.startswith("artifacts/verification-2026-10-03/") and static.quant_calibration.endswith(".json")
        assert spec_slug(static) == spec_slug(base) + "_fp8_tensor_static"
        assert "--quant-calibration" in static.quant_flags() and static.quant_dict()["calibration"] == static.quant_calibration
        for field in ("model_id", "revision", "model_type", "tp", "cp", "sp", "height", "width",
                      "num_frames", "steps", "guidance_scale", "seed", "prompt", "output_kind"):
            assert getattr(base, field) == getattr(static, field), (slug, field)
        assert "static" in static.config_label
