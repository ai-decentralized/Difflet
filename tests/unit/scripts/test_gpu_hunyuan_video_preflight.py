"""CPU checks for geometry and honest SDPA qualification reporting."""

from types import SimpleNamespace
import json

import torch


def test_case_preserves_the_model_layout_and_linear_key_mask():
    from tests.numerical.gpu_hunyuan_video_preflight import make_attention_case

    cfg = SimpleNamespace(
        latent_frames=3,
        latent_height=4,
        latent_width=4,
        patch_size=2,
        text_seq_len=5,
        num_attention_heads=2,
        attention_head_dim=8,
    )
    q, k, v, mask = make_attention_case(cfg, valid_text_tokens=2, device="cpu")
    assert q.shape == k.shape == v.shape == (2, 17, 8)
    assert q.stride() == (8, 16, 1)  # Folded heads after sequence-major QKV projection.
    assert mask.shape == (2, 1, 17) and mask.dtype == torch.bool
    assert mask[0, 0].tolist() == [True] * 14 + [False] * 3


def test_trace_rejects_math_and_unknown_paths_instead_of_claiming_fusion():
    from tests.numerical.gpu_hunyuan_video_preflight import qualify_attention_trace

    assert qualify_attention_trace(["aten::_scaled_dot_product_cudnn_attention"])["qualified"]
    assert qualify_attention_trace(["flash_fwd_kernel"])["qualified"]
    for events in [
        ["aten::bmm"],
        ["aten::_scaled_dot_product_attention_math"],
        ["aten::_scaled_dot_product_flash_attention_for_cpu"],
        ["unknown"],
    ]:
        assert qualify_attention_trace(events)["qualified"] is False


def test_preflight_parser_has_explicit_shape_mode_and_output_directory():
    from tests.numerical.gpu_hunyuan_video_preflight import build_parser

    args = build_parser().parse_args(
        [
            "--height",
            "320",
            "--width",
            "512",
            "--num-frames",
            "61",
            "--mode",
            "both",
            "--output-dir",
            "/tmp/preflight-test",
        ]
    )
    assert (args.height, args.width, args.num_frames, args.mode) == (320, 512, 61, "both")


def test_operator_failure_writes_failed_qualification_before_returning(tmp_path, monkeypatch):
    from tests.numerical import gpu_hunyuan_video_preflight as harness

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic device allocation failure")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
    monkeypatch.setattr(harness, "make_attention_case", fail)
    assert harness.main(["--mode", "eager", "--output-dir", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "preflight.json").read_text())
    assert report["qualified"] is False
    assert report["rows"][0]["qualified"] is False
    assert "allocation failure" in report["rows"][0]["error"]


def test_additive_preflight_matches_boolean_keep_mask_without_quadratic_storage():
    from tests.numerical.gpu_hunyuan_video_preflight import make_attention_case

    cfg = SimpleNamespace(
        latent_frames=3,
        latent_height=4,
        latent_width=4,
        patch_size=2,
        text_seq_len=5,
        num_attention_heads=2,
        attention_head_dim=8,
    )
    *_, bias = make_attention_case(
        cfg, valid_text_tokens=2, device="cpu", mask_representation="additive"
    )
    assert bias.shape == (2, 1, 17) and bias.dtype == torch.bfloat16
    assert bias.stride() == (0, 17, 1)  # One request bias, shared by the heads.
    assert torch.all(bias[..., :14] == 0) and torch.isneginf(bias[..., 14:]).all()
    assert bias.numel() == 34
