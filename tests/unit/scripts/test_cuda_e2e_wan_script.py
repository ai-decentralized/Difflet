"""scripts/cuda_e2e_wan_vs_diffusers.py imports and parses on a CPU host (the run is GPU-only).

The pure parts are tested here: the defaults of the gated run, and the two layouts the
video is compared in — Difflet's ``(B, 3, F, H, W)`` in [-1, 1] and diffusers'
``WanPipeline(output_type="pt")`` ``(B, F, 3, H, W)`` in [0, 1] — both brought to
``(F, H, W, 3)`` in [0, 1].
"""

import os
import pathlib
import subprocess
import sys

import torch

SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import cuda_e2e_vs_diffusers as qwen_e2e  # noqa: E402
import cuda_e2e_wan_vs_diffusers as wan_e2e  # noqa: E402


def test_defaults_are_the_gated_wan22_run():
    args = wan_e2e.build_parser().parse_args(["--model-path", "/snap"])
    assert args.model_path == "/snap" and args.model_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert args.steps == 40 and args.guidance_scale == 4.0 and args.guidance_scale_2 == 3.0
    assert args.seed == 0 and (args.height, args.width, args.num_frames) == (480, 832, 9)
    assert args.out == "/tmp/logs/cuda_e2e_wan.json" and args.images_dir == "/tmp/logs"


def test_metrics_and_png_writer_are_the_qwen_oracle_s():
    assert wan_e2e.image_metrics is qwen_e2e.image_metrics
    assert wan_e2e._save is qwen_e2e._save


def test_both_layouts_reach_the_same_frames_hwc_unit_video():
    torch.manual_seed(0)
    unit = torch.rand(9, 6, 10, 3)  # (F, H, W, 3) in [0, 1]
    difflet = (unit * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, F, H, W) in [-1, 1]
    diffusers = unit.permute(0, 3, 1, 2).unsqueeze(0)  # (1, F, 3, H, W) in [0, 1]
    a = wan_e2e.difflet_unit_video(difflet)
    b = wan_e2e.diffusers_unit_video(diffusers)
    assert a.shape == b.shape == (9, 6, 10, 3)
    assert torch.allclose(a, unit, atol=1e-6) and torch.allclose(b, unit)
    m = wan_e2e.image_metrics(a, b)
    assert m["psnr_db"] > 100 and abs(m["cosine"] - 1.0) < 1e-9


def test_units_are_clamped():
    v = torch.full((1, 3, 1, 2, 2), 3.0)
    assert wan_e2e.difflet_unit_video(v).max() == 1.0
    assert wan_e2e.diffusers_unit_video(-v.permute(0, 2, 1, 3, 4)).min() == 0.0


def test_first_frame_is_chw_for_the_png_writer(tmp_path):
    video = torch.rand(9, 6, 10, 3)
    frame = wan_e2e.first_frame(video)
    assert frame.shape == (3, 6, 10) and torch.equal(frame, video[0].permute(2, 0, 1))
    wan_e2e._save(frame, tmp_path / "f.png")
    assert (tmp_path / "f.png").is_file()


def test_help_runs_on_a_cpu_host():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "cuda_e2e_wan_vs_diffusers.py"), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
    )
    assert proc.returncode == 0 and "--num-frames" in proc.stdout, proc.stderr[-2000:]
