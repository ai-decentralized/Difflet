from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "hv15_frame_metrics.py"


@pytest.fixture(scope="module")
def fm():
    spec = importlib.util.spec_from_file_location("hv15_frame_metrics", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_videos_give_inf_psnr_and_unit_ssim(fm):
    video = torch.rand(3, 3, 24, 32, generator=torch.Generator().manual_seed(0))
    m = fm.frame_metrics(video, video.clone())
    assert m["frames"] == 3
    assert m["psnr_mean"] == math.inf and m["psnr_min"] == math.inf
    assert m["ssim_mean"] == pytest.approx(1.0, abs=1e-12)
    assert m["ssim_min"] == pytest.approx(1.0, abs=1e-12)


def test_constant_offset_matches_hand_computed_values(fm):
    # Flat 0.5 vs flat 0.6: MSE 0.01, so PSNR = 10 log10(1 / 0.01) = 20 dB. Both variances and
    # the covariance are 0, so SSIM = (2 * 0.5 * 0.6 + C1) / (0.5^2 + 0.6^2 + C1) with C1 = 0.01^2
    # = 0.6001 / 0.6101.
    ref = torch.full((2, 3, 16, 16), 0.5, dtype=torch.float64)  # fp32 0.6 is 0.60000002
    cand = torch.full((2, 3, 16, 16), 0.6, dtype=torch.float64)
    m = fm.frame_metrics(ref, cand)
    assert m["psnr_per_frame"] == pytest.approx([20.0, 20.0], abs=1e-9)
    assert m["ssim_per_frame"] == pytest.approx([0.6001 / 0.6101] * 2, abs=1e-9)


def test_vae_layout_tensor_is_read_as_frames(fm, tmp_path):
    vae_out = torch.rand(1, 3, 5, 16, 16)  # [1, C, F, H, W]
    path = tmp_path / "video.pt"
    torch.save(vae_out, path)
    frames = fm.load_frames(path)
    assert frames.shape == (5, 3, 16, 16)
    assert torch.equal(frames[2], vae_out[0, :, 2].to(torch.float64))


def test_rejects_shape_mismatch_and_values_outside_unit_range(fm):
    with pytest.raises(ValueError, match="shape mismatch"):
        fm.frame_metrics(torch.zeros(2, 3, 16, 16), torch.zeros(3, 3, 16, 16))
    with pytest.raises(ValueError, match=r"expected \[0, 1\]"):
        fm.frame_metrics(torch.zeros(1, 3, 16, 16) - 1.0, torch.zeros(1, 3, 16, 16))
