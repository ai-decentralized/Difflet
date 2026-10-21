"""Pure parts of scripts/cuda_e2e_vs_diffusers.py (the GPU run is the script itself)."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

from cuda_e2e_vs_diffusers import image_metrics  # noqa: E402


def test_identical_images_score_perfect():
    a = torch.rand(3, 16, 16)
    m = image_metrics(a, a.clone())
    assert abs(m["cosine"] - 1.0) < 1e-12 and m["mean_abs"] == 0.0 and m["psnr_db"] == float("inf")


def test_noise_lowers_psnr_and_cosine():
    torch.manual_seed(0)
    a = torch.rand(3, 16, 16)
    m = image_metrics(a, (a + 0.1 * torch.randn_like(a)).clamp(0, 1))
    assert 15 < m["psnr_db"] < 30 and 0.9 < m["cosine"] < 1.0 and m["max_abs"] > m["mean_abs"]
