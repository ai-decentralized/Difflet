"""Quantization comparison metrics: tensor error set, PSNR/SSIM, media loading."""

from __future__ import annotations

import math

import pytest
import torch

from difflet.quant import metrics as m


def test_tensor_error_metrics_identity_and_known_offset():
    ref = torch.randn(4, 5)
    same = m.tensor_error_metrics(ref, ref.clone())
    assert same["mse"] == 0.0 and same["max_abs"] == 0.0 and same["rel_l2"] == 0.0
    assert same["cosine"] == pytest.approx(1.0)
    assert same["snr_db"] == math.inf

    ref = torch.ones(10)
    off = m.tensor_error_metrics(ref, ref + 0.1)
    assert off["mse"] == pytest.approx(0.01)
    assert off["mean_abs"] == pytest.approx(0.1)
    assert off["rel_l2"] == pytest.approx(0.1)
    assert off["snr_db"] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        m.tensor_error_metrics(torch.ones(2), torch.ones(3))


def test_psnr_and_ssim_on_identical_and_perturbed_images():
    torch.manual_seed(0)
    img = torch.rand(2, 3, 32, 32)
    assert m.psnr(img, img) == math.inf
    assert m.ssim(img, img) == pytest.approx(1.0, abs=1e-5)
    # mse = 0.01 -> 20 dB for a [0, 1] range
    assert m.psnr(img, (img + 0.1).clamp(max=1.0)) < 20.5
    assert m.psnr(torch.zeros(1, 3, 16, 16), torch.full((1, 3, 16, 16), 0.1)) == pytest.approx(20.0)
    noisy = (img + 0.2 * torch.randn_like(img)).clamp(0, 1)
    assert 0.0 < m.ssim(img, noisy) < 0.9


def test_load_media_normalizes_video_tensor_layouts_and_ranges(tmp_path):
    video = torch.rand(1, 3, 4, 8, 8) * 2 - 1  # Wan [B, C, T, H, W] in [-1, 1]
    path = tmp_path / "frames.pt"
    torch.save(video, path)
    frames = m.load_media(path)
    assert frames.shape == (4, 3, 8, 8)
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    assert torch.allclose(frames, (video[0].permute(1, 0, 2, 3) + 1) / 2, atol=1e-6)

    uint8 = torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8)  # [T, H, W, C]
    torch.save(uint8, tmp_path / "u8.pt")
    loaded = m.load_media(tmp_path / "u8.pt")
    assert loaded.shape == (2, 3, 8, 8)
    assert torch.allclose(loaded, uint8.permute(0, 3, 1, 2).float() / 255)


def test_compare_outputs_on_png_files(tmp_path):
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    b = np.clip(a.astype(np.int16) + rng.integers(-8, 9, a.shape), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(b).save(tmp_path / "b.png")
    same = m.compare_outputs(tmp_path / "a.png", tmp_path / "a.png", lpips_net=None)
    assert same["psnr_db"] == math.inf and same["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert same["frames"] == 1 and same["lpips"] is None
    diff = m.compare_outputs(tmp_path / "a.png", tmp_path / "b.png", lpips_net=None)
    assert 25 < diff["psnr_db"] < 45
    assert diff["pixel"]["max_abs"] <= 8 / 255 + 1e-6


def test_compare_latents_reads_saved_tensors(tmp_path):
    ref = torch.randn(1, 16, 3, 8, 8)
    torch.save(ref, tmp_path / "ref.pt")
    torch.save(ref * 1.01, tmp_path / "test.pt")
    out = m.compare_latents(tmp_path / "ref.pt", tmp_path / "test.pt")
    assert out["shape"] == [1, 16, 3, 8, 8]
    assert out["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert out["rel_l2"] == pytest.approx(0.01, rel=1e-3)
