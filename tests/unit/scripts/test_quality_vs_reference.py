from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "quality_vs_reference.py"


@pytest.fixture(scope="module")
def qvr():
    spec = importlib.util.spec_from_file_location("quality_vs_reference", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identical_latents(qvr):
    ref = torch.randn(1, 16, 8, 8)
    metrics = qvr.latent_metrics(ref, ref.clone())
    assert metrics["rel_l2"] == 0.0
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-12)


def test_scaled_latents(qvr):
    ref = torch.randn(1, 16, 8, 8)
    metrics = qvr.latent_metrics(ref, 1.1 * ref)
    assert metrics["rel_l2"] == pytest.approx(0.1, abs=1e-6)
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_bf16_latents_are_compared_in_fp64(qvr):
    ref = torch.randn(4, 32).to(torch.bfloat16)
    metrics = qvr.latent_metrics(ref, ref.clone())
    assert metrics["rel_l2"] == 0.0


def test_identical_frames(qvr):
    frames = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    metrics = qvr.frame_metrics(frames, frames.copy())
    assert math.isinf(metrics["psnr_mean"])
    assert math.isinf(metrics["psnr_worst"])
    assert metrics["ssim_mean"] == pytest.approx(1.0)


def test_frames_off_by_one(qvr):
    frames = np.full((2, 32, 32, 3), 100, dtype=np.uint8)
    metrics = qvr.frame_metrics(frames, frames + 1)
    assert metrics["psnr_mean"] == pytest.approx(48.1308, abs=1e-3)
    assert metrics["psnr_worst"] == pytest.approx(48.1308, abs=1e-3)


def test_worst_frame_is_the_minimum(qvr):
    ref = np.full((2, 32, 32, 3), 100, dtype=np.uint8)
    cand = ref.copy()
    cand[1] += 4
    metrics = qvr.frame_metrics(ref, cand)
    assert metrics["psnr_worst"] == pytest.approx(10 * math.log10(255**2 / 16), abs=1e-3)
    assert math.isinf(metrics["psnr_mean"])  # frame 0 is identical


def test_ssim_of_constant_frames_has_closed_form(qvr):
    a, b = 100.0, 140.0
    c1 = (0.01 * 255) ** 2
    expected = (2 * a * b + c1) / (a * a + b * b + c1)
    ref = np.full((1, 32, 32, 3), int(a), dtype=np.uint8)
    cand = np.full((1, 32, 32, 3), int(b), dtype=np.uint8)
    assert qvr.frame_metrics(ref, cand)["ssim_mean"] == pytest.approx(expected, abs=1e-6)


def test_ssim_drops_for_noise(qvr):
    rng = np.random.default_rng(1)
    ref = rng.integers(0, 256, (1, 64, 64, 3), dtype=np.uint8)
    noise = rng.normal(0, 40, ref.shape)
    cand = np.clip(ref + noise, 0, 255).astype(np.uint8)
    assert qvr.frame_metrics(ref, cand)["ssim_mean"] < 0.9


def test_frame_shape_mismatch_raises(qvr):
    with pytest.raises(ValueError, match="shape"):
        qvr.frame_metrics(
            np.zeros((1, 8, 8, 3), np.uint8), np.zeros((2, 8, 8, 3), np.uint8)
        )
