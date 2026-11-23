"""MiniMax-H3 first-frame conditioning on CUDA: the keyframe encode through CUDA's tiled VAE
encoder and its noise-augmented anchor rows. CPU-only."""

from __future__ import annotations

import torch
from PIL import Image

from difflet.backends.cuda.minimax_h3.keyframe import encode_keyframe, keyframe_anchor_rows
from difflet.backends.cuda.minimax_h3.vae import VideoEncoder
from difflet.models.minimax_h3.contracts import KEYFRAME_NOISE_AUG
from difflet.models.minimax_h3.keyframes import encode_keyframe_condition, keyframe_pixels
from difflet.models.minimax_h3.pipeline import patchify_video_latents


def test_anchor_mixes_clean_rows_with_frame_zero_of_the_full_seeded_noise():
    clean = torch.randn(1, 24, 1, 4, 6)
    rows = keyframe_anchor_rows(clean, seed=7, num_target_latent_frames=5)

    noise = torch.randn((1, 24, 6, 4, 6), generator=torch.Generator().manual_seed(7))[:, :, :1]
    keep = torch.tensor(KEYFRAME_NOISE_AUG, dtype=torch.float32)
    expected = keep * patchify_video_latents(clean) + (1.0 - keep) * patchify_video_latents(noise)
    assert rows.shape == (1, 6, 96) and torch.equal(rows, expected)
    # Drawing only the first frame would give different noise.
    single = torch.randn((1, 24, 1, 4, 6), generator=torch.Generator().manual_seed(7))
    assert not torch.equal(noise, single)


def _encoder() -> VideoEncoder:
    config = dict(
        in_channels=3,
        latent_channels=24,
        block_out_channels=[8, 8, 8, 8],
        layers_per_block=1,
        spatial_downsample_factors=[2, 2, 2, 2],
        temporal_downsample_factors=[1, 2, 1, 1],
        norm_num_groups=4,
        norm_eps=1e-6,
        spatial_padding_mode="reflect",
        latents_mean=[0.1] * 24,
        latents_std=[1.3] * 24,
    )
    torch.manual_seed(0)
    return VideoEncoder(config).eval()


def test_the_keyframe_encode_samples_and_normalizes_the_tiled_moments():
    """Over 256 pixels on a side, so the encode is tiled and stitched."""
    encoder = _encoder()
    image = Image.fromarray((torch.rand(320, 448, 3) * 255).to(torch.uint8).numpy())
    latent = encode_keyframe(encoder, image)
    assert latent.shape == (1, 24, 1, 20, 28) and latent.device.type == "cpu"
    want = encode_keyframe_condition(
        encoder.moments, keyframe_pixels(image), latents_mean=[0.1] * 24, latents_std=[1.3] * 24
    )
    assert torch.equal(latent, want)
