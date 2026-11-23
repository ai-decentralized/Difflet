"""MiniMax-H3 first-frame conditioning on CUDA: the keyframe's latent and its anchor rows.

The keyframe is put on the canvas and presented to the text encoder with the shared FL2VA
helpers (``difflet.models.minimax_h3.keyframes``), and encoded on the GPU by CUDA's own tiled
VAE encoder (``vae.VideoEncoder``).

The anchor rows mix that latent with a little noise, ``0.999 * clean + 0.001 * noise``. CUDA
draws the noise the way the deployment behind MiniMax's published samples does, so its output
reproduces the official ``assets/i2va.mp4``: frame 0 of a ``target_frames + 1``-frame latent
draw from a generator seeded with the request seed, independent of the video and audio noise.
(The neuron path follows Diffusers' modular pipeline instead, which draws the keyframe, video
and audio noise from one generator in turn.)
"""

from __future__ import annotations

import torch

from difflet.backends.cuda.minimax_h3.vae import VideoEncoder
from difflet.models.minimax_h3.contracts import KEYFRAME_NOISE_AUG
from difflet.models.minimax_h3.keyframes import encode_keyframe_condition, keyframe_pixels
from difflet.models.minimax_h3.pipeline import patchify_video_latents

LATENT_CHANNELS = 24


@torch.no_grad()
def encode_keyframe(encoder: VideoEncoder, image) -> torch.Tensor:
    """The normalized ``(1, 24, 1, h, w)`` CPU latent of one keyframe already on the canvas."""
    # Deterministic cuDNN algorithms without autotuning: one image always gives one latent.
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
        return encode_keyframe_condition(
            lambda pixels: encoder.moments(pixels.to(encoder.quant_conv.weight.device)),
            keyframe_pixels(image),
            latents_mean=encoder.config.latents_mean,
            latents_std=encoder.config.latents_std,
        )


def keyframe_anchor_rows(
    clean_latent: torch.Tensor, *, seed: int, num_target_latent_frames: int
) -> torch.Tensor:
    """The fixed ``(1, rows, 96)`` FP32 anchor rows of a normalized keyframe latent."""
    latent_height, latent_width = clean_latent.shape[-2:]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (1, LATENT_CHANNELS, num_target_latent_frames + 1, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    )[:, :, :1]
    keep = torch.tensor(KEYFRAME_NOISE_AUG, dtype=torch.float32)
    clean = patchify_video_latents(clean_latent.float())
    return keep * clean + (1.0 - keep) * patchify_video_latents(noise)


__all__ = ["LATENT_CHANNELS", "encode_keyframe", "keyframe_anchor_rows"]
