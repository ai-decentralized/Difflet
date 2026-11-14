"""MiniMax-H3 FL2VA (first and/or last keyframe) host-side preparation.

Reproduces the official diffusers ``modular_pipelines/minimax_h3`` recipe:

* ``prepare_keyframes``: the canvas (H3's 768-short-edge geometry for the first keyframe's aspect
  ratio unless the request sets it), the geometry anchor *stretched* onto it, the follower
  *cover-cropped* with the released model's rounding (``MiniMaxH3ResizeStep``).
* ``build_fl2va_presentation``: ``"<Picture i>: "`` plus a vision block per keyframe, then the
  prompt verbatim, no special tokens; the vision block's rows are tagged as video
  (``MiniMaxH3FL2VATextEncoderStep``).
* ``encode_keyframe_condition``: ImageNet-normalized pixels, the VAE posterior *sampled* under its
  own fixed seed, rounded through fp16, then latent-normalized (``encode_vae_condition``).
* ``noise_condition_rows``: one noise draw per keyframe from the request generator (drawn before
  the video and audio noise), mixed to ``t = 0.999`` and patchified
  (``MiniMaxH3PrepareConditionLatentsStep``).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image

from difflet.models.minimax_h3.contracts import (
    KEYFRAME_ENCODE_SEED,
    KEYFRAME_NOISE_AUG,
    PIXEL_MEAN,
    PIXEL_STD,
    TEXT_TAG,
    VIDEO_TAG,
    resolve_canvas_size,
)


def prepare_keyframes(
    image: Image.Image | None,
    last_image: Image.Image | None,
    *,
    height: int | None = None,
    width: int | None = None,
) -> tuple[list[Image.Image], tuple[str, ...], int, int]:
    """``(keyframes, anchors, height, width)``: the keyframes on the canvas, in packed order."""

    pairs = [(anchor, frame) for anchor, frame in (("first", image), ("last", last_image))
             if frame is not None]
    if not pairs:
        raise ValueError("FL2VA needs a first and/or a last keyframe")
    if (height is None) != (width is None):
        raise ValueError("`height` and `width` have to be passed together, or neither of them.")
    keyframes = [frame.convert("RGB") if frame.mode != "RGB" else frame for _, frame in pairs]
    anchors = tuple(anchor for anchor, _ in pairs)
    if height is None:
        height, width = resolve_canvas_size(*keyframes[0].size)

    prepared = []
    for index, keyframe in enumerate(keyframes):
        if keyframe.size == (width, height):
            prepared.append(keyframe)
        elif index == 0:
            # The geometry anchor is stretched (VaeImageProcessor.resize, i.e. PIL LANCZOS).
            prepared.append(keyframe.resize((width, height), Image.Resampling.LANCZOS))
        else:
            # The follower is cover-cropped with the released model's arithmetic: round to size,
            # centre with (src - dst) // 2.
            scale = max(width / keyframe.size[0], height / keyframe.size[1])
            resized_size = (
                max(width, round(keyframe.size[0] * scale)),
                max(height, round(keyframe.size[1] * scale)),
            )
            left = max(0, (resized_size[0] - width) // 2)
            top = max(0, (resized_size[1] - height) // 2)
            resized = keyframe.resize(resized_size, Image.Resampling.LANCZOS)
            prepared.append(resized.crop((left, top, left + width, top + height)))
    return prepared, anchors, int(height), int(width)


def build_fl2va_presentation(
    tokenizer: Any,
    image_processor: Any,
    prompt: str,
    keyframes: Sequence[Image.Image],
) -> tuple[list[int], torch.Tensor, dict[str, torch.Tensor]]:
    """``(token_ids, token_tags, vision_inputs)`` of the FL2VA conditioner presentation."""

    if not isinstance(prompt, str):
        raise ValueError(f"MiniMax-H3 takes one prompt string, got {type(prompt)}")
    vision_inputs: dict[str, torch.Tensor] = {}
    token_ids: list[int] = []
    token_tags: list[int] = []
    if keyframes:
        vision = image_processor(images=list(keyframes), return_tensors="pt")
        image_grid_thw = vision["image_grid_thw"]
        vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": image_grid_thw}
        merge = int(image_processor.merge_size) ** 2
        start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        for index in range(len(keyframes)):
            num_image_tokens = int(image_grid_thw[index].prod()) // merge
            label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = [start] + [pad] * num_image_tokens + [end]
            token_ids += label_ids + vision_ids
            token_tags += [TEXT_TAG] * len(label_ids) + [VIDEO_TAG] * len(vision_ids)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids += prompt_ids
    token_tags += [TEXT_TAG] * len(prompt_ids)
    return token_ids, torch.tensor(token_tags, dtype=torch.long), vision_inputs


def keyframe_pixels(image: Image.Image) -> torch.Tensor:
    """``(1, 3, 1, H, W)`` uint8 pixels of one keyframe."""

    return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None, :, None]


@torch.no_grad()
def encode_keyframe_condition(
    encode_moments: Callable[[torch.Tensor], torch.Tensor],
    pixels: torch.Tensor,
    *,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    encode_seed: int = KEYFRAME_ENCODE_SEED,
) -> torch.Tensor:
    """Normalized ``(1, 24, 1, h, w)`` float32 conditioning latents of one keyframe.

    ``encode_moments`` maps normalized fp32 pixels to the posterior moments
    (``quant_conv(encoder(x))``, ``2 * latent_channels`` channels).
    """

    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
    x = (pixels.to(torch.float32).div(255.0) - mean) / std
    posterior = DiagonalGaussianDistribution(encode_moments(x))
    latents = posterior.sample(generator=torch.Generator().manual_seed(int(encode_seed)))
    latents = latents.to(torch.float16).float().cpu()
    lat_mean = torch.tensor(list(latents_mean)).view(1, -1, 1, 1, 1)
    lat_std = torch.tensor(list(latents_std)).view(1, -1, 1, 1, 1)
    return (latents - lat_mean) / lat_std


def noise_condition_rows(
    condition_latents: Sequence[torch.Tensor],
    generator: torch.Generator,
    scale_noise: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    patchify: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """``(1, num_condition_rows, 96)`` packed keyframe rows held at ``t = 0.999``.

    Draws one noise tensor per condition from the request generator, in packed order; it has to
    run before the video and audio draws.
    """

    packed = []
    for condition in condition_latents:
        noise = torch.randn(condition.shape, generator=generator, dtype=torch.float32)
        noised = scale_noise(condition.float(), torch.tensor(KEYFRAME_NOISE_AUG), noise)
        packed.append(patchify(noised))
    return torch.cat(packed, dim=1)


__all__ = [
    "build_fl2va_presentation",
    "encode_keyframe_condition",
    "keyframe_pixels",
    "noise_condition_rows",
    "prepare_keyframes",
]
