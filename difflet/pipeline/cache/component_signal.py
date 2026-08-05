"""Cheap component-aligned signals for residual cache decisions.

The cache controller should observe the input of the branch it is deciding to
skip.  A small signed/RMS sketch keeps that observation cheap enough to retain
per block and per anchor without copying the full hidden state to the host.
"""

from __future__ import annotations

from typing import Any


def residual_input_sketch(
    hidden_states: Any,
    *,
    text_token_count: int,
    image_height: int,
    image_width: int,
    text_regions: int = 8,
    image_region_rows: int = 4,
    image_region_columns: int = 4,
    channel_groups: int = 32,
):
    """Return signed-mean and RMS sketches of text/image branch inputs.

    ``hidden_states`` must be ``[batch, text + image, channels]`` with image
    tokens in row-major order.  The result is
    ``[batch, text_regions + image_regions, channel_groups, 2]``.  The last
    axis stores signed mean and RMS, preserving both direction and energy.
    """

    import torch

    if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
        raise ValueError("residual input must be a rank-3 torch.Tensor")
    for name, value in (
        ("text_token_count", text_token_count),
        ("image_height", image_height),
        ("image_width", image_width),
        ("text_regions", text_regions),
        ("image_region_rows", image_region_rows),
        ("image_region_columns", image_region_columns),
        ("channel_groups", channel_groups),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    batch, sequence, channels = hidden_states.shape
    image_token_count = image_height * image_width
    if sequence != text_token_count + image_token_count:
        raise ValueError("residual input sequence does not match text + image layout")
    if text_token_count % text_regions:
        raise ValueError("text_token_count must be divisible by text_regions")
    if image_height % image_region_rows or image_width % image_region_columns:
        raise ValueError("image layout must be divisible by the coarse region grid")
    if channels % channel_groups:
        raise ValueError("hidden channels must be divisible by channel_groups")

    channels_per_group = channels // channel_groups
    text_tokens_per_region = text_token_count // text_regions
    text = hidden_states[:, :text_token_count].reshape(
        batch,
        text_regions,
        text_tokens_per_region,
        channel_groups,
        channels_per_group,
    )
    text_mean = text.mean(dim=(2, 4))
    text_rms = text.square().mean(dim=(2, 4)).sqrt()

    image_region_height = image_height // image_region_rows
    image_region_width = image_width // image_region_columns
    image = hidden_states[:, text_token_count:].reshape(
        batch,
        image_region_rows,
        image_region_height,
        image_region_columns,
        image_region_width,
        channel_groups,
        channels_per_group,
    )
    image_mean = image.mean(dim=(2, 4, 6)).reshape(
        batch, image_region_rows * image_region_columns, channel_groups
    )
    image_rms = image.square().mean(dim=(2, 4, 6)).sqrt().reshape(
        batch, image_region_rows * image_region_columns, channel_groups
    )
    signed_mean = torch.cat((text_mean, image_mean), dim=1)
    rms = torch.cat((text_rms, image_rms), dim=1)
    return torch.stack((signed_mean, rms), dim=-1)


def residual_input_samples(
    hidden_states: Any,
    *,
    text_token_count: int,
    image_height: int,
    image_width: int,
    text_regions: int = 8,
    image_region_rows: int = 4,
    image_region_columns: int = 4,
    channel_groups: int = 32,
):
    """Return one representative value per spatial/text and channel region.

    Unlike :func:`residual_input_sketch`, this summary performs no arithmetic:
    it only reshapes and selects the middle token/channel in every region.  It
    is therefore useful when two independently compiled AOT buckets introduce
    a small reduction or normalization floor.  The result shape is
    ``[batch, text_regions + image_regions, channel_groups]``.
    """

    import torch

    if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
        raise ValueError("residual input must be a rank-3 torch.Tensor")
    for name, value in (
        ("text_token_count", text_token_count),
        ("image_height", image_height),
        ("image_width", image_width),
        ("text_regions", text_regions),
        ("image_region_rows", image_region_rows),
        ("image_region_columns", image_region_columns),
        ("channel_groups", channel_groups),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    batch, sequence, channels = hidden_states.shape
    image_token_count = image_height * image_width
    if sequence != text_token_count + image_token_count:
        raise ValueError("residual input sequence does not match text + image layout")
    if text_token_count % text_regions:
        raise ValueError("text_token_count must be divisible by text_regions")
    if image_height % image_region_rows or image_width % image_region_columns:
        raise ValueError("image layout must be divisible by the coarse region grid")
    if channels % channel_groups:
        raise ValueError("hidden channels must be divisible by channel_groups")

    channels_per_group = channels // channel_groups
    text_tokens_per_region = text_token_count // text_regions
    text = hidden_states[:, :text_token_count].reshape(
        batch,
        text_regions,
        text_tokens_per_region,
        channel_groups,
        channels_per_group,
    )
    text_samples = text[
        :, :, text_tokens_per_region // 2, :, channels_per_group // 2
    ]

    image_region_height = image_height // image_region_rows
    image_region_width = image_width // image_region_columns
    image = hidden_states[:, text_token_count:].reshape(
        batch,
        image_region_rows,
        image_region_height,
        image_region_columns,
        image_region_width,
        channel_groups,
        channels_per_group,
    )
    image_samples = image[
        :,
        :,
        image_region_height // 2,
        :,
        image_region_width // 2,
        :,
        channels_per_group // 2,
    ].reshape(
        batch,
        image_region_rows * image_region_columns,
        channel_groups,
    )
    return torch.cat((text_samples, image_samples), dim=1)


def relative_component_drift(current: Any, anchor: Any, *, epsilon: float = 1e-12):
    """Batchwise relative RMS drift for either full tensors or sketches."""

    import torch

    if not torch.is_tensor(current) or not torch.is_tensor(anchor):
        raise TypeError("component drift inputs must be torch.Tensor values")
    if current.shape != anchor.shape or current.ndim < 2:
        raise ValueError("component drift inputs must have equal batch-first shapes")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    reduce_dims = tuple(range(1, current.ndim))
    numerator = (current.float() - anchor.float()).square().mean(dim=reduce_dims).sqrt()
    denominator = anchor.float().square().mean(dim=reduce_dims).sqrt().clamp_min(epsilon)
    return numerator / denominator


__all__ = [
    "relative_component_drift",
    "residual_input_samples",
    "residual_input_sketch",
]
