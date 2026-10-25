"""Fixed-size tiling for the HunyuanVideo VAE decoder: one tile shape for the whole clip.

diffusers' tiled decode (``AutoencoderKLHunyuanVideo._temporal_tiled_decode`` and
``tiled_decode``) walks the latent by the stride and decodes each window as it comes, so the
windows at the clip's end, bottom and right edge are smaller than the tile. A compiler that
builds one graph per input shape then builds one per edge-tile shape (up to 3 x 2 x 2 at
320x512x61). ``fixed_tile_decode`` walks the same windows in the same order and blends them
exactly as diffusers does, but hands ``decode_tile`` every window replicate-padded at its end to
the full tile, together with the window's valid extent; it crops each decoded tile back to the
window's sample extent before blending. Every call therefore has one input shape.

How exact that is depends on ``decode_tile``: a decoder that ignores the extent (the Trainium
AoT path's ``_decode_tile``) normalizes and attends over the padding too, which changes the edge
tiles (40.8-48.8 dB against the unpadded tile at 320x512x61 on the host, in fp32; 42.5 dB for
the whole decode against diffusers' tiled decode). A decoder
that confines its GroupNorm statistics, attention keys and convolution borders to the valid
extent (``HunyuanVideoVAEValidTileDecoder``) decodes the padded window exactly as diffusers
decodes the small one.

The padding is at the end of each axis only. In time that keeps the causal decoder's frames
before the padding unchanged; in space the extent tells the decoder where the border is.

A window that contributes nothing is not decoded: diffusers decodes the last temporal window
even when it is a single latent frame, then drops that window's first decoded frame, which is
all of it. Its slot is an empty tensor, so the blend is unchanged.

Blending runs on the host (the tiles are moved there by ``decode_tile``'s caller), in the
decoded tiles' dtype, with diffusers' own linear ramps.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

#: ``decode_tile(padded_latents, valid) -> decoded``: ``padded_latents`` is the full tile
#: ``[B, C, tile_frames, tile_height, tile_width]`` in latent units, ``valid`` the window's own
#: extent ``(frames, height, width)`` in latent units; ``decoded`` is the full decoded tile.
TileDecoder = Callable[[torch.Tensor, tuple[int, int, int]], torch.Tensor]


def fixed_tile_decode(
    z: torch.Tensor,
    decode_tile: TileDecoder,
    *,
    tile_sample_size: int = 256,
    tile_sample_frames: int = 16,
    stride_sample_size: int = 192,
    stride_sample_frames: int = 12,
    spatial_compression: int = 8,
    temporal_compression: int = 4,
) -> torch.Tensor:
    """Decode ``z`` ([B, C, F, H, W], already divided by the scaling factor) in fixed-size tiles.

    Window positions, crops and blends are diffusers' (``_temporal_tiled_decode`` with spatial
    tiling on); see the module docstring for what differs.
    """
    _, _, num_frames, height, width = z.shape
    s, t = int(spatial_compression), int(temporal_compression)
    tile_h = tile_w = int(tile_sample_size) // s
    tile_t = int(tile_sample_frames) // t + 1  # diffusers' window: min frames + 1 (causal)
    stride_h = int(stride_sample_size) // s
    stride_t = int(stride_sample_frames) // t
    blend_hw = int(tile_sample_size) - int(stride_sample_size)
    blend_frames = int(tile_sample_frames) - int(stride_sample_frames)
    num_sample_frames = (num_frames - 1) * t + 1

    def decode_window(window: torch.Tensor) -> torch.Tensor:
        wt, wh, ww = window.shape[-3:]
        padded = F.pad(window, (0, tile_w - ww, 0, tile_h - wh, 0, tile_t - wt), mode="replicate")
        decoded = decode_tile(padded, (wt, wh, ww))
        return decoded[:, :, : (wt - 1) * t + 1, : wh * s, : ww * s]

    def spatial(window: torch.Tensor) -> torch.Tensor:
        h, w = window.shape[-2:]
        if h <= tile_h and w <= tile_w:  # diffusers decodes a window this small whole
            return decode_window(window)
        rows = [[decode_window(window[:, :, :, i : i + tile_h, j : j + tile_w])
                 for j in range(0, w, stride_h)] for i in range(0, h, stride_h)]
        out_rows = []
        for i, row in enumerate(rows):
            out_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = blend_v(rows[i - 1][j], tile, blend_hw)
                if j > 0:
                    tile = blend_h(row[j - 1], tile, blend_hw)
                out_row.append(tile[:, :, :, :stride_sample_size, :stride_sample_size])
            out_rows.append(torch.cat(out_row, dim=-1))
        return torch.cat(out_rows, dim=3)[:, :, :, : h * s, : w * s]

    row: list[torch.Tensor] = []
    for i in range(0, num_frames, stride_t):
        window = z[:, :, i : i + tile_t]
        if i > 0 and window.shape[2] == 1:  # decodes to one frame, which the crop below drops
            decoded = None
        else:
            decoded = spatial(window)
            if i > 0:
                decoded = decoded[:, :, 1:]
        row.append(decoded)
    ref = next(tile for tile in row if tile is not None)
    row = [ref.new_empty((*ref.shape[:2], 0, *ref.shape[3:])) if tile is None else tile
           for tile in row]

    out = []
    for i, tile in enumerate(row):
        if i > 0:
            tile = blend_t(row[i - 1], tile, blend_frames)
            out.append(tile[:, :, :stride_sample_frames])
        else:
            out.append(tile[:, :, : stride_sample_frames + 1])
    return torch.cat(out, dim=2)[:, :, :num_sample_frames]


def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """diffusers' ``blend_v``: ramp ``a``'s last rows into ``b``'s first (in place on ``b``)."""
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    for y in range(blend_extent):
        b[:, :, :, y, :] = (a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent)
                            + b[:, :, :, y, :] * (y / blend_extent))
    return b


def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """diffusers' ``blend_h``: the same across columns."""
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    for x in range(blend_extent):
        b[:, :, :, :, x] = (a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent)
                            + b[:, :, :, :, x] * (x / blend_extent))
    return b


def blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    """diffusers' ``blend_t``: the same across frames."""
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    for x in range(blend_extent):
        b[:, :, x, :, :] = (a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent)
                            + b[:, :, x, :, :] * (x / blend_extent))
    return b


__all__ = ["TileDecoder", "blend_h", "blend_t", "blend_v", "fixed_tile_decode"]
