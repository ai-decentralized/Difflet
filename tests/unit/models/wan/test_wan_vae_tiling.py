"""Difflet's Wan VAE tiled decode against diffusers' ``AutoencoderKLWan.tiled_decode``.

The tiny model has the real stage layout; 16-pixel tiles with a stride of 8 split its 4 x 6
latent (32 x 48 pixels) into 24 overlapping tiles, so every blend runs. fp32 on the CPU.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit.models.wan.test_wan_vae_regions import _diffusers_vae, _latents, _model_from

TILE, STRIDE = 16, 8


@pytest.mark.parametrize("frames", [1, 3])
def test_tiled_decode_equals_diffusers_tiled_decode(frames):
    vae = _diffusers_vae(0)
    model = _model_from(vae)
    vae.enable_tiling(tile_sample_min_height=TILE, tile_sample_min_width=TILE,
                      tile_sample_stride_height=STRIDE, tile_sample_stride_width=STRIDE)
    model.enable_tiling(TILE, STRIDE)
    z = _latents(frames)
    with torch.no_grad():
        want = vae.tiled_decode(z, return_dict=False)[0]
        got = model(z)
    assert len(model.tile_origins(z)) == 4 * 6
    assert got.shape == want.shape == (1, 3, 1 + 4 * (frames - 1), 32, 48)
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)


def test_tiled_decode_from_separately_decoded_tiles_is_the_same():
    """``blend_tiles`` on tiles decoded one by one elsewhere gives ``tiled_decode``'s frames
    exactly: the path a parallel decode takes."""
    model = _model_from(_diffusers_vae(0))
    model.enable_tiling(TILE, STRIDE)
    z = _latents(3)
    with torch.no_grad():
        want = model.tiled_decode(z)
        tiles = [model.decode_tile(t.clone()) for t in model.tile_latents(z)]
        got = model.blend_tiles(z, tiles)
    assert torch.equal(got, want)


def test_untiled_by_default_and_when_one_tile_covers_the_frame():
    model = _model_from(_diffusers_vae(0))
    z = _latents(3)
    with torch.no_grad():
        untiled = model(z)
        model.enable_tiling()  # 256-pixel tiles: one covers the 32 x 48 frame
        assert torch.equal(model(z), untiled)
