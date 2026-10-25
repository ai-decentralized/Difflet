"""Fixed-size tiling of the HunyuanVideo VAE decode (``vae/tiling.py``) and the valid-extent tile
decoder (``HunyuanVideoVAEValidTileDecoder``) against diffusers' own decoder and tiled decode.

A tiny random ``AutoencoderKLHunyuanVideo`` (the real 8x / 4x compression, 32-pixel and 16-frame
tiles): fp32 on the CPU; the compiled case uses ``aot_eager``. Nothing touches the device.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers")

from difflet.models.hunyuan_video.checkpoint.vae import convert_vae_decoder_state_dict  # noqa: E402
from difflet.models.hunyuan_video.vae.modeling_vae import (  # noqa: E402
    HunyuanVideoVAEDecoderConfig,
    HunyuanVideoVAEDecoderModel,
    HunyuanVideoVAEValidTileDecoder,
)
from difflet.models.hunyuan_video.vae import modeling_vae as MV  # noqa: E402
from difflet.models.hunyuan_video.vae.tiling import fixed_tile_decode  # noqa: E402

TINY = dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 8, 8, 8),
            layers_per_block=1, norm_num_groups=4, scaling_factor=0.476986)
TILE = dict(tile_sample_size=32, tile_sample_frames=16, stride_sample_size=24,
            stride_sample_frames=12)
TILE_LATENT = (5, 4, 4)  # 16 frames -> 4 + 1 latent (the causal first frame), 32 px -> 4


@pytest.fixture(scope="module")
def vae():
    from diffusers import AutoencoderKLHunyuanVideo

    torch.manual_seed(0)
    vae = AutoencoderKLHunyuanVideo(**TINY).eval()
    vae.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32,
                      tile_sample_min_num_frames=16, tile_sample_stride_height=24,
                      tile_sample_stride_width=24, tile_sample_stride_num_frames=12)
    return vae


@pytest.fixture(scope="module")
def tile_decoder(vae):
    model = HunyuanVideoVAEDecoderModel(HunyuanVideoVAEDecoderConfig.from_diffusers_dict(dict(vae.config)))
    model.load_state_dict(convert_vae_decoder_state_dict(vae.state_dict()))
    return HunyuanVideoVAEValidTileDecoder(model.eval()).eval()


def _plain(vae, z):
    with torch.no_grad():
        return vae.decoder(vae.post_quant_conv(z))


def _pad(z):
    t, h, w = z.shape[-3:]
    return F.pad(z, (0, TILE_LATENT[2] - w, 0, TILE_LATENT[1] - h, 0, TILE_LATENT[0] - t),
                 mode="replicate")


@pytest.mark.parametrize("extent", [(5, 4, 4), (4, 4, 4), (5, 2, 4), (5, 4, 1), (2, 3, 2),
                                    (1, 4, 4), (1, 1, 1)])
def test_valid_tile_decoder_on_a_padded_tile_equals_the_decoder_on_the_window(vae, tile_decoder,
                                                                             extent):
    t, h, w = extent
    z = torch.randn(1, 4, t, h, w, generator=torch.Generator().manual_seed(t * 100 + h * 10 + w))
    want = _plain(vae, z)
    with torch.no_grad():
        got = tile_decoder(_pad(z), torch.tensor(extent, dtype=torch.int32))
    assert got.shape == (1, 3, 17, 32, 32)
    got = got[:, :, : (t - 1) * 4 + 1, : h * 8, : w * 8]
    torch.testing.assert_close(got, want, atol=5e-5, rtol=1e-4)


def test_ignoring_the_padding_changes_the_edge_tiles(vae, tile_decoder):
    """What the valid extent is for: the plain decoder on the padded tile (the AoT path) is off."""
    z = torch.randn(1, 4, 4, 2, 4, generator=torch.Generator().manual_seed(7))
    want = _plain(vae, z)
    got = _plain(vae, _pad(z))[:, :, :13, :16, :32]
    assert (got - want).abs().max() > 1e-2


# 33 frames -> 9 latent frames (windows at 0, 3, 6: 5, 5, 3 frames); 40 x 56 latent pixels ->
# 5 x 7 latent (spatial windows at 0 and 3: 4 + 2 rows, 4 + 4 + 1 columns)
SHAPE = (1, 4, 9, 5, 7)


def _diffusers_tiled(vae, z):
    with torch.no_grad():
        return vae.decode(z, return_dict=False)[0]


def test_fixed_tile_decode_equals_diffusers_tiled_decode_with_one_tile_shape(vae, tile_decoder):
    z = torch.randn(SHAPE, generator=torch.Generator().manual_seed(1))
    calls = []

    def decode_tile(padded, valid):
        calls.append((tuple(padded.shape), valid))
        with torch.no_grad():
            return tile_decoder(padded, torch.tensor(valid, dtype=torch.int32))

    got = fixed_tile_decode(z, decode_tile, **TILE)
    want = _diffusers_tiled(vae, z)
    assert got.shape == want.shape == (1, 3, 33, 40, 56)
    torch.testing.assert_close(got, want, atol=5e-5, rtol=1e-4)
    assert {shape for shape, _ in calls} == {(1, 4, *TILE_LATENT)}  # one graph's worth
    assert len(calls) == 3 * 2 * 3
    assert {valid for _, valid in calls} == {(t, h, w) for t in (5, 3) for h in (4, 2)
                                             for w in (4, 1)}


def test_fixed_tile_decode_walks_and_blends_as_diffusers_does(vae):
    """Pins the window order, crops and blends alone: with the plain decoder on padded tiles on
    both sides (diffusers' decoder wrapped to pad and crop), the two decodes agree exactly."""
    z = torch.randn(SHAPE, generator=torch.Generator().manual_seed(2))

    class _PadCrop(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = vae.decoder

        def forward(self, x):  # diffusers calls post_quant_conv (1x1, pointwise) first
            t, h, w = x.shape[-3:]
            return self.decoder(_pad(x))[:, :, : (t - 1) * 4 + 1, : h * 8, : w * 8]

    got = fixed_tile_decode(z, lambda padded, valid: _plain(vae, padded), **TILE)
    original = vae.decoder
    vae.decoder = _PadCrop()
    try:
        want = _diffusers_tiled(vae, z)
    finally:
        vae.decoder = original
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_a_last_window_of_one_latent_frame_is_not_decoded(vae, tile_decoder):
    # 7 latent frames: windows at 0 (5 frames), 3 (4) and 6 (1); diffusers decodes the last and
    # drops its only frame
    z = torch.randn(1, 4, 7, 4, 4, generator=torch.Generator().manual_seed(3))
    calls = []

    def decode_tile(padded, valid):
        calls.append(valid)
        with torch.no_grad():
            return tile_decoder(padded, torch.tensor(valid, dtype=torch.int32))

    got = fixed_tile_decode(z, decode_tile, **TILE)
    want = _diffusers_tiled(vae, z)
    assert got.shape == want.shape == (1, 3, 25, 32, 32)
    torch.testing.assert_close(got, want, atol=5e-5, rtol=1e-4)
    assert calls == [(5, 4, 4), (4, 4, 4)]


def test_compiled_regions_trace_their_graphs_once_for_every_extent(vae, tile_decoder):
    """Compiled region by region (``regions()``), the first window builds every graph; the 17
    windows after it, with seven other extents, build none, and the decode is diffusers'."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    model = HunyuanVideoVAEValidTileDecoder(tile_decoder.model)
    regions = model.regions()
    # the head, then each step, a resnet as its four parts (norm, conv1, norm, conv2 + residual)
    n_resnets = sum(isinstance(s, MV._ValidResnet) for s in model.steps)
    assert regions[0] is model.head and len(regions) == 1 + len(model.steps) + 3 * n_resnets
    for region in regions:
        region.compile(backend=backend, fullgraph=True, dynamic=False)
    z = torch.randn(SHAPE, generator=torch.Generator().manual_seed(4))
    after_each = []

    def decode_tile(padded, valid):
        with torch.no_grad():
            out = model(padded, torch.tensor(valid, dtype=torch.int32))
        after_each.append(len(graphs))
        return out

    got = fixed_tile_decode(z, decode_tile, **TILE)
    torch.testing.assert_close(got, _diffusers_tiled(vae, z), atol=5e-5, rtol=1e-4)
    assert len(after_each) == 18 and set(after_each) == {after_each[0]}
    # fewer graphs than regions: same-class regions at one shape share one
    assert 0 < len(graphs) < len(regions)
