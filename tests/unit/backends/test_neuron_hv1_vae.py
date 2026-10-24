"""HunyuanVideo's VAE decoder application (CPU, fp32) against diffusers' tiled decode.

The tiny ``AutoencoderKLHunyuanVideo`` is ``_neuron_hv1_toy.TINY_HV1_VAE`` (the real 8x spatial
and 4x temporal compression); the application only drives diffusers' own autoencoder, so the
tests run in the pytest process. Nothing touches the device.
"""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.hunyuan_video.vae import NeuronHunyuanVideoVaeDecoderApplication  # noqa: E402
from tests.unit.backends._neuron_hv1_toy import TINY_HV1_VAE  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from diffusers import AutoencoderKLHunyuanVideo

    torch.manual_seed(0)
    path = str(tmp_path_factory.mktemp("hv1vae") / "vae")
    AutoencoderKLHunyuanVideo(**TINY_HV1_VAE).save_pretrained(path, safe_serialization=True)
    return path


def _diffusers_tiled(path, z, *, tile_size, tile_frames):
    from diffusers import AutoencoderKLHunyuanVideo

    ref = AutoencoderKLHunyuanVideo.from_pretrained(path).eval()
    ref.enable_tiling(tile_sample_min_height=tile_size, tile_sample_min_width=tile_size,
                      tile_sample_min_num_frames=tile_frames,
                      tile_sample_stride_height=tile_size * 3 // 4,
                      tile_sample_stride_width=tile_size * 3 // 4,
                      tile_sample_stride_num_frames=tile_frames * 3 // 4)
    with torch.no_grad():
        return ref.decode(z / ref.config.scaling_factor, return_dict=False)[0]


def test_decode_matches_diffusers_tiled_in_space_and_time(tiny):
    # 33 frames -> 9 latent frames, past a 16-frame tile (4 latent): three temporal tiles; a
    # 64 x 96 frame past a 32-pixel tile: 2 x 3 spatial tiles in each
    app = NeuronHunyuanVideoVaeDecoderApplication(model_path=tiny, height=64, width=96,
                                                  num_frames=33, tile_size=32, tile_frames=16,
                                                  device="cpu")
    app.load(skip_warmup=True)
    assert app.dtype == torch.float32 and app.vae.encoder is None
    assert app.latent_shape == (1, 4, 9, 8, 12)
    assert (app.vae.tile_sample_min_height, app.vae.tile_sample_stride_height,
            app.vae.tile_sample_min_num_frames, app.vae.tile_sample_stride_num_frames) == (32, 24, 16, 12)
    z = torch.randn(app.latent_shape, generator=torch.Generator().manual_seed(1))
    ours = app.decode(z)
    want = _diffusers_tiled(tiny, z, tile_size=32, tile_frames=16)
    assert ours.shape == want.shape == (1, 3, 33, 64, 96)
    torch.testing.assert_close(ours, want, atol=1e-5, rtol=1e-5)


def test_warmup_decodes_the_static_shape(tiny):
    app = NeuronHunyuanVideoVaeDecoderApplication(model_path=tiny, height=64, width=64,
                                                  num_frames=17, tile_size=32, device="cpu")
    app.load()
    assert set(app.phase_seconds) == {"load", "warmup decode"}


def test_latent_shape_at_the_default_shape_reads_the_config(tmp_path):
    # the real snapshot's VAE config: 16 latent channels; read before any weight loads
    (tmp_path / "config.json").write_text(json.dumps(dict(
        TINY_HV1_VAE, latent_channels=16, block_out_channels=[128, 256, 512, 512])))
    app = NeuronHunyuanVideoVaeDecoderApplication(model_path=tmp_path, device="cpu")
    assert (app.height, app.width, app.num_frames) == (320, 512, 61)
    assert app.latent_shape == (1, 16, 16, 40, 64)
    assert app.scaling_factor == pytest.approx(0.476986)


@pytest.mark.parametrize("kwargs, match", [
    (dict(tile_size=48), "tile_size=48"),          # a 36-pixel stride is 4.5 latent units
    (dict(tile_frames=8), "tile_frames=8"),        # a 6-frame stride is 1.5 latent frames
    (dict(num_frames=60), "num_frames=60"),
    (dict(height=100), "height=100"),
])
def test_refuses_tiles_and_shapes_off_the_latent_grid(tiny, kwargs, match):
    with pytest.raises(ValueError, match=match):
        NeuronHunyuanVideoVaeDecoderApplication(model_path=tiny, device="cpu", **kwargs)
