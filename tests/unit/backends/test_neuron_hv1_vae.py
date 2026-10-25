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


# ------------------------------------------------- exec_mode="compile": fixed-size tiles, one graph


def _app(tiny, **kwargs):
    return NeuronHunyuanVideoVaeDecoderApplication(model_path=tiny, height=64, width=96,
                                                   num_frames=33, tile_size=32, tile_frames=16,
                                                   device="cpu", **kwargs)


def test_exec_mode_defaults_to_eager_with_diffusers_tiling(tiny):
    app = _app(tiny)
    assert (app.exec_mode, app.tiling) == ("eager", "diffusers")
    assert _app(tiny, exec_mode="compile").tiling == "fixed"
    assert _app(tiny, exec_mode="compile", tiling="diffusers").tiling == "diffusers"
    assert _app(tiny, tiling="fixed").tiling == "fixed"


@pytest.mark.parametrize("kwargs, match", [
    (dict(exec_mode="jit"), "exec_mode='jit'"),
    (dict(tiling="none"), "tiling='none'"),
])
def test_refuses_unknown_exec_modes_and_tilings(tiny, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _app(tiny, **kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(exec_mode="compile"),                       # (i) fixed tiles, one graph
    dict(exec_mode="compile", tiling="diffusers"),   # (ii) diffusers' tiles, a graph per shape
    dict(tiling="fixed"),                            # fixed tiles, eager
])
def test_decode_matches_diffusers_tiled_in_every_mode(tiny, kwargs):
    import torch._dynamo

    torch._dynamo.reset()
    app = _app(tiny, compile_backend="aot_eager", **kwargs)
    app.load(skip_warmup=True)
    z = torch.randn(app.latent_shape, generator=torch.Generator().manual_seed(1))
    ours = app.decode(z)
    want = _diffusers_tiled(tiny, z, tile_size=32, tile_frames=16)
    assert ours.shape == want.shape == (1, 3, 33, 64, 96)
    assert ours.dtype == torch.float32
    torch.testing.assert_close(ours, want, atol=5e-5, rtol=1e-4)


def test_compiled_fixed_tiles_build_their_graphs_in_the_first_window(tiny):
    """The warm-up decode's first window compiles every region's graph; no later window or decode
    adds one (one graph per region class and shape, whatever the window's extent)."""
    import torch._dynamo

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    app = _app(tiny, exec_mode="compile", compile_backend=backend)
    app.load(skip_warmup=True)
    assert app.vae is None  # the tile decoder holds the weights; diffusers' model is dropped
    seen = []
    decode_tile = app._decode_tile

    def counting(padded, valid):
        out = decode_tile(padded, valid)
        seen.append(len(graphs))
        return out

    app._decode_tile = counting
    app.decode(torch.zeros(app.latent_shape))
    app.decode(torch.randn(app.latent_shape, generator=torch.Generator().manual_seed(2)))
    assert len(seen) == 2 * 36 and set(seen) == {seen[0]} and seen[0] > 0  # 3 x (3 x 4) windows


def test_compiled_diffusers_tiles_have_room_for_every_edge_shape(tiny):
    """diffusers' windows at the default 320x512x61 take 12 shapes (3 frame counts x 2 x 2) for
    each of 6 resnet shapes, 72 entries on the resnets' one forward, past Dynamo's default
    recompile_limit of 8; the control mode raises the limit and compiles block by block."""
    import torch._dynamo

    torch._dynamo.reset()
    with torch._dynamo.config.patch(recompile_limit=8):
        app = _app(tiny, exec_mode="compile", tiling="diffusers", compile_backend="aot_eager")
        app.load(skip_warmup=True)
        assert torch._dynamo.config.recompile_limit >= 72
