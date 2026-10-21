"""HunyuanVideo 1.5's VAE decoder application (CPU) against diffusers' tiled decode."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from difflet.backends.neuron.hunyuan_video15.vae import NeuronHunyuanVideo15VaeDecoderApplication  # noqa: E402

TINY = dict(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16, 16, 16, 16),
            layers_per_block=1, spatial_compression_ratio=16, temporal_compression_ratio=4,
            downsample_match_channel=True, upsample_match_channel=True, scaling_factor=0.9)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from diffusers import AutoencoderKLHunyuanVideo15

    torch.manual_seed(0)
    path = str(tmp_path_factory.mktemp("hv15vae") / "vae")
    AutoencoderKLHunyuanVideo15(**TINY).save_pretrained(path, safe_serialization=True)
    return path


def test_decode_matches_diffusers_tiled(tiny):
    from diffusers import AutoencoderKLHunyuanVideo15

    app = NeuronHunyuanVideo15VaeDecoderApplication(model_path=tiny, height=96, width=128, num_frames=5,
                                                    tile_size=64, dtype=torch.float32, device="cpu")
    app.load(skip_warmup=True)
    assert app.vae.encoder is None and app.latent_shape == (1, 4, 2, 6, 8)
    z = torch.randn(app.latent_shape, generator=torch.Generator().manual_seed(1))
    ours = app.decode(z)
    ref = AutoencoderKLHunyuanVideo15.from_pretrained(tiny).eval()
    ref.enable_tiling()
    # 64-pixel tiles: a 48-pixel stride is a whole 3 latent units (32 would give 1.5, truncated)
    ref.tile_sample_min_height = ref.tile_sample_min_width = 64
    ref.tile_latent_min_height = ref.tile_latent_min_width = 4
    with torch.no_grad():
        want = ref.decode(z / ref.config.scaling_factor, return_dict=False)[0]
    assert ours.shape == want.shape == (1, 3, 5, 96, 128)
    torch.testing.assert_close(ours, want, atol=1e-5, rtol=1e-5)
