"""Native spatial VAE tiling, including edge tiles on the real decoder."""

import torch
from diffusers import AutoencoderKLHunyuanVideo15, HunyuanVideo15Pipeline

from tests.unit.models.hunyuan_video.test_cuda_application15 import _application, _pipeline


def _vae(dtype=torch.float32):
    torch.manual_seed(19)
    return (
        AutoencoderKLHunyuanVideo15(
            latent_channels=4,
            block_out_channels=(8, 8, 8, 8, 8),
            layers_per_block=1,
        )
        .to(dtype)
        .eval()
    )


def test_hv15_loading_companions_enables_native_tiling(tmp_path, monkeypatch):
    from difflet.backends.cuda.hunyuan_video_15 import application
    from tests.unit.backends.test_cuda_hunyuan15_transformer import _component

    _, component = _component(tmp_path)
    app = _application(tmp_path)
    app.config = component.dit_config
    app.transformer = component
    app.adapter = application.HunyuanVideo15DiTAdapter(component)
    monkeypatch.setattr(application, "configure_conditioning", lambda pipe: None)
    base = _pipeline(None)
    base.register_modules(
        vae=_vae(), text_encoder=torch.nn.Identity(), text_encoder_2=torch.nn.Identity()
    )
    monkeypatch.setattr(HunyuanVideo15Pipeline, "from_pretrained", lambda *a, **k: base)
    app.load_companions()
    # Diffusers' native tiles: 256 px samples (16 latent), 25% overlap, no slicing.
    vae = app.pipe.vae
    assert vae.use_tiling is True and vae.use_slicing is False
    assert vae.tile_sample_min_height == vae.tile_sample_min_width == 256
    assert vae.tile_latent_min_height == vae.tile_latent_min_width == 16
    assert vae.tile_overlap_factor == 0.25
