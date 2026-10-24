"""The LTX-2 video VAE decoder as a neuron-backend application (CPU only): a tiny diffusers
``AutoencoderKLLTX2Video`` saved to disk, decoded by ``NeuronLtx2VaeDecoderApplication`` on one
gloo rank and by diffusers itself. Nothing here touches the device."""

import pytest

torch = pytest.importorskip("torch")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_ltx2_toy import vae_decoder_worker  # noqa: E402

TINY_VAE = {
    "block_out_channels": [16, 32, 64, 128],
    "decoder_block_out_channels": [16, 32, 64],
    "decoder_causal": False,
    "decoder_inject_noise": [False, False, False, False],
    "decoder_layers_per_block": [1, 1, 1, 1],
    "decoder_spatial_padding_mode": "reflect",
    "decoder_spatio_temporal_scaling": [True, True, True],
    "down_block_types": ["LTX2VideoDownBlock3D"] * 4,
    "downsample_type": ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
    "encoder_causal": True,
    "encoder_spatial_padding_mode": "zeros",
    "in_channels": 3,
    "latent_channels": 8,
    "layers_per_block": [1, 1, 1, 1, 1],
    "out_channels": 3,
    "patch_size": 4,
    "patch_size_t": 1,
    "resnet_norm_eps": 1e-06,
    "scaling_factor": 1.0,
    "spatial_compression_ratio": 32,
    "spatio_temporal_scaling": [True, True, True, True],
    "temporal_compression_ratio": 8,
    "timestep_conditioning": False,
    "upsample_factor": [2, 2, 2],
    "upsample_residual": [True, True, True],
}
LATENT_SHAPE = (1, 8, 2, 2, 3)


@pytest.fixture(scope="module")
def vae_dir(tmp_path_factory):
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLLTX2Video

    root = tmp_path_factory.mktemp("tiny_ltx2_vae")
    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(**TINY_VAE).eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.add_(0.05 * torch.randn_like(p))
        vae.latents_mean.copy_(torch.linspace(-1, 1, 8))
    vae.save_pretrained(str(root))
    return root


def test_device_decoder_matches_diffusers_decode(vae_dir):
    from diffusers import AutoencoderKLLTX2Video

    [result] = run_ranks(vae_decoder_worker, str(vae_dir), LATENT_SHAPE, world_size=1,
                         env={"HF_HUB_OFFLINE": "1"})
    vae = AutoencoderKLLTX2Video.from_pretrained(str(vae_dir), torch_dtype=torch.float32).eval()
    z = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        ref = vae.decode(z, None, return_dict=False)[0]
    out = torch.tensor(result["out"])
    assert tuple(out.shape) == (1, 3, 9, 64, 96)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert result["report"]["missing"] == [] and result["report"]["unexpected"] == []
    skipped = result["report"]["skipped"]
    assert any(k.startswith("encoder.") for k in skipped) and "latents_mean" in skipped
    assert result["warmup_shapes"] == [LATENT_SHAPE]


def test_decoder_refuses_compile_mode_and_wrong_latents(vae_dir):
    from difflet.backends.neuron.ltx_2.vae import NeuronLtx2VaeDecoderApplication

    with pytest.raises(ValueError, match="eager only"):
        NeuronLtx2VaeDecoderApplication(model_path=vae_dir, latent_shape=LATENT_SHAPE,
                                        exec_mode="compile", device="cpu")
    with pytest.raises(ValueError, match="latent_channels=8"):
        NeuronLtx2VaeDecoderApplication(model_path=vae_dir, latent_shape=(1, 4, 2, 2, 3),
                                        device="cpu")
    config = json.loads((Path(vae_dir) / "config.json").read_text())
    noisy = Path(vae_dir).parent / "noisy_vae"
    noisy.mkdir(exist_ok=True)
    (noisy / "config.json").write_text(json.dumps({**config, "timestep_conditioning": True}))
    with pytest.raises(NotImplementedError, match="timestep conditioning"):
        NeuronLtx2VaeDecoderApplication(model_path=noisy, latent_shape=LATENT_SHAPE,
                                        device="cpu")
