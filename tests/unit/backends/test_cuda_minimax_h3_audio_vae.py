"""MiniMax-H3's audio VAE decoder on CUDA, on CPU: CUDA's own BigVGAN (``audio_vae``), its
strict FP32 loading and the decode around it. The neuron/Trainium port is the reference."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from difflet.backends.cuda.minimax_h3 import audio_vae
from difflet.models.minimax_h3.modeling_vae import MiniMaxH3AudioBigVGANDecoder


def _config():
    return dict(
        latent_channels=4,
        latent_dim=8,
        decoder_dim=32,
        decoder_rates=[4, 2],
        decoder_kernel_sizes=[8, 4],
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        latents_mean=[0.1, -0.2, 0.0, 0.3],
        latents_std=[1.5, 0.5, 1.0, 2.0],
        sampling_rate=32000,
    )


def _decoder(seed=0):
    torch.manual_seed(seed)
    decoder = audio_vae.AudioDecoder(_config()).eval()
    with torch.no_grad():
        for name, parameter in decoder.named_parameters():
            parameter.normal_(0.0, 0.5 if name.endswith(("alpha", "beta")) else 0.2)
    return decoder


def test_loads_the_weight_norm_checkpoint_strictly_in_fp32(tmp_path):
    original = _decoder()
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    weights = {k: v.contiguous() for k, v in original.state_dict().items()}
    assert any(k.endswith(".weight_g") for k in weights)  # the checkpoint's weight-norm form
    save_file(weights, str(tmp_path / "model.safetensors"))
    loaded = audio_vae.load_audio_decoder(tmp_path, device="cpu")
    latents = torch.randn(2, 4, 10)
    with torch.no_grad():
        out = loaded(latents)
        assert out.shape == (2, 1, 80)
        assert torch.equal(out, original(latents))
    assert all(p.dtype == torch.float32 for p in loaded.parameters())


def test_the_vocoder_matches_the_reference():
    """The upsampling is a plain transposed convolution here; the reference's is an exact
    rewrite of it, so the two agree to rounding."""
    decoder = _decoder(1)
    c = _config()
    reference = MiniMaxH3AudioBigVGANDecoder(
        in_channels=c["latent_dim"],
        upsample_initial_channel=c["decoder_dim"],
        upsample_rates=tuple(c["decoder_rates"]),
        upsample_kernel_sizes=tuple(c["decoder_kernel_sizes"]),
        resblock_kernel_sizes=tuple(c["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(v) for v in c["resblock_dilation_sizes"]),
    ).eval()
    reference.load_state_dict(decoder.decoder.state_dict(), strict=True)
    x = torch.randn(2, 8, 12)
    with torch.no_grad():
        torch.testing.assert_close(decoder.decoder(x), reference(x))


def test_decode_denormalizes_and_returns_the_stereo_channels():
    seen = []

    class Recorder(torch.nn.Module):
        config = audio_vae.AudioDecoder(_config()).config

        def forward(self, latents):
            seen.append(latents)
            return latents[:, :1].repeat_interleave(3, -1)

    normalized = torch.randn(2, 4, 5)
    audio = audio_vae.decode_audio(Recorder(), normalized)
    mean = torch.tensor(_config()["latents_mean"]).view(1, -1, 1)
    std = torch.tensor(_config()["latents_std"]).view(1, -1, 1)
    assert torch.equal(seen[0], normalized * std + mean)
    assert audio.shape == (1, 2, 15)
    assert torch.equal(audio[0, 1], seen[0][1, 0].repeat_interleave(3))
