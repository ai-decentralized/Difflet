"""MiniMax-H3's visual VAE on CUDA, on CPU: CUDA's own encoder and decoder (``vae``), their
strict FP32 loading, the decoder's arithmetic around its FP16 blocks, and the tiled encode and
decode. The neuron/Trainium port is the reference the modules are checked against."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from difflet.backends.cuda.minimax_h3 import vae
from difflet.models.minimax_h3.host_components import HostVideoEncoder, _HostVideoEncoderModule
from difflet.models.minimax_h3.modeling_vae import MiniMaxH3VideoViTDecoder3d
from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video


def _config():
    return dict(
        in_channels=3,
        latent_channels=24,
        block_out_channels=[8, 8, 8, 8],
        layers_per_block=1,
        spatial_downsample_factors=[2, 2, 2, 2],
        temporal_downsample_factors=[1, 2, 2, 1],
        norm_num_groups=4,
        norm_eps=1e-6,
        spatial_padding_mode="reflect",
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=24,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        decoder_rope_theta=100.0,
        decoder_rope_dim_ratio=0.75,
        decoder_norm_eps=1e-5,
        out_channels=3,
        clip_length=17,
        token_drop=3,
        latents_mean=[0.1] * 24,
        latents_std=[1.3] * 24,
    )


def _randomized(module, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.05)
    return module.eval()


def test_encoder_and_decoder_each_load_their_half_of_the_vae_strictly_in_fp32(tmp_path):
    config = _config()
    encoder = _randomized(vae.VideoEncoder(config), 0)
    decoder = _randomized(vae.VideoDecoder(config), 1)
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = {**encoder.state_dict(), **decoder.state_dict()}
    save_file({k: v.contiguous() for k, v in weights.items()}, str(tmp_path / "model.safetensors"))
    pixels = torch.randn(1, 3, 1, 64, 48)
    latents = torch.randn(1, 24, 2, 2, 3)
    with torch.no_grad():
        loaded = vae.load_video_encoder(tmp_path, device="cpu")
        assert torch.equal(loaded.moments(pixels), encoder.moments(pixels))
        loaded_decoder = vae.load_video_decoder(tmp_path, device="cpu")
        assert torch.equal(loaded_decoder(latents), decoder(latents))
    for module in (loaded, loaded_decoder):
        assert all(p.dtype == torch.float32 for p in module.parameters())
        assert module.config.latents_mean == config["latents_mean"]


def test_the_config_derives_the_compression_from_the_factors():
    decoder = vae.VideoDecoder(_config())
    assert (
        decoder.config.spatial_compression_ratio,
        decoder.config.temporal_compression_ratio,
    ) == (
        16,
        4,
    )
    assert (decoder.decoder.patch_size, decoder.decoder.patch_size_t) == (16, 4)


def test_the_tiled_keyframe_encode_matches_the_reference():
    """Over 256 pixels on a side, so the encode is tiled and stitched."""
    config = _config()
    reference = _randomized(_HostVideoEncoderModule(config), 2)
    encoder = vae.VideoEncoder(config).eval()
    encoder.load_state_dict(reference.state_dict(), strict=True)
    pixels = torch.randn(1, 3, 1, 320, 448)
    with torch.no_grad():
        got = encoder.moments(pixels)
        want = HostVideoEncoder(reference, config).moments(pixels)
    assert got.shape == (1, 48, 1, 20, 28)
    torch.testing.assert_close(got, want)


def test_in_fp32_the_decoder_matches_the_reference():
    config = _config()
    decoder = _randomized(vae.VideoDecoder(config), 3).decoder
    reference = MiniMaxH3VideoViTDecoder3d(
        in_channels=24,
        out_channels=3,
        patch_size=16,
        patch_size_t=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=24,
        num_register_tokens=2,
        ffn_mult=2,
    ).eval()
    reference.load_state_dict(decoder.state_dict(), strict=True)
    x = torch.randn(1, 24, 3, 2, 3)
    with torch.no_grad():
        torch.testing.assert_close(decoder(x), reference(x))


def test_under_autocast_the_projections_stay_fp32_and_the_tables_follow_the_activations(
    monkeypatch,
):
    vit = _randomized(vae.VideoDecoder(_config()), 4).decoder
    seen = []
    linear = vae._fp32_linear

    def spy(module, hidden_states):
        out = linear(module, hidden_states)
        seen.append((module.weight.dtype, hidden_states.dtype, out.dtype))
        return out

    monkeypatch.setattr(vae, "_fp32_linear", spy)
    rope = vit.rope.forward
    tables = []
    monkeypatch.setattr(vit.rope, "forward", lambda ids: tables.append(ids.dtype) or rope(ids))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = vit(torch.randn(1, 24, 3, 2, 3, dtype=torch.bfloat16))
    # proj_in on the BF16 activations and proj_out on the FP32 norm output, both in FP32.
    assert seen == [
        (torch.float32, torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32, torch.float32),
    ]
    assert tables == [torch.bfloat16] and out.dtype == torch.float32


class _Decoder(torch.nn.Module):
    """A deterministic stand-in: each latent tile to pixels, its values traceable."""

    def __init__(self):
        super().__init__()
        self.config = vae._namespace(_config())
        self.tiles = []

    def forward(self, tile):
        self.tiles.append(tuple(tile.shape))
        upsampled = tile[:, :3].repeat_interleave(4, 2).repeat_interleave(16, 3)
        return upsampled.repeat_interleave(16, 4).tanh()


def test_the_tiled_decode_matches_the_reference_chunking_and_tiling():
    latents = torch.randn(1, 24, 9, 16, 28)
    decoder = _Decoder()
    video = vae.decode_video(decoder, latents)
    c = decoder.config
    reference = decode_minimax_h3_video(
        _Decoder(),
        latents,
        latents_mean=c.latents_mean,
        latents_std=c.latents_std,
        clip_length=c.clip_length,
        token_drop=c.token_drop,
        temporal_compression_ratio=c.temporal_compression_ratio,
        spatial_compression_ratio=c.spatial_compression_ratio,
    )
    assert video.shape == (1, 3, 30, 256, 448) and video.device == latents.device
    assert set(decoder.tiles) == {(1, 24, 7, 16, 16)}  # 256-pixel tiles of overlapping clips
    torch.testing.assert_close(video, reference, rtol=0, atol=0)
