from __future__ import annotations

from types import SimpleNamespace

import torch

from difflet.backends.trainium.core.config import NeuronConfig
from difflet.backends.trainium.minimax_h3.vae import (
    MiniMaxH3AudioVAEDecoderInferenceConfig,
    MiniMaxH3VideoVAEDecoderInferenceConfig,
    ModelWrapperMiniMaxH3AudioVAE,
    ModelWrapperMiniMaxH3VideoVAE,
    NeuronMiniMaxH3AudioVAEDecoderApplication,
    NeuronMiniMaxH3VideoVAEDecoderApplication,
)
from difflet.models.minimax_h3.modeling_vae import (
    MiniMaxH3AudioBigVGANDecoder,
    MiniMaxH3AudioUpSample1d,
    MiniMaxH3VideoViTDecoder3d,
)
from difflet.models.minimax_h3.pipeline import (
    _split_tiles,
    decode_minimax_h3_audio,
    decode_minimax_h3_video,
)


def _neuron_config(batch_size=1):
    return NeuronConfig(
        batch_size=batch_size,
        tp_degree=1,
        world_size=1,
        torch_dtype=torch.float32,
    )


def _video_config():
    values = {
        "out_channels": 3,
        "latent_channels": 24,
        "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
        "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
        "decoder_num_layers": 1,
        "decoder_num_attention_heads": 2,
        "decoder_attention_head_dim": 8,
        "decoder_num_register_tokens": 2,
        "decoder_ffn_mult": 2,
        "decoder_rope_theta": 100.0,
        "decoder_rope_dim_ratio": 0.75,
        "decoder_norm_eps": 1e-5,
        "clip_length": 17,
        "token_drop": 3,
        "latents_mean": [0.0] * 24,
        "latents_std": [1.0] * 24,
    }
    return MiniMaxH3VideoVAEDecoderInferenceConfig(
        neuron_config=_neuron_config(),
        load_config=lambda config: config.__dict__.update(values),
        height=768,
        width=1344,
        num_frames=124,
    )


def _audio_config():
    values = {
        "latent_dim": 16,
        "latent_channels": 4,
        "decoder_dim": 128,
        "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
        "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
        "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1]],
        "sampling_rate": 32000,
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
    }
    return MiniMaxH3AudioVAEDecoderInferenceConfig(
        neuron_config=_neuron_config(batch_size=2),
        load_config=lambda config: config.__dict__.update(values),
        num_frames=124,
    )


def test_h3_decoder_only_modules_run_tiny_cpu_forward():
    video = MiniMaxH3VideoViTDecoder3d(
        in_channels=2,
        out_channels=3,
        patch_size=2,
        patch_size_t=2,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        num_register_tokens=2,
        ffn_mult=2,
        rope_dim_ratio=0.75,
        sequence_alignment=8,
    ).eval()
    audio = MiniMaxH3AudioBigVGANDecoder(
        in_channels=4,
        upsample_initial_channel=8,
        upsample_rates=(2,),
        upsample_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
    ).eval()

    assert video(torch.randn(1, 2, 2, 2, 2)).shape == (1, 3, 4, 4, 4)
    assert audio(torch.randn(1, 4, 3)).shape == (1, 1, 6)


def test_h3_audio_polyphase_upsample_matches_transposed_convolution():
    module = MiniMaxH3AudioUpSample1d(ratio=2, kernel_size=12)
    hidden_states = torch.randn(2, 3, 17)
    padded = torch.nn.functional.pad(
        hidden_states,
        (module.pad, module.pad),
        mode="replicate",
    )
    expected = module.ratio * torch.nn.functional.conv_transpose1d(
        padded,
        module.filter.expand(3, -1, -1),
        stride=module.stride,
        groups=3,
    )
    expected = expected[..., module.pad_left : -module.pad_right]

    assert torch.allclose(module(hidden_states), expected, atol=5e-7, rtol=1e-6)


def test_h3_vae_configs_and_wrappers_expose_fixed_graph_shapes():
    video_config = _video_config()
    audio_config = _audio_config()
    video_input = ModelWrapperMiniMaxH3VideoVAE(video_config, None).input_generator()[0][0]
    audio_input = ModelWrapperMiniMaxH3AudioVAE(audio_config, None).input_generator()[0][0]

    assert video_config.tile_latent_frames == 7
    assert video_input.shape == (1, 24, 7, 16, 16)
    assert audio_config.audio_latent_frames == 207
    assert audio_config.audio_num_chunks == 13
    assert audio_config.audio_chunk_latent_frames == 48
    assert audio_input.shape == (2, 4, 48)


def test_h3_vae_checkpoint_conversion_keeps_decoder_weights_only():
    weight_v = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    weight_g = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1)
    state = {
        "encoder.weight": torch.tensor(0),
        "quant_conv.weight": torch.tensor(1),
        "post_quant_conv.weight": torch.tensor(2),
        "decoder.block.weight": torch.tensor(3),
        "decoder.conv.weight_v": weight_v,
        "decoder.conv.weight_g": weight_g,
        "dec_in_proj.weight": torch.tensor(4),
    }

    video = NeuronMiniMaxH3VideoVAEDecoderApplication.convert_hf_to_neuron_state_dict(
        state, SimpleNamespace()
    )
    audio = NeuronMiniMaxH3AudioVAEDecoderApplication.convert_hf_to_neuron_state_dict(
        state, SimpleNamespace()
    )

    assert set(video) == {
        "post_quant_conv.weight",
        "decoder.block.weight",
        "decoder.conv.weight_g",
        "decoder.conv.weight_v",
    }
    assert set(audio) == {
        "dec_in_proj.weight",
        "decoder.block.weight",
        "decoder.conv.weight",
    }
    assert torch.equal(
        audio["decoder.conv.weight"],
        torch._weight_norm(weight_v, weight_g, 0),
    )


def test_h3_official_canvas_tile_plan_covers_without_shape_drift():
    y_starts, _, y_overlaps = _split_tiles(768)
    x_starts, _, x_overlaps = _split_tiles(1344)

    assert y_starts == [0, 160, 336, 512]
    assert y_overlaps == [96, 80, 80]
    assert x_starts[-1] + 256 == 1344
    assert len(x_starts) == 7
    assert all(value % 16 == 0 for value in x_overlaps)


def test_h3_video_decode_controller_restores_124_frames():
    calls = []

    def decoder(tile):
        calls.append(tuple(tile.shape))
        return torch.zeros(1, 3, 28, 32, 32)

    video = decode_minimax_h3_video(
        decoder,
        torch.zeros(1, 24, 37, 2, 2),
        latents_mean=[0.0] * 24,
        latents_std=[1.0] * 24,
        tile_sample_size=32,
        tile_min_overlap=16,
    )

    assert calls == [(1, 24, 7, 2, 2)] * 7
    assert video.shape == (1, 3, 124, 32, 32)
    assert torch.allclose(video[:, 0], torch.full_like(video[:, 0], 0.485))


def test_h3_audio_decode_controller_uses_stereo_as_batch():
    seen = []

    def decoder(latents):
        seen.append(latents)
        return torch.zeros(2, 1, 1600)

    waveform = decode_minimax_h3_audio(
        decoder,
        torch.zeros(2, 4, 2),
        latents_mean=[1.0] * 4,
        latents_std=[2.0] * 4,
    )

    assert torch.all(seen[0] == 1)
    assert waveform.shape == (1, 2, 1600)


def test_h3_audio_decode_controller_runs_fixed_overlapped_neuron_chunks():
    seen = []

    def decoder(latents):
        seen.append(latents.clone())
        return latents[:, :1].repeat_interleave(800, dim=-1)

    waveform = decode_minimax_h3_audio(
        decoder,
        torch.arange(207).view(1, 1, 207).expand(2, 4, -1).float(),
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        chunk_latent_frames=48,
        chunk_core_frames=16,
    )

    assert [tuple(value.shape) for value in seen] == [(2, 4, 48)] * 11
    assert int(seen[0][0, 0, 0]) == 0
    assert int(seen[1][0, 0, 0]) == 16
    assert int(seen[-1][0, 0, 0]) == 159
    assert waveform.shape == (1, 2, 207 * 800)
    assert torch.equal(
        waveform[0, 0],
        torch.arange(207, dtype=torch.float32).repeat_interleave(800),
    )
