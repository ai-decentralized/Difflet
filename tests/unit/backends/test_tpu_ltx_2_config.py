"""Unit tests for the LTX-2 TPU geometry config."""

import json

import pytest
import torch

from difflet.backends.tpu.ltx_2.config import TpuLTX2Config

# Lightricks/LTX-2 transformer/config.json, as shipped (non-underscore keys).
_UPSTREAM = {
    "activation_fn": "gelu-approximate", "attention_bias": True, "attention_head_dim": 128,
    "attention_out_bias": True, "audio_attention_head_dim": 64, "audio_cross_attention_dim": 2048,
    "audio_hop_length": 160, "audio_in_channels": 128, "audio_num_attention_heads": 32,
    "audio_out_channels": 128, "audio_patch_size": 1, "audio_patch_size_t": 1,
    "audio_pos_embed_max_pos": 20, "audio_sampling_rate": 16000, "audio_scale_factor": 4,
    "base_height": 2048, "base_width": 2048, "caption_channels": 3840, "causal_offset": 1,
    "cross_attention_dim": 4096, "cross_attn_timestep_scale_multiplier": 1000, "in_channels": 128,
    "norm_elementwise_affine": False, "norm_eps": 1e-06, "num_attention_heads": 32,
    "num_layers": 48, "out_channels": 128, "patch_size": 1, "patch_size_t": 1,
    "pos_embed_max_pos": 20, "qk_norm": "rms_norm_across_heads", "rope_double_precision": True,
    "rope_theta": 10000.0, "rope_type": "split", "timestep_scale_multiplier": 1000,
    "vae_scale_factors": [8, 32, 32],
}


def _config(**overrides):
    return TpuLTX2Config(dict(_UPSTREAM), **overrides)


def test_default_shape_geometry_matches_the_trainium_config():
    cfg = _config(height=512, width=768, num_frames=121, tp_degree=4)
    assert (cfg.latent_num_frames, cfg.latent_height, cfg.latent_width) == (16, 16, 24)
    assert cfg.video_seq_len == 6144
    # 121 frames / 24 fps * 16000 / 160 / 4 latents per second, rounded
    assert (cfg.audio_num_frames, cfg.audio_seq_len) == (126, 126)
    assert (cfg.video_text_dim, cfg.audio_text_dim) == (3840, 3840)
    assert cfg.vae_scale_factors == (8, 32, 32)
    assert cfg.neuron_config.batch_size == 1  # validate_ltx_2_dit_inputs reads it here


def test_tp_must_divide_both_head_counts():
    with pytest.raises(ValueError, match="num_attention_heads"):
        _config(tp_degree=5).validate()


@pytest.mark.parametrize("changed, match", [
    ({"height": 520}, "spatial scale"),
    ({"num_frames": 120}, "num_frames"),
])
def test_shape_rules(changed, match):
    with pytest.raises(ValueError, match=match):
        _config(**changed).validate()


@pytest.mark.parametrize("mode", ["context_parallel_enabled", "cfg_parallel_enabled", "sp_enabled"])
def test_unimplemented_parallel_modes_fail_at_config_time(mode):
    with pytest.raises(NotImplementedError):
        _config(**{mode: True}).validate()


def test_from_pretrained_reads_the_upstream_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({**_UPSTREAM, "_class_name": "LTX2VideoTransformer3DModel"}))
    cfg = TpuLTX2Config.from_pretrained(tmp_path, tp_degree=4, torch_dtype=torch.bfloat16)
    assert cfg.inner_dim == 4096
    assert not hasattr(cfg, "_class_name")
