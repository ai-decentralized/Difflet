"""Unit tests for the Wan TPU geometry config (plan, Wan port)."""

import pytest
import torch

from difflet.backends.tpu.wan.config import TpuWanConfig
from difflet.backends.tpu.wan.transformer import checkpoint_key


def _config(**overrides):
    return TpuWanConfig(**overrides)


def test_latent_geometry_matches_the_wan_vae_factors():
    cfg = _config(height=480, width=832, num_frames=9)
    # 8x spatial, 4x temporal (+1 for the first frame), patch (1, 2, 2).
    assert (cfg.latent_frames, cfg.latent_height, cfg.latent_width) == (3, 60, 104)
    assert cfg.image_seq_len == 3 * 30 * 52 == 4680


def test_tp_must_divide_the_attention_heads():
    cfg = _config(num_attention_heads=40, tp_degree=4)
    cfg.validate()
    with pytest.raises(ValueError, match="attention heads"):
        _config(num_attention_heads=40, tp_degree=3).validate()


def test_tp_must_divide_the_ffn_dim():
    with pytest.raises(ValueError, match="ffn_dim"):
        _config(ffn_dim=13825, tp_degree=4).validate()


def test_pixel_dims_must_be_aligned_to_the_vae_and_patch():
    """836 // 8 == 104 truncates to the same latents as 832 — reject it.

    Checking the latent grid alone would let the floor division hide the
    mismatch and quietly generate a different size than was asked for.
    """
    with pytest.raises(ValueError, match="width=836 must be divisible by 16"):
        _config(height=480, width=836, tp_degree=1).validate()
    with pytest.raises(ValueError, match="height=484 must be divisible by 16"):
        _config(height=484, width=832, tp_degree=1).validate()


def test_num_frames_must_land_on_a_temporal_boundary():
    with pytest.raises(ValueError, match=r"num_frames - 1"):
        _config(num_frames=10, tp_degree=1).validate()
    _config(num_frames=13, tp_degree=1).validate()  # (13 - 1) % 4 == 0


@pytest.mark.parametrize(
    "mode", ["context_parallel_enabled", "cfg_parallel_enabled", "sp_enabled"]
)
def test_unimplemented_parallel_modes_fail_at_config_time(mode):
    """Better here, with a message, than as a wrong result on device."""
    with pytest.raises(NotImplementedError, match="TPU backend does not implement"):
        _config(**{mode: True}).validate()


def test_from_pretrained_reads_a_diffusers_config(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps({
        "patch_size": [1, 2, 2], "num_attention_heads": 40, "attention_head_dim": 128,
        "in_channels": 16, "out_channels": 16, "text_dim": 4096, "freq_dim": 256,
        "ffn_dim": 13824, "num_layers": 40, "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads", "eps": 1e-06, "image_dim": None,
        "added_kv_proj_dim": None, "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None, "_class_name": "WanTransformer3DModel",
    }))
    cfg = TpuWanConfig.from_pretrained(tmp_path, tp_degree=4, torch_dtype=torch.bfloat16)
    assert cfg.patch_size == (1, 2, 2)      # list -> tuple, or the unpack fails
    assert cfg.inner_dim == 5120
    assert cfg.tp_degree == 4


def test_checkpoint_key_rewrites_only_the_ffn():
    assert checkpoint_key("blocks.0.ffn.net_in.weight") == "blocks.0.ffn.net.0.proj.weight"
    assert checkpoint_key("blocks.0.ffn.net_out.bias") == "blocks.0.ffn.net.2.bias"
    for untouched in (
        "blocks.0.attn1.to_q.weight", "blocks.0.attn1.norm_q.weight",
        "patch_embedding.weight", "condition_embedder.time_proj.weight",
        "scale_shift_table", "proj_out.bias",
    ):
        assert checkpoint_key(untouched) == untouched
