"""Unit tests for the HunyuanVideo TPU geometry config and checkpoint mapping."""

import json

import pytest
import torch

from difflet.backends.tpu.core.checkpoint import CheckpointSlice
from difflet.backends.tpu.hunyuan_video.config import TpuHunyuanVideoConfig
from difflet.backends.tpu.hunyuan_video.transformer import make_checkpoint_key

# The community 1.0 checkpoint's transformer/config.json, as shipped.
_UPSTREAM = {
    "_class_name": "HunyuanVideoTransformer3DModel", "attention_head_dim": 128,
    "guidance_embeds": True, "in_channels": 16, "mlp_ratio": 4.0, "num_attention_heads": 24,
    "num_layers": 20, "num_refiner_layers": 2, "num_single_layers": 40, "out_channels": 16,
    "patch_size": 2, "patch_size_t": 1, "pooled_projection_dim": 768, "qk_norm": "rms_norm",
    "rope_axes_dim": [16, 56, 56], "rope_theta": 256.0, "text_embed_dim": 4096,
}


def _config(**overrides):
    return TpuHunyuanVideoConfig(**overrides)


def test_latent_geometry_matches_the_causal_vae_factors():
    cfg = _config(height=320, width=512, num_frames=61)
    assert (cfg.latent_frames, cfg.latent_height, cfg.latent_width) == (16, 40, 64)
    assert cfg.image_seq_len == 16 * 20 * 32  # patch 2x2 spatially, 1 temporally


def test_tp_must_divide_the_attention_heads():
    with pytest.raises(ValueError, match="attention heads"):
        _config(tp_degree=5).validate()


def test_pixel_dims_must_be_aligned_to_the_vae_and_patch():
    with pytest.raises(ValueError, match="divisible by 16"):
        _config(height=328).validate()


def test_num_frames_must_be_4n_plus_1():
    with pytest.raises(ValueError, match="num_frames"):
        _config(num_frames=60).validate()


@pytest.mark.parametrize(
    "mode", ["context_parallel_enabled", "cfg_parallel_enabled", "sp_enabled"]
)
def test_unimplemented_parallel_modes_fail_at_config_time(mode):
    with pytest.raises(NotImplementedError):
        _config(**{mode: True}).validate()


def test_from_pretrained_reads_the_upstream_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_UPSTREAM))
    cfg = TpuHunyuanVideoConfig.from_pretrained(tmp_path, tp_degree=4, torch_dtype=torch.bfloat16)
    assert cfg.rope_axes_dim == (16, 56, 56)  # list -> tuple
    assert cfg.inner_dim == 3072
    assert (cfg.tp_degree, cfg.num_single_layers) == (4, 40)


def test_checkpoint_key_splits_only_the_single_block_proj_out():
    """Upstream stores one proj_out over cat([attn, mlp]); the modeling keeps
    two row-parallel halves (see HunyuanVideoSingleTransformerBlock), so the
    loader must window the checkpoint tensor — the same split the Trainium
    convert_hf_to_neuron_state_dict does eagerly."""
    key = make_checkpoint_key(3072)
    assert key("single_transformer_blocks.7.proj_out_attn.weight") == CheckpointSlice(
        "single_transformer_blocks.7.proj_out.weight", dim=1, start=0, stop=3072
    )
    assert key("single_transformer_blocks.7.proj_out_mlp.weight") == CheckpointSlice(
        "single_transformer_blocks.7.proj_out.weight", dim=1, start=3072, stop=None
    )
    # skip_bias_add: the bias lives on the attn half; the mlp half has none.
    assert key("single_transformer_blocks.7.proj_out_attn.bias") == (
        "single_transformer_blocks.7.proj_out.bias"
    )
    for untouched in (
        "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.norm1.linear.weight",
        "single_transformer_blocks.7.proj_mlp.weight", "x_embedder.proj.weight",
        "context_embedder.token_refiner.refiner_blocks.0.attn.to_out.0.weight",
        "proj_out.weight", "norm_out.linear.bias",
    ):
        assert key(untouched) == untouched
