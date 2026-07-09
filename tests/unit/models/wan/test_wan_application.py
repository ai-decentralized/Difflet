"""Coverage for difflet.models.wan.application pure helpers + skeleton app.

Skips runtime compile/load (no Neuron). Exercises dtype normalization, the
latent-frame formula, the per-component rank clamp, and a component-less
``NeuronWanApplication`` built against a model path with no config.json (so no
sub-apps are constructed), plus its component declaration and call fallback.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from difflet.models.wan import application as app


_TRANSFORMER_CFG = {
    "num_attention_heads": 4,
    "attention_head_dim": 16,
    "in_channels": 4,
    "out_channels": 4,
    "text_dim": 24,
    "freq_dim": 32,
    "ffn_dim": 48,
    "num_layers": 2,
    "patch_size": [1, 2, 2],
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "eps": 1e-6,
}
_TEXT_ENCODER_CFG = {
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_heads": 4,
    "num_layers": 2,
    "vocab_size": 128,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "is_gated_act": True,
    "dense_act_fn": "gelu_new",
    "feed_forward_proj": "gated-gelu",
    "layer_norm_epsilon": 1e-6,
}
_VAE_CFG = {
    "base_dim": 4,
    "z_dim": 16,
    "dim_mult": [1, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [True],
    "dropout": 0.0,
}


def _write_config(root, subfolder, cfg):
    d = root / subfolder
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))


# ---------------------------------------------------------------------------
# Pure helpers


def test_normalize_dtype_passthrough_and_aliases():
    assert app._normalize_dtype(torch.bfloat16) is torch.bfloat16
    assert app._normalize_dtype("bf16") is torch.bfloat16
    assert app._normalize_dtype("bfloat16") is torch.bfloat16
    assert app._normalize_dtype("torch.bfloat16") is torch.bfloat16
    assert app._normalize_dtype("fp32") is torch.float32
    assert app._normalize_dtype("float32") is torch.float32
    assert app._normalize_dtype("torch.float32") is torch.float32


def test_normalize_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported Wan dtype"):
        app._normalize_dtype("fp8")


def test_latent_num_frames_formula():
    assert app._latent_num_frames(1) == 1
    assert app._latent_num_frames(5) == 2
    assert app._latent_num_frames(9) == 3
    assert app._latent_num_frames(81) == 21


# ---------------------------------------------------------------------------
# Component inference-config builders (config only, no compile/load)


def test_create_wan_backbone_config(tmp_path):
    _write_config(tmp_path, "transformer", _TRANSFORMER_CFG)
    cfg = app.create_wan_backbone_config(
        model_path=str(tmp_path),
        world_size=1,
        tp_degree=1,
        dtype=torch.float32,
        height=64,
        width=64,
        num_frames=3,
        cfg_parallel_enabled=True,
    )
    assert cfg.neuron_config.tp_degree == 1
    assert cfg.neuron_config.skip_sharding is False  # presharding default-on: skip_sharding removed
    assert cfg.cfg_parallel_enabled is True


def test_create_wan_text_encoder_config(tmp_path):
    _write_config(tmp_path, "text_encoder", _TEXT_ENCODER_CFG)
    cfg = app.create_wan_text_encoder_config(
        model_path=str(tmp_path),
        world_size=1,
        tp_degree=1,
        dtype=torch.float32,
        text_seq_len=128,
    )
    assert cfg.text_seq_len == 128
    assert cfg.neuron_config.skip_sharding is False  # presharding default-on: skip_sharding removed


def test_create_wan_vae_decoder_config(tmp_path):
    _write_config(tmp_path, "vae", _VAE_CFG)
    cfg = app.create_wan_vae_decoder_config(
        model_path=str(tmp_path),
        world_size=1,
        tp_degree=1,
        dtype=torch.float32,
        height=64,
        width=64,
        num_frames=5,
    )
    # VAE is pinned to a single logical NeuronCore (LNC=1).
    assert cfg.neuron_config.logical_nc_config == 1


# ---------------------------------------------------------------------------
# _component_load_rank_range


def _component_with_world(world_size):
    return SimpleNamespace(
        config=SimpleNamespace(neuron_config=SimpleNamespace(world_size=world_size))
    )


def test_rank_range_single_core_with_start():
    comp = _component_with_world(1)
    assert app.NeuronWanApplication._component_load_rank_range(
        comp, start_rank_id=3, local_ranks_size=4
    ) == (0, 1)


def test_rank_range_single_core_without_start():
    comp = _component_with_world(1)
    assert app.NeuronWanApplication._component_load_rank_range(
        comp, start_rank_id=None, local_ranks_size=4
    ) == (None, 1)


def test_rank_range_clamps_to_smaller_world():
    comp = _component_with_world(2)
    assert app.NeuronWanApplication._component_load_rank_range(
        comp, start_rank_id=0, local_ranks_size=8
    ) == (0, 2)


def test_rank_range_passes_through_when_world_ge_local():
    comp = _component_with_world(8)
    assert app.NeuronWanApplication._component_load_rank_range(
        comp, start_rank_id=0, local_ranks_size=8
    ) == (0, 8)


def test_rank_range_handles_missing_config():
    comp = SimpleNamespace()  # no .config
    assert app.NeuronWanApplication._component_load_rank_range(
        comp, start_rank_id=2, local_ranks_size=4
    ) == (2, 4)


# ---------------------------------------------------------------------------
# Skeleton application (no config.json → no sub-apps constructed)


def _parallel(**overrides):
    base = dict(
        world_size=1,
        tp_degree=1,
        cp_degree=1,
        cp_mode="gather_kv",
        cfg_parallel_enabled=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_skeleton_app_has_no_components(tmp_path):
    application = app.NeuronWanApplication(
        model_path=str(tmp_path),
        parallel=_parallel(),
        dtype="bf16",
        shape={"height": None, "width": None, "num_frames": None},
    )
    assert application.components() == []
    assert application.transformer is None
    assert application.text_encoder is None
    assert application.vae_decoder is None
    assert application.dtype is torch.bfloat16


def test_skeleton_app_no_components_messages(tmp_path):
    application = app.NeuronWanApplication(
        model_path=str(tmp_path),
        parallel=_parallel(),
        dtype="fp32",
        shape={},
    )
    assert "compile" in application.no_components_message("compile")
    assert "load" in application.no_components_message("load")
    # Unknown action falls through to the base-class default message.
    assert "describe" in application.no_components_message("describe")


def test_skeleton_app_call_returns_zero_latents(tmp_path):
    application = app.NeuronWanApplication(
        model_path=str(tmp_path),
        parallel=_parallel(),
        dtype="bf16",
        shape={"height": 64, "width": 64, "num_frames": 9},
    )
    out = application(
        batch_size=2,
        channels=16,
        num_latent_frames=3,
        latent_height=4,
        latent_width=5,
    )
    assert out.shape == (2, 16, 3, 4, 5)
    assert torch.count_nonzero(out) == 0


def test_components_lists_all_active_subapps(tmp_path):
    application = app.NeuronWanApplication(
        model_path=str(tmp_path),
        parallel=_parallel(),
        dtype="fp32",
        shape={},
    )
    # Inject sentinels so each conditional append in components() is exercised.
    application.text_encoder = SimpleNamespace(name="te")
    application.transformer = SimpleNamespace(name="t1")
    application.transformer_2 = SimpleNamespace(name="t2")
    application.vae_decoder = SimpleNamespace(name="vae")
    names = [spec.name for spec in application.components()]
    assert names == ["text_encoder", "transformer", "transformer_2", "vae_decoder"]


def test_call_routes_to_transformer_with_three_args(tmp_path):
    application = app.NeuronWanApplication(
        model_path=str(tmp_path),
        parallel=_parallel(),
        dtype="fp32",
        shape={},
    )
    sentinel = torch.ones(1, 2)

    def fake_transformer(hidden, timestep, encoder):
        return sentinel

    application.transformer = fake_transformer
    out = application(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert out is sentinel


def test_call_routes_to_pipeline_when_runtime_components_present(tmp_path):
    application = app.NeuronWanApplication(
        model_path=str(tmp_path),
        parallel=_parallel(),
        dtype="fp32",
        shape={"height": 8, "width": 8, "num_frames": 1},
    )
    application.transformer = None

    class _VAE:
        dtype = torch.float32
        config = SimpleNamespace()

        def __call__(self, latents):
            return latents

    application.pipeline.vae_decoder = _VAE()
    assert application.pipeline.has_runtime_components() is True
    out = application(
        prompt_embeds=torch.ones((1, 4, 8)),
        latents=torch.zeros((1, 16, 1, 1, 1)),
        height=8,
        width=8,
        num_frames=1,
        output_type="latent",
    )
    # Routed through the orchestrator (returns a pipeline output, not zeros).
    assert hasattr(out, "frames")


def test_skeleton_app_uses_default_shape_values(tmp_path):
    # height/width/num_frames None → defaults 480/832/9 used internally.
    application = app.NeuronWanApplication(
        model_path=str(tmp_path),
        parallel=_parallel(cfg_parallel_enabled=True),
        dtype="bf16",
        shape={"height": None, "width": None, "num_frames": None},
        batch_size=1,
    )
    # CFG parallel path is taken in __init__ (backbone_batch_size=2) but with no
    # config.json present nothing is built; the orchestrator still attaches.
    assert application.pipeline is not None
    assert application.pipeline.has_runtime_components() is False
