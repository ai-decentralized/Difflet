"""CPU tests for the HunyuanVideo TPU application's construction contract.

No chip: these pin what the application derives from a checkpoint directory
and what the entry point routes, not the forward (the oracle harness does
that on device)."""

from __future__ import annotations

import json

import pytest
import torch

from difflet.models.hunyuan_video.entry import create_hunyuan_video_application
from difflet.models.hunyuan_video.tpu_application import TpuHunyuanVideoApplication
from difflet.pipeline.parallel_config import DiffletParallelConfig

_UPSTREAM = {
    "attention_head_dim": 128, "guidance_embeds": True, "in_channels": 16, "mlp_ratio": 4.0,
    "num_attention_heads": 24, "num_layers": 20, "num_refiner_layers": 2,
    "num_single_layers": 40, "out_channels": 16, "patch_size": 2, "patch_size_t": 1,
    "pooled_projection_dim": 768, "qk_norm": "rms_norm", "rope_axes_dim": [16, 56, 56],
    "rope_theta": 256.0, "text_embed_dim": 4096,
}


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(_UPSTREAM))
    return tmp_path


def test_entry_routes_tpu_to_the_tpu_application(snapshot):
    app = create_hunyuan_video_application(
        model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16", shape={"height": 320, "width": 512, "num_frames": 61},
        backend="tpu", teacache_cadence=2,
    )
    assert isinstance(app, TpuHunyuanVideoApplication)
    assert app.dtype is torch.bfloat16
    assert app.config.tp_degree == 4
    assert app.kwargs["teacache_cadence"] == 2
    assert app.pipeline is None  # built by load_eager, on device


def test_entry_still_refuses_cfg_parallel_on_tpu(snapshot):
    with pytest.raises(NotImplementedError, match="guidance-distilled"):
        create_hunyuan_video_application(
            model_path=str(snapshot),
            parallel=DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True),
            dtype="bf16", shape={"height": 320, "width": 512, "num_frames": 61},
            backend="tpu",
        )


def test_dit_input_contract_matches_the_bundle(snapshot):
    app = create_hunyuan_video_application(
        model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4),
        dtype=torch.bfloat16, shape={"height": 320, "width": 512, "num_frames": 61},
        backend="tpu",
    )
    contract = app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 16, 16, 40, 64)
    assert contract["encoder_hidden_states"]["shape"] == (1, 256, 4096)
    assert contract["encoder_attention_mask"]["dtype"] is torch.int64
    assert contract["pooled_projections"]["shape"] == (1, 768)
    assert set(contract) == {
        "hidden_states", "timestep", "encoder_hidden_states",
        "encoder_attention_mask", "pooled_projections", "guidance",
    }


def test_missing_transformer_config_is_an_explicit_error(tmp_path):
    app = TpuHunyuanVideoApplication(
        model_path=str(tmp_path), parallel=DiffletParallelConfig(tp_degree=4),
        dtype=torch.bfloat16, shape={"height": 320, "width": 512, "num_frames": 61},
    )
    with pytest.raises(NotImplementedError, match="transformer/config.json"):
        app.dit_input_contract()
