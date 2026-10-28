"""LTX-2 tensor-parallel all-reduce precision (bf16 default, fp32 opt-in)."""

from __future__ import annotations

import argparse
import types

import pytest
import torch
from torch import nn

from difflet.backends.trainium.ltx_2 import transformer as lmod
from difflet.pipeline.difflet_pipeline import _cache_application_kwargs


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, in_features, out_features, **kwargs):
        self.calls.append(kwargs)
        return nn.Linear(in_features, out_features, bias=kwargs.get("bias", True))


def _tiny_transformer():
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    return LTX2VideoTransformer3DModel(
        in_channels=8, out_channels=8, patch_size=1, patch_size_t=1,
        num_attention_heads=2, attention_head_dim=4, cross_attention_dim=8,
        vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20, base_height=2048,
        base_width=2048, audio_in_channels=8, audio_out_channels=8,
        audio_patch_size=16, audio_patch_size_t=1, audio_num_attention_heads=2,
        audio_attention_head_dim=4, audio_cross_attention_dim=8, audio_scale_factor=4,
        audio_pos_embed_max_pos=20, audio_sampling_rate=16000, audio_hop_length=160,
        num_layers=1, caption_channels=8, rope_double_precision=False,
        use_prompt_embeddings=True,
    )


def _shard(monkeypatch, **kwargs) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(lmod, "RowParallelLinear", recorder)
    monkeypatch.setattr(lmod, "_patch_ltx2_rope_for_tp", lambda transformer, rank_util: None)
    lmod._shard_ltx2_transformer(_tiny_transformer(), 2, types.SimpleNamespace(), **kwargs)
    return recorder


def test_default_is_bf16(monkeypatch):
    recorder = _shard(monkeypatch)
    assert len(recorder.calls) == len(lmod._LTX2_ATTN_ATTRS) + 2  # + ff, audio_ff
    assert {call["reduce_dtype"] for call in recorder.calls} == {torch.bfloat16}


def test_fp32_opt_in(monkeypatch):
    recorder = _shard(monkeypatch, reduce_dtype=torch.float32)
    assert {call["reduce_dtype"] for call in recorder.calls} == {torch.float32}


def test_helper_requires_reduce_dtype():
    with pytest.raises(TypeError):
        lmod._row_parallel_like(nn.Linear(8, 8), input_is_parallel=True)


def test_transformer_config_carries_reduce_dtype(monkeypatch):
    from difflet.models.ltx_2 import application as app_mod

    monkeypatch.setattr(app_mod, "load_diffusers_config", lambda path: (lambda cfg: None))
    captured = {}

    class _Config:
        def __init__(self, *, neuron_config, **kwargs):
            captured["neuron_config"] = neuron_config

    monkeypatch.setattr(lmod, "LTX2TransformerInferenceConfig", _Config)
    for kwargs, expected in (({}, torch.bfloat16), ({"tp_reduce_dtype": torch.float32}, torch.float32)):
        app_mod.create_ltx_2_transformer_config(
            model_path="/nonexistent", world_size=4, tp_degree=4, dtype=torch.bfloat16,
            height=480, width=704, num_frames=49, **kwargs,
        )
        assert captured["neuron_config"].rpl_reduce_dtype is expected


# Pre-bf16 LTX-2 artifacts hashed no tp_reduce_dtype application kwarg.
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"tp_reduce_dtype": "fp32"}, None),
        ({}, {"tp_reduce_dtype": "bfloat16"}),
        (None, {"tp_reduce_dtype": "bfloat16"}),
        ({"tp_reduce_dtype": "bf16"}, {"tp_reduce_dtype": "bfloat16"}),
        ({"tp_reduce_dtype": "fp32", "teacache_cadence": 2}, {"teacache_cadence": 2}),
    ],
)
def test_cache_application_kwargs(kwargs, expected):
    assert _cache_application_kwargs(kwargs, model_name="ltx_2") == expected


def test_cache_application_kwargs_other_backends_unchanged():
    # model_name is None off-Trainium: no field, raw value never hashed.
    assert _cache_application_kwargs({"tp_reduce_dtype": "fp32"}, model_name=None) is None


def _orch_args(**overrides):
    values = dict(
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False,
        height=480, width=704, num_frames=49, cache_dir=None, force=False, revision=None,
        tp_reduce_dtype=None, teacache_speedup=None, teacache_calibration=None,
        teacache_cadence=None, teacache_online_delta=None, dp=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("value", [None, "fp32"])
def test_compile_passes_dtype_to_precompile(monkeypatch, value):
    from difflet.cli.orchestrators import ltx_2 as orch_mod
    from difflet.pipeline import difflet_pipeline, path_resolver

    seen = {}
    monkeypatch.setattr(path_resolver, "resolve_model_path", lambda *a, **k: "/m")
    monkeypatch.setattr(
        difflet_pipeline.DiffletPipeline, "precompile",
        classmethod(lambda cls, model_id, **kwargs: seen.update(kwargs)),
    )
    orch = orch_mod.LTX2Orchestrator.__new__(orch_mod.LTX2Orchestrator)
    orch.args = _orch_args(tp_reduce_dtype=value)
    monkeypatch.setattr(orch, "_adaptive_teacache_kwargs", lambda shape: {})
    monkeypatch.setattr(orch, "_resolved_shape", lambda: {})
    orch.compile()
    assert seen["application_kwargs"]["tp_reduce_dtype"] == (value or "bf16")
