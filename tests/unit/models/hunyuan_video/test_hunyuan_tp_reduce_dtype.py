"""HunyuanVideo tensor-parallel all-reduce precision (bf16 default, fp32 opt-in)."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import types

import pytest
import torch
from torch import nn

_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.hunyuan_video.modeling_hunyuan_video as hv  # noqa: E402

importlib.reload(hv)
if _ORIG_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _ORIG_BACKEND

import difflet.cli.orchestrators.hunyuan_video as hv_orch_mod  # noqa: E402
from difflet.cli.orchestrators.base import stage_cache_key  # noqa: E402
from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator  # noqa: E402
from difflet.pipeline.difflet_pipeline import _cache_application_kwargs  # noqa: E402

# Pre-bf16 key of the HunyuanVideo DiT stage at 320x512x61 / tp4, toolchain
# stubbed to _STUB_TOOLCHAIN. --tp-reduce-dtype fp32 must keep hashing to it.
EXPECTED_LEGACY_KEY = "f61bd88411cadb3b"
_STUB_TOOLCHAIN = {"stub": "toolchain"}


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DIFFLET_BACKEND", None)
        else:
            os.environ["DIFFLET_BACKEND"] = prev


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, in_features, out_features, **kwargs):
        self.calls.append(kwargs)
        return nn.Linear(in_features, out_features, bias=kwargs.get("bias", True))


_TINY = dict(
    in_channels=2, out_channels=2, num_attention_heads=2, attention_head_dim=4,
    num_layers=1, num_single_layers=1, num_refiner_layers=1, mlp_ratio=2.0,
    patch_size=2, patch_size_t=1, guidance_embeds=True, text_embed_dim=6,
    pooled_projection_dim=5, rope_axes_dim=(2, 2, 0),
)
# refiner self-attn + FF, dual-stream to_out + to_add_out + ff + ff_context,
# single-stream proj_out_attn + proj_out_mlp
_SITES = 2 + 4 + 2


def _build(monkeypatch, *args, **kwargs) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(hv, "RowParallelLinear", recorder)
    hv.HunyuanVideoTransformer3DModel(*args, **kwargs)
    return recorder


def test_default_is_bf16(monkeypatch):
    recorder = _build(monkeypatch, **_TINY)
    assert len(recorder.calls) == _SITES
    assert {call["reduce_dtype"] for call in recorder.calls} == {torch.bfloat16}


def test_fp32_opt_in(monkeypatch):
    recorder = _build(monkeypatch, **_TINY, reduce_dtype=torch.float32)
    assert {call["reduce_dtype"] for call in recorder.calls} == {torch.float32}


@pytest.mark.parametrize(
    ("rpl_reduce_dtype", "expected"),
    [(None, torch.bfloat16), (torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16)],
)
def test_config_object_reads_neuron_config(monkeypatch, rpl_reduce_dtype, expected):
    config = types.SimpleNamespace(
        **dataclasses.asdict(hv.HunyuanVideoTransformerConfig(**_TINY)),
        neuron_config=types.SimpleNamespace(rpl_reduce_dtype=rpl_reduce_dtype),
    )
    recorder = _build(monkeypatch, config)
    assert {call["reduce_dtype"] for call in recorder.calls} == {expected}


def test_backbone_config_carries_reduce_dtype(monkeypatch):
    from difflet.backends.trainium.hunyuan_video import backbone
    from difflet.models.hunyuan_video import application as app_mod

    monkeypatch.setattr(app_mod, "_load_diffusers_config", lambda path: None)
    captured = {}

    class _Config:
        def __init__(self, *, neuron_config, **kwargs):
            captured["neuron_config"] = neuron_config

    monkeypatch.setattr(backbone, "HunyuanVideoBackboneInferenceConfig", _Config)
    for kwargs, expected in (({}, torch.bfloat16), ({"tp_reduce_dtype": torch.float32}, torch.float32)):
        app_mod.create_hunyuan_video_backbone_config(
            model_path="/nonexistent", world_size=4, tp_degree=4, dtype=torch.bfloat16,
            height=320, width=512, num_frames=61, **kwargs,
        )
        assert captured["neuron_config"].rpl_reduce_dtype is expected


def _args(**overrides):
    values = dict(
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", sp_enabled=False, cfg_parallel=False,
        height=320, width=512, num_frames=61, shapes=None, teacache_speedup=None,
        teacache_calibration=None, teacache_cadence=None, teacache_online_delta=None,
        model_id="hunyuanvideo-community/HunyuanVideo", revision=None, cache_dir=None,
        tp_reduce_dtype=None, prompt=None, output=None, steps=30, guidance_scale=None,
        seed=42,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _generate_key(monkeypatch, **overrides) -> str:
    monkeypatch.setattr(hv_orch_mod, "stage_toolchain_versions", lambda: _STUB_TOOLCHAIN)
    args = _args(**overrides)
    return stage_cache_key(HunyuanVideoOrchestrator(args)._stage_cache_inputs("generate", args))


def test_legacy_key_unchanged(monkeypatch):
    assert _generate_key(monkeypatch, tp_reduce_dtype="fp32") == EXPECTED_LEGACY_KEY


def test_bf16_key_differs(monkeypatch):
    assert _generate_key(monkeypatch) != EXPECTED_LEGACY_KEY
    assert _generate_key(monkeypatch, tp_reduce_dtype="bf16") == _generate_key(monkeypatch)


def test_stage_argv_carries_flag():
    argv = HunyuanVideoOrchestrator(_args(tp_reduce_dtype="fp32"))._shared_cli_args("generate")
    i = argv.index("--tp-reduce-dtype")
    assert argv[i + 1] == "fp32"


def test_stage_argv_omits_absent_flag():
    argv = HunyuanVideoOrchestrator(_args())._shared_cli_args("generate")
    assert "--tp-reduce-dtype" not in argv


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"tp_reduce_dtype": "fp32"}, {"attention_mask_accumulation": "fp32-v1"}),
        ({}, {"attention_mask_accumulation": "fp32-v1", "tp_reduce_dtype": "bfloat16"}),
    ],
)
def test_difflet_pipeline_cache_kwargs(kwargs, expected):
    assert _cache_application_kwargs(kwargs, model_name="hunyuan_video") == expected
