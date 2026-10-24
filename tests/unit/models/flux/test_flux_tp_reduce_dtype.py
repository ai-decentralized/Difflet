"""Flux tensor-parallel all-reduce precision: T5/CLIP encoders and the DiT."""

from __future__ import annotations

import argparse
import types

import pytest
import torch
from torch import nn

from difflet.models.flux.clip import modeling_clip as clip_mod
from difflet.models.flux.t5 import modeling_t5 as t5_mod
from difflet.pipeline.difflet_pipeline import _cache_application_kwargs


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, in_features, out_features, **kwargs):
        self.calls.append(kwargs)
        return nn.Linear(in_features, out_features, bias=kwargs.get("bias", True))


def _column(in_features, out_features, **kwargs):
    return nn.Linear(in_features, out_features, bias=kwargs.get("bias", True))


def _neuron_config(rpl_reduce_dtype):
    return types.SimpleNamespace(rpl_reduce_dtype=rpl_reduce_dtype, torch_dtype=torch.bfloat16)


def _t5_config(rpl_reduce_dtype):
    return types.SimpleNamespace(
        d_model=8, d_ff=16, dropout_rate=0.0, dense_act_fn="gelu_new",
        neuron_config=_neuron_config(rpl_reduce_dtype),
    )


def _clip_config(rpl_reduce_dtype):
    return types.SimpleNamespace(
        hidden_size=8, num_attention_heads=2, attention_dropout=0.0,
        intermediate_size=16, hidden_act="quick_gelu",
        neuron_config=_neuron_config(rpl_reduce_dtype),
    )


@pytest.fixture
def recorders(monkeypatch):
    t5_rec, clip_rec = _Recorder(), _Recorder()
    monkeypatch.setattr(t5_mod, "RowParallelLinear", t5_rec)
    monkeypatch.setattr(t5_mod, "ColumnParallelLinear", _column)
    monkeypatch.setattr(clip_mod, "RowParallelLinear", clip_rec)
    monkeypatch.setattr(clip_mod, "ColumnParallelLinear", _column)
    monkeypatch.setattr(clip_mod, "get_tensor_model_parallel_size", lambda: 1)
    return t5_rec, clip_rec


@pytest.mark.parametrize(
    ("rpl_reduce_dtype", "expected"),
    [(None, torch.bfloat16), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)],
)
def test_encoders_follow_neuron_config(recorders, rpl_reduce_dtype, expected):
    t5_rec, clip_rec = recorders
    t5_mod.NeuronT5DenseActDense(_t5_config(rpl_reduce_dtype))
    t5_mod.NeuronT5DenseGatedActDense(_t5_config(rpl_reduce_dtype))
    clip_mod.NeuronCLIPAttention(_clip_config(rpl_reduce_dtype))
    clip_mod.NeuronCLIPMLP(_clip_config(rpl_reduce_dtype))
    assert len(t5_rec.calls) == 2 and len(clip_rec.calls) == 2
    assert {c["reduce_dtype"] for c in t5_rec.calls + clip_rec.calls} == {expected}


@pytest.mark.parametrize(
    ("neuron_config", "expected"),
    [
        (_neuron_config(None), torch.bfloat16),  # torch_dtype, as before
        (_neuron_config(torch.float32), torch.float32),
        (types.SimpleNamespace(torch_dtype=torch.bfloat16), torch.bfloat16),
    ],
)
def test_dit_reduce_dtype(neuron_config, expected):
    from difflet.models.flux.modeling_flux import _flux_reduce_dtype

    assert _flux_reduce_dtype(neuron_config) is expected


def test_create_flux_config_sets_every_tp_component(monkeypatch):
    from difflet.models.flux import application as app_mod

    seen = []
    monkeypatch.setattr(app_mod, "NeuronConfig", lambda **kw: seen.append(kw) or types.SimpleNamespace(**kw))
    monkeypatch.setattr(app_mod, "load_pretrained_config", lambda path: None)
    monkeypatch.setattr(app_mod, "load_diffusers_config", lambda path: None)
    for name in ("CLIPInferenceConfig", "T5InferenceConfig", "FluxBackboneInferenceConfig"):
        monkeypatch.setattr(
            app_mod, name, lambda **kw: types.SimpleNamespace(in_channels=64, **kw)
        )

    class _Stop(Exception):
        pass

    def _vae(**kw):
        raise _Stop

    monkeypatch.setattr(app_mod, "VAEDecoderInferenceConfig", _vae)
    with pytest.raises(_Stop):
        app_mod.create_flux_config(
            model_path="/m", world_size=4, backbone_tp_degree=4, dtype=torch.bfloat16,
            height=1024, width=1024, tp_reduce_dtype=torch.float32,
        )
    # clip, t5, backbone carry the dtype; the VAE decoder has no TP all-reduce
    assert [kw.get("rpl_reduce_dtype") for kw in seen] == [torch.float32] * 3 + [None]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"tp_reduce_dtype": "bf16"}, {"tp_reduce_dtype": "bfloat16"}),
        ({}, {"tp_reduce_dtype": "bfloat16"}),
        ({"tp_reduce_dtype": "fp32"}, {"tp_reduce_dtype": "float32"}),
        ({"tp_reduce_dtype": "bf16", "taef1": True}, {"taef1": True, "tp_reduce_dtype": "bfloat16"}),
    ],
)
def test_cache_application_kwargs(kwargs, expected):
    # Pre-bf16 Flux artifacts mixed a bf16 DiT with fp32 encoders: every Flux
    # artifact now names its dtype, so none collides with a pre-change one.
    assert _cache_application_kwargs(kwargs, model_name="flux") == expected


@pytest.mark.parametrize("value", [None, "fp32"])
def test_orchestrator_passes_dtype_to_compile_and_load(value):
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    orch = FluxOrchestrator.__new__(FluxOrchestrator)
    orch.args = argparse.Namespace(
        tp_reduce_dtype=value, teacache_cadence=None, teacache_online_delta=None,
        taef1=False, teacache_speedup=None,
    )
    assert orch._model_kwargs()["application_kwargs"] == {"tp_reduce_dtype": value or "bf16"}
