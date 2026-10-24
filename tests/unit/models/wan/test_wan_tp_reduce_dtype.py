"""Wan tensor-parallel all-reduce precision: umT5 encoder and the DiT."""

from __future__ import annotations

import argparse
import types

import pytest
import torch
from torch import nn

import difflet.cli.orchestrators.wan as wan_orch_mod
from difflet.cli.orchestrators.base import stage_cache_key
from difflet.cli.orchestrators.wan import WanOrchestrator
from difflet.models.wan import modeling_wan as wmod
from difflet.models.wan.umt5 import modeling_umt5 as umod
from difflet.pipeline.difflet_pipeline import _cache_application_kwargs

# Pre-bf16 key of the Wan 2.2 transformer stage (DiT + umT5) at 480x832x9 /
# tp4, toolchain stubbed. That artifact mixed a bf16 DiT with an fp32 umT5, so
# neither dtype may map back to it.
LEGACY_KEY = "29de50915cb58b48"
_STUB_TOOLCHAIN = {"stub": "toolchain"}


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, in_features, out_features, **kwargs):
        self.calls.append(kwargs)
        return nn.Linear(in_features, out_features, bias=kwargs.get("bias", True))


def _column(in_features, out_features, **kwargs):
    return nn.Linear(in_features, out_features, bias=kwargs.get("bias", True))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, torch.bfloat16), ({"tp_reduce_dtype": torch.float32}, torch.float32)],
)
def test_umt5_ffn_reduce_dtype(monkeypatch, kwargs, expected):
    recorder = _Recorder()
    monkeypatch.setattr(umod, "RowParallelLinear", recorder)
    monkeypatch.setattr(umod, "ColumnParallelLinear", _column)
    cfg = umod.WanUmT5Config(d_model=8, d_ff=16, num_heads=2, num_layers=1, **kwargs)
    umod.WanUmT5DenseGatedActDense(cfg)
    assert [c["reduce_dtype"] for c in recorder.calls] == [expected]


def test_text_encoder_wrapper_passes_neuron_config_dtype():
    from difflet.backends.trainium.wan.text_encoder import _umt5_reduce_dtype

    config = types.SimpleNamespace(neuron_config=types.SimpleNamespace(rpl_reduce_dtype=None))
    assert _umt5_reduce_dtype(config) is torch.bfloat16
    config.neuron_config.rpl_reduce_dtype = torch.float32
    assert _umt5_reduce_dtype(config) is torch.float32


@pytest.mark.parametrize(
    ("dtype", "reduce_dtype", "expected"),
    [(None, None, None), (torch.bfloat16, None, torch.bfloat16), (None, torch.float32, torch.float32)],
)
def test_dit_layers_reduce_dtype(monkeypatch, dtype, reduce_dtype, expected):
    recorder = _Recorder()
    monkeypatch.setattr(wmod, "RowParallelLinear", recorder)
    monkeypatch.setattr(wmod, "ColumnParallelLinear", _column)
    wmod.WanFeedForward(8, 16, dtype=dtype, reduce_dtype=reduce_dtype)
    wmod.WanAttention(8, 2, 4, dtype=dtype, reduce_dtype=reduce_dtype)
    assert len(recorder.calls) == 2
    assert {c["reduce_dtype"] for c in recorder.calls} == {expected}


@pytest.mark.parametrize(
    ("rpl_reduce_dtype", "dtype", "expected"),
    [(None, None, None), (torch.float32, None, torch.float32), (torch.bfloat16, None, torch.bfloat16)],
)
def test_dit_model_reads_neuron_config(rpl_reduce_dtype, dtype, expected):
    config = types.SimpleNamespace(neuron_config=types.SimpleNamespace(rpl_reduce_dtype=rpl_reduce_dtype))
    assert wmod._wan_reduce_dtype(config, dtype) is expected


@pytest.mark.parametrize("factory", ["create_wan_backbone_config", "create_wan_text_encoder_config"])
def test_configs_carry_reduce_dtype(monkeypatch, factory):
    from difflet.backends.trainium.wan import backbone, text_encoder
    from difflet.models.wan import application as app_mod

    seen = {}

    class _Config:
        def __init__(self, *, neuron_config, **kwargs):
            seen["neuron_config"] = neuron_config

    monkeypatch.setattr(app_mod, "load_diffusers_config", lambda path: None)
    monkeypatch.setattr(backbone, "WanBackboneInferenceConfig", _Config)
    monkeypatch.setattr(text_encoder, "WanTextEncoderInferenceConfig", _Config)
    shape = (
        dict(height=480, width=832, num_frames=9)
        if factory == "create_wan_backbone_config"
        else {}
    )
    for kwargs, expected in (({}, torch.bfloat16), ({"tp_reduce_dtype": torch.float32}, torch.float32)):
        getattr(app_mod, factory)(
            model_path="/m", world_size=4, tp_degree=4, dtype=torch.bfloat16, **shape, **kwargs
        )
        assert seen["neuron_config"].rpl_reduce_dtype is expected


def _args(**overrides):
    values = dict(
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", sp_enabled=False, cfg_parallel=False,
        height=480, width=832, num_frames=9, shapes=None, teacache_speedup=None,
        teacache_calibration=None, teacache_cadence=None, teacache_online_delta=None,
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", revision=None, cache_dir=None,
        tp_reduce_dtype=None, prompt=None, output=None, steps=40, guidance_scale=None,
        seed=42,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _transformer_key(monkeypatch, **overrides) -> str:
    monkeypatch.setattr(wan_orch_mod, "stage_toolchain_versions", lambda: _STUB_TOOLCHAIN)
    args = _args(**overrides)
    return stage_cache_key(WanOrchestrator(args)._stage_cache_inputs("transformer", args))


def test_transformer_key_names_its_dtype(monkeypatch):
    bf16 = _transformer_key(monkeypatch)
    fp32 = _transformer_key(monkeypatch, tp_reduce_dtype="fp32")
    assert len({bf16, fp32, LEGACY_KEY}) == 3
    assert _transformer_key(monkeypatch, tp_reduce_dtype="bf16") == bf16


def test_stage_argv_carries_flag():
    argv = WanOrchestrator(_args(tp_reduce_dtype="fp32"))._shared_cli_args("generate")
    i = argv.index("--tp-reduce-dtype")
    assert argv[i + 1] == "fp32"


def test_stage_argv_omits_absent_flag():
    argv = WanOrchestrator(_args())._shared_cli_args("generate")
    assert "--tp-reduce-dtype" not in argv


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, {"tp_reduce_dtype": "bfloat16"}), ({"tp_reduce_dtype": "fp32"}, {"tp_reduce_dtype": "float32"})],
)
def test_difflet_pipeline_cache_kwargs(kwargs, expected):
    assert _cache_application_kwargs(kwargs, model_name="wan") == expected
