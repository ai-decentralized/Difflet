"""Qwen-Image tensor-parallel all-reduce precision (bf16 default, fp32 opt-in)."""

from __future__ import annotations

import argparse
import json

import pytest
import torch
from torch import nn

import difflet.cli.orchestrators.qwen_image as qwen_orch_mod
from difflet.cli.orchestrators.base import stage_cache_key
from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
from difflet.models.qwen_image import modeling_qwen_image as qmod
from difflet.models.qwen_image.application import create_qwen_image_transformer_config

# Key of the Trainium Qwen DiT stage at 1024x1024 / tp4 with --tp-reduce-dtype fp32 (which adds
# no field: the legacy reduce), toolchain stubbed to _STUB_TOOLCHAIN. It was the pre-bf16 key
# e8af81d4e33a97f8 until the DiT masked the prompt padding ("text_mask"), which changed the graph
# no older artifact may be reused for.
EXPECTED_LEGACY_KEY = "0ec2a0022defa6a9"
_STUB_TOOLCHAIN = {"stub": "toolchain"}


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, in_features, out_features, **kwargs):
        self.calls.append(kwargs)
        return nn.Linear(in_features, out_features, bias=kwargs.get("bias", True))


def _tiny_transformer():
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModel,
    )

    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=8,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=24,
        guidance_embeds=False,
        axes_dims_rope=(2, 4, 2),
    )


def _shard(monkeypatch, **kwargs) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(qmod, "RowParallelLinear", recorder)
    monkeypatch.setattr(qmod, "_safe_tensor_parallel_size", lambda: 4)
    qmod._replace_qwen_linears_for_tp(_tiny_transformer(), **kwargs)
    return recorder


def test_default_is_bf16(monkeypatch):
    recorder = _shard(monkeypatch)
    assert len(recorder.calls) == 4 * 2  # to_out, to_add_out, img_mlp, txt_mlp per block
    assert {call["reduce_dtype"] for call in recorder.calls} == {torch.bfloat16}


def test_fp32_opt_in(monkeypatch):
    recorder = _shard(monkeypatch, reduce_dtype=torch.float32)
    assert {call["reduce_dtype"] for call in recorder.calls} == {torch.float32}


def test_helper_requires_reduce_dtype():
    with pytest.raises(TypeError):
        qmod._row_parallel_like(nn.Linear(8, 8), input_is_parallel=True)


def _write_transformer_config(model_dir):
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    config = {
        "_class_name": "QwenImageTransformer2DModel",
        "patch_size": 2,
        "in_channels": 64,
        "out_channels": 16,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "guidance_embeds": False,
        "axes_dims_rope": [2, 2, 4],
    }
    (transformer_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model_dir)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, torch.bfloat16), ({"tp_reduce_dtype": torch.float32}, torch.float32)],
)
def test_transformer_config_carries_reduce_dtype(tmp_path, kwargs, expected):
    config = create_qwen_image_transformer_config(
        model_path=_write_transformer_config(tmp_path / "Qwen-Image"),
        world_size=4,
        tp_degree=4,
        dtype=torch.bfloat16,
        height=64,
        width=64,
        text_seq_len=16,
        **kwargs,
    )
    assert config.neuron_config.rpl_reduce_dtype is expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, torch.bfloat16), ({"tp_reduce_dtype": "bf16"}, torch.bfloat16),
     ({"tp_reduce_dtype": "fp32"}, torch.float32)],
)
def test_application_maps_kwarg(tmp_path, monkeypatch, kwargs, expected):
    from difflet.models.qwen_image.application import NeuronQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    app = NeuronQwenImageApplication(
        model_path=_write_transformer_config(tmp_path / "Qwen-Image"),
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        shape={"height": 64, "width": 64},
        text_seq_len=8,
        **kwargs,
    )
    assert app.transformer.config.neuron_config.rpl_reduce_dtype is expected


def test_application_rejects_unknown_kwarg_value(tmp_path, monkeypatch):
    from difflet.models.qwen_image.application import NeuronQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    with pytest.raises(ValueError, match="bf16, fp32"):
        NeuronQwenImageApplication(
            model_path=_write_transformer_config(tmp_path / "Qwen-Image"),
            parallel=DiffletParallelConfig(tp_degree=4),
            dtype="bf16",
            shape={"height": 64, "width": 64},
            text_seq_len=8,
            tp_reduce_dtype="fp16",
        )


def _args(**overrides):
    values = dict(
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", sp_enabled=False, height=1024,
        width=1024, shapes=None, tp_reduce_dtype=None, model_id="Qwen/Qwen-Image",
        revision=None, cache_dir=None, prompt=None, output=None, steps=20,
        guidance_scale=4.0, seed=42, teacache_speedup=None, teacache_calibration=None,
        teacache_cadence=None, teacache_online_delta=None, vae_tp_degree=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _generate_key(monkeypatch, **overrides) -> str:
    monkeypatch.setattr(qwen_orch_mod, "stage_toolchain_versions", lambda: _STUB_TOOLCHAIN)
    args = _args(**overrides)
    return stage_cache_key(QwenImageOrchestrator(args)._stage_cache_inputs("generate", args))


def test_legacy_key_unchanged(monkeypatch):
    assert _generate_key(monkeypatch, tp_reduce_dtype="fp32") == EXPECTED_LEGACY_KEY


def test_bf16_key_differs(monkeypatch):
    assert _generate_key(monkeypatch) != EXPECTED_LEGACY_KEY
    assert _generate_key(monkeypatch, tp_reduce_dtype="bf16") == _generate_key(monkeypatch)


def test_stage_argv_carries_flag():
    argv = QwenImageOrchestrator(_args(tp_reduce_dtype="fp32"))._shared_cli_args("generate")
    i = argv.index("--tp-reduce-dtype")
    assert argv[i + 1] == "fp32"


def test_stage_argv_omits_absent_flag():
    argv = QwenImageOrchestrator(_args())._shared_cli_args("generate")
    assert "--tp-reduce-dtype" not in argv
