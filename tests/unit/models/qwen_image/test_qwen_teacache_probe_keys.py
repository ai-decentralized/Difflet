"""Weight-name contract between the Qwen-Image TeaCache probe and the backbone.

Same property as tests/unit/models/flux/test_flux_teacache_probe_keys.py:
the probe's traced weight names ARE the backbone's shard keys (plus the
declared NEFF-state tensor prev_mod), so the shared weight store can serve
the probe from the backbone's pre-sharded checkpoint with no layout tag
(3f04080 / issue #39). Qwen adds one wrinkle the flux probe does not have:
the backbone's trace module prefixes diffusers keys with ``transformer.`` and
adds ``global_rank.rank`` only under context parallelism — both must flow
through the probe unchanged.

CPU only — tiny diffusers config, torch-native backend.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from difflet.backends.trainium.core.application_base import (
    NeuronApplicationBase,
    checkpoint_missing_weights,
)
from difflet.backends.trainium.qwen_image import teacache_probe_fused as probe_mod
from difflet.backends.trainium.qwen_image.transformer import (
    NeuronQwenImageTransformerApplication,
    _QwenImageTransformerTraceModule,
)
from difflet.models.qwen_image.application import create_qwen_image_transformer_config

STATE = probe_mod.NeuronQwenImageTeacacheProbeFusedApplication.state_tensor_names
HEADS, HEAD_DIM = 2, 8
INNER = HEADS * HEAD_DIM
TEXT_SEQ_LEN = 16


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    torch.manual_seed(0)


def _write_transformer_config(model_dir):
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImageTransformer2DModel",
                "patch_size": 2,
                "in_channels": 64,
                "out_channels": 16,
                "num_layers": 1,
                "attention_head_dim": HEAD_DIM,
                "num_attention_heads": HEADS,
                "joint_attention_dim": 32,
                "guidance_embeds": False,
                "axes_dims_rope": [2, 2, 4],
            }
        ),
        encoding="utf-8",
    )
    return str(model_dir)


def _config(tmp_path, *, context_parallel_enabled=False):
    model_path = _write_transformer_config(tmp_path / "Qwen-Image")
    return create_qwen_image_transformer_config(
        model_path=model_path,
        world_size=2 if context_parallel_enabled else 1,
        tp_degree=1,
        dtype=torch.float32,
        height=64,
        width=64,
        text_seq_len=TEXT_SEQ_LEN,
        context_parallel_enabled=context_parallel_enabled,
        cp_mode="gather_kv",
    )


def _build(config):
    backbone = _QwenImageTransformerTraceModule(config).eval()
    probe = probe_mod.QwenImageTeacacheProbeFusedModel(
        config, seq_len=probe_mod._seq_len(config), inner_dim=INNER, batch_size=1
    ).eval()
    return backbone, probe


def _hf_state_dict(backbone) -> dict:
    """The diffusers checkpoint as loaded from disk: no ``transformer.`` prefix."""
    prefix = "transformer."
    return {
        k[len(prefix):]: v.clone()
        for k, v in backbone.state_dict().items()
        if k.startswith(prefix)
    }


# ------------------------------------------------------------- construction

def test_probe_model_is_the_backbone_trace_module():
    assert issubclass(probe_mod.QwenImageTeacacheProbeFusedModel, _QwenImageTransformerTraceModule)
    assert NeuronQwenImageTransformerApplication._model_cls is _QwenImageTransformerTraceModule
    assert probe_mod.NeuronQwenImageTeacacheProbeFusedApplication._model_cls is (
        probe_mod.QwenImageTeacacheProbeFusedModel
    )


def test_probe_declares_exactly_prev_mod_as_state():
    assert STATE == frozenset({"prev_mod"})
    assert NeuronApplicationBase.state_tensor_names == frozenset()
    assert NeuronQwenImageTransformerApplication.state_tensor_names == frozenset()


# --------------------------------------------------- traced parameter names

def test_probe_parameter_names_equal_backbone_names_plus_declared_state(tmp_path):
    # Built without CP: the CP branch of the trace module's __init__ needs the
    # NxD parallel state (init_parallel_mesh), which this CPU test does not
    # have. The CP-only additions (global_rank, cp_group) come from the same
    # base __init__ for backbone and probe alike, so the name equality holds
    # there by the same construction; the converter test below covers CP.
    config = _config(tmp_path)
    backbone, probe = _build(config)

    backbone_names = set(backbone.state_dict())
    probe_names = set(probe.state_dict())
    assert probe_names - STATE == backbone_names
    assert probe_names - backbone_names == STATE
    assert all(k.startswith("transformer.") or k == "global_rank.rank" for k in backbone_names)
    assert not any(k.startswith("trace_module.") for k in probe_names)


def test_probe_prev_mod_shape_matches_mod_input(tmp_path):
    config = _config(tmp_path)
    _, probe = _build(config)
    assert tuple(probe.prev_mod.shape) == (1, int(config.image_seq_len), INNER)
    assert probe.prev_mod.requires_grad is False


# ------------------------------------------------------ checkpoint converter

def test_probe_converter_is_the_backbone_converter():
    probe_fn = probe_mod.NeuronQwenImageTeacacheProbeFusedApplication.convert_hf_to_neuron_state_dict
    backbone_fn = NeuronQwenImageTransformerApplication.convert_hf_to_neuron_state_dict
    assert probe_fn is backbone_fn


@pytest.mark.parametrize("context_parallel_enabled", [False, True])
def test_probe_converted_keys_equal_backbone_converted_keys(tmp_path, context_parallel_enabled):
    # The checkpoint tensors come from a non-CP build (see above); the
    # converter itself is exercised with the CP flag on and off.
    backbone, _ = _build(_config(tmp_path / "build"))
    hf = _hf_state_dict(backbone)
    config = _config(tmp_path / "convert", context_parallel_enabled=context_parallel_enabled)

    probe_keys = set(
        probe_mod.NeuronQwenImageTeacacheProbeFusedApplication.convert_hf_to_neuron_state_dict(
            dict(hf), config
        )
    )
    backbone_keys = set(
        NeuronQwenImageTransformerApplication.convert_hf_to_neuron_state_dict(dict(hf), config)
    )
    assert probe_keys == backbone_keys
    assert ("global_rank.rank" in probe_keys) is context_parallel_enabled
    assert not any(k.startswith("trace_module.") for k in probe_keys)
    assert not (STATE & probe_keys)


# ----------------------------------------------- the loader-facing invariant

def test_backbone_checkpoint_serves_every_probe_weight(tmp_path):
    config = _config(tmp_path)
    backbone, probe = _build(config)
    converted = NeuronQwenImageTransformerApplication.convert_hf_to_neuron_state_dict(
        _hf_state_dict(backbone), config
    )

    assert checkpoint_missing_weights(probe, converted, STATE) == set()
    assert checkpoint_missing_weights(probe, converted) == {"prev_mod"}
    # The wrapper-era layout is reported as missing — the on-device failure.
    nested = {f"trace_module.{k}": v for k, v in converted.items()}
    assert "transformer.img_in.weight" in checkpoint_missing_weights(probe, nested, STATE)


def test_aliased_tensors_are_exactly_the_declared_state(tmp_path):
    config = _config(tmp_path)
    wrapper = probe_mod.ModelWrapperQwenImageTeacacheProbeFused(
        config,
        probe_mod.QwenImageTeacacheProbeFusedModel,
        tag="probe",
        compiler_args="",
        priority_model_idx=0,
    )
    instance = wrapper.get_model_instance()
    instance.load_module()
    module, aliases = instance.get(0)

    assert isinstance(module, _QwenImageTransformerTraceModule)
    aliased = {
        name for name, param in module.named_parameters() if any(param is a for a in aliases)
    }
    assert aliased == STATE
    assert set(aliases.values()) == {1}


# ------------------------------------------------------------ forward parity

def test_probe_forward_matches_backbone_hook_and_updates_from_prev_mod(tmp_path):
    config = _config(tmp_path)
    backbone, probe = _build(config)
    missing, unexpected = probe.load_state_dict(backbone.state_dict(), strict=False)
    assert set(missing) == STATE and unexpected == []

    hidden = torch.randn(1, int(config.image_seq_len), int(config.in_channels))
    timestep = torch.rand(1)
    encoder_hidden_states = torch.randn(1, TEXT_SEQ_LEN, int(config.joint_attention_dim))
    mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.int64)
    guidance = torch.ones(1)

    with torch.no_grad():
        expected = backbone.teacache_mod_input(hidden, timestep, encoder_hidden_states, guidance)
        rel_l1, mod_input = probe(hidden, timestep, encoder_hidden_states, mask, guidance)

    assert torch.equal(mod_input, expected)
    assert tuple(mod_input.shape) == tuple(probe.prev_mod.shape)
    assert torch.isfinite(rel_l1) and rel_l1.item() > 0
    with torch.no_grad():
        probe.prev_mod.copy_(mod_input)
        again, _ = probe(hidden, timestep, encoder_hidden_states, mask, guidance)
    assert again.item() == pytest.approx(0.0, abs=1e-6)
