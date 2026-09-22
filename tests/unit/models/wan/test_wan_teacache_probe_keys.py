"""Weight-name contract between the Wan TeaCache probe and the backbone.

The shared weight store hands every application that shares
(source, dtype, topology) the same pre-sharded checkpoint, and the NEFF looks
weights up by the traced module path. These tests pin the property that makes
that correct for the probe without a layout tag or a second ~28 GB copy of the
expert: **same weights => same names by construction**.

Flux hit the opposite case on device 2026-08-30 with a wrapper design:
``Missing weight tensor with key trace_module.transformer.transformer_blocks.0.norm1.linear.bias``
(GitHub issue #39).

CPU only — models are built on the torch-native backend with tiny dims; no
Trainium, no compile.
"""

from __future__ import annotations

import importlib
import os

import pytest

_PREV_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch  # noqa: E402

import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
# Rebind the wan modeling module onto the cpu backend, then the probe module so
# its subclass derives from the *reloaded* WanTransformer3DModel.
mw = importlib.reload(importlib.import_module("difflet.models.wan.modeling_wan"))
backbone_mod = importlib.reload(
    importlib.import_module("difflet.backends.trainium.wan.backbone")
)
probe_mod = importlib.reload(
    importlib.import_module("difflet.backends.trainium.wan.teacache_probe_fused")
)

from difflet.backends.trainium.core.application_base import (  # noqa: E402
    NeuronApplicationBase,
    checkpoint_missing_weights,
)
from difflet.backends.trainium.core.config import NeuronConfig  # noqa: E402

if _PREV_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _PREV_BACKEND

HEADS = 4
HEAD_DIM = 16
INNER = HEADS * HEAD_DIM
STATE = probe_mod.NeuronWanTeacacheProbeFusedApplication.state_tensor_names


@pytest.fixture(autouse=True)
def _cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    torch.manual_seed(0)
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def _config(**overrides):
    neuron_config = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    kwargs = dict(
        neuron_config=neuron_config,
        patch_size=(1, 2, 2),
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        in_channels=4,
        out_channels=4,
        text_dim=24,
        freq_dim=32,
        ffn_dim=48,
        num_layers=2,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        text_seq_len=8,
        height=64,
        width=64,
        num_frames=4,
    )
    kwargs.update(overrides)
    return backbone_mod.WanBackboneInferenceConfig(**kwargs)


def _build(config):
    backbone = mw.WanTransformer3DModel(config).eval()
    probe = probe_mod.WanTeacacheProbeFusedModel(
        config,
        seq_len=probe_mod.probe_seq_len(config),
        inner_dim=INNER,
        batch_size=1,
    ).eval()
    return backbone, probe


# ------------------------------------------------------------- construction


def test_probe_model_is_the_backbone_transformer():
    # Subclass, not wrapper: that is the whole mechanism. A wrapper would move
    # every parameter one attribute level deeper and change its traced name.
    assert issubclass(probe_mod.WanTeacacheProbeFusedModel, mw.WanTransformer3DModel)
    assert probe_mod.NeuronWanTeacacheProbeFusedApplication._model_cls is (
        probe_mod.WanTeacacheProbeFusedModel
    )


def test_probe_declares_exactly_prev_mod_as_state():
    assert STATE == frozenset({"prev_mod"})
    assert NeuronApplicationBase.state_tensor_names == frozenset()
    assert backbone_mod.NeuronWanBackboneApplication.state_tensor_names == frozenset()


# --------------------------------------------------- traced parameter names


def test_probe_parameter_names_equal_backbone_names_plus_declared_state():
    config = _config()
    backbone, probe = _build(config)

    backbone_names = set(backbone.state_dict())
    probe_names = set(probe.state_dict())

    assert probe_names - STATE == backbone_names
    assert probe_names - backbone_names == STATE
    assert not (STATE & backbone_names)


def test_probe_prev_mod_shape_matches_mod_input():
    config = _config()
    _, probe = _build(config)

    seq = probe_mod.probe_seq_len(config)
    assert tuple(probe.prev_mod.shape) == (1, seq, INNER)
    assert probe.prev_mod.requires_grad is False

    # And the declared shape is the shape the signal actually has.
    hidden = torch.randn(1, config.in_channels, config.num_frames, 64 // 8, 64 // 8)
    timestep = torch.ones(1)
    encoder = torch.randn(1, config.text_seq_len, config.text_dim)
    mod_input = probe.teacache_mod_input(hidden, timestep, encoder)
    assert mod_input.shape == probe.prev_mod.shape


def test_probe_seq_len_follows_the_patchified_latent_grid():
    config = _config(patch_size=(1, 2, 2), num_frames=4, height=64, width=64)
    # latent grid is 4 x (64//8) x (64//8) = 4 x 8 x 8, patched by (1, 2, 2).
    assert probe_mod.probe_seq_len(config) == 4 * 4 * 4


# ----------------------------------------------------------- probe forward


def test_forward_returns_rel_l1_and_mod_input():
    config = _config()
    _, probe = _build(config)

    hidden = torch.randn(1, config.in_channels, config.num_frames, 8, 8)
    timestep = torch.ones(1)
    encoder = torch.randn(1, config.text_seq_len, config.text_dim)

    rel_l1, mod_input = probe(hidden, timestep, encoder)
    assert rel_l1.shape == ()
    assert mod_input.shape == probe.prev_mod.shape
    # prev_mod starts zero-filled (NxD StateInitializer), so the first delta is
    # the garbage step-0 value the controller's warmup window absorbs.
    assert float(rel_l1) > 0.0


def test_rel_l1_matches_the_host_shadow_formula():
    """The device scalar is the same number the host path computed itself."""
    config = _config()
    _, probe = _build(config)

    hidden = torch.randn(1, config.in_channels, config.num_frames, 8, 8)
    timestep = torch.ones(1)
    encoder = torch.randn(1, config.text_seq_len, config.text_dim)

    prev = torch.randn_like(probe.prev_mod)
    with torch.no_grad():
        probe.prev_mod.copy_(prev)

    rel_l1, mod_input = probe(hidden, timestep, encoder)

    # difflet/models/wan/pipeline.py host branch: mean|cur-prev| / mean|prev|.
    cur = mod_input.detach().float()
    denom = prev.detach().float().abs().mean().clamp_min(1e-8)
    expected = float((cur - prev.float()).abs().mean() / denom)
    assert float(rel_l1) == pytest.approx(expected, rel=1e-5)


def test_probe_signal_equals_the_backbone_prefix():
    """Probe and backbone compute the identical block-0 modulated input.

    They must: the probe inherits ``teacache_mod_input`` rather than
    reimplementing it, so this pins that no override creeps in later.
    """
    config = _config()
    backbone, probe = _build(config)
    probe.load_state_dict(
        {k: v for k, v in backbone.state_dict().items()}, strict=False
    )

    hidden = torch.randn(1, config.in_channels, config.num_frames, 8, 8)
    timestep = torch.ones(1)
    encoder = torch.randn(1, config.text_seq_len, config.text_dim)

    expected = backbone.teacache_mod_input(hidden, timestep, encoder)
    _, mod_input = probe(hidden, timestep, encoder)
    assert torch.allclose(mod_input, expected, atol=1e-5)


# ------------------------------------------------------ checkpoint converter


def test_probe_converter_is_the_backbone_converter():
    probe_fn = probe_mod.NeuronWanTeacacheProbeFusedApplication.convert_hf_to_neuron_state_dict
    backbone_fn = backbone_mod.NeuronWanBackboneApplication.convert_hf_to_neuron_state_dict
    assert probe_fn is backbone_fn


# ----------------------------------------------- the loader-facing invariant


def test_backbone_checkpoint_serves_every_probe_weight():
    """Every non-state tensor in the probe's state dict is present, under the
    same key, in the checkpoint the backbone shards. This is the host-side form
    of the on-device hard error ("Missing weight tensor with key ...")."""
    config = _config()
    backbone, probe = _build(config)
    converted = backbone_mod.NeuronWanBackboneApplication.convert_hf_to_neuron_state_dict(
        dict(backbone.state_dict()), config
    )

    assert checkpoint_missing_weights(probe, converted, STATE) == set()
    # And the check is not vacuous: forgetting to declare prev_mod as state
    # reports it as a missing weight, exactly like the wrapper-era keys would.
    assert checkpoint_missing_weights(probe, converted) == {"prev_mod"}
    nested = {f"trace_module.transformer.{k}": v for k, v in converted.items()}
    assert checkpoint_missing_weights(probe, nested, STATE)


def test_aliased_tensors_are_exactly_the_declared_state():
    """What the ModelInstance aliases is what NxD turns into INPUT_STATE, so the
    declared state set must be precisely the aliased parameter names."""
    config = _config()
    _, probe = _build(config)
    instance = probe_mod._WanFusedProbeModelInstance(module_builder=lambda: probe)
    instance.module = probe
    _module, aliases = instance.get(bucket_rank=0)

    names = {
        name for name, param in probe.named_parameters() if any(param is a for a in aliases)
    }
    assert names == set(STATE)
