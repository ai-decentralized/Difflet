"""Weight-name contract between the LTX-2 TeaCache probe and the backbone.

The shared weight store keys only on (source, dtype, tp_degree, world_size,
context_parallel, sequence_parallel, cfg_parallel) — no component tag — so a
probe built from the backbone's own config lands on the backbone's store entry
and is handed its shards. That is only correct if the probe's traced parameter
names are byte-identical to the backbone's.

LTX-2's single-mode backbone is itself a wrapper (``_LTX2TransformerTraceModule``
holds the diffusers model at ``self.transformer``) and its converter bakes that
prefix into every key. So the probe subclasses the trace module and adds only
``prev_mod``. Adding another wrapper layer would nest everything one level
deeper and fail on device with ``Missing weight tensor with key ...`` — the
failure Flux hit on 2026-08-30 (GitHub issue #39).

CPU only — tiny dims, no Trainium, no compile.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

transformer_mod = pytest.importorskip("difflet.backends.trainium.ltx_2.transformer")
probe_mod = pytest.importorskip("difflet.backends.trainium.ltx_2.teacache_probe_fused")

from difflet.backends.trainium.core.application_base import (  # noqa: E402
    NeuronApplicationBase,
    checkpoint_missing_weights,
)
from difflet.backends.trainium.core.config import NeuronConfig  # noqa: E402

HEADS = 2
HEAD_DIM = 8
INNER = HEADS * HEAD_DIM
STATE = probe_mod.NeuronLTX2TeacacheProbeFusedApplication.state_tensor_names


def _config(**overrides):
    neuron_config = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    kwargs = dict(
        neuron_config=neuron_config,
        in_channels=4,
        out_channels=4,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        cross_attention_dim=INNER,
        vae_scale_factors=(8, 32, 32),
        audio_in_channels=4,
        audio_out_channels=4,
        audio_patch_size=1,
        audio_patch_size_t=1,
        audio_num_attention_heads=HEADS,
        audio_attention_head_dim=HEAD_DIM,
        audio_cross_attention_dim=INNER,
        audio_scale_factor=1,
        audio_sampling_rate=16000,
        audio_hop_length=256,
        num_layers=1,
        caption_channels=16,
        text_seq_len=8,
        height=64,
        width=64,
        num_frames=9,
    )
    kwargs.update(overrides)
    return transformer_mod.LTX2TransformerInferenceConfig(**kwargs)


def _build(config):
    backbone = transformer_mod._LTX2TransformerTraceModule(config).eval()
    probe = probe_mod.LTX2TeacacheProbeFusedModel(
        config,
        seq_len=int(config.video_seq_len),
        inner_dim=probe_mod.probe_inner_dim(config),
        batch_size=1,
    ).eval()
    return backbone, probe


# ------------------------------------------------------------- construction


def test_probe_model_is_the_backbone_trace_module():
    # Subclass, not another wrapper. LTX-2's backbone is already one level deep
    # (``transformer.*``); the probe must not add a second level.
    assert issubclass(
        probe_mod.LTX2TeacacheProbeFusedModel, transformer_mod._LTX2TransformerTraceModule
    )
    assert probe_mod.NeuronLTX2TeacacheProbeFusedApplication._model_cls is (
        probe_mod.LTX2TeacacheProbeFusedModel
    )


def test_probe_declares_exactly_prev_mod_as_state():
    assert STATE == frozenset({"prev_mod"})
    assert NeuronApplicationBase.state_tensor_names == frozenset()
    assert transformer_mod.NeuronLTX2TransformerApplication.state_tensor_names == frozenset()


# --------------------------------------------------- traced parameter names


def test_probe_parameter_names_equal_backbone_names_plus_declared_state():
    config = _config()
    backbone, probe = _build(config)

    backbone_names = set(backbone.state_dict())
    probe_names = set(probe.state_dict())

    assert probe_names - STATE == backbone_names
    assert probe_names - backbone_names == STATE
    assert not (STATE & backbone_names)


def test_probe_keeps_the_inner_transformer_prefix():
    """The converter prefixes every diffusers key with ``transformer.``; the
    probe must keep the inner model at that exact attribute name."""
    config = _config()
    _, probe = _build(config)
    names = set(probe.state_dict())
    assert any(name.startswith("transformer.") for name in names)
    assert not any(name.startswith("trace_module.") for name in names)
    assert not any(name.startswith("transformer.transformer.") for name in names)


def test_probe_prev_mod_shape_matches_mod_input():
    config = _config()
    _, probe = _build(config)

    seq = int(config.video_seq_len)
    assert tuple(probe.prev_mod.shape) == (1, seq, INNER)
    assert probe.prev_mod.requires_grad is False

    hidden = torch.randn(1, seq, config.in_channels)
    mod_input = probe.teacache_mod_input(hidden, torch.ones(1))
    assert mod_input.shape == probe.prev_mod.shape


# ----------------------------------------------------------- probe forward


def test_forward_returns_rel_l1_and_mod_input():
    config = _config()
    _, probe = _build(config)

    hidden = torch.randn(1, int(config.video_seq_len), config.in_channels)
    rel_l1, mod_input = probe(hidden, torch.ones(1))
    assert rel_l1.shape == ()
    assert mod_input.shape == probe.prev_mod.shape
    assert float(rel_l1) > 0.0


def test_rel_l1_matches_the_host_formula():
    """The device scalar is the same number the host path computed itself."""
    config = _config()
    _, probe = _build(config)

    hidden = torch.randn(1, int(config.video_seq_len), config.in_channels)
    prev = torch.randn_like(probe.prev_mod)
    with torch.no_grad():
        probe.prev_mod.copy_(prev)

    rel_l1, mod_input = probe(hidden, torch.ones(1))

    # difflet/models/ltx_2/pipeline.py host branch: mean|cur-prev| / mean|prev|.
    cur = mod_input.detach().float()
    denom = prev.detach().float().abs().mean().clamp_min(1e-8)
    expected = float((cur - prev.float()).abs().mean() / denom)
    assert float(rel_l1) == pytest.approx(expected, rel=1e-5)


def test_probe_signal_matches_the_host_cpu_path():
    """Device probe and the host CPU transformer compute the same signal.

    Both replicate ``norm1(proj_in(latent)) * (1 + scale_msa) + shift_msa`` on
    the same weights, so the probe can replace the host path without changing
    the controller's decisions.
    """
    config = _config()
    _, probe = _build(config)

    model = probe.transformer
    hidden = torch.randn(1, int(config.video_seq_len), config.in_channels)
    timestep = torch.ones(1)

    projected = model.proj_in(hidden)
    temb, _ = model.time_embed(
        timestep.flatten(), batch_size=1, hidden_dtype=projected.dtype
    )
    temb = temb.view(1, -1, temb.size(-1))
    block0 = model.transformer_blocks[0]
    params = block0.get_mod_params(block0.scale_shift_table, temb, 1)
    expected = block0.norm1(projected) * (1 + params[1]) + params[0]

    _, mod_input = probe(hidden, timestep)
    assert torch.allclose(mod_input, expected, atol=1e-5)


# ------------------------------------------------------ checkpoint converter


def test_probe_converter_is_the_backbone_converter():
    probe_fn = probe_mod.NeuronLTX2TeacacheProbeFusedApplication.convert_hf_to_neuron_state_dict
    backbone_fn = transformer_mod.NeuronLTX2TransformerApplication.convert_hf_to_neuron_state_dict
    assert probe_fn is backbone_fn


# ----------------------------------------------- the loader-facing invariant


def test_backbone_checkpoint_serves_every_probe_weight():
    """Every non-state tensor in the probe's state dict is present, under the
    same key, in the checkpoint the backbone shards — the host-side form of the
    on-device hard error ("Missing weight tensor with key ...")."""
    config = _config()
    backbone, probe = _build(config)
    # The converter consumes diffusers-shaped keys (no `transformer.` prefix),
    # which is what the inner model's own state dict is.
    converted = transformer_mod.NeuronLTX2TransformerApplication.convert_hf_to_neuron_state_dict(
        dict(backbone.transformer.state_dict()), config
    )

    assert checkpoint_missing_weights(probe, converted, STATE) == set()
    # Not vacuous: undeclared prev_mod reports as a missing weight.
    assert checkpoint_missing_weights(probe, converted) == {"prev_mod"}
    nested = {f"trace_module.{k}": v for k, v in converted.items()}
    assert checkpoint_missing_weights(probe, nested, STATE)


def test_aliased_tensors_are_exactly_the_declared_state():
    config = _config()
    _, probe = _build(config)
    instance = probe_mod._LTX2FusedProbeModelInstance(module_builder=lambda: probe)
    instance.module = probe
    _module, aliases = instance.get(bucket_rank=0)

    names = {
        name for name, param in probe.named_parameters() if any(param is a for a in aliases)
    }
    assert names == set(STATE)
