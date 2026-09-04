"""Weight-name contract between the Flux TeaCache probe and the backbone.

The shared weight store (``core/shared_weights.py``) hands every application
that shares (source, dtype, topology) the same pre-sharded checkpoint, and the
NEFF looks weights up by the traced module path. These tests pin the property
that makes that correct for the probe without a layout tag or a second copy of
the transformer: **same weights => same names by construction**.

Found on device 2026-08-30 with the previous wrapper design:
``Missing weight tensor with key trace_module.transformer.transformer_blocks.0.norm1.linear.bias``
(campaign fix ``3f04080`` / GitHub issue #39, superseded by this contract).

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

import difflet.backends.trainium.core.config as _config_mod  # noqa: E402
import difflet.layers.normalization as _norm_mod  # noqa: E402

# Rebind the flux modeling module onto the cpu backend, then the probe module
# so its subclass derives from the *reloaded* NeuronFluxTransformer2DModel.
mf = importlib.reload(importlib.import_module("difflet.models.flux.modeling_flux"))
probe_mod = importlib.reload(
    importlib.import_module("difflet.backends.trainium.flux.teacache_probe_fused")
)
from difflet.backends.trainium.core.application_base import (  # noqa: E402
    NeuronApplicationBase,
    checkpoint_missing_weights,
)

if _PREV_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _PREV_BACKEND

HEADS = 2
HEAD_DIM = 8
INNER = HEADS * HEAD_DIM
STATE = probe_mod.NeuronFluxTeacacheProbeFusedApplication.state_tensor_names


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


class _IdentityMarker:
    def __call__(self, *args):
        return args[0] if len(args) == 1 else args


@pytest.fixture
def patched_markers(monkeypatch):
    monkeypatch.setattr(mf, "ModuleMarkerStartWrapper", _IdentityMarker)
    monkeypatch.setattr(mf, "ModuleMarkerEndWrapper", _IdentityMarker)
    monkeypatch.setattr(_norm_mod, "ModuleMarkerStartWrapper", _IdentityMarker)
    monkeypatch.setattr(_norm_mod, "ModuleMarkerEndWrapper", _IdentityMarker)
    yield


def _config(guidance_embeds=True, num_single_layers=1, **overrides):
    nc = _config_mod.NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    kwargs = dict(
        neuron_config=nc,
        attention_head_dim=HEAD_DIM,
        guidance_embeds=guidance_embeds,
        in_channels=4,
        joint_attention_dim=INNER,
        num_attention_heads=HEADS,
        num_layers=1,
        num_single_layers=num_single_layers,
        patch_size=1,
        pooled_projection_dim=8,
        height=32,
        width=32,
        vae_scale_factor=8,
        out_channels=4,
    )
    kwargs.update(overrides)
    return mf.FluxBackboneInferenceConfig(**kwargs)


def _seq_len(config):
    return probe_mod._seq_len(config)


def _hf_state_dict(backbone: torch.nn.Module, config) -> dict:
    """A diffusers-shaped checkpoint: single-block proj_out still fused."""
    sd = {k: v.clone() for k, v in backbone.state_dict().items()}
    for i in range(config.num_single_layers):
        attn_w = sd.pop(f"single_transformer_blocks.{i}.proj_out_attn.weight")
        attn_b = sd.pop(f"single_transformer_blocks.{i}.proj_out_attn.bias")
        mlp_w = sd.pop(f"single_transformer_blocks.{i}.proj_out_mlp.weight")
        sd[f"single_transformer_blocks.{i}.proj_out.weight"] = torch.cat([attn_w, mlp_w], dim=1)
        sd[f"single_transformer_blocks.{i}.proj_out.bias"] = attn_b
    return sd


def _build(config):
    backbone = mf.NeuronFluxTransformer2DModel(config).eval()
    probe = probe_mod.FluxTeacacheProbeFusedModel(
        config, seq_len=_seq_len(config), inner_dim=INNER, batch_size=1
    ).eval()
    return backbone, probe


# ------------------------------------------------------------- construction

def test_probe_model_is_the_backbone_transformer():
    # Subclass, not wrapper: that is the whole mechanism. A wrapper would move
    # every parameter one attribute level deeper and change its traced name.
    assert issubclass(probe_mod.FluxTeacacheProbeFusedModel, mf.NeuronFluxTransformer2DModel)
    assert probe_mod.NeuronFluxTeacacheProbeFusedApplication._model_cls is (
        probe_mod.FluxTeacacheProbeFusedModel
    )


def test_probe_declares_exactly_prev_mod_as_state():
    assert STATE == frozenset({"prev_mod"})
    assert NeuronApplicationBase.state_tensor_names == frozenset()
    assert mf.NeuronFluxBackboneApplication.state_tensor_names == frozenset()


# --------------------------------------------------- traced parameter names

@pytest.mark.parametrize("guidance_embeds", [True, False])
def test_probe_parameter_names_equal_backbone_names_plus_declared_state(guidance_embeds):
    config = _config(guidance_embeds=guidance_embeds)
    backbone, probe = _build(config)

    backbone_names = set(backbone.state_dict())
    probe_names = set(probe.state_dict())

    # Everything the probe's traced graph could look up in the shard table is
    # a backbone name; the only additions are the declared state tensors.
    assert probe_names - STATE == backbone_names
    assert probe_names - backbone_names == STATE
    assert not (STATE & backbone_names)


def test_probe_prev_mod_shape_matches_mod_input():
    config = _config()
    _, probe = _build(config)
    assert tuple(probe.prev_mod.shape) == (1, _seq_len(config), INNER)
    assert probe.prev_mod.requires_grad is False


# ------------------------------------------------------ checkpoint converter

def test_probe_converter_is_the_backbone_converter():
    probe_fn = probe_mod.NeuronFluxTeacacheProbeFusedApplication.convert_hf_to_neuron_state_dict
    backbone_fn = mf.NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict
    assert probe_fn is backbone_fn


def test_probe_converted_keys_equal_backbone_converted_keys():
    config = _config()
    backbone, _ = _build(config)
    sd = _hf_state_dict(backbone, config)

    probe_keys = set(
        probe_mod.NeuronFluxTeacacheProbeFusedApplication.convert_hf_to_neuron_state_dict(
            dict(sd), config
        )
    )
    backbone_keys = set(
        mf.NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(dict(sd), config)
    )
    assert probe_keys == backbone_keys
    assert "global_rank.rank" in probe_keys  # rank marker rides along unchanged
    assert not any(k.startswith("trace_module.") for k in probe_keys)
    assert not (STATE & probe_keys)  # state is never expected from the shards


# ----------------------------------------------- the loader-facing invariant

def test_backbone_checkpoint_serves_every_probe_weight():
    """Every non-state tensor in the probe's state dict is present, under the
    same key, in the checkpoint the backbone shards. This is the host-side
    form of the on-device hard error ("Missing weight tensor with key ...")."""
    config = _config()
    backbone, probe = _build(config)
    converted = mf.NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(
        _hf_state_dict(backbone, config), config
    )

    assert checkpoint_missing_weights(probe, converted, STATE) == set()
    # And the check is not vacuous: forgetting to declare prev_mod as state
    # reports it as a missing weight, exactly like the wrapper-era keys would.
    assert checkpoint_missing_weights(probe, converted) == {"prev_mod"}
    nested = {f"trace_module.transformer.{k}": v for k, v in converted.items()}
    assert "transformer_blocks.0.norm1.linear.bias" in checkpoint_missing_weights(
        probe, nested, STATE
    )


def test_aliased_tensors_are_exactly_the_declared_state():
    """What the ModelInstance aliases is what NxD turns into INPUT_STATE, so
    the declared state set must be precisely the aliased parameter names."""
    config = _config()
    wrapper = probe_mod.ModelWrapperFluxTeacacheProbeFused(
        config,
        probe_mod.FluxTeacacheProbeFusedModel,
        tag="probe",
        compiler_args="",
        priority_model_idx=0,
    )
    instance = wrapper.get_model_instance()
    instance.load_module()
    module, aliases = instance.get(0)

    assert isinstance(module, mf.NeuronFluxTransformer2DModel)
    aliased = {
        name for name, param in module.named_parameters() if any(param is a for a in aliases)
    }
    assert aliased == STATE
    assert set(aliases.values()) == {1}  # mod_input is output index 1


# ------------------------------------------------------------ forward parity

@pytest.mark.parametrize("guidance_embeds", [True, False])
def test_probe_forward_matches_backbone_prefix_and_updates_from_prev_mod(
    patched_markers, guidance_embeds
):
    config = _config(guidance_embeds=guidance_embeds)
    backbone, probe = _build(config)
    # Same weights: load the backbone's tensors into the probe under the
    # backbone's own names (strict apart from the declared state).
    missing, unexpected = probe.load_state_dict(backbone.state_dict(), strict=False)
    assert set(missing) == STATE and unexpected == []

    seq = _seq_len(config)
    hidden = torch.randn(1, seq, config.in_channels)
    timestep = torch.rand(1)
    pooled = torch.randn(1, config.pooled_projection_dim)
    guidance = torch.rand(1) if guidance_embeds else torch.tensor([])

    with torch.no_grad():
        hs = backbone.x_embedder(hidden)
        if guidance_embeds:
            temb = backbone.time_text_embed(timestep * 1000, guidance * 1000, pooled)
        else:
            temb = backbone.time_text_embed(timestep * 1000, pooled)
        expected, *_ = backbone.transformer_blocks[0].norm1(hs, emb=temb, hlomarker=True)

        rel_l1, mod_input = probe(hidden, timestep, pooled, guidance)

    assert torch.equal(mod_input, expected)
    assert tuple(mod_input.shape) == tuple(probe.prev_mod.shape)
    # prev_mod is zero at init, so the signal is |mod| / (0 + eps): finite, > 0.
    assert torch.isfinite(rel_l1) and rel_l1.item() > 0
    with torch.no_grad():
        probe.prev_mod.copy_(mod_input)
        again, _ = probe(hidden, timestep, pooled, guidance)
    assert again.item() == pytest.approx(0.0, abs=1e-6)
