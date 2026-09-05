"""Megatron-SP wiring + equivalence for difflet.models.qwen_image.modeling_qwen.

On the CPU backend every sequence collective is identity (tp==1), so the SP
fork with the same weights as the diffusers parent must produce bit-identical
output. That equivalence is the host-side proof that the fork's forward is a
faithful copy and that the SP insertions no-op cleanly at tp==1. Real
multi-rank numerics (the g/ḡ placement over the joint attention and per-stream
MLPs, the _sp_unbias correction) are covered on device by
``scripts/qwen_sp_parity_smoke.sh``.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

_ORIG_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"

import difflet.ops as _ops  # noqa: E402

importlib.reload(_ops)
import difflet.models.qwen_image.modeling_qwen as qwen  # noqa: E402

importlib.reload(qwen)

if _ORIG_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _ORIG_BACKEND


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


def _tiny_kwargs(**overrides):
    base = dict(
        patch_size=2,
        in_channels=8,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=24,
        # The M4a Trainium config: guidance_embeds=False, guidance=None. (In
        # diffusers 0.38 the guidance branch of QwenTimestepProjEmbeddings is
        # unreachable upstream — passing guidance crashes identically in the
        # parent and the fork.)
        guidance_embeds=False,
        axes_dims_rope=(2, 4, 2),  # exactly 3 even entries (QwenEmbedRope uses [0..2])
    )
    base.update(overrides)
    return base


def _forward(model):
    torch.manual_seed(0)
    hidden = torch.randn(1, 16, 8)
    timestep = torch.tensor([0.7])
    encoder = torch.randn(1, 8, 24)
    mask = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        out = model(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            encoder_hidden_states_mask=mask,
            timestep=timestep,
            img_shapes=[[(1, 4, 4)]],  # per-sample list of frame tuples
            guidance=None,
            return_dict=False,
        )[0]
    return out


# ------------------------------------------------------------------ threading

def test_sp_model_subclasses_diffusers_parent():
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModel,
    )

    model = qwen.QwenImageSPTransformer2DModel(**_tiny_kwargs())
    assert isinstance(model, QwenImageTransformer2DModel)
    assert type(model.transformer_blocks[0]) is qwen.QwenImageSPTransformerBlock


def test_sp_flag_degrades_to_dense_at_tp1():
    # On the CPU reference (tp==1) sp_enabled is forced off: collectives are
    # identity, so the fork must behave exactly like the dense parent.
    model = qwen.QwenImageSPTransformer2DModel(**_tiny_kwargs(), sp_enabled=True)
    assert model.sp_enabled is False
    assert model.transformer_blocks[0].sp_enabled is False


def test_state_dict_keys_match_diffusers_parent():
    # Weight-loading contract: identical submodule names/layout so diffusers'
    # from_pretrained checkpoints load into the SP fork unchanged.
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModel,
    )

    dense = QwenImageTransformer2DModel(**_tiny_kwargs())
    fork = qwen.QwenImageSPTransformer2DModel(**_tiny_kwargs(), sp_enabled=True)
    assert list(fork.state_dict().keys()) == list(dense.state_dict().keys())


def test_registry_pins_qwen_sp_support():
    from difflet.registry import resolve_model

    entry = resolve_model("Qwen/Qwen-Image", model_type="qwen_image")
    assert entry.capabilities.supports_sp is True


# ----------------------------------------------------------------- equivalence

def test_sp_fork_matches_dense_parent_bit_identical():
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModel,
    )

    torch.manual_seed(0)
    dense = QwenImageTransformer2DModel(**_tiny_kwargs()).eval()
    fork = qwen.QwenImageSPTransformer2DModel(**_tiny_kwargs(), sp_enabled=True).eval()
    fork.load_state_dict(dense.state_dict())

    out_dense = _forward(dense)
    out_fork = _forward(fork)
    assert out_fork.shape == out_dense.shape
    assert torch.equal(out_dense, out_fork)


def test_sp_unbias_is_identity_without_bias_or_tp1():
    torch.manual_seed(1)
    linear = torch.nn.Linear(8, 8, bias=True).eval()
    x = torch.randn(1, 4, 8)
    assert torch.equal(qwen._sp_unbias(x, linear), x)  # tp==1 on CPU

    no_bias = torch.nn.Linear(8, 8, bias=False).eval()
    assert torch.equal(qwen._sp_unbias(x, no_bias), x)


def test_sp_unbias_subtracts_overcount_at_tp2(monkeypatch):
    monkeypatch.setattr(qwen, "_safe_tp_size", lambda: 2)
    torch.manual_seed(2)
    linear = torch.nn.Linear(8, 8, bias=True).eval()
    x = torch.randn(1, 4, 8)
    expected = x - (2 - 1) * linear.bias
    assert torch.allclose(qwen._sp_unbias(x, linear), expected, atol=1e-7)


def test_shard_sequence_rejects_uneven_shards(monkeypatch):
    from difflet.ops import SPMDRank

    monkeypatch.setattr(qwen, "_safe_tp_size", lambda: 2)
    util = SPMDRank(world_size=2)
    x = torch.randn(1, 5, 8)  # 5 % 2 != 0
    with pytest.raises(ValueError, match="divide tp"):
        qwen._shard_sequence(x, util, what="image")


def test_shard_sequence_identity_without_util():
    torch.manual_seed(3)
    x = torch.randn(1, 6, 8)
    out = qwen._shard_sequence(x, None, what="image")
    assert out is x  # CPU reference / tp==1: untouched


def test_sp_rejects_zero_cond_t(monkeypatch):
    # zero_cond_t's modulate_index is per-token and rank-dependent; SP must
    # refuse rather than silently mis-shard. The guard fires before any
    # collective, so a tp>1 stub is enough to observe it.
    monkeypatch.setattr(qwen, "_safe_tp_size", lambda: 2)
    model = qwen.QwenImageSPTransformer2DModel(
        **_tiny_kwargs(zero_cond_t=True), sp_enabled=True)
    assert model.sp_enabled is True
    with pytest.raises(NotImplementedError, match="zero_cond_t"):
        _forward(model)
