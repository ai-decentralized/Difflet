"""Experiment E1 in modeling_wan (DIFFLET_EXP_ATTN_CONSTMAX): each expert's table and per-layer
allowlist on the self-attentions, picked by the directory the expert loads; only attn1's kernel
calls carry the sigma; unset, nothing changes. CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import difflet.models.wan.modeling_wan as wan

ENV = "DIFFLET_EXP_ATTN_CONSTMAX"
TABLES = json.loads(Path(wan.__file__).with_name("attn_constmax.json").read_text())["tables"]
WAN21 = "/c/weights-bf16/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/transformer"


def _model(num_layers=40):
    torch.manual_seed(0)
    return wan.WanTransformer3DModel(wan.WanTransformerConfig(
        patch_size=(1, 2, 2), num_attention_heads=4, attention_head_dim=16, in_channels=4,
        out_channels=4, text_dim=24, freq_dim=32, ffn_dim=48, num_layers=num_layers))


@pytest.mark.parametrize("path, key", [
    (WAN21, "38ec498c/transformer"),
    ("/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/"
     "5be7df9619b54f4e2667b2755bc6a756675b5cd7/transformer", "5be7df96/transformer"),
    ("/c/quantized/Wan-AI--Wan2.2-T2V-A14B-Diffusers__5be7df96/transformer_2/mxfp8-03f6d63f26aab3cf",
     "5be7df96/transformer_2"),
])
def test_each_expert_takes_its_own_tables_allowlist(monkeypatch, path, key):
    monkeypatch.setenv(ENV, "1")
    model = _model()
    assert wan.add_constmax_sigma_(model, path) == 36
    routed = [i for i, b in enumerate(model.blocks) if hasattr(b.attn1, "constmax_sigma")]
    assert routed == [i for i in range(40) if i not in (33, 34, 38, 39)]  # the gap rule's failures
    for i in routed:
        sigma = model.blocks[i].attn1.constmax_sigma
        assert sigma.shape == (128, 1)
        assert torch.all(sigma == torch.tensor(TABLES[key]["layers"][f"blocks.{i}.attn1"]["sigma"]))
    assert not any(hasattr(b.attn2, "constmax_sigma") for b in model.blocks)  # cross-attn: cte


def test_env_unset_adds_nothing(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    model = _model()
    assert wan.add_constmax_sigma_(model, WAN21) == 0
    assert not any(hasattr(b.attn1, "constmax_sigma") for b in model.blocks)


def test_a_checkpoint_without_a_table_is_refused(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError, match="no single const-max table"):
        wan.add_constmax_sigma_(_model(), "/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/0/transformer")


def test_only_the_self_attention_calls_carry_the_sigma(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    model = _model(num_layers=2)
    wan.add_constmax_sigma_(model, WAN21)
    calls = []

    def spy(q, k, v, *, scale, **kw):
        calls.append(kw.get("constmax_sigma"))
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    monkeypatch.setattr(wan, "attention", spy)
    g = torch.Generator().manual_seed(0)
    model(torch.randn(1, 4, 2, 8, 8, generator=g), torch.tensor([3.0]), torch.randn(1, 5, 24, generator=g))
    assert [c is model.blocks[i // 2].attn1.constmax_sigma if i % 2 == 0 else c is None
            for i, c in enumerate(calls)] == [True] * 4  # attn1, attn2 of each block
