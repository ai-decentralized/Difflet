"""--attention const-max for MiniMax-H3 (DIFFLET_EXP_ATTN_CONSTMAX): the committed table, the
blocks it routes, and the bounded attention call carrying their sigma. CPU, no checkpoint."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from difflet.backends.neuron.minimax_h3 import transformer as h3  # noqa: E402
from difflet.models.minimax_h3.modeling_minimax_h3_tp import MiniMaxH3TPAttnProcessor  # noqa: E402

_TABLE = json.loads(h3.CONSTMAX_TABLE.read_text())
#: The four blocks whose gap exceeds 80 at Sk 5120 (scripts/attn_constmax/table.py minimax_h3).
_UNROUTED = {44, 45, 48, 49}


def _blocks(n=50):
    return SimpleNamespace(transformer_blocks=[SimpleNamespace(attn=nn.Module()) for _ in range(n)])


def test_the_table_covers_the_50_blocks_at_the_fleet_shape():
    assert _TABLE["model"] == "minimax_h3" and _TABLE["d"] == 128
    assert _TABLE["reference_sk"] == 5120 and _TABLE["gap_limit"] == 80.0
    assert set(_TABLE["layers"]) == {f"transformer_blocks.{i}.attn" for i in range(50)}
    failing = {i for i in range(50) if not _TABLE["layers"][f"transformer_blocks.{i}.attn"]["ok"]}
    assert failing == _UNROUTED


def test_the_env_routes_every_block_that_passes_the_gap_rule(monkeypatch):
    monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", "1")
    transformer = _blocks()

    routed = h3.add_constmax_sigma_(transformer)

    assert set(routed) == set(range(50)) - _UNROUTED
    for i, block in enumerate(transformer.transformer_blocks):
        sigma = getattr(block.attn, "constmax_sigma", None)
        if i in _UNROUTED:
            assert sigma is None
        else:
            assert sigma.shape == (128, 1)
            assert float(sigma[0, 0]) == pytest.approx(
                _TABLE["layers"][f"transformer_blocks.{i}.attn"]["sigma"])
            assert "constmax_sigma" not in block.attn.state_dict()  # non-persistent


@pytest.mark.parametrize("env", [None, "0"])
def test_without_the_env_no_block_carries_a_sigma(monkeypatch, env):
    if env is None:
        monkeypatch.delenv("DIFFLET_EXP_ATTN_CONSTMAX", raising=False)
    else:
        monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", env)
    transformer = _blocks()

    assert h3.add_constmax_sigma_(transformer) == []
    assert not any(hasattr(b.attn, "constmax_sigma") for b in transformer.transformer_blocks)


class _Attn(nn.Module):
    """Two heads of 128: the processor's projections, identity norms, no output change."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.to_q, self.to_k, self.to_v = (nn.Linear(256, 256) for _ in range(3))
        self.norm_q = self.norm_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Linear(256, 256), nn.Identity()])


@pytest.mark.parametrize("routed", [True, False])
def test_the_bounded_call_carries_the_sigma_of_a_routed_block(routed):
    attn = _Attn()
    if routed:
        attn.register_buffer("constmax_sigma", torch.full((128, 1), 3.0), persistent=False)
    seen = {}

    def fake_attention(q, k, v, **kwargs):
        seen.update(kwargs)
        return torch.zeros_like(q)

    processor = MiniMaxH3TPAttnProcessor()
    processor._attention = fake_attention
    mask = torch.tensor([[1] * 3 + [0] * 1])  # a live prefix of 3 keys out of 4

    processor(attn, torch.randn(1, 4, 256), attention_mask=mask)

    assert seen["scale"] == pytest.approx(1 / math.sqrt(128))
    assert int(seen["bound_min"].max()) == 0 and int(seen["bound_max"].min()) == 3  # key prefix
    if routed:
        assert torch.equal(seen["constmax_sigma"], attn.constmax_sigma)
    else:
        assert "constmax_sigma" not in seen
