"""Experiment E1: LTX-2's const-max table and per-layer allowlist (DIFFLET_EXP_ATTN_CONSTMAX), on
the tiny diffusers-layout LTX-2 at the real depth; CPU only."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.unit.backends._neuron_ltx2_toy import tiny_ltx2_config  # noqa: E402


def _blocks(monkeypatch, env):
    if env:
        monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", env)
    else:
        monkeypatch.delenv("DIFFLET_EXP_ATTN_CONSTMAX", raising=False)
    from difflet.backends.neuron.ltx_2.transformer import _trace_module_class

    return _trace_module_class()(tiny_ltx2_config(num_layers=48, checkpoint_num_layers=48)).transformer.transformer_blocks


def test_layers_past_the_gap_rule_stay_on_cte(monkeypatch):
    import difflet.models.ltx_2.modeling_ltx_2 as modeling

    table = json.loads(Path(modeling.__file__).with_name("attn_constmax.json").read_text())["layers"]
    blocks = _blocks(monkeypatch, "1")
    routed = [i for i, b in enumerate(blocks) if hasattr(b.attn1, "constmax_sigma")]
    assert routed == [i for i in range(48) if i not in (44, 45, 47)]  # the table's 3 failures
    for i in routed:
        assert torch.all(blocks[i].attn1.constmax_sigma == torch.tensor(table[f"transformer_blocks.{i}.attn1"]["sigma"]))
    assert not any(hasattr(b.audio_attn1, "constmax_sigma") for b in blocks)  # video attn1 only


def test_env_unset_adds_nothing(monkeypatch):
    assert not any(hasattr(b.attn1, "constmax_sigma") for b in _blocks(monkeypatch, None))
