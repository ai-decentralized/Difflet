"""Coverage for difflet.models.wan.checkpoint.backbone key remapping.

Complements the existing conversion tests by exercising the regex rename
behaviour and the tp_rank_util injection branch directly with mock dicts.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from difflet.models.wan.checkpoint.backbone import (
    BACKBONE_KEY_RENAMES,
    convert_backbone_state_dict,
)


def test_key_renames_table_shape():
    # Two ordered substring rewrites: up-proj and down-proj.
    assert len(BACKBONE_KEY_RENAMES) == 2
    patterns = [p for p, _ in BACKBONE_KEY_RENAMES]
    assert any("net" in p for p in patterns)


def test_renames_apply_per_block_index():
    raw = {
        "blocks.7.ffn.net.0.proj.weight": torch.zeros(1),
        "blocks.7.ffn.net.2.bias": torch.zeros(1),
        "blocks.12.ffn.net.0.proj.bias": torch.zeros(1),
    }
    out = convert_backbone_state_dict(raw)
    assert set(out) == {
        "blocks.7.ffn.net_in.weight",
        "blocks.7.ffn.net_out.bias",
        "blocks.12.ffn.net_in.bias",
    }


def test_non_ffn_keys_pass_through_unchanged():
    raw = {
        "patch_embedding.weight": torch.zeros(1),
        "blocks.0.attn1.to_q.weight": torch.zeros(1),
        "scale_shift_table": torch.zeros(1),
    }
    out = convert_backbone_state_dict(raw)
    assert set(out) == set(raw)
    for key in raw:
        assert out[key].data_ptr() == raw[key].data_ptr()


def test_tp_rank_util_injected_when_tp_degree_gt_one():
    config = SimpleNamespace(
        context_parallel_enabled=False,
        cfg_parallel_enabled=False,
        neuron_config=SimpleNamespace(world_size=4, tp_degree=4),
    )
    out = convert_backbone_state_dict({"a.weight": torch.zeros(1)}, config=config)
    assert "tp_rank_util.rank" in out
    assert torch.equal(out["tp_rank_util.rank"], torch.arange(0, 4, dtype=torch.int32))
    # tp-only (no CP/CFG) → no global_rank buffer.
    assert "global_rank.rank" not in out


def test_tp_rank_util_absent_at_tp_degree_one():
    config = SimpleNamespace(
        context_parallel_enabled=False,
        cfg_parallel_enabled=False,
        neuron_config=SimpleNamespace(world_size=1, tp_degree=1),
    )
    out = convert_backbone_state_dict({"a.weight": torch.zeros(1)}, config=config)
    assert "tp_rank_util.rank" not in out


def test_both_global_rank_and_tp_rank_injected_together():
    config = SimpleNamespace(
        context_parallel_enabled=True,
        cfg_parallel_enabled=False,
        neuron_config=SimpleNamespace(world_size=8, tp_degree=2),
    )
    out = convert_backbone_state_dict({"a.weight": torch.zeros(1)}, config=config)
    assert torch.equal(out["global_rank.rank"], torch.arange(0, 8, dtype=torch.int32))
    assert torch.equal(out["tp_rank_util.rank"], torch.arange(0, 2, dtype=torch.int32))
