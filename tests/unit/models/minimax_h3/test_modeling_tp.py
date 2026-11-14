"""Helpers of the backend-neutral TP rewrite (``difflet/models/minimax_h3/modeling_minimax_h3_tp.py``).
The Trainium re-export check is in ``test_trainium_transformer.py`` (needs the Neuron SDK)."""

from __future__ import annotations

import torch

from difflet.models.minimax_h3 import modeling_minimax_h3_tp as neutral


def test_ff_prefixes_cover_refiner_then_blocks():
    assert neutral.ff_prefixes(1, 2) == [
        "transformer.token_refiner.refiner_blocks.0.ff",
        "transformer.transformer_blocks.0.ff",
        "transformer.transformer_blocks.1.ff",
    ]
    assert neutral.ff_prefixes(0, 1, root="") == ["transformer_blocks.0.ff"]


def test_split_fused_swiglu_keys_is_in_place_and_skips_missing():
    state = {
        "a.ff.net.0.proj.weight": torch.arange(8).reshape(4, 2),
        "a.ff.net.2.weight": torch.zeros(2, 2),
    }
    out = neutral.split_fused_swiglu_keys(state, ["a.ff", "missing.ff"])
    assert out is state
    assert sorted(state) == ["a.ff.down_proj.weight", "a.ff.gate_proj.weight", "a.ff.up_proj.weight"]
    assert state["a.ff.up_proj.weight"].tolist() == [[0, 1], [2, 3]]
