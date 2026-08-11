"""Equivalence tests for the precomputed-AdaLN inference path.

The 13B adaln_proj branch depends only on (timestep, modality). These tests
prove that dropping it from the resident model and feeding the offline table
instead reproduces the resident computation exactly, and that the table
builder's row layout matches what `adaln_proj` emits.
"""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from difflet.models.minimax_h3.modeling_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Transformer3DModel,
)
from difflet.models.minimax_h3.pipeline import build_adaln_modulation_table

_TINY = dict(
    num_attention_heads=2,
    attention_head_dim=8,
    hidden_size=16,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=32,
    in_channels=2,
    audio_in_channels=3,
    patch_size=(1, 1, 1),
    text_dim=4,
    freq_dim=4,
    time_embed_hidden_dim=16,
    time_embed_dim=8,
    rope_freq_dim=1,
)

_FORWARD_INPUTS = dict(
    hidden_states=torch.randn(1, 2, 2),
    audio_hidden_states=torch.randn(1, 2, 3),
    encoder_hidden_states=torch.randn(1, 2, 4),
    timestep=torch.tensor([0.3, 0.7]),
    timestep_indices=torch.tensor([1, 1, 0, 0, 1, 1]),
    token_tags=torch.tensor([1, 1, 2, 2, 0, 0]),
    position_ids=torch.arange(6).unsqueeze(1).expand(-1, 3).float(),
    video_indices=torch.tensor([4, 5]),
    audio_indices=torch.tensor([2, 3]),
    text_indices=torch.tensor([0, 1]),
)


def _modulation_from_reference(model: MiniMaxH3Transformer3DModel) -> torch.Tensor:
    """Build [num_layers, 6, rows, hidden] straight from the resident branch."""

    timestep = _FORWARD_INPUTS["timestep"]
    temb = model.time_embedder(model.time_proj(timestep).to(torch.float32))
    layers = []
    for block in model.transformer_blocks:
        layers.append(torch.stack(block.adaln_proj(temb), dim=0))
    return torch.stack(layers, dim=0)


def test_precomputed_path_matches_resident_adaln_exactly():
    torch.manual_seed(0)
    reference = MiniMaxH3Transformer3DModel(**_TINY).eval()

    precomputed = MiniMaxH3Transformer3DModel(**_TINY, precomputed_adaln=True).eval()
    missing, unexpected = precomputed.load_state_dict(reference.state_dict(), strict=False)
    assert not missing
    assert all(".adaln_proj." in key for key in unexpected), unexpected
    assert all(block.adaln_proj is None for block in precomputed.transformer_blocks)

    with torch.no_grad():
        expected = reference(**_FORWARD_INPUTS, return_dict=False)
        table = _modulation_from_reference(reference)
        actual = precomputed(**_FORWARD_INPUTS, block_modulation=table, return_dict=False)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_precomputed_model_validates_modulation_presence_and_shape():
    model = MiniMaxH3Transformer3DModel(**_TINY, precomputed_adaln=True).eval()
    try:
        model(**_FORWARD_INPUTS, return_dict=False)
        raise AssertionError("expected a missing-modulation error")
    except ValueError as error:
        assert "block_modulation" in str(error)

    resident = MiniMaxH3Transformer3DModel(**_TINY).eval()
    try:
        resident(
            **_FORWARD_INPUTS,
            block_modulation=torch.zeros(2, 6, 6, 16),
            return_dict=False,
        )
        raise AssertionError("expected a rejection on the resident path")
    except ValueError as error:
        assert "precomputed_adaln" in str(error)


def test_table_builder_matches_adaln_proj_row_layout(tmp_path):
    torch.manual_seed(1)
    model = MiniMaxH3Transformer3DModel(**_TINY).eval()

    # Persist only what the builder streams: time_embedder + per-block adaln.
    tensors = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("time_embedder.") or ".adaln_proj." in key
    }
    shard = "diffusion_pytorch_model.safetensors"
    save_file(tensors, str(tmp_path / shard))
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in tensors}})
    )

    steps = torch.tensor([[0.9, 0.8], [0.5, 0.4], [0.1, 0.05]], dtype=torch.float32)
    table = build_adaln_modulation_table(
        str(tmp_path),
        steps,
        num_layers=_TINY["num_layers"],
        freq_dim=_TINY["freq_dim"],
        time_embed_hidden_dim=_TINY["time_embed_hidden_dim"],
        time_embed_dim=_TINY["time_embed_dim"],
        hidden_size=_TINY["hidden_size"],
        dtype=torch.float32,
    )
    assert table.shape == (
        3,
        _TINY["num_layers"],
        6,
        2 * MINIMAX_H3_MODALITY_NUM,
        _TINY["hidden_size"],
    )

    with torch.no_grad():
        for step in range(steps.shape[0]):
            temb = model.time_embedder(model.time_proj(steps[step]).to(torch.float32))
            for layer, block in enumerate(model.transformer_blocks):
                expected = torch.stack(block.adaln_proj(temb), dim=0)
                # The builder batches all steps through one GEMM, the module runs
                # per-step: same math, different fp32 accumulation order.
                torch.testing.assert_close(table[step, layer], expected)
