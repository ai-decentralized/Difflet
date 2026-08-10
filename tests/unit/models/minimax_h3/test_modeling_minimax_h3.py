from __future__ import annotations

import torch

from difflet.models.minimax_h3.modeling_minimax_h3 import MiniMaxH3Transformer3DModel


def _tiny_model():
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=1,
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
    ).eval()


def test_tiny_h3_transformer_runs_joint_text_audio_video_forward():
    model = _tiny_model()
    output = model(
        hidden_states=torch.randn(1, 2, 2),
        audio_hidden_states=torch.randn(1, 2, 3),
        encoder_hidden_states=torch.randn(1, 2, 4),
        timestep=torch.tensor([0.5]),
        timestep_indices=torch.zeros(6, dtype=torch.long),
        token_tags=torch.tensor([1, 1, 2, 2, 0, 0]),
        position_ids=torch.arange(6).unsqueeze(1).expand(-1, 3).float(),
        video_indices=torch.tensor([4, 5]),
        audio_indices=torch.tensor([2, 3]),
        text_indices=torch.tensor([0, 1]),
        return_dict=False,
    )

    assert output[0].shape == (1, 2, 2)
    assert output[1].shape == (1, 2, 3)
    assert torch.isfinite(output[0]).all()
    assert torch.isfinite(output[1]).all()


def test_h3_checkpoint_module_names_match_official_diffusers_port():
    keys = _tiny_model().state_dict()

    assert "token_refiner.refiner_blocks.0.ff.net.0.proj.weight" in keys
    assert "transformer_blocks.0.adaln_proj.linear.weight" in keys
    assert "transformer_blocks.0.attn.to_q.weight" in keys
    assert "audio_proj_out.weight" in keys
