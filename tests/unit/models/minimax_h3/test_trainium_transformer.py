from __future__ import annotations

from types import SimpleNamespace

import torch

from difflet.backends.trainium.minimax_h3.transformer import (
    MiniMaxH3TransformerInferenceConfig,
    ModelWrapperMiniMaxH3Transformer,
    NeuronMiniMaxH3TransformerApplication,
    _prefix_bounds,
)
from difflet.backends.trainium.core.config import NeuronConfig


def _config(*, text_seq_len=8):
    values = {
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "hidden_size": 16,
        "num_layers": 1,
        "num_refiner_layers": 1,
        "ffn_dim": 32,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 16,
        "freq_dim": 4,
        "time_embed_hidden_dim": 16,
        "time_embed_dim": 8,
        "rope_freq_dim": 1,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }
    return MiniMaxH3TransformerInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=4,
            world_size=4,
            torch_dtype=torch.bfloat16,
        ),
        load_config=lambda config: config.__dict__.update(values),
        height=32,
        width=32,
        num_frames=124,
        text_seq_len=text_seq_len,
    )


def test_h3_transformer_config_derives_fixed_t2va_shapes():
    config = _config()

    assert config.num_video_latent_frames == 37
    assert config.video_seq_len == 37
    assert config.video_patch_dim == 96
    assert config.audio_seq_len == 414
    assert config.packed_seq_len == 512


def test_h3_model_wrapper_generates_compact_mask_and_runtime_indices():
    config = _config(text_seq_len=128)
    wrapper = ModelWrapperMiniMaxH3Transformer(config, model_cls=None)
    inputs = wrapper.input_generator()[0]

    assert inputs[0].shape == (1, 37, 96)
    assert inputs[1].shape == (1, 414, 32)
    assert inputs[2].shape == (1, 128, 16)
    assert inputs[6].shape == (640, 3)
    assert inputs[9].shape == (128,)
    assert inputs[10].sum().item() == 64
    assert inputs[11].sum().item() == 64 + 37 + 414


def test_h3_prefix_mask_lowers_to_per_head_attention_bounds():
    mask = torch.tensor([[True, True, True, False, False]])
    bound_min, bound_max = _prefix_bounds(mask, num_heads=2, query_length=5)

    assert bound_min.shape == (2, 5, 1)
    assert bound_max.shape == (2, 5, 1)
    assert torch.all(bound_min == 0)
    assert torch.all(bound_max == 3)


def test_h3_checkpoint_conversion_splits_swiglu_in_official_order():
    config = SimpleNamespace(num_refiner_layers=1, num_layers=1)
    state = {}
    for prefix in (
        "token_refiner.refiner_blocks.0.ff",
        "transformer_blocks.0.ff",
    ):
        state[f"{prefix}.net.0.proj.weight"] = torch.arange(16).reshape(4, 4)
        state[f"{prefix}.net.2.weight"] = torch.ones(4, 2)

    converted = NeuronMiniMaxH3TransformerApplication.convert_hf_to_neuron_state_dict(state, config)

    prefix = "transformer.transformer_blocks.0.ff"
    assert converted[f"{prefix}.up_proj.weight"].tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert converted[f"{prefix}.gate_proj.weight"].tolist() == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert f"{prefix}.net.0.proj.weight" not in converted
