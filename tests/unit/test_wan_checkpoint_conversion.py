"""Unit tests for difflet.models.wan.checkpoint conversion (W3b)."""

from __future__ import annotations

import torch


def test_backbone_renames_ffn_proj_keys():
    from difflet.models.wan.checkpoint import convert_backbone_state_dict

    raw = {
        "blocks.0.ffn.net.0.proj.weight": torch.zeros(1),
        "blocks.0.ffn.net.0.proj.bias": torch.zeros(1),
        "blocks.0.ffn.net.2.weight": torch.zeros(1),
        "blocks.0.ffn.net.2.bias": torch.zeros(1),
        "blocks.39.ffn.net.0.proj.weight": torch.zeros(1),
        "blocks.39.ffn.net.2.bias": torch.zeros(1),
    }
    out = convert_backbone_state_dict(raw)
    expected = {
        "blocks.0.ffn.net_in.weight",
        "blocks.0.ffn.net_in.bias",
        "blocks.0.ffn.net_out.weight",
        "blocks.0.ffn.net_out.bias",
        "blocks.39.ffn.net_in.weight",
        "blocks.39.ffn.net_out.bias",
    }
    assert set(out) == expected


def test_backbone_preserves_attention_and_other_keys_verbatim():
    from difflet.models.wan.checkpoint import convert_backbone_state_dict

    raw = {
        "patch_embedding.weight": torch.zeros(1),
        "patch_embedding.bias": torch.zeros(1),
        "condition_embedder.time_embedder.linear_1.weight": torch.zeros(1),
        "condition_embedder.text_embedder.linear_1.weight": torch.zeros(1),
        "blocks.0.attn1.to_q.weight": torch.zeros(1),
        "blocks.0.attn1.norm_q.weight": torch.zeros(1),
        "blocks.0.attn2.to_k.weight": torch.zeros(1),
        "blocks.0.scale_shift_table": torch.zeros(1),
        "blocks.0.norm2.weight": torch.zeros(1),
        "blocks.0.norm2.bias": torch.zeros(1),
        "norm_out.weight": torch.zeros(1),  # FP32LayerNorm without affine has no weight; here just guarding
        "proj_out.weight": torch.zeros(1),
        "proj_out.bias": torch.zeros(1),
        "scale_shift_table": torch.zeros(1),
    }
    out = convert_backbone_state_dict(raw)
    # Every key carries over unchanged (no FFN keys here).
    assert set(out) == set(raw)


def test_text_encoder_conversion_is_identity():
    from difflet.models.wan.checkpoint import convert_text_encoder_state_dict

    raw = {
        "shared.weight": torch.zeros(1),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.zeros(1),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": torch.zeros(1),
        "encoder.block.0.layer.0.layer_norm.weight": torch.zeros(1),
        "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": torch.zeros(1),
        "encoder.block.0.layer.1.DenseReluDense.wi_1.weight": torch.zeros(1),
        "encoder.block.0.layer.1.DenseReluDense.wo.weight": torch.zeros(1),
        "encoder.block.23.layer.0.SelfAttention.q.weight": torch.zeros(1),
        "encoder.final_layer_norm.weight": torch.zeros(1),
    }
    out = convert_text_encoder_state_dict(raw)
    assert set(out) == set(raw)
    # Tensors pass through (identity)
    for k in raw:
        assert out[k].data_ptr() == raw[k].data_ptr()


def test_vae_decoder_conversion_filters_encoder_and_kl_keys():
    from difflet.models.wan.checkpoint import convert_vae_decoder_state_dict

    raw = {
        "post_quant_conv.weight": torch.zeros(1),
        "post_quant_conv.bias": torch.zeros(1),
        "decoder.conv_in.weight": torch.zeros(1),
        "decoder.up_blocks.0.resnets.0.conv1.weight": torch.zeros(1),
        "encoder.conv_in.weight": torch.zeros(1),
        "quant_conv.weight": torch.zeros(1),
    }

    out = convert_vae_decoder_state_dict(raw)

    assert set(out) == {
        "post_quant_conv.weight",
        "post_quant_conv.bias",
        "decoder.conv_in.weight",
        "decoder.up_blocks.0.resnets.0.conv1.weight",
    }
    for key in out:
        assert out[key].data_ptr() == raw[key].data_ptr()


def test_convert_diffusers_checkpoint_handles_missing_components(tmp_path):
    """When ``model_dir`` has no convertable components, the helper raises
    a clear ``FileNotFoundError`` rather than silently doing nothing.
    """
    import pytest
    from difflet.models.wan.checkpoint import convert_diffusers_checkpoint

    empty = tmp_path / "snapshot"
    empty.mkdir()
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="convertable components"):
        convert_diffusers_checkpoint(model_dir=empty, out_dir=out)


def test_component_converters_registry_covers_known_subdirs():
    from difflet.models.wan.checkpoint.cli import COMPONENT_CONVERTERS

    # The Wan2.2 14B snapshot has these directories that we currently know
    # how to handle.
    assert "transformer" in COMPONENT_CONVERTERS
    assert "transformer_2" in COMPONENT_CONVERTERS
    assert "text_encoder" in COMPONENT_CONVERTERS
    assert "vae" in COMPONENT_CONVERTERS


def test_trainium_wan_backbone_app_uses_checkpoint_converter():
    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication

    raw = {
        "blocks.0.ffn.net.0.proj.weight": torch.zeros(1),
        "blocks.0.ffn.net.2.bias": torch.zeros(1),
        "blocks.0.attn1.to_q.weight": torch.zeros(1),
    }

    out = NeuronWanBackboneApplication.convert_hf_to_neuron_state_dict(raw, config=None)

    assert "blocks.0.ffn.net_in.weight" in out
    assert "blocks.0.ffn.net_out.bias" in out
    assert "blocks.0.attn1.to_q.weight" in out
    assert "blocks.0.ffn.net.0.proj.weight" not in out


def test_trainium_wan_text_encoder_app_uses_identity_checkpoint_converter():
    from difflet.backends.trainium.wan.text_encoder import NeuronWanTextEncoderApplication

    raw = {
        "shared.weight": torch.zeros(1),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.zeros(1),
    }

    out = NeuronWanTextEncoderApplication.convert_hf_to_neuron_state_dict(
        raw, config=None
    )

    assert set(out) == set(raw)
    for key in raw:
        assert out[key].data_ptr() == raw[key].data_ptr()


def test_trainium_wan_vae_app_uses_decoder_checkpoint_converter():
    from difflet.backends.trainium.wan.vae import NeuronWanVAEDecoderApplication

    raw = {
        "post_quant_conv.weight": torch.zeros(1),
        "decoder.conv_out.bias": torch.zeros(1),
        "encoder.conv_in.weight": torch.zeros(1),
    }

    out = NeuronWanVAEDecoderApplication.convert_hf_to_neuron_state_dict(
        raw, config=None
    )

    assert set(out) == {"post_quant_conv.weight", "decoder.conv_out.bias"}
