"""FP8 PTQ wiring for Qwen-Image: HF checkpoint -> Difflet names, config, compiler args."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from difflet.quant import checkpoint as ckpt
from difflet.quant.spec import QuantSpec

_TINY_QWEN_CONFIG = {
    "_class_name": "QwenImageTransformer2DModel",
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "guidance_embeds": False,
    "axes_dims_rope": [2, 2, 4],
}


def _qwen_like_hf_state_dict() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    return {
        "transformer_blocks.0.attn.to_q.weight": r(16, 16),
        "transformer_blocks.0.attn.add_q_proj.weight": r(16, 16),
        "transformer_blocks.0.attn.to_out.0.weight": r(16, 16),
        "transformer_blocks.0.attn.to_add_out.weight": r(16, 16),
        "transformer_blocks.0.img_mlp.net.0.proj.weight": r(32, 16),
        "transformer_blocks.0.txt_mlp.net.2.weight": r(16, 32),
        "transformer_blocks.0.img_mod.1.weight": r(96, 16),
        "transformer_blocks.0.txt_mod.1.weight": r(96, 16),
        "img_in.weight": r(16, 64),
        "txt_in.weight": r(16, 32),
        "norm_out.linear.weight": r(32, 16),
        "proj_out.weight": r(64, 16),
        "time_text_embed.timestep_embedder.linear_1.weight": r(16, 16),
    }


def test_quantized_hf_checkpoint_maps_onto_difflet_qwen_names():
    """Targets get fp8 weights + scales and the converter's ``transformer.`` prefix
    is applied to the scales too; modulation, embedders, norm_out and proj_out stay bf16."""
    from difflet.backends.trainium.qwen_image.transformer import NeuronQwenImageTransformerApplication

    quantized, report = ckpt.quantize_state_dict(_qwen_like_hf_state_dict(), QuantSpec.for_model("qwen_image"))
    assert report["num_quantized"] == 6
    renamed = {k.replace(".weight_scale", ".scale"): v for k, v in quantized.items()}  # get_state_dict
    config = SimpleNamespace(neuron_config=SimpleNamespace(world_size=1, tp_degree=1))
    out = NeuronQwenImageTransformerApplication.convert_hf_to_neuron_state_dict(renamed, config)
    assert out["transformer.transformer_blocks.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn
    assert out["transformer.transformer_blocks.0.attn.to_q.scale"].dtype == torch.float32
    assert out["transformer.transformer_blocks.0.img_mlp.net.0.proj.scale"].shape == (1,)
    assert not any(k.startswith("transformer_blocks.") for k in out)  # everything prefixed
    for stays in ("transformer.img_in.weight", "transformer.txt_in.weight", "transformer.norm_out.linear.weight",
                  "transformer.proj_out.weight", "transformer.transformer_blocks.0.img_mod.1.weight",
                  "transformer.time_text_embed.timestep_embedder.linear_1.weight"):
        assert out[stays].dtype == torch.bfloat16
        assert stays.replace(".weight", ".scale") not in out


def test_qwen_transformer_config_carries_the_quant_fields(tmp_path, monkeypatch):
    pytest.importorskip("neuronx_distributed")
    from difflet.models.qwen_image import application as qapp

    import json

    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(_TINY_QWEN_CONFIG))
    plain = qapp.create_qwen_image_transformer_config(
        model_path=str(tmp_path), world_size=4, tp_degree=4, dtype=torch.bfloat16,
        height=64, width=64, text_seq_len=8)
    assert not getattr(plain.neuron_config, "quantized", False)
    fp8 = qapp.create_qwen_image_transformer_config(
        model_path=str(tmp_path), world_size=4, tp_degree=4, dtype=torch.bfloat16,
        height=64, width=64, text_seq_len=8,
        quant=QuantSpec.for_model("qwen_image", activation="none"), quant_checkpoint_dir=tmp_path / "q")
    nc = fp8.neuron_config
    assert nc.quantized and nc.quantized_checkpoints_path == str(tmp_path / "q")
    assert nc.quant_targets == list(QuantSpec.for_model("qwen_image").targets)
    with pytest.raises(ValueError, match="quant_checkpoint_dir"):
        qapp.create_qwen_image_transformer_config(
            model_path=str(tmp_path), world_size=4, tp_degree=4, dtype=torch.bfloat16,
            height=64, width=64, text_seq_len=8, quant=QuantSpec.for_model("qwen_image"))


def test_qwen_compiler_args_carry_the_fp8_flag_only_when_quantized():
    from difflet.backends.trainium.qwen_image.transformer import NeuronQwenImageTransformerApplication

    def args_for(quantized):
        app = NeuronQwenImageTransformerApplication.__new__(NeuronQwenImageTransformerApplication)
        app.config = SimpleNamespace(neuron_config=SimpleNamespace(
            world_size=4, quantized=quantized, quantization_dtype="f8e4m3"))
        return app.get_compiler_args()

    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" not in args_for(False)
    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" in args_for(True)
