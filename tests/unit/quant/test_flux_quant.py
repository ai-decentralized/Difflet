"""FP8 PTQ wiring for FLUX.1-dev: HF checkpoint -> Difflet names, config, app kwargs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from difflet.quant import checkpoint as ckpt
from difflet.quant.spec import QuantSpec


def _flux_like_hf_state_dict() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    return {
        "transformer_blocks.0.attn.to_q.weight": r(16, 16),
        "transformer_blocks.0.attn.add_q_proj.weight": r(16, 16),
        "transformer_blocks.0.ff.net.0.proj.weight": r(32, 16),
        "transformer_blocks.0.ff_context.net.2.weight": r(16, 32),
        "transformer_blocks.0.norm1.linear.weight": r(96, 16),
        "single_transformer_blocks.0.attn.to_q.weight": r(16, 16),
        "single_transformer_blocks.0.proj_mlp.weight": r(64, 16),
        "single_transformer_blocks.0.proj_out.weight": r(16, 80),
        "single_transformer_blocks.0.proj_out.bias": r(16),
        "single_transformer_blocks.0.norm.linear.weight": r(48, 16),
        "x_embedder.weight": r(16, 64),
        "proj_out.weight": r(64, 16),
        "norm_out.linear.weight": r(32, 16),
    }


def test_quantized_hf_checkpoint_maps_onto_difflet_flux_names():
    """The offline quantizer (HF names) + the FLUX converter: fp8 targets with
    scales, the fused single-block proj_out split with its scale copied to both
    halves, and embedders / modulation / root proj_out left bf16 (Review Focus 2)."""
    from difflet.models.flux.modeling_flux import NeuronFluxBackboneApplication

    quantized, report = ckpt.quantize_state_dict(_flux_like_hf_state_dict(), QuantSpec.for_model("flux"))
    assert report["num_quantized"] == 7
    renamed = {k.replace(".weight_scale", ".scale"): v for k, v in quantized.items()}  # get_state_dict
    config = SimpleNamespace(
        num_attention_heads=1, attention_head_dim=16, num_single_layers=1,
        neuron_config=SimpleNamespace(world_size=1),
    )
    out = NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(renamed, config)
    attn = out["single_transformer_blocks.0.proj_out_attn.weight"]
    mlp = out["single_transformer_blocks.0.proj_out_mlp.weight"]
    assert attn.dtype == torch.float8_e4m3fn and attn.shape == (16, 16)
    assert mlp.dtype == torch.float8_e4m3fn and mlp.shape == (16, 64)
    assert torch.equal(out["single_transformer_blocks.0.proj_out_attn.scale"],
                       out["single_transformer_blocks.0.proj_out_mlp.scale"])
    assert "single_transformer_blocks.0.proj_out_attn.bias" in out
    assert "single_transformer_blocks.0.proj_out.weight" not in out
    assert out["transformer_blocks.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn
    assert out["transformer_blocks.0.attn.to_q.scale"].dtype == torch.float32
    for stays in ("proj_out.weight", "x_embedder.weight", "norm_out.linear.weight",
                  "transformer_blocks.0.norm1.linear.weight", "single_transformer_blocks.0.norm.linear.weight"):
        assert out[stays].dtype == torch.bfloat16
        assert stays.replace(".weight", ".scale") not in out


def test_flux_backbone_neuron_config_carries_the_quant_fields(tmp_path):
    pytest.importorskip("neuronx_distributed")
    from difflet.models.flux.application import backbone_neuron_config

    plain = backbone_neuron_config(tp_degree=4, world_size=4, dtype=torch.bfloat16)
    assert not getattr(plain, "quantized", False)
    fp8 = backbone_neuron_config(tp_degree=4, world_size=4, dtype=torch.bfloat16,
                                 quant=QuantSpec.for_model("flux"),
                                 quant_checkpoint_dir=tmp_path / "q")
    assert fp8.quantized and fp8.quantization_type == "per_tensor_symmetric"
    assert fp8.quantized_checkpoints_path == str(tmp_path / "q")
    assert fp8.quant_targets == list(QuantSpec.for_model("flux").targets)
    assert fp8.activation_quantization_type == "dynamic"  # always W8A8
    with pytest.raises(ValueError, match="quant_checkpoint_dir"):
        backbone_neuron_config(tp_degree=4, world_size=4, dtype=torch.bfloat16,
                               quant=QuantSpec.for_model("flux"))


def test_flux_compiler_args_carry_the_fp8_flag_only_when_quantized():
    from difflet.models.flux.modeling_flux import NeuronFluxBackboneApplication

    def args_for(quantized):
        app = NeuronFluxBackboneApplication.__new__(NeuronFluxBackboneApplication)
        app.config = SimpleNamespace(neuron_config=SimpleNamespace(
            world_size=4, quantized=quantized, quantization_dtype="f8e4m3"))
        app.context_parallel_enabled = False
        return app.get_compiler_args()

    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" not in args_for(False)
    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" in args_for(True)
    assert "--verify-hlo=true" in args_for(True)
