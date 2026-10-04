"""FP8 PTQ wiring for HunyuanVideo 1.0: HF checkpoint -> Difflet names, config, compiler args."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from difflet.quant import checkpoint as ckpt
from difflet.quant.spec import QuantSpec

_TINY_HV_CONFIG = {
    "_class_name": "HunyuanVideoTransformer3DModel",
    "attention_head_dim": 8,
    "guidance_embeds": True,
    "in_channels": 16,
    "mlp_ratio": 4.0,
    "num_attention_heads": 2,
    "num_layers": 1,
    "num_refiner_layers": 1,
    "num_single_layers": 1,
    "out_channels": 16,
    "patch_size": 2,
    "patch_size_t": 1,
    "pooled_projection_dim": 8,
    "qk_norm": "rms_norm",
    "rope_axes_dim": [2, 3, 3],
    "rope_theta": 256.0,
    "text_embed_dim": 32,
}


def _hv_like_hf_state_dict() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    return {
        "transformer_blocks.0.attn.to_q.weight": r(16, 16),
        "transformer_blocks.0.attn.add_q_proj.weight": r(16, 16),
        "transformer_blocks.0.ff.net.0.proj.weight": r(64, 16),
        "transformer_blocks.0.ff_context.net.2.weight": r(16, 64),
        "transformer_blocks.0.norm1.linear.weight": r(96, 16),
        "single_transformer_blocks.0.attn.to_q.weight": r(16, 16),
        "single_transformer_blocks.0.proj_mlp.weight": r(64, 16),
        "single_transformer_blocks.0.proj_out.weight": r(16, 80),
        "single_transformer_blocks.0.proj_out.bias": r(16),
        "single_transformer_blocks.0.norm.linear.weight": r(48, 16),
        # Token refiner (context_embedder) linears share the attn.to_q spelling
        # but are NOT in the target set (anchored globs): they stay bf16.
        "context_embedder.token_refiner.refiner_blocks.0.attn.to_q.weight": r(16, 16),
        "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj.weight": r(64, 16),
        "context_embedder.proj_in.weight": r(16, 32),
        "x_embedder.proj.weight": r(16, 16),
        "proj_out.weight": r(64, 16),
        "norm_out.linear.weight": r(32, 16),
    }


def test_quantized_hf_checkpoint_maps_onto_difflet_hunyuan_names():
    """Offline quantizer (HF names) + the HunyuanVideo converter: fp8 targets with
    scales, the fused single-block proj_out split (attn half keeps the bias, scale
    copied to both halves), token refiner / embedders / modulation / root proj_out bf16."""
    from difflet.backends.trainium.hunyuan_video.backbone import NeuronHunyuanVideoBackboneApplication

    quantized, report = ckpt.quantize_state_dict(_hv_like_hf_state_dict(), QuantSpec.for_model("hunyuan_video"))
    assert report["num_quantized"] == 6  # add_q_proj stays bf16 (tp4 NaN, see difflet.quant.targets)
    assert quantized["transformer_blocks.0.attn.add_q_proj.weight"].dtype == torch.bfloat16
    renamed = {k.replace(".weight_scale", ".scale"): v for k, v in quantized.items()}  # get_state_dict
    config = SimpleNamespace(
        num_attention_heads=1, attention_head_dim=16, num_single_layers=1,
        neuron_config=SimpleNamespace(world_size=1),
    )
    out = NeuronHunyuanVideoBackboneApplication.convert_hf_to_neuron_state_dict(renamed, config)
    attn = out["single_transformer_blocks.0.proj_out_attn.weight"]
    mlp = out["single_transformer_blocks.0.proj_out_mlp.weight"]
    assert attn.dtype == torch.float8_e4m3fn and attn.shape == (16, 16)
    assert mlp.dtype == torch.float8_e4m3fn and mlp.shape == (16, 64)
    assert torch.equal(out["single_transformer_blocks.0.proj_out_attn.scale"],
                       out["single_transformer_blocks.0.proj_out_mlp.scale"])
    assert "single_transformer_blocks.0.proj_out_attn.bias" in out
    assert "single_transformer_blocks.0.proj_out_mlp.bias" not in out
    assert "single_transformer_blocks.0.proj_out.weight" not in out
    assert out["transformer_blocks.0.attn.to_q.weight"].dtype == torch.float8_e4m3fn
    assert out["transformer_blocks.0.attn.to_q.scale"].dtype == torch.float32
    for stays in ("proj_out.weight", "x_embedder.proj.weight", "norm_out.linear.weight",
                  "context_embedder.proj_in.weight",
                  "context_embedder.token_refiner.refiner_blocks.0.attn.to_q.weight",
                  "context_embedder.token_refiner.refiner_blocks.0.ff.net.0.proj.weight",
                  "transformer_blocks.0.norm1.linear.weight", "single_transformer_blocks.0.norm.linear.weight"):
        assert out[stays].dtype == torch.bfloat16, stays
        assert stays.replace(".weight", ".scale") not in out


def test_bf16_checkpoint_takes_the_same_converter_path():
    from difflet.backends.trainium.hunyuan_video.backbone import NeuronHunyuanVideoBackboneApplication

    config = SimpleNamespace(
        num_attention_heads=1, attention_head_dim=16, num_single_layers=1,
        neuron_config=SimpleNamespace(world_size=1),
    )
    out = NeuronHunyuanVideoBackboneApplication.convert_hf_to_neuron_state_dict(_hv_like_hf_state_dict(), config)
    assert out["single_transformer_blocks.0.proj_out_attn.weight"].shape == (16, 16)
    assert out["single_transformer_blocks.0.proj_out_mlp.weight"].shape == (16, 64)
    assert "single_transformer_blocks.0.proj_out_attn.bias" in out
    assert not any(k.endswith(".scale") for k in out)


def test_hunyuan_backbone_config_carries_the_quant_fields(tmp_path):
    pytest.importorskip("neuronx_distributed")
    from difflet.models.hunyuan_video import application as happ

    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(_TINY_HV_CONFIG))
    common = dict(model_path=str(tmp_path), world_size=4, tp_degree=4, dtype=torch.bfloat16,
                  height=64, width=64, num_frames=5, text_seq_len=8)
    plain = happ.create_hunyuan_video_backbone_config(**common)
    assert not getattr(plain.neuron_config, "quantized", False)
    fp8 = happ.create_hunyuan_video_backbone_config(
        **common, quant=QuantSpec.for_model("hunyuan_video"), quant_checkpoint_dir=tmp_path / "q")
    nc = fp8.neuron_config
    assert nc.quantized and nc.quantized_checkpoints_path == str(tmp_path / "q")
    assert nc.quant_targets == list(QuantSpec.for_model("hunyuan_video").targets)
    with pytest.raises(ValueError, match="quant_checkpoint_dir"):
        happ.create_hunyuan_video_backbone_config(**common, quant=QuantSpec.for_model("hunyuan_video"))


def test_hunyuan_compiler_args_carry_the_fp8_flag_only_when_quantized():
    from difflet.backends.trainium.hunyuan_video.backbone import NeuronHunyuanVideoBackboneApplication

    def args_for(quantized):
        app = NeuronHunyuanVideoBackboneApplication.__new__(NeuronHunyuanVideoBackboneApplication)
        app.config = SimpleNamespace(neuron_config=SimpleNamespace(
            world_size=4, quantized=quantized, quantization_dtype="f8e4m3"))
        return app.get_compiler_args()

    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" not in args_for(False)
    assert "--verify-hlo=true" in args_for(False)
    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" in args_for(True)


def test_hunyuan_application_rejects_the_probe_with_quant(tmp_path):
    """The TeaCache probe traces its own backbone copy without the quant hook, so
    adaptive TeaCache + --quant fails at construction (before any compile)."""
    from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    common = dict(model_path=str(tmp_path), parallel=DiffletParallelConfig(tp_degree=4), dtype=torch.bfloat16,
                  shape={"height": 320, "width": 512, "num_frames": 61}, enable_transformer=False,
                  quant=QuantSpec.for_model("hunyuan_video").to_dict(), quant_cache_dir=str(tmp_path))
    app = NeuronHunyuanVideoApplication(**common, enable_teacache_probe=False)
    assert app.quant_spec == QuantSpec.for_model("hunyuan_video") and app.quant_tag == "hunyuan_video"
    with pytest.raises(NotImplementedError, match="adaptive TeaCache"):
        NeuronHunyuanVideoApplication(**common, teacache_speedup=1.5)
    with pytest.raises(NotImplementedError, match="adaptive TeaCache"):
        NeuronHunyuanVideoApplication(**common, teacache_fused=True)
    # bf16 keeps the Python-API default (probe enabled) untouched.
    assert NeuronHunyuanVideoApplication(
        model_path=str(tmp_path), parallel=DiffletParallelConfig(tp_degree=4), dtype=torch.bfloat16,
        shape={"height": 320, "width": 512, "num_frames": 61}, enable_transformer=False).quant_spec is None
