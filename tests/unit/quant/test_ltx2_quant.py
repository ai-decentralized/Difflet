"""FP8 PTQ wiring for LTX-2 (single-transformer mode): names, config, compiler args, segmented rejection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from difflet.quant import checkpoint as ckpt
from difflet.quant.spec import QuantSpec


def _ltx2_like_hf_state_dict() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    return {
        "transformer_blocks.0.attn1.to_q.weight": r(16, 16),
        "transformer_blocks.0.attn2.to_out.0.weight": r(16, 16),
        "transformer_blocks.0.audio_attn1.to_v.weight": r(8, 8),
        "transformer_blocks.0.audio_to_video_attn.to_k.weight": r(16, 8),
        "transformer_blocks.0.video_to_audio_attn.to_out.0.weight": r(8, 8),
        "transformer_blocks.0.ff.net.0.proj.weight": r(32, 16),
        "transformer_blocks.0.audio_ff.net.2.weight": r(8, 16),
        "transformer_blocks.0.scale_shift_table": r(6, 16),
        "proj_out.weight": r(64, 16),
        "caption_projection.linear_1.weight": r(16, 32),
        "adaln_single.linear.weight": r(96, 16),
    }


def test_quantized_hf_checkpoint_maps_onto_difflet_ltx2_names():
    from difflet.backends.trainium.ltx_2.transformer import NeuronLTX2TransformerApplication

    quantized, report = ckpt.quantize_state_dict(_ltx2_like_hf_state_dict(), QuantSpec.for_model("ltx_2"))
    assert report["num_quantized"] == 7
    renamed = {k.replace(".weight_scale", ".scale"): v for k, v in quantized.items()}
    config = SimpleNamespace(neuron_config=SimpleNamespace(world_size=1, tp_degree=1))
    out = NeuronLTX2TransformerApplication.convert_hf_to_neuron_state_dict(renamed, config)
    assert out["transformer.transformer_blocks.0.audio_to_video_attn.to_k.weight"].dtype == torch.float8_e4m3fn
    assert out["transformer.transformer_blocks.0.audio_to_video_attn.to_k.scale"].dtype == torch.float32
    assert out["transformer.transformer_blocks.0.audio_ff.net.2.scale"].shape == (1,)
    assert not any(k.startswith("transformer_blocks.") for k in out)
    for stays in ("transformer.proj_out.weight", "transformer.caption_projection.linear_1.weight",
                  "transformer.adaln_single.linear.weight"):
        assert out[stays].dtype == torch.bfloat16
        assert stays.replace(".weight", ".scale") not in out


def test_ltx2_compiler_args_carry_the_fp8_flag_only_when_quantized():
    from difflet.backends.trainium.ltx_2.transformer import NeuronLTX2TransformerApplication

    def args_for(quantized):
        app = NeuronLTX2TransformerApplication.__new__(NeuronLTX2TransformerApplication)
        app.config = SimpleNamespace(neuron_config=SimpleNamespace(
            world_size=4, quantized=quantized, quantization_dtype="f8e4m3"))
        return app.get_compiler_args()

    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" not in args_for(False)
    assert "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" in args_for(True)


def test_ltx2_transformer_neuron_config_carries_the_quant_fields(tmp_path):
    pytest.importorskip("neuronx_distributed")
    from difflet.models.ltx_2.application import transformer_neuron_config

    plain = transformer_neuron_config(tp_degree=4, world_size=4, dtype=torch.bfloat16, batch_size=1)
    assert not getattr(plain, "quantized", False)
    fp8 = transformer_neuron_config(tp_degree=4, world_size=4, dtype=torch.bfloat16, batch_size=1,
                                    quant=QuantSpec.for_model("ltx_2"),
                                    quant_checkpoint_dir=tmp_path / "q")
    assert fp8.quantized and fp8.quantized_checkpoints_path == str(tmp_path / "q")
    assert fp8.quant_targets == list(QuantSpec.for_model("ltx_2").targets)
    with pytest.raises(ValueError, match="quant_checkpoint_dir"):
        transformer_neuron_config(tp_degree=4, world_size=4, dtype=torch.bfloat16, batch_size=1,
                                  quant=QuantSpec.for_model("ltx_2"))


def test_ltx2_segmented_mode_rejects_quant():
    """Review Focus 4: the segmented runtime bypasses the quantized checkpoint
    (plain nn.Linear, its own loader), so --quant must fail before any compile."""
    from difflet.models.ltx_2.application import reject_quant_for_segmented

    reject_quant_for_segmented("single", QuantSpec.for_model("ltx_2"))
    reject_quant_for_segmented("segmented", None)
    with pytest.raises(ValueError, match="segmented.*--quant"):
        reject_quant_for_segmented("segmented", QuantSpec.for_model("ltx_2"))
