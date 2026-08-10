from __future__ import annotations

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.registry import resolve_model


def test_minimax_h3_registry_contract():
    entry = resolve_model("MiniMaxAI/MiniMax-H3")

    assert entry.name == "minimax_h3"
    assert entry.default_parallel == DiffletParallelConfig(tp_degree=4)
    assert entry.default_shape == {"height": 768, "width": 1344, "num_frames": 124}
    patterns = entry.download_patterns or ()
    assert "transformer/diffusion_pytorch_model*.safetensors" in patterns
    assert "text_encoder/model*.safetensors" in patterns
    assert "vae/diffusion_pytorch_model*.safetensors" in patterns
    assert "audio_vae/diffusion_pytorch_model*.safetensors" in patterns
    assert not any(pattern.startswith("transformer_ref/") for pattern in patterns)
    assert not any(pattern.startswith("FL2VA/") for pattern in patterns)


def test_minimax_h3_detector_accepts_normalized_name():
    assert resolve_model("some-org/minimax_h3-experiment").name == "minimax_h3"
