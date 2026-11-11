"""L1 (CPU): the H3 video VAE decoder as a neuron application, against the reference decoder.

A tiny decoder (2 blocks, 2 heads x 16) with random weights, saved in the checkpoint layout
(encoder keys included, to be skipped). The application pads the tile's 1797 tokens to 1920 and
attends through ``difflet.ops.attention`` with key bounds; the reference is the unpadded host
decoder with diffusers' attention. One rank, CPU, fp32.
"""

import pytest

torch = pytest.importorskip("torch")

import json  # noqa: E402

from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_h3_toy import vae_worker  # noqa: E402

TINY_VAE = {
    "out_channels": 3, "latent_channels": 24,
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
    "decoder_num_layers": 2, "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 16, "decoder_num_register_tokens": 4, "decoder_ffn_mult": 2,
    "decoder_rope_theta": 100.0, "decoder_rope_dim_ratio": 0.75, "decoder_norm_eps": 1e-5,
    "clip_length": 17, "token_drop": 3, "latents_mean": [0.0] * 24, "latents_std": [1.0] * 24,
}


def _write(root):
    from safetensors.torch import save_file

    from difflet.models.minimax_h3.host_components import _HostVideoDecoder

    torch.manual_seed(0)
    module = _HostVideoDecoder(TINY_VAE)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0.0, 0.2)
    vae = root / "vae"
    vae.mkdir()
    state = {k: v.contiguous() for k, v in module.state_dict().items()}
    state["encoder.dummy.weight"] = torch.zeros(2)
    state["quant_conv.weight"] = torch.zeros(2)
    save_file(state, str(vae / "diffusion_pytorch_model.safetensors"))
    (vae / "config.json").write_text(json.dumps(TINY_VAE))
    return vae


@pytest.mark.parametrize("exec_mode", ["eager", "compile"])
def test_device_vae_matches_reference_decoder(tmp_path, exec_mode):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    vae = _write(tmp_path)
    (r,) = run_ranks(vae_worker, str(vae), exec_mode, world_size=1)
    assert r["shape"] == [1, 3, 28, 256, 256] and r["dtype"] == "torch.float32"
    assert r["max_abs"] < 1e-4, r["max_abs"]
    assert r["report"]["missing"] == [] and r["report"]["unexpected"] == []
    assert sorted(r["report"]["skipped"]) == ["encoder.dummy.weight", "quant_conv.weight"]
    assert r["norm_dtype"] == "torch.float32"
