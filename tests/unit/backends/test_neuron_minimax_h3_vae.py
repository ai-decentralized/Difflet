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


def test_tiles_spread_over_ranks_match_one_process(tmp_path):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    from tests.unit.backends._neuron_h3_toy import vae_ranks_worker

    vae = _write(tmp_path)
    r0, r1 = run_ranks(vae_ranks_worker, str(vae), str(tmp_path / "tiles"), world_size=2)
    assert r0["equal"] and r0["shape"][:2] == [1, 3] and r0["leftovers"] == []
    assert r1["video_is_none"]
    assert r0["tiles_decoded_here"] > 0 and r1["tiles_decoded_here"] > 0


@pytest.mark.parametrize("world, cp", [(2, 1), (4, 2)])
def test_tp_sharded_decoder_matches_the_whole_one(tmp_path, world, cp):
    """The decoder head-sharded over TP (the ranks of a TP group decode a tile together,
    the groups split the tiles) gives the whole decoder's video."""
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    from tests.unit.backends._neuron_h3_toy import vae_ranks_worker

    vae = _write(tmp_path)
    results = run_ranks(vae_ranks_worker, str(vae), str(tmp_path / "tiles"), True, cp,
                        world_size=world)
    r0 = results[0]
    assert r0["max_abs"] < 1e-5, r0["max_abs"]
    assert r0["shape"][:2] == [1, 3] and r0["leftovers"] == []
    assert r0["groups"] == cp and r0["local_heads"] == TINY_VAE["decoder_num_attention_heads"] // (world // cp)
    assert r0["report"]["missing"] == [] and r0["report"]["unexpected"] == []
    assert all(r["video_is_none"] for r in results[1:])
    # every rank runs its group's tiles
    assert all(r["tiles_decoded_here"] > 0 for r in results)


@pytest.mark.parametrize("frames,height,width", [(124, 768, 1344), (124, 256, 448), (61, 512, 512),
                                                 (17, 256, 256), (30, 384, 640)])
def test_video_tiles_are_the_decodes_tiles(frames, height, width):
    """``video_tiles`` lists exactly the tiles ``decode_minimax_h3_video`` hands its decoder,
    bit for bit and in its call order, so the ranks can split them without a recording pass."""
    from difflet.backends.neuron.minimax_h3.vae import video_tiles
    from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video

    latent_frames = (frames - 1) // 4 + 1
    gen = torch.Generator().manual_seed(frames + height + width)
    latents = torch.randn(1, 24, latent_frames, height // 16, width // 16, generator=gen)
    kwargs = dict(latents_mean=[0.1 * i for i in range(24)],
                  latents_std=[1.0 + 0.05 * i for i in range(24)], clip_length=17, token_drop=3)
    seen = []

    def record(tile):
        seen.append(tile.clone())
        b, _, t, h, w = tile.shape
        return torch.zeros(b, 3, t * 4, h * 16, w * 16)

    decode_minimax_h3_video(record, latents, **kwargs)
    tiles = video_tiles(latents, **kwargs)
    assert len(tiles) == len(seen) > 0
    for listed, decoded in zip(tiles, seen):
        assert listed.shape == decoded.shape
        assert torch.equal(listed, decoded)
