"""CPU parity of the Flux TP recipe: at tp=1 the sharded module (TP attention
processor, split proj_out, parallel linears) must reproduce diffusers' own
forward exactly; the checkpoint-key windows must tile proj_out."""

from __future__ import annotations

import pytest
import torch

from difflet.backends.tpu.core.checkpoint import CheckpointSlice
from difflet.backends.tpu.flux.config import FLUX_1_DEV_CONFIG, TpuFluxConfig
from difflet.backends.tpu.flux.transformer import make_checkpoint_key
from difflet.backends.tpu.ops_impl import parallel_mesh as pm
from difflet.models.flux.tp_sharding import build_flux_transformer, shard_flux_transformer
from difflet.pipeline.parallel_mesh import MeshSpec

_TINY = {
    "attention_head_dim": 8, "guidance_embeds": True, "in_channels": 8, "joint_attention_dim": 16,
    "num_attention_heads": 2, "num_layers": 1, "num_single_layers": 1, "patch_size": 1,
    "pooled_projection_dim": 8, "axes_dims_rope": [2, 2, 4], "out_channels": None,
}


@pytest.fixture(autouse=True)
def _mesh():
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=1))
    yield
    pm.destroy_parallel_mesh()


def _inputs(cfg):
    torch.manual_seed(0)
    h, w = 4, 6  # latent grid (2x2-packed) -> 6 image tokens
    hidden = torch.randn(1, (h // 2) * (w // 2), cfg.in_channels)
    text = torch.randn(1, 5, cfg.joint_attention_dim)
    pooled = torch.randn(1, cfg.pooled_projection_dim)
    t = torch.tensor([0.5])
    img_ids = torch.zeros((h // 2) * (w // 2), 3)
    img_ids[:, 1] = torch.arange(h // 2).repeat_interleave(w // 2)
    img_ids[:, 2] = torch.arange(w // 2).repeat(h // 2)
    txt_ids = torch.zeros(5, 3)
    guidance = torch.tensor([3.5])
    return hidden, text, pooled, t, img_ids, txt_ids, guidance


def test_sharded_forward_matches_diffusers_at_tp1():
    cfg = TpuFluxConfig(_TINY, height=32, width=48, tp_degree=1, torch_dtype=torch.float32)
    ref = build_flux_transformer(cfg).eval()
    sharded = build_flux_transformer(cfg).eval()
    sharded.load_state_dict(ref.state_dict())
    # Shard at "tp=2 on a tp=1 mesh" is meaningless; shard at tp=1 is a no-op.
    # Force the module surgery anyway by sharding at 1 with the processor/split
    # applied, which is what the TPU app does at tp>1: emulate with tp=1 pieces.
    from difflet.models.flux import tp_sharding as S

    # Apply the surgery manually at degree 1 so the code paths run.
    for block in sharded.single_transformer_blocks:
        dim = block.proj_out.out_features
        w, b = block.proj_out.weight.data.clone(), block.proj_out.bias.data.clone()
        block.proj_out_attn = S.RowParallelLinear(dim, dim, bias=True, input_is_parallel=True,
                                                  reduce_output=False, skip_bias_add=True)
        block.proj_out_mlp = S.RowParallelLinear(int(block.mlp_hidden_dim), dim, bias=False,
                                                 input_is_parallel=True, reduce_output=False)
        block.proj_out_attn.weight.data.copy_(w[:, :dim])
        block.proj_out_attn.bias.data.copy_(b)
        block.proj_out_mlp.weight.data.copy_(w[:, dim:])
        del block.proj_out
        block.forward = S._single_block_forward.__get__(block, type(block))
        block.attn.processor = S.FluxTPAttnProcessor()
    for block in sharded.transformer_blocks:
        block.attn.processor = S.FluxTPAttnProcessor()

    inputs = _inputs(cfg)
    with torch.no_grad():
        expected = ref(*inputs, return_dict=False)[0]
        actual = sharded(*inputs, return_dict=False)[0]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_shard_flux_transformer_rewires_every_block():
    cfg = TpuFluxConfig(_TINY, height=32, width=48, tp_degree=2, torch_dtype=torch.float32)
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=2))
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        m = build_flux_transformer(cfg)
        shard_flux_transformer(m, 2)
    joint = m.transformer_blocks[0]
    single = m.single_transformer_blocks[0]
    assert joint.attn.heads == 1 and single.attn.heads == 1
    assert joint.attn.to_q.weight.shape == (8, 16)          # 2 heads x 8 / tp 2 = 8 rows
    assert joint.attn.to_out[0].weight.shape == (16, 8)     # row-parallel input shard
    assert single.proj_out_attn.weight.shape == (16, 8)
    assert single.proj_out_mlp.weight.shape == (16, 32)     # mlp 64 / tp 2
    assert not hasattr(single, "proj_out")


def test_checkpoint_key_windows_the_single_proj_out():
    key = make_checkpoint_key(3072)
    assert key("single_transformer_blocks.3.proj_out_attn.weight") == CheckpointSlice(
        "single_transformer_blocks.3.proj_out.weight", dim=1, start=0, stop=3072)
    assert key("single_transformer_blocks.3.proj_out_mlp.weight") == CheckpointSlice(
        "single_transformer_blocks.3.proj_out.weight", dim=1, start=3072, stop=None)
    assert key("single_transformer_blocks.3.proj_out_attn.bias") == "single_transformer_blocks.3.proj_out.bias"
    assert key("transformer_blocks.0.attn.to_q.weight") == "transformer_blocks.0.attn.to_q.weight"


def test_flux_1_dev_geometry():
    cfg = TpuFluxConfig(FLUX_1_DEV_CONFIG, height=1024, width=1024, tp_degree=4)
    cfg.validate()
    assert cfg.inner_dim == 3072 and cfg.image_seq_len == 4096
    with pytest.raises(ValueError, match="divisible by 16"):
        TpuFluxConfig(FLUX_1_DEV_CONFIG, height=1000, tp_degree=4).validate()
