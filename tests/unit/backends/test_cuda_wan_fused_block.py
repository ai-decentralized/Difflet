"""Wan's fused eager Triton kernels against the modeling's unfused ops.

Run with ``DIFFLET_BACKEND=cuda`` (the modeling's ``difflet.ops`` bind at import) on a CUDA
host, or on any host with Triton under ``TRITON_INTERPRET=1``: the kernels then run in Triton's
interpreter on CPU tensors, which checks the block against the modeling block and the swap. The
per-kernel tests bound every value to one bf16 rounding step and need a CUDA device: the
interpreter casts fp32 to bf16 by truncation, not round-to-nearest. Hosts without Triton (CI's
CPU stacks install torch's CPU wheel) skip the file.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET else "cuda"

pytestmark = pytest.mark.skipif(
    os.environ.get("DIFFLET_BACKEND") != "cuda"
    or importlib.util.find_spec("triton") is None
    or not (INTERPRET or torch.cuda.is_available()),
    reason="needs DIFFLET_BACKEND=cuda, Triton, and a CUDA device or TRITON_INTERPRET=1",
)

# The interpreter truncates fp32 to bf16 (about half of all values land one step toward zero),
# so the one-step bounds hold only where the cast rounds to nearest.
hardware_rounding = pytest.mark.skipif(
    INTERPRET, reason="Triton's interpreter casts fp32 to bf16 by truncation"
)

BF16 = torch.bfloat16
DIM, HEAD_DIM = 5120, 128


def _ulp_close(got, want, *, share=1e-3):
    """Equal up to one-bf16-rounding differences in a small share of values."""
    assert got.dtype == want.dtype == BF16 and got.shape == want.shape
    diff = (got.float() - want.float()).abs()
    assert bool((diff <= want.float().abs().clamp_min(1e-2) * 2**-7).all())
    assert (diff > 0).float().mean().item() <= share


@pytest.fixture
def fused():
    from difflet.backends.cuda.ops_impl import parallel_mesh as pm
    from difflet.backends.cuda.wan import fused_block
    from difflet.pipeline.parallel_mesh import MeshSpec

    # The modeling's parallel linears read the process mesh (tp=1 here).
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(cfg=1, tp=1))
    torch.manual_seed(0)
    yield fused_block
    pm.destroy_parallel_mesh()


def _modulation(batch=1):
    """Six (B, 1, DIM) fp32 chunks, as the block slices them from its table plus temb."""
    table = torch.randn(1, 6, DIM, device=DEVICE) / DIM**0.5
    temb = 0.1 * torch.randn(batch, 6, DIM, device=DEVICE, dtype=BF16)
    return (table + temb.float()).chunk(6, dim=1)


@hardware_rounding
@pytest.mark.parametrize("batch", [1, 2])
def test_layer_norm_modulates_like_the_block(fused, batch):
    x = torch.randn(batch, 300, DIM, device=DEVICE, dtype=BF16)
    shift, scale = _modulation(batch)[:2]
    want = (F.layer_norm(x.float(), (DIM,), eps=1e-6) * (1 + scale) + shift).type_as(x)
    _ulp_close(fused.layer_norm(x, 1e-6, shift=shift, scale=scale), want)


@hardware_rounding
def test_affine_layer_norm_matches_fp32_layer_norm(fused):
    x = torch.randn(1, 300, DIM, device=DEVICE, dtype=BF16)
    weight = (1 + 0.1 * torch.randn(DIM, device=DEVICE)).to(BF16)
    bias = (0.1 * torch.randn(DIM, device=DEVICE)).to(BF16)
    want = F.layer_norm(x.float(), (DIM,), weight.float(), bias.float(), 1e-6).type_as(x)
    _ulp_close(fused.layer_norm(x, 1e-6, weight=weight, bias=bias), want)


@hardware_rounding
def test_gated_residual_matches_the_block(fused):
    residual, update = (torch.randn(1, 300, DIM, device=DEVICE, dtype=BF16) for _ in range(2))
    gate = _modulation()[2]
    want = (residual.float() + update * gate).type_as(residual)
    _ulp_close(fused.gated_residual(residual, update, gate), want)


@hardware_rounding
def test_rms_norm_rotary_matches_the_norm_then_the_rotation(fused):
    from difflet.backends._torch_megatron.embeddings import apply_rotary_emb
    from difflet.models.wan.modeling_wan import WanRotaryPosEmbed

    rope = WanRotaryPosEmbed(HEAD_DIM, (1, 2, 2), 1024).to(DEVICE)
    cos, sin = rope(torch.zeros(1, 16, 3, 20, 26, device=DEVICE))  # 3 x 10 x 13 tokens
    tokens = cos.shape[1]
    # A strided q, as a projection of a wider fused output would give.
    wide = torch.randn(1, tokens, 2 * DIM, device=DEVICE, dtype=BF16)
    x = wide[..., :DIM]
    weight = (1 + 0.1 * torch.randn(DIM, device=DEVICE)).to(BF16)
    normed = F.rms_norm(x, (DIM,), weight=weight, eps=1e-6)
    heads = normed.unflatten(-1, (DIM // HEAD_DIM, HEAD_DIM))
    want = apply_rotary_emb(heads, cos, sin).flatten(-2)
    got = fused.rms_norm_rotary(x, weight, 1e-6, cos, sin, HEAD_DIM)
    # The RMS reduces in another order, so a normalized value can round one bf16 step apart;
    # the rotation then carries that step at the size of the pair it mixes, which for a small
    # output (two products that nearly cancel) is several of the output's own steps.
    pair = normed.float().abs() + normed.float().unflatten(-1, (-1, 2)).flip(-1).flatten(-2).abs()
    diff = (got.float() - want.float()).abs()
    assert got.shape == want.shape and bool((diff <= pair * 2**-7).all())
    assert (diff > 0).float().mean().item() <= 1e-3


def _block(dim=256, heads=2):
    from difflet.models.wan.modeling_wan import WanRotaryPosEmbed, WanTransformerBlock

    torch.manual_seed(0)
    block = WanTransformerBlock(dim, 2 * dim, heads, cross_attn_norm=True, eps=1e-6)
    block = block.to(DEVICE, BF16).eval().requires_grad_(False)
    block.scale_shift_table.data = block.scale_shift_table.data.float()
    head_dim = dim // heads
    rope = WanRotaryPosEmbed(head_dim, (1, 2, 2), 1024).to(DEVICE)
    rotary = rope(torch.zeros(1, 16, 3, 8, 12, device=DEVICE))  # 3 x 4 x 6 tokens
    hidden = torch.randn(1, rotary[0].shape[1], dim, device=DEVICE, dtype=BF16)
    text = torch.randn(1, 7, dim, device=DEVICE, dtype=BF16)
    temb = 0.1 * torch.randn(1, 6, dim, device=DEVICE, dtype=BF16)
    return block, (hidden, text, temb, rotary)


def test_the_fused_block_matches_the_modeling_block(fused):
    block, inputs = _block()
    with torch.no_grad():
        want = block(*inputs)
        got = fused.fused_block_forward(block, *inputs)
    cosine = F.cosine_similarity(got.float().flatten(), want.float().flatten(), dim=0)
    assert cosine.item() > 0.99999
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


def test_fuse_blocks_swaps_every_block_and_the_fused_path_runs(fused, monkeypatch):
    from types import SimpleNamespace

    block, inputs = _block()
    model = SimpleNamespace(blocks=[block])
    assert fused.fuse_blocks(model) == 1
    calls = []
    real = fused.layer_norm
    monkeypatch.setattr(fused, "layer_norm", lambda *a, **k: calls.append(1) or real(*a, **k))
    with torch.no_grad():
        block(*inputs)
    assert len(calls) == 3  # norm1, norm2, norm3


@pytest.mark.skipif(INTERPRET, reason="loads and compiles an expert on a CUDA device")
def test_the_cuda_expert_fuses_in_eager_only(tmp_path):
    from diffusers import WanTransformer3DModel

    from difflet.backends.cuda.wan.fused_block import fused_block_forward
    from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication
    from difflet.models.wan.dit_config import WanDiTConfig

    torch.manual_seed(0)
    WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=128, in_channels=4,
        out_channels=4, text_dim=16, freq_dim=32, ffn_dim=512, num_layers=2,
        cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6,
    ).save_pretrained(str(tmp_path / "transformer"))  # fmt: skip
    cfg = WanDiTConfig.from_pretrained(
        tmp_path / "transformer", height=64, width=96, num_frames=9, text_seq_len=6
    )
    apps = {}
    for compiled in (False, True):
        app = CudaWanTransformerApplication(
            model_path=str(tmp_path / "transformer"), config=cfg, device=DEVICE,
            torch_compile=compiled,
        )  # fmt: skip
        app.compile(str(tmp_path / f"cache{compiled}"))
        app.load(str(tmp_path / f"cache{compiled}"))
        apps[compiled] = app
    fused_forward = [getattr(b.forward, "__func__", None) for b in apps[False].module.blocks]
    assert fused_forward == [fused_block_forward] * 2
    # Compiled blocks are torch.compile wrappers around the modeling's own block.
    assert all("forward" not in vars(b._orig_mod) for b in apps[True].module.blocks)
    x = torch.randn(1, 4, 3, 8, 12, device=DEVICE, dtype=BF16)
    t = torch.tensor([500.0], device=DEVICE)
    text = torch.randn(1, 6, 16, device=DEVICE, dtype=BF16)
    with torch.no_grad():
        eager, compiled = apps[False](x, t, text), apps[True](x, t, text)
    cosine = F.cosine_similarity(eager.float().flatten(), compiled.float().flatten(), dim=0)
    assert cosine.item() > 0.9999
