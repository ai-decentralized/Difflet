"""CPU check of HunyuanVideo's sequence-parallel single-stream blocks (backends/tpu/hunyuan_video/sp.py).

The collective matmuls are JAX rings that only run on the chips; here they are
replaced by their tp=1 meaning (all-gather = identity, reduce-scatter = the
matmul(s) plus the bias once). A chain of three SP single blocks -- the first
cuts the joint [latent | text] sequence into its chunk, the last gathers it back
-- must reproduce the shared dense blocks, including a padded text mask.
"""

from __future__ import annotations

import importlib
import types

import pytest
import torch
import torch.nn.functional as F

from difflet.pipeline.parallel_mesh import MeshSpec


@pytest.fixture(autouse=True)
def _mesh(monkeypatch):
    # The shared HunyuanVideo layers resolve their implementation per backend; the
    # TPU one needs no Neuron SDK.
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    pm = importlib.import_module("difflet.backends.tpu.ops_impl.parallel_mesh")
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=1))
    yield
    pm.destroy_parallel_mesh()


def _fake_rings(monkeypatch, cm):
    def ag(x, linears, head_dims=None):  # [B, S, N]: the heads-first path takes it too
        return [F.linear(x, lin.weight, lin.bias) for lin in linears]

    def rs2(x0, l0, x1, l1, bias=None):
        out = F.linear(x0, l0.weight) + F.linear(x1, l1.weight)
        return out + bias if bias is not None else out

    monkeypatch.setattr(cm, "all_gather_matmul", ag)
    monkeypatch.setattr(cm, "matmul2_reduce_scatter", rs2)
    monkeypatch.setattr(cm, "prepare", lambda ops=None: None)


def _blocks(n=3, heads=2, head_dim=8):
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoSingleTransformerBlock,
    )

    torch.manual_seed(0)
    blocks = [HunyuanVideoSingleTransformerBlock(heads, head_dim, mlp_ratio=2.0,
                                                 reduce_dtype=torch.float64)
              for _ in range(n)]
    for b in blocks:
        with torch.no_grad():
            for p in b.parameters():  # zero-initialised biases / gates would hide bugs
                p.copy_(torch.randn_like(p) * 0.2)
    return [b.double().eval() for b in blocks]


@pytest.mark.parametrize("heads_first", [False, True])
def test_sp_single_blocks_match_dense(monkeypatch, heads_first):
    sp = importlib.import_module("difflet.backends.tpu.hunyuan_video.sp")
    _fake_rings(monkeypatch, sp.cm)
    heads, head_dim, s_lat, s_txt, batch = 2, 8, 12, 4, 2
    hidden = heads * head_dim
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(batch, s_lat, hidden, generator=g, dtype=torch.float64)
    text = torch.randn(batch, s_txt, hidden, generator=g, dtype=torch.float64)
    temb = torch.randn(batch, hidden, generator=g, dtype=torch.float64)
    mask = torch.ones(batch, 1, 1, s_lat + s_txt, dtype=torch.bool)
    mask[1, ..., -2:] = False  # row 1: the last two text tokens are padding
    cos = torch.randn(s_lat, head_dim, generator=g, dtype=torch.float64)
    sin = torch.randn(s_lat, head_dim, generator=g, dtype=torch.float64)

    dense = _blocks()
    h, e = latent, text
    for b in dense:
        h, e = b(h, e, temb, mask, (cos, sin))

    sp_blocks = _blocks()  # same seed: same weights
    for i, b in enumerate(sp_blocks):
        b._difflet_sp_single = (i == 0, i == len(sp_blocks) - 1, s_txt)
        b.forward = types.MethodType(sp._single_block_forward, b)
    sp_blocks[0].register_buffer("_difflet_sp_rows", torch.arange(s_lat + s_txt), persistent=False)
    if heads_first:  # q/k permuted for split-halves RoPE, attention on [B, H, S, D]
        monkeypatch.setenv(sp.HEADS_FIRST_ENV, "1")
        sp._setup_heads_first([b.attn for b in sp_blocks])
    h2, e2 = latent, text
    for b in sp_blocks:
        h2, e2 = b(h2, e2, temb, mask, (cos, sin))

    # diffusers' RMSNorm takes the mean of squares in float32 even for float64
    # input, and the permutation reorders that sum: float32-level agreement.
    tol = 1e-6 if heads_first else 1e-10
    torch.testing.assert_close(h2, h, rtol=tol, atol=tol)
    torch.testing.assert_close(e2, e, rtol=tol, atol=tol)


@pytest.mark.parametrize("masked", [False, True])
def test_dual_attention_heads_first_matches_shared(monkeypatch, masked):
    """The dual-stream attention, heads first with q/k permuted, against the shared
    ``HunyuanVideoAttention.forward`` on the stock weights (tp=1, float64)."""
    import copy

    from difflet.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoAttention

    sp = importlib.import_module("difflet.backends.tpu.hunyuan_video.sp")
    heads, head_dim, s_lat, s_txt, batch = 2, 8, 12, 4, 2
    hidden = heads * head_dim
    torch.manual_seed(0)
    attn = HunyuanVideoAttention(hidden_size=hidden, num_attention_heads=heads,
                                 attention_head_dim=head_dim, added_kv_proj_dim=hidden,
                                 context_pre_only=False, reduce_dtype=torch.float64)
    with torch.no_grad():
        for p in attn.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    attn = attn.double().eval()
    g = torch.Generator().manual_seed(1)
    latent = torch.randn(batch, s_lat, hidden, generator=g, dtype=torch.float64)
    text = torch.randn(batch, s_txt, hidden, generator=g, dtype=torch.float64)
    mask = None
    if masked:
        mask = torch.ones(batch, 1, 1, s_lat + s_txt, dtype=torch.bool)
        mask[1, ..., -2:] = False
    rope = (torch.randn(s_lat, head_dim, generator=g, dtype=torch.float64),
            torch.randn(s_lat, head_dim, generator=g, dtype=torch.float64))
    want_latent, want_text = attn(hidden_states=latent, encoder_hidden_states=text,
                                  attention_mask=mask, image_rotary_emb=rope)

    ours = copy.deepcopy(attn)
    monkeypatch.setenv(sp.HEADS_FIRST_ENV, "1")
    sp._setup_heads_first([ours])
    qkv = (ours.to_q(latent), ours.to_k(latent), ours.to_v(latent))
    got_latent, got_text = sp._dual_attention(ours, qkv, text, mask, rope)
    # float32-level: diffusers' RMSNorm sums the squares in float32 (permuted order)
    torch.testing.assert_close(ours.to_out[0](got_latent), want_latent, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_text, want_text, rtol=1e-6, atol=1e-6)
