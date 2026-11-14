"""CPU check of Wan's padded sequence-parallel layout (backends/tpu/wan/sp.py,
``DIFFLET_TPU_WAN_SP_PAD=1``).

The collective matmuls are JAX rings that only run on the chips; here they are
replaced by their tp=1 meaning (all-gather = identity, reduce-scatter = the
matmul plus the bias). A 24-token sequence is padded to 32: the first block
pads and selects the rank's rows, attention ignores the pad keys, the last block
cuts the pad. With split-halves RoPE the q/k channels are permuted at install and
rotated as halves. Every combination must match the plain model's output.

The q/k RMSNorm is swapped for a float64 one: the stock one computes in float32,
where the permuted channels' different summation order shows at ~1e-7.
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
    # The shared Wan layers resolve their implementation per backend; the TPU one
    # needs no Neuron SDK.
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    pm = importlib.import_module("difflet.backends.tpu.ops_impl.parallel_mesh")
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=1))
    yield
    pm.destroy_parallel_mesh()


def _fake_rings(monkeypatch, cm):
    def ag(x, linears, head_dims=None):  # [B, S, N]: the heads-first path takes it too
        return [F.linear(x, lin.weight, lin.bias) for lin in linears]

    def rs(x, linear):
        return F.linear(x, linear.weight, linear.bias)

    monkeypatch.setattr(cm, "all_gather_matmul", ag)
    monkeypatch.setattr(cm, "matmul_reduce_scatter", rs)
    monkeypatch.setattr(cm, "prepare", lambda ops=None: None)


def _config(wan, sp_enabled):
    cfg = wan.WanTransformerConfig(patch_size=(1, 2, 2), num_attention_heads=4,
                                   attention_head_dim=16, in_channels=4, out_channels=4,
                                   text_dim=24, freq_dim=32, ffn_dim=48, num_layers=3)
    cfg.sp_enabled = sp_enabled
    return cfg


def _rms_norm_float64(x, weight, eps):
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight if weight is not None else out).to(x.dtype)


@pytest.mark.parametrize("pad", ["1", "0"])
@pytest.mark.parametrize("split_rope", ["1", "0"])
@pytest.mark.parametrize("heads_first", ["1", "0"])
def test_padded_sp_matches_plain_model(monkeypatch, pad, split_rope, heads_first):
    wan = importlib.import_module("difflet.models.wan.modeling_wan")
    sp = importlib.import_module("difflet.backends.tpu.wan.sp")
    _fake_rings(monkeypatch, sp.cm)
    monkeypatch.setenv(sp.PAD_ENV, pad)
    monkeypatch.setenv(sp.SPLIT_ROPE_ENV, split_rope)
    monkeypatch.setenv(sp.HEADS_FIRST_ENV, heads_first)
    monkeypatch.setattr(wan, "rms_norm_fp32", _rms_norm_float64)

    torch.manual_seed(0)
    plain = wan.WanTransformer3DModel(_config(wan, False)).double().eval()
    with torch.no_grad():
        for p in plain.parameters():  # zero-initialised biases / gates would hide bugs
            p.copy_(torch.randn_like(p) * 0.2)
    fused = wan.WanTransformer3DModel(_config(wan, True)).double().eval()
    fused.load_state_dict(plain.state_dict())

    seq_len = 2 * 3 * 4  # frames x (6 / 2) x (8 / 2)
    assert sp.install(fused, seq_len=seq_len)
    assert fused.sp_enabled is (pad == "0")  # padded: the blocks scatter and gather
    if pad == "1":
        assert sp.padded_length(seq_len, 1) == 32

    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(1, 4, 2, 6, 8, generator=g, dtype=torch.float64)
    timestep = torch.tensor([3.0], dtype=torch.float64)
    text = torch.randn(1, 5, 24, generator=g, dtype=torch.float64)
    with torch.no_grad():
        expected = plain(hidden, timestep, text)
        got = fused(hidden, timestep, text)
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)


def test_heads_first_global_rms_norm_matches_token_major(monkeypatch):
    """The tp>1 branch of the heads-first global qk RMSNorm against the shared
    ``_global_rms_norm`` on the token-major view; the cross-rank sum and the weight
    scatter replaced by the same stand-ins in both (a 2-rank group whose other rank
    holds the same squares; rank 1's half of the weight)."""
    wan = importlib.import_module("difflet.models.wan.modeling_wan")
    sp = importlib.import_module("difflet.backends.tpu.wan.sp")
    monkeypatch.setattr(wan, "reduce_from_tensor_model_parallel_region", lambda t: 2 * t)
    monkeypatch.setattr(wan, "scatter_to_process_group_spmd",
                        lambda w, dim, rank, group: w.chunk(2, dim=dim)[rank])
    heads, head_dim = 3, 8
    norm = wan.RMSNorm(2 * heads * head_dim, eps=1e-6)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    attn = types.SimpleNamespace(tp_degree=2, _rank_util=types.SimpleNamespace(get_rank=lambda: 1))
    attn._global_rms_norm = types.MethodType(wan.WanAttention._global_rms_norm, attn)
    x = torch.randn(2, heads, 5, head_dim)
    want = attn._global_rms_norm(norm, x.transpose(1, 2).flatten(2, 3))
    got = sp._global_rms_norm_heads_first(attn, norm, x)
    torch.testing.assert_close(got.transpose(1, 2).flatten(2, 3), want, rtol=1e-6, atol=1e-6)
