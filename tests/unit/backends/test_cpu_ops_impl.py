"""Direct unit tests for difflet.backends.cpu.ops_impl.* implementations."""

import pytest
import torch

from difflet.backends.cpu.ops_impl import (
    attention as cpu_attn,
)
from difflet.backends.cpu.ops_impl import (
    collectives as cpu_col,
)
from difflet.backends.cpu.ops_impl import (
    embeddings as cpu_emb,
)
from difflet.backends.cpu.ops_impl import (
    linear as cpu_lin,
)
from difflet.backends.cpu.ops_impl import (
    mx as cpu_mx,
)
from difflet.backends.cpu.ops_impl import (
    norm as cpu_norm,
)
from difflet.backends.cpu.ops_impl import (
    platform as cpu_plat,
)


# ---------------- linear ----------------


def test_column_parallel_linear_forward():
    torch.manual_seed(0)
    layer = cpu_lin.ColumnParallelLinear(4, 6, gather_output=False, reduce_dtype=torch.float32)
    out = layer(torch.randn(2, 4))
    assert out.shape == (2, 6)


def test_row_parallel_linear_forward():
    torch.manual_seed(0)
    layer = cpu_lin.RowParallelLinear(4, 6, input_is_parallel=True)
    out = layer(torch.randn(2, 4))
    assert out.shape == (2, 6)


def test_parallel_embedding_forward():
    torch.manual_seed(0)
    layer = cpu_lin.ParallelEmbedding(10, 4, shard_across_embedding=True, pad=True)
    out = layer(torch.tensor([0, 5, 9]))
    assert out.shape == (3, 4)


# ---------------- norm ----------------


def test_layer_norm_alias_is_nn_layernorm():
    assert cpu_norm.LayerNorm is torch.nn.LayerNorm


def test_rmsnorm_forward_matches_reference():
    torch.manual_seed(0)
    norm = cpu_norm.RMSNorm(4, eps=1e-6)
    x = torch.randn(2, 3, 4)
    out = norm(x)
    var = x.float().pow(2).mean(-1, keepdim=True)
    expected = x * torch.rsqrt(var + 1e-6)
    assert torch.allclose(out, expected, atol=1e-5)


def test_custom_rmsnorm_is_rmsnorm():
    assert cpu_norm.CustomRMSNorm is cpu_norm.RMSNorm


# ---------------- attention ----------------


def test_attention_matches_reference():
    torch.manual_seed(0)
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    out = cpu_attn.attention(q, k, v, scale=0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * 0.5
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    expected = torch.matmul(probs, v)
    assert torch.allclose(out, expected, atol=1e-6)


def test_attention_default_scale_is_one():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4)
    out = cpu_attn.attention(q, q, q)
    scores = torch.matmul(q, q.transpose(-1, -2))
    expected = torch.matmul(torch.softmax(scores.float(), -1).to(q.dtype), q)
    assert torch.allclose(out, expected, atol=1e-6)


def test_attention_causal_masks_future():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    out = cpu_attn.attention(q, q, q, scale=1.0, causal=True)
    # first query position attends only to itself -> equals v[0]
    assert torch.allclose(out[:, 0], q[:, 0], atol=1e-5)


def test_attention_bool_mask():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4)
    mask = torch.tensor([[True, False], [True, True]])
    out = cpu_attn.attention(q, q, q, scale=1.0, attention_mask=mask)
    assert out.shape == q.shape


def test_attention_additive_mask():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4)
    mask = torch.zeros(2, 2)
    mask[0, 1] = float("-inf")
    out = cpu_attn.attention(q, q, q, scale=1.0, attention_mask=mask)
    assert out.shape == q.shape


def test_cross_attention():
    torch.manual_seed(0)
    q = torch.randn(1, 3, 4)
    kv = torch.randn(1, 5, 4)
    out = cpu_attn.cross_attention(q, kv, kv, scale=0.5)
    assert out.shape == (1, 3, 4)


def test_ring_attention_equals_plain_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 3, 4, 8)
    v = torch.randn(2, 3, 4, 8)
    out = cpu_attn.ring_attention(q, k, v, scale=0.3)
    # reference: plain attention per (b,h)
    b, h, s, d = q.shape
    ref = cpu_attn.attention(
        q.reshape(b * h, s, d), k.reshape(b * h, s, d), v.reshape(b * h, s, d), scale=0.3
    ).reshape(b, h, s, d)
    assert torch.allclose(out, ref, atol=1e-6)


def test_joint_ring_attention_equals_full_joint():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 8)
    ik = torch.randn(1, 2, 4, 8)
    iv = torch.randn(1, 2, 4, 8)
    tk = torch.randn(1, 2, 2, 8)
    tv = torch.randn(1, 2, 2, 8)
    out = cpu_attn.joint_ring_attention(q, ik, iv, tk, tv, scale=0.3)
    full_k = torch.cat([ik, tk], dim=2)
    full_v = torch.cat([iv, tv], dim=2)
    b, h, s_q, d = q.shape
    s_k = full_k.shape[2]
    ref = cpu_attn.attention(
        q.reshape(b * h, s_q, d),
        full_k.reshape(b * h, s_k, d),
        full_v.reshape(b * h, s_k, d),
        scale=0.3,
    ).reshape(b, h, s_q, d)
    assert torch.allclose(out, ref, atol=1e-6)


# ---------------- embeddings ----------------


def test_apply_rotary_emb_reference():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 2, 4)
    cos = torch.randn(1, 1, 2, 4)
    sin = torch.randn(1, 1, 2, 4)
    out = cpu_emb.apply_rotary_emb(x, cos, sin)
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    c = cos[..., 0::2]
    s = sin[..., 1::2]
    expected = torch.empty_like(x)
    expected[..., 0::2] = x1 * c - x2 * s
    expected[..., 1::2] = x1 * s + x2 * c
    assert torch.allclose(out, expected, atol=1e-6)
    assert out.dtype == x.dtype


# ---------------- collectives ----------------


def test_collectives_identity_ops():
    t = torch.arange(12).reshape(3, 4)
    assert cpu_col.gather_tp_dim(t, dim=1) is t
    assert cpu_col.gather_from_tensor_model_parallel_region_with_dim(t, 0) is t
    assert cpu_col.reduce_tp(t) is t
    assert cpu_col.reduce_from_tensor_model_parallel_region(t) is t
    assert cpu_col.scatter_tp_dim(t, dim=0) is t
    assert cpu_col.scatter_to_tensor_model_parallel_region(t) is t
    assert cpu_col.scatter_to_process_group_spmd(t, 1, foo=2) is t


def test_collectives_scalars():
    assert cpu_col.get_tp_size() == 1
    assert cpu_col.get_tp_rank() == 0
    assert cpu_col.get_tensor_model_parallel_size() == 1
    assert cpu_col.get_tensor_model_parallel_rank() == 0
    assert cpu_col.get_dp_rank_spmd() == 0


def test_collectives_process_groups():
    assert cpu_col.get_data_parallel_group().size() == 1
    assert cpu_col.get_world_group().size() == 1


def test_spmd_rank():
    r = cpu_col.SPMDRank(world_size=8)
    assert r.get_rank() == 0
    assert r.world_size == 8


# ---------------- platform ----------------


def test_platform_target_and_hardware_enum():
    assert cpu_plat.get_platform_target() is cpu_plat.hardware.CPU
    assert cpu_plat.hardware.CPU.value == "cpu"
    assert cpu_plat.hardware.TRN1.value == "trn1"
    assert cpu_plat.hardware.TRN2.value == "trn2"


# ---------------- mx ----------------


def test_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    data, scale = cpu_mx.quantize_mx(x)
    assert data.dtype == torch.uint32
    assert scale.dtype == torch.uint8
    deq = cpu_mx.dequantize_mx(data, scale)
    assert deq.shape == x.shape
    # mxfp8 is lossy but should be roughly close
    assert torch.allclose(deq.float(), x.float(), atol=0.5, rtol=0.3)


def test_quantize_validation_errors():
    with pytest.raises(TypeError):
        cpu_mx.quantize_mx(torch.randn(8, 8))  # float32 not allowed
    with pytest.raises(ValueError):
        cpu_mx.quantize_mx(torch.randn(8, dtype=torch.bfloat16))  # 1D
    with pytest.raises(ValueError):
        cpu_mx.quantize_mx(torch.randn(7, 8, dtype=torch.bfloat16))  # p%8!=0
    with pytest.raises(NotImplementedError):
        cpu_mx.quantize_mx(torch.randn(8, 8, dtype=torch.bfloat16), dtype="bogus")
    with pytest.raises(NotImplementedError):
        cpu_mx.quantize_mx(torch.randn(8, 8, dtype=torch.bfloat16), group_size=16)


def test_dequantize_validation_errors():
    data, scale = cpu_mx.quantize_mx(torch.randn(8, 8, dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        cpu_mx.dequantize_mx(data.to(torch.int32), scale)
    with pytest.raises(TypeError):
        cpu_mx.dequantize_mx(data, scale.to(torch.int32))
    with pytest.raises(ValueError):
        cpu_mx.dequantize_mx(data.reshape(-1), scale)  # 1D
    with pytest.raises(ValueError):
        cpu_mx.dequantize_mx(data, scale[:, :1])  # bad scale shape


def test_matmul_mx_roundtrip():
    torch.manual_seed(0)
    a = torch.randn(8, 8, dtype=torch.bfloat16)
    b = torch.randn(8, 8, dtype=torch.bfloat16)
    a_mx, a_scale = cpu_mx.quantize_mx(a)
    b_mx, b_scale = cpu_mx.quantize_mx(b)
    out = cpu_mx.matmul_mx(a_mx, a_scale, b_mx, b_scale)
    assert out.shape == (8, 8)


def test_matmul_mx_contraction_mismatch():
    a = torch.randn(8, 8, dtype=torch.bfloat16)
    b = torch.randn(16, 8, dtype=torch.bfloat16)
    a_mx, a_scale = cpu_mx.quantize_mx(a)
    b_mx, b_scale = cpu_mx.quantize_mx(b)
    with pytest.raises(ValueError, match="contraction mismatch"):
        cpu_mx.matmul_mx(a_mx, a_scale, b_mx, b_scale)


def test_linear_mx_reference_close_to_fp_matmul():
    torch.manual_seed(0)
    # M=128, K multiple of 512, N=512
    inp = torch.randn(128, 512, dtype=torch.bfloat16)
    w = torch.randn(512, 512, dtype=torch.bfloat16)
    bias = torch.randn(512, dtype=torch.bfloat16)
    out = cpu_mx.linear_mx(inp, w, bias, out_dtype=torch.float32)
    assert out.shape == (128, 512)
    ref = inp.float() @ w.float() + bias.float()
    # quantized; loose tolerance relative to magnitude
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.1


def test_linear_mx_validation_errors():
    good_w = torch.randn(512, 512, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        cpu_mx.linear_mx(torch.randn(128, 512), good_w)  # fp32 input
    with pytest.raises(ValueError):
        cpu_mx.linear_mx(torch.randn(64, 512, dtype=torch.bfloat16), good_w)  # M!=128
    with pytest.raises(ValueError):
        cpu_mx.linear_mx(
            torch.randn(128, 256, dtype=torch.bfloat16),
            torch.randn(256, 512, dtype=torch.bfloat16),
        )  # K not multiple of 512


def test_linear_mx_outer_n_multiple_tiles():
    torch.manual_seed(0)
    inp = torch.randn(128, 512, dtype=torch.bfloat16)
    w = torch.randn(512, 1024, dtype=torch.bfloat16)  # N=1024 -> 2 tiles
    out = cpu_mx.linear_mx(inp, w, out_dtype=torch.float32)
    assert out.shape == (128, 1024)


def test_dequantize_hardware_tile():
    torch.manual_seed(0)
    x = torch.randn(8, 8, dtype=torch.bfloat16)
    data, scale = cpu_mx.quantize_mx(x)
    tile = cpu_mx.dequantize_mx_hardware_tile(data, scale)
    assert tile.shape == (8, 2, 4)


def test_dequantize_hardware_tile_errors():
    data, scale = cpu_mx.quantize_mx(torch.randn(8, 8, dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        cpu_mx.dequantize_mx_hardware_tile(data.to(torch.int32), scale)
    with pytest.raises(TypeError):
        cpu_mx.dequantize_mx_hardware_tile(data, scale.to(torch.int32))
    with pytest.raises(ValueError):
        cpu_mx.dequantize_mx_hardware_tile(data.unsqueeze(0), scale)  # 3D
    with pytest.raises(ValueError):
        cpu_mx.dequantize_mx_hardware_tile(data, scale[:, :1])  # bad shape


def test_matmul_mx_single_tile_reference():
    torch.manual_seed(0)
    stat = torch.randn(8, 8, dtype=torch.bfloat16)
    mov = torch.randn(8, 8, dtype=torch.bfloat16)
    s_mx, s_sc = cpu_mx.quantize_mx(stat)
    m_mx, m_sc = cpu_mx.quantize_mx(mov)
    out = cpu_mx.matmul_mx_single_tile_reference(s_mx, s_sc, m_mx, m_sc)
    assert out.shape == (2, 2)


def test_single_tile_from_dequantized_mismatch():
    stat = torch.randn(8, 2, 4)
    mov = torch.randn(4, 2, 4)
    with pytest.raises(ValueError, match="contraction mismatch"):
        cpu_mx.matmul_mx_single_tile_reference_from_dequantized(stat, mov)


def test_matmul_mx_k_tiles_reference():
    torch.manual_seed(0)
    # 2 K-tiles, each [8,8]
    stat = torch.randn(2, 8, 8, dtype=torch.bfloat16)
    mov = torch.randn(2, 8, 8, dtype=torch.bfloat16)
    s_mx, s_sc = cpu_mx.quantize_mx(stat)
    m_mx, m_sc = cpu_mx.quantize_mx(mov)
    out = cpu_mx.matmul_mx_k_tiles_reference(s_mx, s_sc, m_mx, m_sc)
    assert out.shape == (2, 2)


def test_matmul_mx_k_tiles_validation():
    s2 = torch.randn(2, 8, 8, dtype=torch.bfloat16)
    s_mx, s_sc = cpu_mx.quantize_mx(s2)
    with pytest.raises(ValueError, match="3D"):
        cpu_mx.matmul_mx_k_tiles_reference(
            s_mx[0], s_sc[0], s_mx, s_sc
        )
    # K_tiles mismatch
    s3 = torch.randn(3, 8, 8, dtype=torch.bfloat16)
    t_mx, t_sc = cpu_mx.quantize_mx(s3)
    with pytest.raises(ValueError, match="K_tiles mismatch"):
        cpu_mx.matmul_mx_k_tiles_reference(s_mx, s_sc, t_mx, t_sc)


def test_pack_linear_mx_input_validation_branches():
    good_w = torch.randn(512, 512, dtype=torch.bfloat16)
    # weight dtype
    with pytest.raises(TypeError, match="weight must be"):
        cpu_mx.pack_linear_mx_inputs(
            torch.randn(128, 512, dtype=torch.bfloat16), good_w.float()
        )
    # input ndim
    with pytest.raises(ValueError, match="input must be 2D"):
        cpu_mx.pack_linear_mx_inputs(
            torch.randn(2, 128, 512, dtype=torch.bfloat16), good_w
        )
    # weight ndim
    with pytest.raises(ValueError, match="weight must be 2D"):
        cpu_mx.pack_linear_mx_inputs(
            torch.randn(128, 512, dtype=torch.bfloat16),
            torch.randn(512, 512, 1, dtype=torch.bfloat16),
        )
    # contraction mismatch
    with pytest.raises(ValueError, match="contraction mismatch"):
        cpu_mx.pack_linear_mx_inputs(
            torch.randn(128, 512, dtype=torch.bfloat16),
            torch.randn(1024, 512, dtype=torch.bfloat16),
        )
    # N != 512
    with pytest.raises(ValueError, match="N=512"):
        cpu_mx.pack_linear_mx_inputs(
            torch.randn(128, 1024, dtype=torch.bfloat16),
            torch.randn(1024, 1024, dtype=torch.bfloat16),
        )
    # bias dtype + shape
    with pytest.raises(TypeError, match="bias must be"):
        cpu_mx.pack_linear_mx_inputs(
            torch.randn(128, 512, dtype=torch.bfloat16),
            good_w,
            torch.randn(512).float(),
        )
    with pytest.raises(ValueError, match="bias must be"):
        cpu_mx.pack_linear_mx_inputs(
            torch.randn(128, 512, dtype=torch.bfloat16),
            good_w,
            torch.randn(256, dtype=torch.bfloat16),
        )


def test_linear_mx_outer_n_validation_branches():
    inp = torch.randn(128, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="weight must be 2D"):
        cpu_mx.linear_mx(inp, torch.randn(512, 512, 1, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="N multiple of 512"):
        cpu_mx.linear_mx(inp, torch.randn(512, 600, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bias must be"):
        cpu_mx.linear_mx(
            inp,
            torch.randn(512, 512, dtype=torch.bfloat16),
            torch.randn(256, dtype=torch.bfloat16),
        )


def test_matmul_mx_k_tiles_scale_mismatch():
    stat = torch.randn(2, 8, 8, dtype=torch.bfloat16)
    mov = torch.randn(2, 8, 8, dtype=torch.bfloat16)
    s_mx, s_sc = cpu_mx.quantize_mx(stat)
    m_mx, m_sc = cpu_mx.quantize_mx(mov)
    with pytest.raises(ValueError, match="stationary scale K_tiles"):
        cpu_mx.matmul_mx_k_tiles_reference(s_mx, s_sc[:1], m_mx, m_sc)
    with pytest.raises(ValueError, match="moving scale K_tiles"):
        cpu_mx.matmul_mx_k_tiles_reference(s_mx, s_sc, m_mx, m_sc[:1])


def test_supported_dtypes_constant():
    assert "float8_e4m3fn_x4" in cpu_mx.SUPPORTED_DTYPES
    assert "float8_e5m2_x4" in cpu_mx.SUPPORTED_DTYPES
