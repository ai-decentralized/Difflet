"""``difflet.ops.attention_heads``: attention over ``(B, H, S, D)`` inputs of any
strides. The CPU reference and the TPU/Trainium backends fold the heads into the batch axis
inside the op — the very ops the model code used to do before the call, so their bits do not
move; the CUDA backend hands the strided views to SDPA as they are (no copies)."""

import math

import pytest
import torch

import difflet.ops as ops
from difflet.backends.cpu.ops_impl import attention as cpu_attn
from difflet.backends.cuda.ops_impl import attention as cuda_attn
from difflet.backends.tpu.ops_impl import attention as tpu_attn

B, H, S, SK, D = 2, 3, 5, 7, 4


def _qkv(dtype=torch.float32):
    torch.manual_seed(0)
    # (B, S, H, D) projections, as every model produces them, then the transposed VIEW
    # (B, H, S, D) the models used to reshape (copy) into (B*H, S, D).
    q = torch.randn(B, S, H, D, dtype=dtype).transpose(1, 2)
    k = torch.randn(B, SK, H, D, dtype=dtype).transpose(1, 2)
    v = torch.randn(B, SK, H, D, dtype=dtype).transpose(1, 2)
    assert not q.is_contiguous()
    return q, k, v


def _folded_reference(impl, q, k, v, **kw):
    """What the model code did before: fold heads into the batch, call ``attention``, unfold."""
    out = impl.attention(
        q.reshape(B * H, S, D), k.reshape(B * H, SK, D), v.reshape(B * H, SK, D), **kw
    )
    return out.reshape(B, H, S, D)


KW = dict(scale=1.0 / math.sqrt(D), causal=False, attention_mask=None, tp_q=True, tp_k=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_reference_is_the_folded_path_bit_for_bit(dtype):
    q, k, v = _qkv(dtype)
    out = cpu_attn.attention_heads(q, k, v, **KW, tp_out=False)
    assert tuple(out.shape) == (B, H, S, D) and out.dtype == dtype
    assert torch.equal(out, _folded_reference(cpu_attn, q, k, v, **KW, tp_out=False))


def test_tpu_backend_folds_inside_the_op_bit_for_bit():
    q, k, v = _qkv()
    out = tpu_attn.attention_heads(q, k, v, **KW, tp_out=False)
    assert torch.equal(out, _folded_reference(tpu_attn, q, k, v, **KW, tp_out=False))


def test_cuda_backend_passes_the_strided_views_to_sdpa_without_a_copy(monkeypatch):
    import torch.nn.functional as F

    seen = {}
    real = F.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen["q"], seen["k"], seen["v"] = q, k, v
        return real(q, k, v, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    q, k, v = _qkv()
    out = cuda_attn.attention_heads(q, k, v, **KW, tp_out=False)
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert seen[name].data_ptr() == t.data_ptr() and seen[name].stride() == t.stride(), name
    assert seen["q"].ndim == 4
    assert torch.allclose(out, _folded_reference(cpu_attn, q, k, v, **KW, tp_out=False), atol=1e-6)


def test_trainium_backend_folds_inside_the_op():
    pytest.importorskip("nkilib")
    from difflet.backends.trainium.ops_impl import attention as trn_attn

    assert callable(trn_attn.attention_heads)


def test_the_op_is_on_the_surface():
    from difflet.ops import _EXPORTS

    assert _EXPORTS["attention_heads"] == ("attention", "attention_heads")
    assert callable(ops.attention_heads)
