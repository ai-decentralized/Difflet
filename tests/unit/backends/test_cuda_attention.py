"""CUDA attention ops.

Runs on CPU tensors: SDPA has a CPU path, so the contract — scale=None means 1.0,
masks, bounds, causal — is pinned here; the kernel choice is only observable on a GPU.
Mirrors the attention block of test_tpu_layers.py.
"""

import pytest
import torch

from difflet.backends.cpu.ops_impl import attention as CPU_A
from difflet.backends.cuda.ops_impl import attention as A


def _qkv(b=1, h=2, s=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(b, h, s, d, generator=g) for _ in range(3))


def test_attention_matches_cpu_reference_unmasked():
    q, k, v = _qkv()
    assert torch.allclose(
        A.attention(q, k, v, scale=0.3), CPU_A.attention(q, k, v, scale=0.3), atol=1e-5
    )


def test_attention_scale_none_means_one_not_sdpa_default():
    q, k, v = _qkv()
    ref = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)).float(), -1), v)
    assert torch.allclose(A.attention(q, k, v, scale=None), ref, atol=1e-5)


def test_attention_causal_masks_the_future():
    q, k, v = _qkv(h=1)
    out = A.attention(q, k, v, scale=1.0, causal=True)
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-5)
    assert torch.allclose(out, CPU_A.attention(q, k, v, scale=1.0, causal=True), atol=1e-5)


def test_attention_bool_mask_and_causal_combine():
    q, k, v = _qkv(h=1)
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[:, 1] = False  # nobody may attend to key 1
    out = A.attention(q, k, v, scale=1.0, causal=True, attention_mask=mask)
    ref = CPU_A.attention(q, k, v, scale=1.0, causal=True, attention_mask=mask)
    assert torch.allclose(out, ref, atol=1e-5)
    # query 2 sees keys {0, 2} only: never key 1 (mask) nor key 3 (causal)
    assert torch.allclose(out[0, 0, 1], v[0, 0, 0], atol=1e-5)


def test_attention_additive_mask_uses_finfo_min():
    q, k, v = _qkv(h=1)
    additive = torch.zeros(4, 4)
    additive[0, :] = torch.finfo(torch.float32).min  # query 0 fully masked
    out = A.attention(q, k, v, scale=1.0, causal=True, attention_mask=additive)
    assert torch.isfinite(out).all()
    ref = CPU_A.attention(q, k, v, scale=1.0, causal=True, attention_mask=additive)
    assert torch.allclose(out[0, 0, 1:], ref[0, 0, 1:], atol=1e-5)


def test_attention_bounds_select_a_key_window():
    q, k, v = _qkv(h=1)
    bound_min = torch.zeros(1, 1, 1, dtype=torch.long)
    bound_max = torch.full((1, 1, 1), 2, dtype=torch.long)
    out = A.attention(q, k, v, scale=1.0, bound_min=bound_min, bound_max=bound_max)
    ref = CPU_A.attention(q, k, v, scale=1.0, bound_min=bound_min, bound_max=bound_max)
    assert torch.allclose(out, ref, atol=1e-5)


def test_attention_bounds_require_both_ends():
    q, k, v = _qkv(h=1)
    with pytest.raises(ValueError, match="both be provided"):
        A.attention(q, k, v, bound_min=torch.zeros(1, 1, 1, dtype=torch.long))


def test_attention_bounds_and_mask_are_mutually_exclusive():
    q, k, v = _qkv(h=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        A.attention(
            q,
            k,
            v,
            bound_min=torch.zeros(1, 1, 1, dtype=torch.long),
            bound_max=torch.full((1, 1, 1), 4, dtype=torch.long),
            attention_mask=torch.ones(4, 4, dtype=torch.bool),
        )


def test_attention_accepts_heads_folded_into_batch():
    # difflet model code passes (B*heads, S, D); SDPA wants a batch of matrices either way.
    q, k, v = (t.reshape(2, 4, 8) for t in _qkv())
    assert torch.allclose(
        A.attention(q, k, v, scale=1.0), CPU_A.attention(q, k, v, scale=1.0), atol=1e-5
    )


def test_cross_attention_forces_tp_flags_off_and_no_causal():
    q, k, v = _qkv(h=1)
    out = A.cross_attention(q, k, v, scale=1.0, tp_q=True, tp_k=True, tp_out=True)
    assert torch.allclose(out, CPU_A.attention(q, k, v, scale=1.0), atol=1e-5)


@pytest.mark.parametrize(
    "name",
    ["ring_attention", "joint_ring_attention", "ulysses_attention", "joint_ulysses_attention"],
)
def test_context_parallel_modes_fail_loudly(name):
    # Context parallel across cards is not supported on the CUDA backend yet (TP is not an attention
    # mode).
    with pytest.raises(NotImplementedError, match="not supported"):
        getattr(A, name)()


def test_heads_folded_inputs_reach_sdpa_as_four_dimensional(monkeypatch):
    # On a GPU, SDPA picks the flash kernel for (B, H, S, D) but falls back to the
    # math path (materialised S×S scores) for the (B*H, S, D) layout difflet's model code
    # passes. The backend must present 4-D tensors to SDPA whatever the caller's layout.
    import torch.nn.functional as F

    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        seen.append(q.ndim)
        return real(q, k, v, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    q, k, v = (t.reshape(2, 4, 8) for t in _qkv())
    out = A.attention(q, k, v, scale=1.0)
    assert seen == [4], seen
    assert tuple(out.shape) == (2, 4, 8)
    assert torch.allclose(out, CPU_A.attention(q, k, v, scale=1.0), atol=1e-5)
