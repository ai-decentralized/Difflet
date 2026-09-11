"""Unit tests for the TPU attention op's kernel dispatch.

The fused Pallas path itself cannot be exercised here — it needs a TPU and a
jax-capable toolchain — so these cover the decision logic around it: when it is
chosen, when it is declined, and that declining always falls back rather than
raising. The measured numbers behind each threshold are in the module's
constants and in benchmark/v5e/wan_2_2.md.
"""

import torch

# --------------------------------------------------------------- flash kernel


def _reset_flash_cache():
    from difflet.backends.tpu.ops_impl import attention as A

    A._FLASH_KERNEL = None


def test_align_rounds_up_to_the_kernel_block(monkeypatch):
    from difflet.backends.tpu.ops_impl import attention as A

    assert A._align(512) == 512
    assert A._align(4680) == 5120     # Wan's sequence
    assert A._align(5120) == 5120     # Qwen's is already aligned
    assert A._align(1) == 512


def test_flash_is_declined_when_the_kernel_is_unavailable(monkeypatch):
    """Off a jax-capable toolchain every call must fall back, not raise."""
    from difflet.backends.tpu.ops_impl import attention as A

    monkeypatch.setattr(A, "_FLASH_KERNEL", False)
    q = torch.zeros(10, 4680, 128)
    k = torch.zeros(10, 4680, 128)
    assert A._should_flash(q, k, causal=False) is False


def test_flash_declines_small_shapes_causal_and_odd_head_dims(monkeypatch):
    """Each guard exists for a measured reason; see the module constants."""
    from difflet.backends.tpu.ops_impl import attention as A

    monkeypatch.setattr(A, "_FLASH_KERNEL", object())

    big_q = torch.zeros(10, 4680, 128)
    big_k = torch.zeros(10, 4680, 128)
    assert A._should_flash(big_q, big_k, causal=False) is True

    # Wan's cross-attention: 24 M score elements, where SDPA is 1.7x faster.
    small_k = torch.zeros(10, 512, 128)
    assert A._should_flash(big_q, small_k, causal=False) is False

    # Causal is declined by this wrapper, not by the kernel.
    assert A._should_flash(big_q, big_k, causal=True) is False

    odd = torch.zeros(10, 4680, 96)
    assert A._should_flash(odd, odd, causal=False) is False


def test_env_var_disables_the_kernel(monkeypatch):
    from difflet.backends.tpu.ops_impl import attention as A

    monkeypatch.setenv("DIFFLET_TPU_FLASH", "0")
    _reset_flash_cache()
    try:
        assert A._flash_kernel() is None
    finally:
        _reset_flash_cache()


def test_masked_calls_never_reach_the_kernel(monkeypatch):
    """An explicit attention_mask stays on the SDPA branch."""
    from difflet.backends.tpu.ops_impl import attention as A

    def _explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("masked attention must not use the fused kernel")

    monkeypatch.setattr(A, "_flash_attention", _explode)
    monkeypatch.setattr(A, "_flash_attention_key_bounds", _explode)
    monkeypatch.setattr(A, "_FLASH_KERNEL", object())
    q = torch.randn(2, 8, 16)
    mask = torch.ones(8, 8, dtype=torch.bool)
    A.attention(q, q, q, scale=1.0, attention_mask=mask)


def test_key_bounds_take_the_fused_path_at_flash_shapes(monkeypatch):
    """HunyuanVideo's joint attention arrives as [bound_min, bound_max) key
    windows; at flash-worthy shapes they must go to the segment-id kernel path,
    not to a materialized mask (1.3 GB of scores at 10 496 tokens)."""
    from difflet.backends.tpu.ops_impl import attention as A

    calls = []

    def _fake_bounds(q, k, v, *, scale, bound_min, bound_max):
        calls.append((tuple(bound_min.shape), tuple(bound_max.shape)))
        return q

    monkeypatch.setattr(A, "_flash_attention_key_bounds", _fake_bounds)
    monkeypatch.setattr(A, "_should_flash", lambda q, k, causal: True)
    monkeypatch.setattr(A, "_FLASH_KERNEL", object())
    q = torch.randn(6, 64, 128)
    lo = torch.zeros(6, 64, 1, dtype=torch.int32)
    hi = torch.full((6, 64, 1), 40, dtype=torch.int32)
    A.attention(q, q, q, scale=1.0, bound_min=lo, bound_max=hi)
    assert calls == [((6, 64, 1), (6, 64, 1))]


def test_key_bounds_fall_back_to_the_mask_below_flash_shapes(monkeypatch):
    """Small shapes keep the exact reference behaviour: bounds -> bool mask -> SDPA."""
    from difflet.backends.tpu.ops_impl import attention as A

    def _explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("small bounded attention must not use the kernel")

    monkeypatch.setattr(A, "_flash_attention_key_bounds", _explode)
    monkeypatch.setattr(A, "_FLASH_KERNEL", object())
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 8, 16) for _ in range(3))
    lo = torch.zeros(2, 8, 1, dtype=torch.int32)
    hi = torch.full((2, 8, 1), 5, dtype=torch.int32)
    out = A.attention(q, k, v, scale=0.25, bound_min=lo, bound_max=hi)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k[:, :5], v[:, :5], scale=0.25)
    assert torch.allclose(out, ref, atol=1e-5)


def test_key_bounds_segment_ids_mark_the_window(monkeypatch):
    """The kernel wrapper's segment ids: 0 inside [lo, hi) and for real
    queries, 1 for keys outside the window and for the block padding."""
    from difflet.backends.tpu.ops_impl import attention as A

    seen = {}

    class _Kernel:
        @staticmethod
        def flash_attention(q4, k4, v4, *, causal, sm_scale, q_segment_ids, kv_segment_ids):
            seen["q"] = q_segment_ids
            seen["kv"] = kv_segment_ids
            seen["shape"] = tuple(q4.shape)
            return q4

    monkeypatch.setattr(A, "_FLASH_KERNEL", _Kernel)
    monkeypatch.setattr(A, "FLASH_BLOCK", 8)
    q = torch.randn(3, 6, 4)  # 3 rows, 6 queries/keys, padded to 8
    lo = torch.zeros(3, 6, 1, dtype=torch.int32)
    hi = torch.tensor([4, 6, 2], dtype=torch.int32).view(3, 1, 1).expand(3, 6, 1).contiguous()
    out = A._flash_attention_key_bounds(q, q, q, scale=1.0, bound_min=lo, bound_max=hi)
    assert out.shape == q.shape
    assert seen["shape"] == (3, 1, 8, 4)  # batch=rows so ids can differ per row
    assert seen["kv"].tolist() == [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 1],
    ]
    assert seen["q"].tolist() == [[0, 0, 0, 0, 0, 0, 1, 1]] * 3
