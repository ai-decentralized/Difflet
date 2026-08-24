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


def test_masked_and_bounded_calls_never_reach_the_kernel(monkeypatch):
    """The fused path handles neither; they must stay on the SDPA branch."""
    from difflet.backends.tpu.ops_impl import attention as A

    def _explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("masked attention must not use the fused kernel")

    monkeypatch.setattr(A, "_flash_attention", _explode)
    monkeypatch.setattr(A, "_FLASH_KERNEL", object())
    q = torch.randn(2, 8, 16)
    mask = torch.ones(8, 8, dtype=torch.bool)
    A.attention(q, q, q, scale=1.0, attention_mask=mask)
