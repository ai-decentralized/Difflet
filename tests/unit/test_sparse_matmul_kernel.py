"""NKI compilation tests for sparse matmul kernels.

The ``nc_matmul_sparse`` private ISA instruction is only callable on hardware
(baremetal), not in the NKI simulator. These tests verify the kernels are
well-formed via import and attribute checks. Full numerical validation is
done via hardware probe scripts in ``scripts/``.
"""
import importlib.util
import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("nki") is None,
    reason="NKI is not installed",
)
def test_sparse_matmul_kernels_importable():
    """Both kernels import cleanly and have expected structure."""
    from difflet.backends.trainium.nki_kernels.sparse_matmul import (
        sparse_matmul_bf16_kernel,
        sparse_matmul_fp8_kernel,
    )

    # Both are valid NKI kernel objects
    for kernel, name in [
        (sparse_matmul_bf16_kernel, "sparse_matmul_bf16_kernel"),
        (sparse_matmul_fp8_kernel, "sparse_matmul_fp8_kernel"),
    ]:
        assert hasattr(kernel, "__name__"), f"{name} missing __name__"
        assert kernel.__name__ == name, f"{name} __name__ mismatch: {kernel.__name__}"
        # Kernel should be registered (index 1 for single-kernel, [1] for base)
        # The jit decorator creates a Kernel object
        assert hasattr(kernel, 'func'), f"{name} missing 'func' attribute"
