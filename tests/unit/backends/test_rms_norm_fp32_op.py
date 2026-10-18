"""``difflet.ops.rms_norm_fp32`` (CUDA plan F4b): the reference on every backend without a fused
kernel, and the CUDA backend's fused ``F.rms_norm`` against it."""

import pytest
import torch

import difflet.ops as ops
from difflet.backends._torch_megatron.norm import rms_norm_fp32 as reference
from difflet.backends.cpu.ops_impl import norm as cpu_norm
from difflet.backends.cuda.ops_impl import norm as cuda_norm
from difflet.backends.tpu.ops_impl import norm as tpu_norm


def _manual(x, weight, eps):
    sq = x.float().pow(2).sum(dim=-1, keepdim=True)
    out = x.float() * torch.rsqrt(sq / x.shape[-1] + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_is_the_fp32_math_bit_for_bit(dtype):
    torch.manual_seed(0)
    x = torch.randn(3, 7, 32).to(dtype)
    weight = (torch.rand(32) + 0.5).to(dtype)
    assert torch.equal(reference(x, weight, 1e-6), _manual(x, weight, 1e-6))
    assert torch.equal(reference(x, None, 1e-6), _manual(x, None, 1e-6))
    assert reference(x, weight, 1e-6).dtype == dtype


def test_cpu_and_tpu_backends_export_the_reference_and_the_op_surface_has_it():
    assert cpu_norm.rms_norm_fp32 is reference
    assert tpu_norm.rms_norm_fp32 is reference
    assert callable(ops.rms_norm_fp32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_fused_path_matches_the_reference(dtype):
    # F.rms_norm computes in fp32 and casts once: the reference up to the reduction order.
    torch.manual_seed(0)
    x = torch.randn(3, 7, 32).to(dtype)
    weight = (torch.rand(32) + 0.5).to(dtype)
    fused = cuda_norm.rms_norm_fp32(x, weight, 1e-6)
    ref = reference(x, weight, 1e-6)
    assert fused.dtype == dtype
    tol = 1e-6 if dtype is torch.float32 else 1 / 64  # within one bf16 ulp of values ~<2
    assert torch.allclose(fused.float(), ref.float(), atol=tol, rtol=tol)
    assert torch.allclose(
        cuda_norm.rms_norm_fp32(x, None, 1e-6).float(),
        reference(x, None, 1e-6).float(),
        atol=tol,
        rtol=tol,
    )
