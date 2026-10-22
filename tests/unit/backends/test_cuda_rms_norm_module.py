"""The CUDA backend's ``RMSNorm`` module: one fused ``F.rms_norm`` with fp32 statistics,
normalisation and affine — diffusers' ``nn.RMSNorm`` numerics — where the shared pure-torch
module normalises in the input dtype in several passes. Flux's and Qwen-Image's q/k norms are
this module (``difflet.ops.RMSNorm`` / ``CustomRMSNorm``); the DiT bench put the Flux gap in
exactly those passes. The CPU reference backend keeps the shared module."""

import pytest
import torch

from difflet.backends._torch_megatron.norm import RMSNorm as SharedRMSNorm
from difflet.backends._torch_megatron.norm import rms_norm_fp32
from difflet.backends.cuda.ops_impl import norm as cuda_norm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_rms_norm_matches_the_fp32_reference(dtype):
    torch.manual_seed(0)
    m = cuda_norm.RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        m.weight.copy_(torch.rand(32) + 0.5)
    m = m.to(dtype)
    x = torch.randn(3, 7, 32).to(dtype)
    out = m(x)
    ref = rms_norm_fp32(x, m.weight, 1e-6)
    assert out.dtype == dtype and out.shape == x.shape
    tol = 1e-6 if dtype is torch.float32 else 1 / 64
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol)


def test_cuda_rms_norm_state_dict_and_eps_match_the_shared_module():
    shared = SharedRMSNorm(16, eps=1e-5, extra="ignored")
    fused = cuda_norm.RMSNorm(16, eps=1e-5, extra="ignored")
    assert list(fused.state_dict()) == list(shared.state_dict()) == ["weight"]
    assert fused.eps == shared.eps == 1e-5
    fused.load_state_dict(shared.state_dict())  # the checkpoint loader's path
