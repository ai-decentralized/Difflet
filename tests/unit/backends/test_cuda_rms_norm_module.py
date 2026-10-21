"""The CUDA backend's ``RMSNorm`` module: one fused ``F.rms_norm`` with fp32 statistics,
normalisation and affine — diffusers' ``nn.RMSNorm`` numerics — where the shared pure-torch
module normalises in the input dtype in several passes. Flux's and Qwen-Image's q/k norms are
this module (``difflet.ops.RMSNorm`` / ``CustomRMSNorm``); the DiT bench put the Flux gap in
exactly those passes. The CPU reference backend keeps the shared module."""

import pytest
import torch

from difflet.backends._torch_megatron.norm import RMSNorm as SharedRMSNorm
from difflet.backends._torch_megatron.norm import rms_norm_fp32
from difflet.backends.cpu.ops_impl import norm as cpu_norm
from difflet.backends.cuda.ops_impl import norm as cuda_norm


def test_cuda_rms_norm_is_its_own_fused_module_and_the_cpu_keeps_the_shared_math():
    assert cuda_norm.RMSNorm is not SharedRMSNorm
    assert cuda_norm.CustomRMSNorm is cuda_norm.RMSNorm
    # The CPU reference backend's module is its own class with the shared module's math
    # (input-dtype normalisation), untouched by the fused CUDA module.
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16).to(torch.bfloat16)
    assert torch.equal(cpu_norm.RMSNorm(16)(x), SharedRMSNorm(16)(x))


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


def test_cuda_rms_norm_is_what_the_op_surface_hands_the_models(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    import importlib

    from difflet.ops import _dispatch

    mod = importlib.import_module("difflet.backends.cuda.ops_impl.norm")
    assert mod.RMSNorm is cuda_norm.RMSNorm
