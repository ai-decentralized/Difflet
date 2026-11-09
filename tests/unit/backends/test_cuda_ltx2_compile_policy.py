"""Keep native RMSNorm rounding inside compiled LTX-2 attention graphs."""

import pytest
import torch

from difflet.backends.cuda.ltx_2.compile_policy import preserve_native_rms_norm


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_native_norm_preserves_values_parameters_and_fake_contract(affine, noncontiguous):
    norm = torch.nn.RMSNorm(32, eps=1e-6, elementwise_affine=affine).to(torch.bfloat16)
    norm.requires_grad_(False)
    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    if noncontiguous:
        x = x.transpose(0, 1)
    original = x.clone()
    expected = norm(x)
    state = {key: value.clone() for key, value in norm.state_dict().items()}
    weight = norm.weight
    assert preserve_native_rms_norm(norm) == 1
    assert norm.weight is weight
    assert torch.equal(norm(x), expected)
    assert torch.equal(x, original)
    assert all(torch.equal(norm.state_dict()[key], value) for key, value in state.items())
    from difflet.backends.cuda.ltx_2.compile_policy import _native_rms_norm

    torch.library.opcheck(_native_rms_norm, (x, norm.weight, norm.eps))


def test_only_stock_last_dimension_rms_norms_are_replaced():
    class CustomNorm(torch.nn.RMSNorm):
        def forward(self, x):
            return super().forward(x) + 1

    custom = CustomNorm(32)
    wide = torch.nn.RMSNorm((2, 32))
    root = torch.nn.ModuleList([torch.nn.RMSNorm(32), custom, wide, torch.nn.LayerNorm(32)])
    original = custom.forward
    assert preserve_native_rms_norm(root) == 1
    assert custom.forward == original
    assert wide.forward.__func__ is torch.nn.RMSNorm.forward


def test_ltx_compile_policy_preserves_eager_gemm_selection():
    from difflet.backends.cuda.ltx_2.transformer import CudaLtx2TransformerApplication

    options = CudaLtx2TransformerApplication.compile_options
    assert options["emulate_precision_casts"] is True
    assert options["max_autotune"] is False
    assert options["max_autotune_gemm"] is False


def test_one_compiler_policy_on_every_device():
    from difflet.backends.cuda.ltx_2.dit_config import Ltx2DiTConfig
    from difflet.backends.cuda.ltx_2.transformer import CudaLtx2TransformerApplication

    app = CudaLtx2TransformerApplication(model_path="unused", config=Ltx2DiTConfig(), device="cpu")
    assert app.compile_options is CudaLtx2TransformerApplication.compile_options


@pytest.mark.parametrize("compiled,expected", [(False, 0), (True, 1)])
def test_native_norm_policy_is_scoped_to_compiled_blocks(compiled, expected):
    from difflet.backends.cuda.ltx_2.dit_config import Ltx2DiTConfig
    from difflet.backends.cuda.ltx_2.transformer import CudaLtx2TransformerApplication

    app = CudaLtx2TransformerApplication(
        model_path="unused", config=Ltx2DiTConfig(), device="cpu", torch_compile=compiled
    )
    norm = torch.nn.RMSNorm(32)
    original = norm.forward
    app.apply_load_time_swaps(norm)
    assert int(norm.forward != original) == expected


@pytest.mark.parametrize("affine,bias", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_diffusers_manual_norm_preserves_its_original_arithmetic(affine, bias, weight_dtype):
    from diffusers.models.normalization import RMSNorm

    norm = RMSNorm(32, eps=1e-6, elementwise_affine=affine, bias=bias).to(weight_dtype)
    norm.requires_grad_(False)
    x = torch.randn(3, 2, 32, dtype=torch.bfloat16).transpose(0, 1)
    expected = norm(x)
    state = {key: value.clone() for key, value in norm.state_dict().items()}
    weight, original_bias = norm.weight, norm.bias
    assert preserve_native_rms_norm(norm) == 1
    assert norm.weight is weight and norm.bias is original_bias
    actual = norm(x)
    assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    assert all(torch.equal(norm.state_dict()[key], value) for key, value in state.items())
    from difflet.backends.cuda.ltx_2.compile_policy import _diffusers_rms_norm

    torch.library.opcheck(_diffusers_rms_norm, (x, norm.weight, norm.bias, norm.eps))


def test_custom_diffusers_norm_is_not_replaced():
    from diffusers.models.normalization import RMSNorm

    class Custom(RMSNorm):
        def forward(self, x):
            return super().forward(x) + 1

    norm = Custom(32, eps=1e-6)
    original = norm.forward
    assert preserve_native_rms_norm(norm) == 0 and norm.forward == original
