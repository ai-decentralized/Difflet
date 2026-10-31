"""HunyuanVideo's compiled blocks must retain BF16 intermediate rounding."""

from types import SimpleNamespace

import torch
from torch import nn

from difflet.backends.cuda.hunyuan_video.transformer import (
    CudaHunyuanVideoTransformerApplication,
)


class _ResidualAffine(nn.Module):
    def forward(self, x, gate, residual, scale, shift):
        return (x * gate + residual) * (1 + scale) + shift


class _RepeatedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_ResidualAffine()])
        self.single_transformer_blocks = nn.ModuleList()

    def forward(self, *inputs):
        return self.transformer_blocks[0](*inputs)


class _TinyHunyuan(CudaHunyuanVideoTransformerApplication):
    def build_module(self):
        return _RepeatedBlock()

    def load_weights(self, module):
        pass

    def apply_load_time_swaps(self, module):
        pass


def test_compiled_hunyuan_retains_bfloat16_residual_affine_rounding(tmp_path):
    generator = torch.Generator().manual_seed(2)
    inputs = [torch.randn(256, 64, generator=generator, dtype=torch.bfloat16) for _ in range(5)]
    app = _TinyHunyuan(
        model_path="unused",
        config=SimpleNamespace(torch_dtype=torch.bfloat16),
        device="cpu",
        torch_compile=True,
    )
    app.compile(str(tmp_path))
    app.load(str(tmp_path), skip_warmup=True)
    with torch.no_grad():
        reference = _ResidualAffine()(*inputs)
        actual = app(*inputs)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_native_compile_boundaries_preserve_state_and_share_wrappers():
    from difflet.backends.cuda.hunyuan_video import numerics

    blocks = nn.ModuleList(
        [nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)), nn.Sequential(nn.Linear(8, 8))]
    ).eval()
    state = {name: tensor.clone() for name, tensor in blocks.state_dict().items()}
    parameters = dict(blocks.named_parameters())
    count = numerics.preserve_reference_compiled_ops([blocks], tp_degree=1)
    assert count == {"linear": 2, "norm": 1}
    assert blocks[0][0].forward.__func__ is blocks[1][0].forward.__func__
    for layer in (blocks[0][0], blocks[0][1], blocks[1][0]):
        assert getattr(layer.forward, "_torchdynamo_disable", False)
        assert layer.training is False
    assert dict(blocks.named_parameters()).keys() == parameters.keys()
    assert all(dict(blocks.named_parameters())[name] is value for name, value in parameters.items())
    assert all(torch.equal(blocks.state_dict()[name], value) for name, value in state.items())
    assert numerics.preserve_reference_compiled_ops([blocks], tp_degree=1) == {
        "linear": 0,
        "norm": 0,
    }


def test_native_boundaries_keep_real_compiled_pointwise_work():
    from diffusers.models.normalization import RMSNorm
    from difflet.backends.cuda.hunyuan_video import numerics

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.norm = nn.LayerNorm(8)
            self.rms = RMSNorm(8, eps=1e-6)

        def forward(self, x, scale, residual):
            x = self.rms(self.norm(self.linear(x)))
            return x * scale + residual, torch.compiler.is_compiling()

    block = Block().to(dtype=torch.bfloat16).eval().requires_grad_(False)
    inputs = [torch.randn(1, 3, 8, dtype=torch.bfloat16) for _ in range(3)]
    reference, _ = block(*inputs)
    numerics.preserve_reference_compiled_ops([nn.ModuleList([block])], tp_degree=1)
    compiled = torch.compile(
        block,
        options={"emulate_precision_casts": True, "max_autotune_gemm_backends": "ATEN"},
        fullgraph=False,
    )
    actual, is_compiling = compiled(*inputs)
    assert is_compiling is True
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
