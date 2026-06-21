import pytest


def test_cpu_quantize_dequantize_roundtrip(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.ops.mx import dequantize_mx, quantize_mx

    generator = torch.Generator().manual_seed(0)
    x = torch.randn((16, 64), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )

    data, scale = quantize_mx(x)
    out = dequantize_mx(data, scale, output_dtype=torch.float32)

    assert data.dtype == torch.uint32
    assert data.shape == (16, 16)
    assert scale.dtype == torch.uint8
    assert scale.shape == (2, 16)
    assert torch.nn.functional.cosine_similarity(
        x.float().reshape(1, -1), out.reshape(1, -1)
    ).item() >= 0.99


def test_cpu_matmul_mx_matches_dequantized_matmul(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.ops.mx import dequantize_mx, matmul_mx, quantize_mx

    generator = torch.Generator().manual_seed(1)
    a = torch.randn((16, 64), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    b = torch.randn((64, 32), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )

    a_data, a_scale = quantize_mx(a)
    b_data, b_scale = quantize_mx(b)

    out = matmul_mx(
        a_data,
        a_scale,
        b_data,
        b_scale,
        out_dtype=torch.float32,
    )
    expected = dequantize_mx(a_data, a_scale, output_dtype=torch.float32) @ dequantize_mx(
        b_data,
        b_scale,
        output_dtype=torch.float32,
    )

    assert out.shape == (16, 32)
    assert torch.equal(out, expected)


def test_cpu_hardware_tile_reference_matches_explicit_einsum(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.backends.cpu.ops_impl.mx import (
        dequantize_mx_hardware_tile,
        matmul_mx_single_tile_reference,
    )
    from difflet.ops.mx import quantize_mx

    generator = torch.Generator().manual_seed(2)
    stationary = torch.randn((128, 512), generator=generator).to(torch.bfloat16)
    moving = torch.randn((128, 2048), generator=generator).to(torch.bfloat16)
    stationary_mx, stationary_scale = quantize_mx(stationary)
    moving_mx, moving_scale = quantize_mx(moving)

    out = matmul_mx_single_tile_reference(
        stationary_mx,
        stationary_scale,
        moving_mx,
        moving_scale,
        out_dtype=torch.float32,
    )
    expected = torch.einsum(
        "kmq,knq->mn",
        dequantize_mx_hardware_tile(stationary_mx, stationary_scale),
        dequantize_mx_hardware_tile(moving_mx, moving_scale),
    )

    assert out.shape == (128, 512)
    assert torch.equal(out, expected)


def test_cpu_k_tiles_reference_matches_sum_of_single_tiles(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.backends.cpu.ops_impl.mx import (
        matmul_mx_k_tiles_reference,
        matmul_mx_single_tile_reference,
    )
    from difflet.ops.mx import quantize_mx

    generator = torch.Generator().manual_seed(3)
    stationary = (0.1 * torch.randn((2, 128, 512), generator=generator)).to(
        torch.bfloat16
    )
    moving = (0.1 * torch.randn((2, 128, 2048), generator=generator)).to(
        torch.bfloat16
    )
    stationary_mx, stationary_scale = quantize_mx(stationary)
    moving_mx, moving_scale = quantize_mx(moving)

    out = matmul_mx_k_tiles_reference(
        stationary_mx,
        stationary_scale,
        moving_mx,
        moving_scale,
        out_dtype=torch.float32,
    )
    expected = torch.zeros_like(out)
    for k_idx in range(2):
        expected += matmul_mx_single_tile_reference(
            stationary_mx[k_idx],
            stationary_scale[k_idx],
            moving_mx[k_idx],
            moving_scale[k_idx],
            out_dtype=torch.float32,
        )

    assert out.shape == (128, 512)
    assert torch.equal(out, expected)


def test_cpu_linear_mx_reference_matches_logical_linear(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.backends.cpu.ops_impl.mx import linear_mx_reference

    generator = torch.Generator().manual_seed(4)
    input_bf16 = (0.05 * torch.randn((128, 2048), generator=generator)).to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn((2048, 512), generator=generator)).to(
        torch.bfloat16
    )
    bias = (0.01 * torch.randn((512,), generator=generator)).to(torch.bfloat16)

    out = linear_mx_reference(input_bf16, weight, bias)
    expected = (input_bf16.float() @ weight.float() + bias.float()).to(torch.bfloat16)
    diff = (expected.float() - out.float()).abs()

    assert out.shape == (128, 512)
    assert torch.nn.functional.cosine_similarity(
        expected.float().reshape(1, -1), out.float().reshape(1, -1)
    ).item() >= 0.999
    assert diff.mean().item() <= 0.01
    assert diff.max().item() <= 0.05


def test_cpu_linear_mx_public_op_dispatches(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.ops.mx import linear_mx

    generator = torch.Generator().manual_seed(5)
    input_bf16 = (0.05 * torch.randn((128, 512), generator=generator)).to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn((512, 512), generator=generator)).to(torch.bfloat16)

    out = linear_mx(input_bf16, weight)

    assert out.shape == (128, 512)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("k_dim", [512, 2048])
def test_cpu_linear_mx_outer_n_reference_matches_logical_linear(monkeypatch, k_dim):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.backends.cpu.ops_impl.mx import linear_mx_outer_n_reference
    from difflet.ops.mx import linear_mx

    generator = torch.Generator().manual_seed(40 + k_dim)
    input_bf16 = (0.05 * torch.randn((128, k_dim), generator=generator)).to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn((k_dim, 2048), generator=generator)).to(
        torch.bfloat16
    )

    out = linear_mx_outer_n_reference(input_bf16, weight)
    dispatched = linear_mx(input_bf16, weight)
    expected = (input_bf16.float() @ weight.float()).to(torch.bfloat16)
    diff = (expected.float() - out.float()).abs()

    assert out.shape == (128, 2048)
    assert torch.equal(out, dispatched)
    assert torch.nn.functional.cosine_similarity(
        expected.float().reshape(1, -1), out.float().reshape(1, -1)
    ).item() >= 0.999
    assert diff.mean().item() <= 0.01
    assert diff.max().item() <= 0.05


def test_cpu_linear_mx_outer_n_adds_bias_by_tile(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.backends.cpu.ops_impl.mx import linear_mx_outer_n_reference

    generator = torch.Generator().manual_seed(43)
    input_bf16 = (0.05 * torch.randn((128, 512), generator=generator)).to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn((512, 1024), generator=generator)).to(torch.bfloat16)
    bias = (0.01 * torch.randn((1024,), generator=generator)).to(torch.bfloat16)

    out = linear_mx_outer_n_reference(input_bf16, weight, bias)
    no_bias = linear_mx_outer_n_reference(input_bf16, weight)
    expected = (no_bias.float() + bias.float()).to(torch.bfloat16)
    diff = (out.float() - expected.float()).abs()

    assert out.shape == (128, 1024)
    assert diff.max().item() <= 0.001


def test_pack_linear_mx_inputs_uses_activation_as_stationary(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.backends.cpu.ops_impl.mx import (
        matmul_mx_single_tile_reference,
        pack_linear_mx_inputs,
    )

    generator = torch.Generator().manual_seed(6)
    input_bf16 = (0.05 * torch.randn((128, 512), generator=generator)).to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn((512, 512), generator=generator)).to(torch.bfloat16)

    stationary_mx, stationary_scale, moving_mx, moving_scale = pack_linear_mx_inputs(
        input_bf16,
        weight,
    )
    out = matmul_mx_single_tile_reference(
        stationary_mx[0],
        stationary_scale[0],
        moving_mx[0],
        moving_scale[0],
    )
    expected = (input_bf16.float() @ weight.float()).to(torch.bfloat16)

    assert stationary_mx.shape == (1, 128, 128)
    assert moving_mx.shape == (1, 128, 512)
    assert torch.nn.functional.cosine_similarity(
        expected.float().reshape(1, -1), out.float().reshape(1, -1)
    ).item() >= 0.999


def test_cpu_quantize_rejects_unsupported_dtype(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.ops.mx import quantize_mx

    x = torch.ones((8, 4), dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="float8_e4m3fn_x4"):
        quantize_mx(x, dtype="float4_e2m1fn_x4")


def test_cpu_quantize_requires_supported_tile_shape(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    import torch

    from difflet.ops.mx import quantize_mx

    x = torch.ones((7, 4), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="multiples"):
        quantize_mx(x)
