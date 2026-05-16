import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("nki") is None,
    reason="NKI is not installed",
)
def test_matmul_mx_single_tile_kernel_simulates_zero_tile():
    import numpy as np
    from nki.simulator import simulate_kernel

    from nova.backends.trainium.nki_kernels.mx import matmul_mx_single_tile_kernel

    stationary = np.zeros((128, 128), dtype=np.uint32)
    stationary_scale = np.full((16, 128), 127, dtype=np.uint8)
    moving = np.zeros((128, 512), dtype=np.uint32)
    moving_scale = np.full((16, 512), 127, dtype=np.uint8)

    out = simulate_kernel(
        matmul_mx_single_tile_kernel,
        (stationary, stationary_scale, moving, moving_scale),
        {},
    )

    assert out.shape == (128, 512)
    assert np.max(np.abs(out.astype(np.float32))) == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("nki") is None,
    reason="NKI is not installed",
)
def test_matmul_mx_k_tiles_kernel_simulates_zero_tile():
    import numpy as np
    from nki.simulator import simulate_kernel

    from nova.backends.trainium.nki_kernels.mx import matmul_mx_k_tiles_kernel

    stationary = np.zeros((2, 128, 128), dtype=np.uint32)
    stationary_scale = np.full((2, 16, 128), 127, dtype=np.uint8)
    moving = np.zeros((2, 128, 512), dtype=np.uint32)
    moving_scale = np.full((2, 16, 512), 127, dtype=np.uint8)

    out = simulate_kernel(
        matmul_mx_k_tiles_kernel,
        (stationary, stationary_scale, moving, moving_scale),
        {},
    )

    assert out.shape == (128, 512)
    assert np.max(np.abs(out.astype(np.float32))) == 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("nki") is None,
    reason="NKI is not installed",
)
@pytest.mark.parametrize("k_tiles", [2, 8])
def test_matmul_mx_k_tiles_kernel_simulates_numerical_parity(k_tiles):
    import numpy as np
    import torch
    from nki.simulator import simulate_kernel

    from nova.backends.cpu.ops_impl.mx import (
        matmul_mx_k_tiles_reference,
        quantize_mx,
    )
    from nova.backends.trainium.nki_kernels.mx import matmul_mx_k_tiles_kernel

    generator = torch.Generator().manual_seed(k_tiles)
    stationary = (0.1 * torch.randn((k_tiles, 128, 512), generator=generator)).to(
        torch.bfloat16
    )
    moving = (0.1 * torch.randn((k_tiles, 128, 2048), generator=generator)).to(
        torch.bfloat16
    )
    stationary_mx, stationary_scale = quantize_mx(stationary)
    moving_mx, moving_scale = quantize_mx(moving)
    expected = matmul_mx_k_tiles_reference(
        stationary_mx,
        stationary_scale,
        moving_mx,
        moving_scale,
        out_dtype=torch.float32,
    )

    out = simulate_kernel(
        matmul_mx_k_tiles_kernel,
        (
            stationary_mx.numpy(),
            stationary_scale.numpy(),
            moving_mx.numpy(),
            moving_scale.numpy(),
        ),
        {},
    )
    observed = torch.from_numpy(out).to(torch.bfloat16).float()
    expected = expected.to(torch.bfloat16).float()
    diff = (expected - observed).abs()

    assert torch.nn.functional.cosine_similarity(
        expected.reshape(1, -1), observed.reshape(1, -1)
    ).item() >= 0.999
    assert diff.mean().item() <= 0.01
    assert diff.max().item() <= 0.05


@pytest.mark.skipif(
    importlib.util.find_spec("nki") is None,
    reason="NKI is not installed",
)
def test_matmul_mx_k_tiles_kernel_simulates_logical_linear_parity():
    import torch
    from nki.simulator import simulate_kernel

    from nova.backends.cpu.ops_impl.mx import (
        linear_mx_reference,
        pack_linear_mx_inputs,
    )
    from nova.backends.trainium.nki_kernels.mx import matmul_mx_k_tiles_kernel

    generator = torch.Generator().manual_seed(7)
    input_bf16 = (0.05 * torch.randn((128, 2048), generator=generator)).to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn((2048, 512), generator=generator)).to(
        torch.bfloat16
    )
    bias = (0.01 * torch.randn((512,), generator=generator)).to(torch.bfloat16)
    expected = linear_mx_reference(input_bf16, weight, bias)
    packed = pack_linear_mx_inputs(input_bf16, weight, bias)

    out = simulate_kernel(
        matmul_mx_k_tiles_kernel,
        tuple(tensor.numpy() for tensor in packed),
        {},
    )
    observed = (torch.from_numpy(out).float() + bias.float()).to(torch.bfloat16)
    diff = (expected.float() - observed.float()).abs()

    assert torch.nn.functional.cosine_similarity(
        expected.float().reshape(1, -1), observed.float().reshape(1, -1)
    ).item() >= 0.999
    assert diff.mean().item() <= 0.01
    assert diff.max().item() <= 0.05


@pytest.mark.skipif(
    importlib.util.find_spec("nki") is None,
    reason="NKI is not installed",
)
@pytest.mark.parametrize("k_dim", [512, 2048])
def test_matmul_mx_k_tiles_kernel_simulates_outer_n_linear_parity(k_dim):
    import torch
    from nki.simulator import simulate_kernel

    from nova.backends.cpu.ops_impl.mx import (
        linear_mx_outer_n_reference,
        pack_linear_mx_inputs,
    )
    from nova.backends.trainium.nki_kernels.mx import matmul_mx_k_tiles_kernel

    generator = torch.Generator().manual_seed(70 + k_dim)
    input_bf16 = (0.05 * torch.randn((128, k_dim), generator=generator)).to(
        torch.bfloat16
    )
    weight = (0.05 * torch.randn((k_dim, 2048), generator=generator)).to(
        torch.bfloat16
    )
    expected = linear_mx_outer_n_reference(input_bf16, weight)

    outputs = []
    for n_start in range(0, weight.shape[1], 512):
        packed = pack_linear_mx_inputs(input_bf16, weight[:, n_start : n_start + 512])
        out = simulate_kernel(
            matmul_mx_k_tiles_kernel,
            tuple(tensor.numpy() for tensor in packed),
            {},
        )
        outputs.append(torch.from_numpy(out).to(torch.bfloat16))
    observed = torch.cat(outputs, dim=1)
    diff = (expected.float() - observed.float()).abs()

    assert observed.shape == (128, 2048)
    assert torch.nn.functional.cosine_similarity(
        expected.float().reshape(1, -1), observed.float().reshape(1, -1)
    ).item() >= 0.999
    assert diff.mean().item() <= 0.01
    assert diff.max().item() <= 0.05


@pytest.mark.skipif(
    importlib.util.find_spec("nki") is None,
    reason="NKI is not installed",
)
def test_quantize_mx_single_tile_kernel_simulates_zero_tile():
    import numpy as np
    from nki.simulator import simulate_kernel

    from nova.backends.trainium.nki_kernels.mx import (
        quantize_mx_e4m3_single_tile_kernel,
    )

    x = np.zeros((128, 512), dtype=np.float16)

    data, scale = simulate_kernel(quantize_mx_e4m3_single_tile_kernel, (x,), {})

    assert data.shape == (128, 128)
    assert scale.shape == (16, 128)
