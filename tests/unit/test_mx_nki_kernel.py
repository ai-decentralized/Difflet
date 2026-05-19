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
        (stationary, stationary_scale, moving, moving_scale, stationary.shape[0]),
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
            k_tiles,
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
        (*tuple(tensor.numpy() for tensor in packed), packed[0].shape[0]),
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
            (*tuple(tensor.numpy() for tensor in packed), packed[0].shape[0]),
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
def test_linear_mx_prequant_kernel_simulates_logical_linear_parity():
    import torch
    from nki.simulator import simulate_kernel

    from nova.backends.cpu.ops_impl.mx import linear_mx_reference, quantize_mx
    from nova.backends.trainium.nki_kernels.mx import linear_mx_prequant_kernel

    generator = torch.Generator().manual_seed(91)
    input_fp16 = (0.05 * torch.randn((128, 2048), generator=generator)).to(torch.float16)
    weight = (0.05 * torch.randn((2048, 512), generator=generator)).to(torch.float16)

    moving_native_tiles = []
    for k_start in range(0, input_fp16.shape[1], 512):
        weight_tile = weight[k_start : k_start + 512, :].contiguous()
        moving_native_tiles.append(
            weight_tile.reshape(128, 4, 512)
            .permute(0, 2, 1)
            .reshape(128, 512 * 4)
            .contiguous()
        )

    moving_native = torch.stack(moving_native_tiles, dim=0)
    moving_mx, moving_scale = quantize_mx(moving_native)
    expected = linear_mx_reference(input_fp16, weight, out_dtype=torch.float32)

    out = simulate_kernel(
        linear_mx_prequant_kernel,
        (
            input_fp16.numpy(),
            moving_mx.numpy(),
            moving_scale.numpy(),
            input_fp16.shape[1] // 512,
            input_fp16.shape[1],
            "float16",
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
def test_linear_mx_prequant_group2_kernel_simulates_two_linear_parity():
    import torch
    from nki.simulator import simulate_kernel

    from nova.backends.cpu.ops_impl.mx import linear_mx_reference, quantize_mx
    from nova.backends.trainium.nki_kernels.mx import linear_mx_prequant_group2_kernel

    generator = torch.Generator().manual_seed(97)
    input_fp16 = (0.05 * torch.randn((128, 2048), generator=generator)).to(torch.float16)
    weight0 = (0.05 * torch.randn((2048, 512), generator=generator)).to(torch.float16)
    weight1 = (0.05 * torch.randn((2048, 512), generator=generator)).to(torch.float16)

    moving_args = []
    for weight in (weight0, weight1):
        moving_native_tiles = []
        for k_start in range(0, input_fp16.shape[1], 512):
            weight_tile = weight[k_start : k_start + 512, :].contiguous()
            moving_native_tiles.append(
                weight_tile.reshape(128, 4, 512)
                .permute(0, 2, 1)
                .reshape(128, 512 * 4)
                .contiguous()
            )
        moving_native = torch.stack(moving_native_tiles, dim=0)
        moving_args.append(quantize_mx(moving_native))

    expected0 = linear_mx_reference(input_fp16, weight0, out_dtype=torch.float32)
    expected1 = linear_mx_reference(input_fp16, weight1, out_dtype=torch.float32)

    out0, out1 = simulate_kernel(
        linear_mx_prequant_group2_kernel,
        (
            input_fp16.numpy(),
            moving_args[0][0].numpy(),
            moving_args[0][1].numpy(),
            moving_args[1][0].numpy(),
            moving_args[1][1].numpy(),
            input_fp16.shape[1] // 512,
            input_fp16.shape[1],
            "float16",
        ),
        {},
    )

    for expected, observed_np in ((expected0, out0), (expected1, out1)):
        observed = torch.from_numpy(observed_np).to(torch.bfloat16).float()
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
def test_linear_mx_prequant_native_weight_kernel_simulates_parity():
    import torch
    from nki.simulator import simulate_kernel

    from nova.backends.cpu.ops_impl.mx import linear_mx_reference, quantize_mx
    from nova.backends.trainium.ltx_2.segmented import _expand_compact_mx_scale_to_native
    from nova.backends.trainium.nki_kernels.mx import linear_mx_prequant_native_weight_kernel

    generator = torch.Generator().manual_seed(101)
    input_fp16 = (0.05 * torch.randn((128, 2048), generator=generator)).to(torch.float16)
    weight = (0.05 * torch.randn((2048, 512), generator=generator)).to(torch.float16)

    moving_tiles = []
    moving_scales = []
    for k_start in range(0, input_fp16.shape[1], 512):
        weight_tile = weight[k_start : k_start + 512, :].contiguous()
        weight_native = (
            weight_tile.reshape(128, 4, 512)
            .permute(0, 2, 1)
            .reshape(128, 512 * 4)
            .contiguous()
        )
        moving_mx, moving_scale = quantize_mx(weight_native)
        moving_tiles.append(moving_mx)
        moving_scales.append(_expand_compact_mx_scale_to_native(moving_scale))

    expected = linear_mx_reference(input_fp16, weight, out_dtype=torch.float32)
    out = simulate_kernel(
        linear_mx_prequant_native_weight_kernel,
        (
            input_fp16.numpy(),
            torch.stack(moving_tiles, dim=0).numpy(),
            torch.stack(moving_scales, dim=0).numpy(),
            input_fp16.shape[1] // 512,
            input_fp16.shape[1],
            "float16",
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
def test_quantize_mx_linear_activation_kernel_simulates_cpu_pack_parity():
    import torch
    from nki.simulator import simulate_kernel

    from nova.backends.cpu.ops_impl.mx import quantize_mx
    from nova.backends.trainium.nki_kernels.mx import quantize_mx_linear_activation_kernel

    generator = torch.Generator().manual_seed(93)
    input_fp16 = (0.05 * torch.randn((128, 2048), generator=generator)).to(torch.float16)

    activation_native_tiles = []
    for k_start in range(0, input_fp16.shape[1], 512):
        input_tile = input_fp16[:, k_start : k_start + 512].contiguous()
        activation_native_tiles.append(
            input_tile.reshape(128, 128, 4)
            .permute(1, 0, 2)
            .reshape(128, 128 * 4)
            .contiguous()
        )

    activation_native = torch.stack(activation_native_tiles, dim=0)
    expected_mx, expected_scale = quantize_mx(activation_native)

    observed_mx, observed_scale = simulate_kernel(
        quantize_mx_linear_activation_kernel,
        (
            input_fp16.numpy(),
            input_fp16.shape[1] // 512,
            input_fp16.shape[1],
            "float16",
        ),
        {},
    )

    assert torch.equal(torch.from_numpy(observed_mx), expected_mx)
    assert torch.equal(torch.from_numpy(observed_scale), expected_scale)


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

    data, scale = simulate_kernel(
        quantize_mx_e4m3_single_tile_kernel,
        (x, x.shape[1], "float16"),
        {},
    )

    assert data.shape == (128, 128)
    assert scale.shape == (16, 128)
