"""MXFP8 law and layout against nkilib's own numpy reference (``_swizzle``, ``quantize_to_mx``)
when the Neuron SDK is installed; the pure-torch properties always."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from difflet.quant import mx
from difflet.quant.fp8 import FP8_DTYPE

nkilib_torch = pytest.importorskip("nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch",
                                   reason="nkilib (Neuron SDK) is not installed")
from nkilib.core.utils.mx_torch_common import quantize_to_mx  # noqa: E402
import nki.language as nl  # noqa: E402


@pytest.mark.parametrize("k", [512, 3072, 768, 128, 256, 1536])
def test_swizzle_perm_matches_nkilib(k):
    """``t[:, perm]`` reshaped to ``[R, K/4, 4]`` is nkilib's ``_swizzle(t.T)`` transposed."""
    f = 8
    t = torch.arange(k * f, dtype=torch.float32).reshape(f, k)
    perm = mx.swizzle_perm(k)
    assert sorted(perm.tolist()) == list(range(k))
    ours = t[:, perm].reshape(f, k // 4, 4).permute(1, 0, 2).reshape(k // 4, f * 4)
    theirs = nkilib_torch._swizzle(t.t().contiguous().numpy())
    assert np.array_equal(ours.numpy(), theirs)


def test_quantize_mx_matches_nkilib_on_the_swizzled_weight():
    torch.manual_seed(0)
    n, k = 16, 1536
    w = torch.randn(n, k) * 0.05
    w[3, 100] = 4.0  # an outlier in one block only
    q, s = mx.quantize_mx(w)
    assert q.dtype == FP8_DTYPE and q.shape == (n, k)
    assert s.dtype == torch.uint8 and s.shape == (n, k // 32)
    data, scales_hw = mx.kernel_layout(q, s)
    assert data.dtype == torch.int32 and data.shape == (k // 4, n)
    assert scales_hw.shape == (k // 4, n)

    sw = nkilib_torch._swizzle(w.t().contiguous().numpy())            # [K/4, N*4]
    packed, scale_block = quantize_to_mx(sw, nl.float8_e4m3fn_x4)   # [K/4, N] x4, [K/32, N]
    theirs = torch.from_numpy(np.ascontiguousarray(packed).view(np.int32).copy())
    assert torch.equal(data, theirs)
    assert np.array_equal(s.t().numpy(), scale_block)
    # Hardware quadrant layout: four rows per 32 partitions.
    assert np.array_equal(scales_hw[0:4].numpy(), scale_block[0:4])
    assert np.array_equal(scales_hw[32:36].numpy(), scale_block[4:8])
    assert int(scales_hw[4:32].abs().sum()) == 0


def test_dequantize_round_trip_and_outlier_isolation():
    torch.manual_seed(0)
    x = torch.randn(4, 3072)
    x[0, 7] = 1000.0
    back = mx.fake_quantize_mx(x)
    rel = ((back - x).norm() / x.norm()).item()
    assert rel < 0.05
    # The outlier's block is coarse; a block far from it keeps fp8's ~3-bit precision.
    far = slice(2048, 2048 + 32)
    assert ((back[1, far] - x[1, far]).abs() / x[1, far].abs()).max().item() < 0.07
    # Static per-tensor at the same amax would zero most of the row instead.
    assert (back[0] != 0).float().mean().item() > 0.9


def test_blocks_are_the_swizzled_groups_not_contiguous_k():
    k = 512
    perm = mx.swizzle_perm(k)
    x = torch.zeros(1, k)
    x[0, perm[:32]] = 1.0  # one hardware block
    x[0, perm[0]] = 100.0
    q, s = mx.quantize_mx(x)
    assert (s[0] != s[0, 1]).sum().item() == 1, "exactly one block scaled up"
    assert s[0, 0] > s[0, 1]
