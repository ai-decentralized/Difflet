"""scripts/mx_attention/membership.py: decoding the device's quantize_mx tiles and comparing them
with the CPU quantizer under a block membership. No device: the tiles are encoded here from the
CPU quantizer in the layout the script assumes (byte j of row r is index perm[4r + j]; each block's
scale in its quadrant's first four rows)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from difflet.quant.mx import quantize_mx, swizzle_perm

REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "mx_attention_membership", REPO / "scripts" / "mx_attention" / "membership.py")
membership = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(membership)


def _device_tiles(x: torch.Tensor, perm: torch.Tensor):
    fp8, scales = quantize_mx(x, perm)
    m, k = x.shape
    data = (fp8.view(torch.uint8).view(m, k // 4, 4).permute(1, 0, 2).contiguous()
            .view(torch.int32).squeeze(-1))
    tile = torch.zeros(k // 4, m, dtype=torch.uint8)
    tile[membership.scale_rows(k)] = scales.t()
    return data, tile


def test_scale_rows_are_the_quadrant_layout():
    assert membership.scale_rows(512).tolist() == [0, 1, 2, 3, 32, 33, 34, 35, 64, 65, 66, 67,
                                                   96, 97, 98, 99]
    assert membership.scale_rows(128).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("name", sorted(membership.OPERANDS))
def test_compare_tells_the_layouts_membership_from_another(name):
    x = membership.logical(name, membership.make_inputs())
    contiguous = torch.arange(x.shape[1])
    data, tile = _device_tiles(x, contiguous)
    same = membership.compare(x, data, tile, contiguous)
    assert same == {"scales": x.numel() // 32, "scale_mismatches": 0, "fp8": x.numel(),
                    "fp8_mismatches": 0, "nonzero_scale_rows_outside": 0}
    other = membership.compare(x, data, tile, swizzle_perm(x.shape[1]))
    assert other["scale_mismatches"] > other["scales"] // 4
    assert other["fp8_mismatches"] > other["fp8"] // 4


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="NKI is not installed")
def test_the_kernels_loaders_simulated_match_the_cpu_quantizer():
    """The membership kernel runs the fork's own MM2 loaders (``_mx_quantize_p``, ``_mx_load_v``)
    on the NKI simulator: every operand, the partial 500-key tiles included (keys past the end
    zero), matches ``quantize_mx`` under the kernel's perms bit for bit, and not under
    ``swizzle_perm``."""
    from difflet.backends.neuron.ops_impl.mx_attention_kernel import pv_perm, qk_perm

    import numpy as np
    from nki._simulator import simulate_kernel

    inputs = membership.make_inputs()
    outs = simulate_kernel(membership.quantize_operands,
                           tuple(membership.to_numpy(inputs[n]) for n in membership.SHAPES), {})
    got = membership.unpack([torch.from_numpy(np.asarray(t)) for t in outs])
    assert set(got) == set(membership.OPERANDS)
    for name, (data, scales) in got.items():
        x = membership.logical(name, inputs)
        r = membership.compare(x, data, scales, (qk_perm if name == "q" else pv_perm)(x.shape[1]))
        assert r["scale_mismatches"] == 0 and r["fp8_mismatches"] == 0, (name, r)
        assert r["nonzero_scale_rows_outside"] == 0, (name, r)
        assert membership.compare(x, data, scales, swizzle_perm(x.shape[1]))["scale_mismatches"] > 0
