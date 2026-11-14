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
    assert same == {"scales": x.numel() // 32, "scale_mismatches": 0, "zero_block_scale_mismatches": 0,
                    "fp8": x.numel(), "fp8_mismatches": 0, "nonzero_scale_rows_outside": 0}


def test_compare_counts_the_all_zero_blocks_scale_apart():
    """The device writes 2^-126 (byte 1) as an all-zero block's scale, the CPU quantizer 2^-127
    (byte 0); its data bytes are zero either way. compare() counts those mismatches apart."""
    x = membership.logical("p", membership.make_inputs())
    x[3, 64:96] = 0  # block 2 of row 3, contiguous membership
    contiguous = torch.arange(x.shape[1])
    data, tile = _device_tiles(x, contiguous)
    assert tile[membership.scale_rows(x.shape[1])[2], 3] == 0
    tile[membership.scale_rows(x.shape[1])[2], 3] = 1
    r = membership.compare(x, data, tile, contiguous)
    assert r["scale_mismatches"] == r["zero_block_scale_mismatches"] == 1 and r["fp8_mismatches"] == 0
    other = membership.compare(x, data, tile, swizzle_perm(x.shape[1]))
    assert other["scale_mismatches"] > other["scales"] // 4
    assert other["fp8_mismatches"] > other["fp8"] // 4


def test_mm1_operands_split_into_the_quadrants_mm1_reads():
    """Each MM1 tile is checked per row quadrant: Q's group in every quadrant, K's 512-key tile t
    in quadrant t, rows past the valid count zero; the inputs pair zeros (+0 and -0) and bf16
    subnormals with normal values in the fp32 view the loaders transpose."""
    inputs = membership.make_inputs()
    kq = inputs["kq"]
    assert (kq[0::3, 1::2] == 0).all() and (kq[1::3, 1::4].view(torch.int16) == -32768).all()
    assert ((kq[2::3, 1::2].float().abs() < 1.18e-38) & (kq[2::3, 1::2] != 0)).all()
    assert (kq[:, 0::2] != 0).all()
    for name, rows in (("q_mm1", 128), ("q_mm1_tail", 128), ("k_mm1", 2048), ("k_mm1_tail", 2048)):
        f = rows // 4 if name.startswith("k") else 128
        data, scales = torch.zeros(128, f, dtype=torch.int32), torch.zeros(128, f, dtype=torch.uint8)
        tiles = membership.tiles(name, inputs, data, scales)
        assert len(tiles) == 4
        valid = membership.OPERANDS[name][1]
        for t, (x, d, s) in enumerate(tiles):
            assert d.shape == s.shape == (32, f) and x.shape == (f, 128)
            lo = 0 if name.startswith("q") else 512 * t
            expect = kq[lo:lo + f].clone()
            expect[max(valid - lo, 0):] = 0
            assert torch.equal(x, expect), (name, t)


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="NKI is not installed")
def test_the_kernels_loaders_simulated_match_the_cpu_quantizer():
    """The membership kernel runs the fork's own MM2 loaders (``_mx_quantize_p``, ``_mx_load_v``)
    and MM1 loaders (``_mx_load_q``, ``_mx_load_k``) on the NKI simulator: every operand and, for
    MM1, every row quadrant, the partial tiles included (rows or keys past the end zero), matches
    ``quantize_mx`` under the kernel's perms bit for bit, and not under ``swizzle_perm``."""
    from difflet.backends.neuron.ops_impl.mx_attention_kernel import pv_perm, qk_perm

    import numpy as np
    from nki._simulator import simulate_kernel

    inputs = membership.make_inputs()
    outs = simulate_kernel(membership.quantize_operands,
                           tuple(membership.to_numpy(inputs[n]) for n in membership.SHAPES), {})
    got = membership.unpack([torch.from_numpy(np.asarray(t)) for t in outs])
    assert set(got) == set(membership.OPERANDS)
    for name, (data, scales) in got.items():
        perm = qk_perm if name == "q" or name in membership.MM1 else pv_perm
        for x, d, s in membership.tiles(name, inputs, data, scales):
            r = membership.compare(x, d, s, perm(x.shape[1]))
            assert r["scale_mismatches"] == 0 and r["fp8_mismatches"] == 0, (name, r)
            assert r["nonzero_scale_rows_outside"] == 0, (name, r)
            if x.any():  # an all-zero tile (k_mm1_tail's fourth) quantizes alike under any perm
                assert membership.compare(x, d, s, swizzle_perm(x.shape[1]))["scale_mismatches"] > 0
