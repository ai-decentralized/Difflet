"""scripts/mx_attention/row_tiling.py (plan Task 4 Step 1): the micro-kernel's operand layouts, its
reference and the kernel itself on the NKI simulator (CPU; the device and BIRSim
runs are evidence, not tests)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "mx_attention_row_tiling", REPO / "scripts" / "mx_attention" / "row_tiling.py")
rt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rt)


def test_tile_t_sits_in_quadrant_t_in_the_group_of_4_layout():
    """Partition 32 t + g, element 4 m + j = row m of tile t, head-dim index 4 g + j."""
    q, k = rt.make_inputs()
    qs, ks = rt.operands("keys", q, k)
    assert qs.shape == (128, 512) and ks.shape == (128, 2048)
    t, g, n, j = 3, 5, 7, 2
    assert ks[32 * t + g, 4 * n + j] == k[512 * t + n, 4 * g + j]
    assert qs[32 * t + g, 4 * n + j] == q[n, 4 * g + j]  # the same q tile in every quadrant
    qs, ks = rt.operands("queries", q, k)
    assert qs[32 * t + g, 4 * n + j] == q[128 * t + n, 4 * g + j]
    assert ks[32 * t + g, 4 * n + j] == k[n, 4 * g + j]


@pytest.mark.parametrize("variant", sorted(rt.SHAPES))
def test_the_simulated_kernel_matches_the_reference(variant, tmp_path):
    pytest.importorskip("nki")
    import numpy as np

    q, k = rt.make_inputs()
    got = torch.from_numpy(rt.run("sim", variant, 2, False, tmp_path).astype(np.float64))
    ref = rt.reference(variant, q, k, 2)
    assert ((got - ref).norm() / ref.norm()).item() < 1e-6
    if variant != "bf16":  # and it is the MX product, not the bf16 one
        exact = rt.reference(variant, q, k, 2, quantized=False)
        assert ((got - exact).norm() / exact.norm()).item() > 1e-2
