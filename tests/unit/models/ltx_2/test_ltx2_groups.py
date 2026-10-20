"""LTX-2 CFG collectives must be wired to the cfg axis ops, not dp.

The cfg axis is ONLY cond/uncond (size 2); STG's extra guidance branches are
rejected at module construction and never map onto the cfg axis.
"""

import inspect

import pytest

# The trace module (and its CFG split) moved to the backend-neutral modeling module; the
# Trainium transformer module re-exports it.
ltx2 = pytest.importorskip("difflet.models.ltx_2.modeling_ltx_2")


def test_ltx2_uses_axis_ops_not_dp():
    src = inspect.getsource(ltx2)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "data_parallel_group" not in src
    assert "get_cfg_group" in src
    assert "get_cfg_rank_spmd" in src
    assert "init_parallel_mesh" in src
