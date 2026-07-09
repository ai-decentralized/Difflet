"""LTX-2 CFG collectives must be wired to the cfg axis ops, not dp.

The cfg axis is ONLY cond/uncond (size 2); STG's extra guidance branches are
rejected at module construction and never map onto the cfg axis.
"""

import inspect

import pytest

ltx2 = pytest.importorskip("difflet.backends.trainium.ltx_2.transformer")


def test_ltx2_uses_axis_ops_not_dp():
    src = inspect.getsource(ltx2)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "data_parallel_group" not in src
    assert "get_cfg_group" in src
    assert "get_cfg_rank_spmd" in src
    assert "init_parallel_mesh" in src
