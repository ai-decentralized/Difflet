"""Wan CFG/CP collectives must be wired to the cfg/cp axis ops, not dp.

Static wiring check on the module source: after the mesh refactor the CFG
merge rides ``cfg_group`` (size 2) and CP rides ``cp_group``; nothing may
reference the legacy merged-axis dp helpers.
"""

import inspect

import difflet.models.wan.modeling_wan as wan


def test_wan_uses_axis_ops_not_dp():
    src = inspect.getsource(wan)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "data_parallel_group" not in src
    assert "get_cfg_group" in src
    assert "get_cp_group" in src
    assert "get_cfg_rank_spmd" in src
    assert "get_cp_rank_spmd" in src
    assert "init_parallel_mesh" in src
