"""Flux CFG/CP collectives must be wired to the cfg/cp axis ops, not dp.

The dormant Python-API CFG-parallel path stays in the code (policy: FLUX.1-dev
is guidance-distilled and the CLI blocks it) but it must ride cfg_group.
"""

import inspect

import difflet.models.flux.modeling_flux as flux


def test_flux_uses_axis_ops_not_dp():
    src = inspect.getsource(flux)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "data_parallel_group" not in src
    assert "get_cfg_group" in src
    assert "get_cp_group" in src
    assert "get_cfg_rank_spmd" in src
    assert "get_cp_rank_spmd" in src
    assert "init_parallel_mesh" in src
