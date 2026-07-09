"""HunyuanVideo CP collectives must be wired to the cp axis ops, not dp."""

import inspect

import difflet.models.hunyuan_video.modeling_hunyuan_video as hyv


def test_hunyuan_uses_axis_ops_not_dp():
    src = inspect.getsource(hyv)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "data_parallel_group" not in src
    assert "get_cp_group" in src
    assert "get_cp_rank_spmd" in src
    assert "init_parallel_mesh" in src
