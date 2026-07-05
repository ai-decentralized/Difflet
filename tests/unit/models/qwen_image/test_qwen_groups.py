"""Qwen-Image CP collectives must be wired to the cp axis ops, not dp."""

import inspect

import pytest

qwen = pytest.importorskip("difflet.backends.trainium.qwen_image.transformer")


def test_qwen_uses_axis_ops_not_dp():
    src = inspect.getsource(qwen)
    assert "get_data_parallel_group" not in src
    assert "get_dp_rank_spmd" not in src
    assert "data_parallel_group" not in src
    assert "get_cp_group" in src
    assert "get_cp_rank_spmd" in src
    assert "init_parallel_mesh" in src
