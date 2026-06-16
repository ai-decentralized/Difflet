"""NovaParallelConfig cp_degree tests.

cp_degree=1 means disabled; world_size scales as tp_degree * cp_degree.
"""

import pytest

from nova.pipeline.parallel_config import NovaParallelConfig


def test_default_has_cp_degree_one():
    cfg = NovaParallelConfig(tp_degree=4)
    assert cfg.cp_degree == 1
    assert cfg.world_size == 4


def test_cp_degree_two_sets_world_size():
    cfg = NovaParallelConfig(tp_degree=4, cp_degree=2)
    assert cfg.cp_degree == 2
    assert cfg.world_size == 8


def test_cp_degree_sets_world_size():
    cfg = NovaParallelConfig(tp_degree=4, cp_degree=4)
    assert cfg.world_size == 16


def test_cp_degree_one_explicit_is_disabled():
    cfg = NovaParallelConfig(tp_degree=2, cp_degree=1)
    assert cfg.world_size == 2


def test_cp_degree_below_one_rejected():
    with pytest.raises(ValueError, match="cp_degree must be >= 1"):
        NovaParallelConfig(tp_degree=2, cp_degree=0)


def test_cp_and_cfg_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        NovaParallelConfig(tp_degree=2, cp_degree=2, cfg_parallel_enabled=True)


def test_cache_dict_includes_all_fields():
    d = NovaParallelConfig(tp_degree=4).to_cache_dict()
    assert d == {"tp_degree": 4, "cp_degree": 1, "cfg_parallel_enabled": False}


def test_cache_dict_reflects_cp_degree():
    d = NovaParallelConfig(tp_degree=4, cp_degree=4).to_cache_dict()
    assert d["cp_degree"] == 4
