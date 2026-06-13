"""CP-degree generalization of NovaParallelConfig.

cp_degree is additive: default (cp_degree=1) is byte-identical to legacy,
cp_enabled=True remains a back-compat alias for cp_degree=2, and world_size
scales as tp_degree * cp_degree.
"""

import pytest

from nova.pipeline.parallel_config import NovaParallelConfig


def test_default_has_cp_degree_one_and_disabled():
    cfg = NovaParallelConfig(tp_degree=4)
    assert cfg.cp_degree == 1
    assert cfg.cp_enabled is False
    assert cfg.world_size == 4


def test_cp_enabled_true_aliases_to_degree_two():
    cfg = NovaParallelConfig(tp_degree=4, cp_enabled=True)
    assert cfg.cp_degree == 2
    assert cfg.cp_enabled is True
    assert cfg.world_size == 8


def test_cp_degree_sets_enabled_and_world_size():
    cfg = NovaParallelConfig(tp_degree=4, cp_degree=4)
    assert cfg.cp_enabled is True
    assert cfg.world_size == 16


def test_cp_degree_one_explicit_is_disabled():
    cfg = NovaParallelConfig(tp_degree=2, cp_degree=1)
    assert cfg.cp_enabled is False
    assert cfg.world_size == 2


def test_cp_degree_below_one_rejected():
    with pytest.raises(ValueError, match="cp_degree must be >= 1"):
        NovaParallelConfig(tp_degree=2, cp_degree=0)


def test_cp_and_cfg_mutually_exclusive_via_degree():
    with pytest.raises(ValueError, match="mutually exclusive"):
        NovaParallelConfig(tp_degree=2, cp_degree=2, cfg_parallel_enabled=True)


def test_conflicting_cp_enabled_and_degree_rejected():
    # cp_enabled=True with an explicit cp_degree>2 is contradictory.
    with pytest.raises(ValueError, match="cp_enabled is a degree-2 alias"):
        NovaParallelConfig(tp_degree=2, cp_enabled=True, cp_degree=4)


def test_cache_dict_byte_identical_to_legacy_when_disabled():
    # cp_degree=1 default must not perturb the compile-cache key.
    legacy = {"tp_degree": 4, "cp_enabled": False, "cfg_parallel_enabled": False}
    assert NovaParallelConfig(tp_degree=4).to_cache_dict() == legacy


def test_cache_dict_includes_degree_when_enabled():
    d = NovaParallelConfig(tp_degree=4, cp_degree=4).to_cache_dict()
    assert d["cp_degree"] == 4
