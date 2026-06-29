import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig


def test_cp_mode_defaults_to_gather_kv():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=1)
    assert cfg.cp_mode == "gather_kv"


def test_cp_mode_ring_requires_cp_degree_gt_1():
    with pytest.raises(ValueError, match="cp_mode='ring' requires cp_degree > 1"):
        DiffletParallelConfig(tp_degree=4, cp_degree=1, cp_mode="ring")


def test_cp_mode_ring_with_cp_degree_2_is_valid():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="ring")
    assert cfg.cp_mode == "ring"


def test_cp_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="cp_mode must be one of"):
        DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="bogus")


def test_cache_dict_omits_cp_mode_when_gather_kv():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="gather_kv")
    assert "cp_mode" not in cfg.to_cache_dict()


def test_cache_dict_includes_cp_mode_when_ring():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="ring")
    assert cfg.to_cache_dict()["cp_mode"] == "ring"


def test_cache_dict_default_is_byte_identical_to_legacy_keys():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=1)
    assert set(cfg.to_cache_dict()) == {"tp_degree", "cp_degree", "cfg_parallel_enabled"}
