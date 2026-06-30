"""Config-level coverage for Megatron-style sequence parallelism (`sp_enabled`)."""

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig


def test_sp_defaults_to_disabled():
    cfg = DiffletParallelConfig(tp_degree=4)
    assert cfg.sp_enabled is False


def test_sp_enabled_is_accepted():
    cfg = DiffletParallelConfig(tp_degree=4, sp_enabled=True)
    assert cfg.sp_enabled is True


def test_sp_does_not_change_world_size():
    # SP reuses the TP group; it must not introduce a new world-size axis.
    base = DiffletParallelConfig(tp_degree=4)
    with_sp = DiffletParallelConfig(tp_degree=4, sp_enabled=True)
    assert with_sp.world_size == base.world_size == 4


def test_sp_with_cfg_parallel_keeps_cfg_world_size():
    cfg = DiffletParallelConfig(tp_degree=4, cfg_parallel_enabled=True, sp_enabled=True)
    assert cfg.world_size == 8  # cfg doubles; sp does not multiply


def test_sp_and_cp_degree_gt_1_are_mutually_exclusive():
    with pytest.raises(ValueError, match="sp_enabled and cp_degree > 1 are mutually exclusive"):
        DiffletParallelConfig(tp_degree=4, cp_degree=2, sp_enabled=True)


def test_sp_with_cp_degree_1_is_valid():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=1, sp_enabled=True)
    assert cfg.sp_enabled is True


def test_sp_allowed_with_tp_degree_1_for_host_tests():
    # On the CPU backend tp==1 and SP is a numeric no-op; constructing it must
    # not raise so the SP path can be exercised on a host.
    cfg = DiffletParallelConfig(tp_degree=1, sp_enabled=True)
    assert cfg.sp_enabled is True


def test_cache_dict_omits_sp_when_disabled():
    cfg = DiffletParallelConfig(tp_degree=4, sp_enabled=False)
    assert "sp_enabled" not in cfg.to_cache_dict()


def test_cache_dict_includes_sp_when_enabled():
    cfg = DiffletParallelConfig(tp_degree=4, sp_enabled=True)
    assert cfg.to_cache_dict()["sp_enabled"] is True


def test_cache_dict_default_is_byte_identical_to_legacy_keys():
    # A default (SP-off) config must not leak the new key into the cache hash.
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=1)
    assert set(cfg.to_cache_dict()) == {"tp_degree", "cp_degree", "cfg_parallel_enabled"}


def test_cache_dict_sp_enabled_is_additive_over_legacy():
    cfg = DiffletParallelConfig(tp_degree=4, sp_enabled=True)
    assert set(cfg.to_cache_dict()) == {
        "tp_degree",
        "cp_degree",
        "cfg_parallel_enabled",
        "sp_enabled",
    }
