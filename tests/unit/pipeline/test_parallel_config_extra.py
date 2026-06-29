"""Extra tests for difflet.pipeline.parallel_config covering validation,
world_size, and cache-dict branches."""

import pytest

from difflet.pipeline.parallel_config import CandidateConfig, DiffletParallelConfig


def test_default_world_size_is_one():
    cfg = DiffletParallelConfig()
    assert cfg.world_size == 1
    assert cfg.cp_mode == "gather_kv"


def test_world_size_multiplies_tp_cp():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=2)
    assert cfg.world_size == 8


def test_cfg_parallel_doubles_world_size():
    cfg = DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True)
    assert cfg.world_size == 4


def test_tp_degree_must_be_positive():
    with pytest.raises(ValueError):
        DiffletParallelConfig(tp_degree=0)


def test_cp_degree_must_be_positive():
    with pytest.raises(ValueError):
        DiffletParallelConfig(cp_degree=0)


def test_cp_and_cfg_mutually_exclusive():
    with pytest.raises(ValueError):
        DiffletParallelConfig(cp_degree=2, cfg_parallel_enabled=True)


def test_invalid_cp_mode_rejected():
    with pytest.raises(ValueError):
        DiffletParallelConfig(cp_mode="bogus")


def test_ring_requires_cp_degree_gt_one():
    with pytest.raises(ValueError):
        DiffletParallelConfig(cp_mode="ring", cp_degree=1)


def test_ring_valid_with_cp_degree():
    cfg = DiffletParallelConfig(cp_mode="ring", cp_degree=2)
    assert cfg.cp_mode == "ring"


def test_to_cache_dict_omits_default_cp_mode():
    d = DiffletParallelConfig(tp_degree=2).to_cache_dict()
    assert "cp_mode" not in d
    assert d["tp_degree"] == 2


def test_to_cache_dict_includes_ring_cp_mode():
    d = DiffletParallelConfig(cp_mode="ring", cp_degree=2).to_cache_dict()
    assert d["cp_mode"] == "ring"


def test_candidate_defaults_trivial():
    c = CandidateConfig()
    assert c.is_trivial is True
    assert c.active_candidates == 1


def test_candidate_active_defaults_to_max():
    c = CandidateConfig(max_candidates=4)
    assert c.active_candidates == 4
    assert c.is_trivial is False


def test_candidate_max_must_be_positive():
    with pytest.raises(ValueError):
        CandidateConfig(max_candidates=0)


def test_candidate_active_must_be_within_range():
    with pytest.raises(ValueError):
        CandidateConfig(max_candidates=2, active_candidates=5)
    with pytest.raises(ValueError):
        CandidateConfig(max_candidates=2, active_candidates=0)


def test_candidate_world_size_is_parallel_world_size():
    parallel = DiffletParallelConfig(tp_degree=4)
    c = CandidateConfig(max_candidates=8)
    assert c.world_size(parallel) == parallel.world_size


def test_candidate_to_cache_dict_only_max():
    c = CandidateConfig(max_candidates=4, active_candidates=2)
    assert c.to_cache_dict() == {"max_candidates": 4}
