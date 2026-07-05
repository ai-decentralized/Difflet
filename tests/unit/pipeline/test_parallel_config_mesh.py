"""dp_degree axis + mesh_spec derivation on DiffletParallelConfig."""

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.parallel_mesh import MeshSpec


def test_default_dp_degree_is_one_and_cache_key_unchanged():
    cfg = DiffletParallelConfig(tp_degree=2)
    assert cfg.dp_degree == 1
    assert "dp_degree" not in cfg.to_cache_dict()


def test_dp_degree_in_cache_dict_when_set():
    cfg = DiffletParallelConfig(tp_degree=2, dp_degree=4)
    assert cfg.to_cache_dict()["dp_degree"] == 4


def test_dp_degree_validation():
    with pytest.raises(ValueError):
        DiffletParallelConfig(dp_degree=0)


def test_world_size_includes_dp():
    assert DiffletParallelConfig(tp_degree=2, dp_degree=4).world_size == 8


def test_mesh_spec_cfg_parallel():
    cfg = DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True)
    assert cfg.mesh_spec == MeshSpec(dp=1, cfg=2, cp=1, tp=2)
    assert cfg.mesh_spec.world_size == cfg.world_size


def test_mesh_spec_cp():
    cfg = DiffletParallelConfig(tp_degree=2, cp_degree=2)
    assert cfg.mesh_spec == MeshSpec(dp=1, cfg=1, cp=2, tp=2)
    assert cfg.mesh_spec.world_size == cfg.world_size


def test_mesh_spec_dp_and_cfg():
    cfg = DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True, dp_degree=2)
    assert cfg.mesh_spec == MeshSpec(dp=2, cfg=2, cp=1, tp=2)
    assert cfg.world_size == 8


def test_cfg_cp_still_mutually_exclusive():
    with pytest.raises(ValueError):
        DiffletParallelConfig(cp_degree=2, cfg_parallel_enabled=True)
