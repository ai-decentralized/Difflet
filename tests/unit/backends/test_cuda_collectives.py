"""CUDA parallel mesh + collectives at world size 1 (CUDA plan Phase 2a, D5).

Mirrors tests/unit/backends/test_tpu_collectives.py. Release 1 runs one process on one
device: every collective is the identity, and anything that would need a second process
raises naming follow-up F2 rather than silently running on one rank.
"""

import os

import pytest
import torch

from difflet.backends.cuda.ops_impl import collectives as C
from difflet.backends.cuda.ops_impl import parallel_mesh as pm
from difflet.pipeline.parallel_mesh import AXES, MeshSpec


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def test_mesh_not_initialized_is_an_explicit_error():
    with pytest.raises(RuntimeError, match="not initialized"):
        pm.get_tp_size()


def test_tp_only_mesh_groups():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    assert pm.get_tp_groups() == [[0, 1]]
    assert pm.get_tp_size() == 2


@pytest.mark.parametrize("axis", AXES)
def test_every_axis_covers_every_replica_exactly_once(axis):
    spec = MeshSpec(dp=1, cfg=2, cp=2, tp=2)
    pm.init_parallel_mesh(spec)
    groups = pm.axis_replica_groups(axis)
    assert sorted(r for g in groups for r in g) == list(range(spec.world_size))


def test_mesh_reinit_with_a_different_spec_is_rejected():
    pm.init_parallel_mesh(MeshSpec(tp=1))
    pm.init_parallel_mesh(MeshSpec(tp=1))  # idempotent for an equal spec
    with pytest.raises(RuntimeError, match="already initialized"):
        pm.init_parallel_mesh(MeshSpec(tp=2))


def test_mesh_from_config_reads_world_size_from_env(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    spec = pm.mesh_spec_from_config(type("Cfg", (), {"tp_degree": 1})())
    assert spec == MeshSpec(tp=1)
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert pm.mesh_spec_from_config(type("Cfg", (), {"tp_degree": 2})()) == MeshSpec(tp=2)


def test_axis_rank_reads_rank_from_env(monkeypatch):
    pm.init_parallel_mesh(MeshSpec(tp=2))
    monkeypatch.setenv("RANK", "1")
    assert pm.get_tp_rank() == 1


def test_tp_degree_one_collectives_are_identity():
    pm.init_parallel_mesh(MeshSpec(tp=1))
    t = torch.arange(8.0).reshape(2, 4)
    for fn in (
        lambda x: C.gather_tp_dim(x, dim=-1),
        C.reduce_tp,
        lambda x: C.scatter_tp_dim(x, dim=-1),
        lambda x: C.scatter_to_sequence_parallel_region(x, dim=0),
        lambda x: C.gather_from_sequence_parallel_region(x, dim=0),
        lambda x: C.reduce_scatter_to_sequence_parallel_region(x, dim=0),
        C.reduce_from_tensor_model_parallel_region,
        C.scatter_to_tensor_model_parallel_region,
        lambda x: C.gather_from_tensor_model_parallel_region_with_dim(x, 1),
        lambda x: C.scatter_to_process_group_spmd(x, 1, 0),
    ):
        assert torch.equal(fn(t), t)
    assert C.get_tp_size() == 1 and C.get_tp_rank() == 0
    assert C.get_tensor_model_parallel_size() == 1 and C.get_tensor_model_parallel_rank() == 0
    assert C.get_world_group().size() == 1 and C.get_world_group().rank() == 0
    assert C.SPMDRank(1).get_rank() == 0
    assert C.get_cp_rank_spmd(0) == 0 and C.get_cfg_rank_spmd(0) == 0
    assert C.get_cfg_group() == [[0]] and C.get_cp_group() == [[0]]
    C.init_parallel_mesh(MeshSpec(tp=1))  # exported, idempotent


def test_public_surface_matches_the_ops_contract():
    from difflet.ops import _EXPORTS

    expected = {name for name, (module, _) in _EXPORTS.items() if module == "collectives"}
    assert expected <= set(C.__all__)
    for name in expected:
        assert hasattr(C, name), name


def test_multi_rank_collectives_raise_f2():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    t = torch.zeros(2, 4)
    for fn in (
        lambda x: C.gather_tp_dim(x, dim=-1),
        C.reduce_tp,
        lambda x: C.reduce_scatter_to_sequence_parallel_region(x, dim=0),
        lambda x: C.gather_from_tensor_model_parallel_region_with_dim(x, 1),
    ):
        with pytest.raises(NotImplementedError, match="F2"):
            fn(t)


def test_scatter_rejects_indivisible_dimension():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    with pytest.raises(ValueError, match="cannot scatter"):
        C.scatter_tp_dim(torch.zeros(2, 3), dim=-1)


def test_import_does_not_initialise_torch_distributed():
    assert not torch.distributed.is_initialized()
    assert "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) == 1


def test_reduce_from_tensor_model_parallel_region_takes_a_process_group():
    # modeling_flux's single block reduces over ``proj_out_attn.tensor_parallel_group``.
    pm.init_parallel_mesh(MeshSpec(tp=1))
    t = torch.arange(8.0).reshape(2, 4)
    assert C.reduce_from_tensor_model_parallel_region(t, process_group=None) is t
    assert C.reduce_from_tensor_model_parallel_region(t, process_group=[[0]]) is t
    with pytest.raises(NotImplementedError, match="F2"):
        C.reduce_from_tensor_model_parallel_region(t, process_group=[[0, 1]])


def test_reduce_from_tensor_model_parallel_region_defaults_to_the_tp_group():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    with pytest.raises(NotImplementedError, match="F2"):
        C.reduce_from_tensor_model_parallel_region(torch.zeros(2, 4), process_group=None)
