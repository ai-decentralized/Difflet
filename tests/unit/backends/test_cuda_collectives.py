"""CUDA parallel mesh + collectives.

Mirrors tests/unit/backends/test_tpu_collectives.py. At world size 1 every collective
is the identity and no ``torch.distributed`` group exists. At world size > 1 the mesh
builds ``torch.distributed`` groups for the cfg/cp/dp axes (exercised here with a
fake ``torch.distributed`` and for real under gloo in test_cuda_multirank.py), and
tensor parallel still raises.
"""

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


def test_mesh_from_config_without_a_launcher_is_the_product_of_the_factors(monkeypatch):
    # The CLI parent compiles (writes the manifest) before torchrun starts the
    # ranks, with WORLD_SIZE unset and no process group: the mesh is what the config
    # asks for (tp x cfg x dp, cp=1), not a world of one that cfg=2 cannot divide.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert not torch.distributed.is_initialized()

    def cfg(**flags):
        return type("Cfg", (), {"tp_degree": 1, **flags})()

    assert pm.mesh_spec_from_config(cfg(cfg_parallel_enabled=True)) == MeshSpec(cfg=2)
    assert pm.mesh_spec_from_config(cfg(dp_degree=2)) == MeshSpec(dp=2)
    assert pm.mesh_spec_from_config(cfg(dp_degree=2, cfg_parallel_enabled=True)) == MeshSpec(
        dp=2, cfg=2
    )
    assert pm.mesh_spec_from_config(cfg(context_parallel_enabled=True)) == MeshSpec()
    # A launcher that *set* a world size the config cannot fill is still refused.
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="not divisible"):
        pm.mesh_spec_from_config(cfg(cfg_parallel_enabled=True))
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="world_size 4"):
        pm.mesh_spec_from_config(cfg(cfg_parallel_enabled=True))


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


def test_reduce_from_tensor_model_parallel_region_takes_a_process_group():
    # modeling_flux's single block reduces over ``proj_out_attn.tensor_parallel_group``.
    pm.init_parallel_mesh(MeshSpec(tp=1))
    t = torch.arange(8.0).reshape(2, 4)
    assert C.reduce_from_tensor_model_parallel_region(t, process_group=None) is t
    assert C.reduce_from_tensor_model_parallel_region(t, process_group=[[0]]) is t
    with pytest.raises(NotImplementedError, match="not supported"):
        C.reduce_from_tensor_model_parallel_region(t, process_group=[[0, 1]])


def test_world_size_one_creates_no_process_group(monkeypatch):
    def _no_new_group(*a, **kw):
        raise AssertionError("world size 1 must not create a torch.distributed group")

    monkeypatch.setattr(torch.distributed, "new_group", _no_new_group)
    pm.init_parallel_mesh(MeshSpec(tp=1))
    assert pm.get_cfg_group() == [[0]] and pm.get_cp_group() == [[0]]
