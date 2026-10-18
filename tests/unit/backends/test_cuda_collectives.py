"""CUDA parallel mesh + collectives (CUDA plan Phase 2a, D5; follow-up F2).

Mirrors tests/unit/backends/test_tpu_collectives.py. At world size 1 every collective
is the identity and no ``torch.distributed`` group exists. At world size > 1 the mesh
builds ``torch.distributed`` groups for the cfg/cp/dp axes (F2; exercised here with a
fake ``torch.distributed`` and for real under gloo in test_cuda_multirank.py), and
tensor parallel still raises naming follow-up F6.
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


def test_tensor_parallel_collectives_raise_f6():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    t = torch.zeros(2, 4)
    for fn in (
        lambda x: C.gather_tp_dim(x, dim=-1),
        C.reduce_tp,
        lambda x: C.reduce_scatter_to_sequence_parallel_region(x, dim=0),
        lambda x: C.gather_from_sequence_parallel_region(x, dim=0),
        lambda x: C.gather_from_tensor_model_parallel_region_with_dim(x, 1),
        C.reduce_from_tensor_model_parallel_region,
    ):
        with pytest.raises(NotImplementedError, match="F6"):
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
    with pytest.raises(NotImplementedError, match="F6"):
        C.reduce_from_tensor_model_parallel_region(t, process_group=[[0, 1]])


def test_reduce_from_tensor_model_parallel_region_defaults_to_the_tp_group():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    with pytest.raises(NotImplementedError, match="F6"):
        C.reduce_from_tensor_model_parallel_region(torch.zeros(2, 4), process_group=None)


# --------------------------------------------------------------------------
# World size > 1 under a fake torch.distributed (F2). The real collectives run
# under gloo in test_cuda_multirank.py; these pin the group bookkeeping.
# --------------------------------------------------------------------------


class _FakeGroup:
    def __init__(self, ranks):
        self.ranks = list(ranks)

    def __repr__(self):
        return f"_FakeGroup({self.ranks})"


@pytest.fixture
def fake_dist(monkeypatch):
    """A 4-rank ``torch.distributed`` that records ``new_group`` calls."""
    state = {"world": 4, "new_group": []}

    def new_group(ranks=None, **kw):
        state["new_group"].append(list(ranks))
        return _FakeGroup(ranks)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: (state["world"] if group is None else len(group.ranks)),
    )
    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: list(group.ranks)
    )
    return state


def test_world_size_one_creates_no_process_group(monkeypatch):
    def _no_new_group(*a, **kw):
        raise AssertionError("world size 1 must not create a torch.distributed group")

    monkeypatch.setattr(torch.distributed, "new_group", _no_new_group)
    pm.init_parallel_mesh(MeshSpec(tp=1))
    assert pm.get_cfg_group() == [[0]] and pm.get_cp_group() == [[0]]


def test_mesh_builds_one_group_per_replica_list_on_every_rank(fake_dist, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    pm.init_parallel_mesh(MeshSpec(cfg=2, cp=2))
    # Every rank enters new_group for every ranks list, axes in cfg, cp, dp order.
    assert fake_dist["new_group"] == [[0, 2], [1, 3], [0, 1], [2, 3]]
    # Each rank keeps the group it belongs to; replica lists are unchanged.
    assert pm.get_cfg_group().ranks == [1, 3]
    assert pm.get_cp_group().ranks == [0, 1]
    assert pm.axis_replica_groups("cfg") == [[0, 2], [1, 3]]
    assert pm.get_cp_mesh() == [[0, 1], [2, 3]]


def test_trivial_axes_get_no_group(fake_dist, monkeypatch):
    fake_dist["world"] = 2
    monkeypatch.setenv("RANK", "0")
    pm.init_parallel_mesh(MeshSpec(dp=2))
    assert fake_dist["new_group"] == [[0, 1]]  # the dp axis only
    assert pm.get_cfg_group() == [[0], [1]] and pm.get_cp_group() == [[0], [1]]


def test_mesh_refuses_tensor_parallel_across_processes(fake_dist, monkeypatch):
    fake_dist["world"] = 2
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(NotImplementedError, match="F6"):
        pm.init_parallel_mesh(MeshSpec(tp=2))
    assert fake_dist["new_group"] == []
    with pytest.raises(RuntimeError, match="not initialized"):
        pm.get_mesh_spec()


def test_mesh_world_size_must_match_the_process_group(fake_dist, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="world size"):
        pm.init_parallel_mesh(MeshSpec(cfg=2))  # fake world is 4


def test_destroy_drops_the_groups(fake_dist, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    pm.init_parallel_mesh(MeshSpec(cfg=2, cp=2))
    pm.destroy_parallel_mesh()
    assert pm._AXIS_GROUPS == {}


def test_collectives_accept_a_process_group(fake_dist, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    pm.init_parallel_mesh(MeshSpec(cfg=2, cp=2))
    t = torch.arange(8.0).reshape(2, 4)
    one = _FakeGroup([0])
    assert C._group_size(one) == 1 and C._group_ranks(one) == [0]
    assert C.gather_from_tensor_model_parallel_region_with_dim(t, 1, process_group=one) is t
    assert C.reduce_from_tensor_model_parallel_region(t, process_group=one) is t
    assert C.scatter_to_process_group_spmd(t, 1, 0, process_group=one) is t
    cfg = pm.get_cfg_group()
    assert C._group_size(cfg) == 2 and C._group_ranks(cfg) == [0, 2]
    assert torch.equal(C.scatter_to_process_group_spmd(t, 1, 1, process_group=cfg), t[:, 2:])
    # A replica list resolves to the list this rank belongs to.
    assert C._group_size([[0, 2], [1, 3]]) == 2 and C._group_ranks([[0, 2], [1, 3]]) == [0, 2]
