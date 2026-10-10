"""Unit tests for the neuron backend's TP collectives and mesh; CPU only, no Neuron device.

Single-process tests cover the mesh state machine and the tp == 1 identities; the
gloo tests run the real functional collectives on 4 CPU ranks, eagerly and under
``torch.compile(backend="aot_eager", fullgraph=True)``.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from difflet.backends import registry  # noqa: E402
from difflet.backends.neuron.ops_impl import collectives as C  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh as pm  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.pipeline.parallel_mesh import MeshSpec  # noqa: E402
from tests.unit.backends._neuron_gloo import run_ranks  # noqa: E402
from tests.unit.backends._neuron_workers import c5_collectives_worker  # noqa: E402

TRAINIUM_COLLECTIVES = {
    "SPMDRank",
    "gather_from_sequence_parallel_region",
    "gather_from_tensor_model_parallel_region_with_dim",
    "gather_tp_dim",
    "get_cfg_group",
    "get_cfg_rank_spmd",
    "get_cp_group",
    "get_cp_rank_spmd",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
    "get_tp_rank",
    "get_tp_size",
    "get_world_group",
    "init_parallel_mesh",
    "reduce_from_tensor_model_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "reduce_tp",
    "scatter_to_process_group_spmd",
    "scatter_to_sequence_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "scatter_tp_dim",
}
EXTRAS = {"destroy_parallel_mesh", "get_mesh_spec", "get_tp_group", "is_mesh_initialized"}


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def test_import_needs_neither_torch_xla_nor_nxd():
    program = """
import importlib.abc, sys

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch_xla", "neuronx_distributed"):
            raise ImportError(f"{name} is blocked for this test")
        return None

sys.meta_path.insert(0, _Block())
from difflet.backends.neuron.ops_impl import collectives
from difflet.pipeline.parallel_mesh import MeshSpec
collectives.init_parallel_mesh(MeshSpec(tp=4))
assert collectives.get_tp_size() == 4
bad = sorted(m for m in sys.modules if m.split(".")[0] in ("torch_xla", "neuronx_distributed"))
assert not bad, bad
print("OK")
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_surface_is_the_trainium_collectives_plus_mesh_extras():
    assert set(C.__all__) == TRAINIUM_COLLECTIVES | EXTRAS
    for name in C.__all__:
        assert callable(getattr(C, name)), name


def test_difflet_ops_dispatches_collectives_to_this_module(monkeypatch):
    import difflet.ops as ops
    from difflet.ops import _EXPORTS

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    registry._get_backend_by_name.cache_clear()
    try:
        names = [name for name, (module, _) in _EXPORTS.items() if module == "collectives"]
        assert set(names) == TRAINIUM_COLLECTIVES
        for name in names:
            assert getattr(ops, name) is getattr(C, name), name
    finally:
        registry._get_backend_by_name.cache_clear()


def test_uninitialized_single_process_is_tp1():
    assert not pm.is_mesh_initialized()
    assert C.get_tp_size() == 1
    assert C.get_tp_rank() == 0
    assert C.get_tp_group() is None
    with pytest.raises(RuntimeError, match="not initialized"):
        C.get_mesh_spec()


@pytest.mark.parametrize(
    "config, expected",
    [
        (MeshSpec(tp=4), MeshSpec(tp=4)),
        (DiffletParallelConfig(tp_degree=4), MeshSpec(tp=4)),
        (SimpleNamespace(tp_degree=2), MeshSpec(tp=2)),
        (SimpleNamespace(), MeshSpec()),
    ],
)
def test_mesh_spec_from_config(config, expected):
    assert pm.mesh_spec_from_config(config) == expected


@pytest.mark.parametrize(
    "config, unsupported",
    [
        (DiffletParallelConfig(tp_degree=2, cp_degree=2), "cp=2"),
        (DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True), "cfg=2"),
        (DiffletParallelConfig(dp_degree=2), "dp=2"),
        (MeshSpec(tp=2, cp=2), "cp=2"),
        (SimpleNamespace(tp_degree=1, cfg_parallel_enabled=True), "cfg=2"),
    ],
)
def test_non_tp_modes_are_rejected(config, unsupported):
    match = f"tensor parallelism only; unsupported: {unsupported}"
    with pytest.raises(NotImplementedError, match=match):
        C.init_parallel_mesh(config)
    assert not pm.is_mesh_initialized()


def test_reinit_is_idempotent_for_an_equal_spec_only():
    C.init_parallel_mesh(MeshSpec(tp=4))
    C.init_parallel_mesh(DiffletParallelConfig(tp_degree=4))
    with pytest.raises(RuntimeError, match="already initialized"):
        C.init_parallel_mesh(MeshSpec(tp=2))


def test_spec_only_mesh_knows_sizes_but_refuses_ranks_and_collectives():
    C.init_parallel_mesh(MeshSpec(tp=4))
    assert pm.is_mesh_initialized()
    assert C.get_mesh_spec() == MeshSpec(tp=4)
    assert C.get_tp_size() == C.get_tensor_model_parallel_size() == 4
    with pytest.raises(RuntimeError, match="spec-only"):
        C.get_tp_rank()
    with pytest.raises(RuntimeError, match="spec-only"):
        C.reduce_tp(torch.ones(4))
    with pytest.raises(RuntimeError, match="spec-only"):
        C.gather_tp_dim(torch.ones(4), dim=0)
    with pytest.raises(RuntimeError, match="spec-only"):
        C.scatter_tp_dim(torch.ones(8), dim=0)


def test_tp1_collectives_are_the_identity():
    C.init_parallel_mesh(MeshSpec(tp=1))
    sentinel = object()
    assert C.gather_tp_dim(sentinel, dim=0) is sentinel
    assert C.reduce_tp(sentinel) is sentinel
    assert C.scatter_tp_dim(sentinel, dim=0) is sentinel
    assert C.scatter_to_sequence_parallel_region(sentinel, dim=1) is sentinel
    assert C.gather_from_sequence_parallel_region(sentinel, dim=1) is sentinel
    assert C.reduce_scatter_to_sequence_parallel_region(sentinel, dim=1) is sentinel
    assert C.gather_from_tensor_model_parallel_region_with_dim(sentinel, 0) is sentinel
    assert C.reduce_from_tensor_model_parallel_region(sentinel) is sentinel
    assert C.scatter_to_tensor_model_parallel_region(sentinel) is sentinel
    assert C.scatter_to_process_group_spmd(sentinel, 0, 0) is sentinel


def test_scatter_validates_divisibility_before_needing_a_rank():
    C.init_parallel_mesh(MeshSpec(tp=4))
    with pytest.raises(ValueError, match="cannot scatter dimension 0 of size 6 across 4 ranks"):
        C.scatter_tp_dim(torch.zeros(6, 2), dim=0)
    with pytest.raises(ValueError, match="cannot reduce-scatter dimension 1 of size 6"):
        C.reduce_scatter_to_sequence_parallel_region(torch.zeros(2, 6), dim=-1)


def test_scatter_to_process_group_takes_the_given_ranks_slice():
    # A local narrow: works on a spec-only mesh because the rank is an argument.
    C.init_parallel_mesh(MeshSpec(tp=4))
    full = torch.arange(8).reshape(8, 1)
    shard = C.scatter_to_process_group_spmd(full, partition_dim=0, rank=2)
    assert shard.flatten().tolist() == [4, 5]
    with pytest.raises(TypeError, match="Python int"):
        C.scatter_to_process_group_spmd(full, partition_dim=0, rank=torch.tensor(2))
    with pytest.raises(ValueError, match="outside the tensor-parallel group"):
        C.scatter_to_process_group_spmd(full, partition_dim=0, rank=4)


def test_explicit_process_groups():
    C.init_parallel_mesh(MeshSpec(tp=4))
    x = torch.ones(4, 2)
    trivial = C.get_cp_group()
    assert C.gather_from_tensor_model_parallel_region_with_dim(x, 0, process_group=trivial) is x
    assert C.scatter_to_process_group_spmd(x, 0, 3, process_group=trivial) is x
    two = SimpleNamespace(size=lambda: 2)
    with pytest.raises(NotImplementedError, match="tensor parallelism only"):
        C.gather_from_tensor_model_parallel_region_with_dim(x, 0, process_group=two)
    with pytest.raises(NotImplementedError, match="tensor parallelism only"):
        C.scatter_to_process_group_spmd(x, 0, 0, process_group=two)


@pytest.mark.parametrize("initialized", [False, True])
def test_cfg_and_cp_groups_are_trivial(initialized):
    if initialized:
        C.init_parallel_mesh(MeshSpec(tp=4))
    for group in (C.get_cfg_group(), C.get_cp_group(), C.get_world_group()):
        assert group.size() == 1
        assert group.rank() == 0
    assert C.get_cfg_rank_spmd(3) == 0
    assert C.get_cp_rank_spmd(3) == 0
    got = C.get_cp_rank_spmd(torch.tensor(3))
    assert got.dtype == torch.int32 and int(got) == 0
    assert int(C.get_cfg_rank_spmd(torch.tensor(3))) == 0


def test_spmd_rank_is_a_parameterless_constant():
    rank_util = C.SPMDRank(world_size=4)
    assert isinstance(rank_util, torch.nn.Module)
    assert rank_util.world_size == 4
    assert rank_util.state_dict() == {}
    assert rank_util.get_rank() == 0


@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_gloo_tp4_collectives_are_exact(mode):
    results = run_ranks(c5_collectives_worker, mode, world_size=4)
    failed = {
        rank: [name for name, ok in checks.items() if not ok] for rank, checks in enumerate(results)
    }
    assert all(not names for names in failed.values()), failed
    assert len(results[0]) >= 20
