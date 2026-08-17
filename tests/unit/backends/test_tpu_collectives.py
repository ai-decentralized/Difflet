"""Unit tests for the TPU collectives + parallel mesh (plan Phase 2a).

These run anywhere: the module imports ``torch_xla`` lazily, inside the custom
op bodies, so mesh math and replica-group validation are testable without a
TPU. The on-device numerics are covered by a 4-process run on real hardware
(see the plan's Phase 2a notes), not here.
"""

import pytest

from difflet.backends.tpu.ops_impl import collectives as C
from difflet.backends.tpu.ops_impl import parallel_mesh as pm
from difflet.pipeline.parallel_mesh import MeshSpec


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def test_import_does_not_require_torch_xla():
    # The whole point of the lazy imports: a non-TPU host (CI, dev laptop)
    # must be able to import the module to run these tests at all.
    #
    # Checking `"torch_xla" not in sys.modules` would be wrong: that is a
    # property of the whole process, and in a Neuron venv torch_xla is
    # legitimately imported by neuronx_distributed. Block the import in a
    # subprocess instead, which tests what this actually claims.
    import subprocess
    import sys

    program = """
import sys

class _Block:
    def find_module(self, name, path=None):
        return self if name == "torch_xla" or name.startswith("torch_xla.") else None
    def load_module(self, name):
        raise ImportError(f"{name} is blocked for this test")

sys.meta_path.insert(0, _Block())
from difflet.backends.tpu.ops_impl import collectives, linear, norm, parallel_mesh
from difflet.pipeline.parallel_mesh import MeshSpec
parallel_mesh.init_parallel_mesh(MeshSpec(tp=2))
assert parallel_mesh.get_tp_groups() == [[0, 1]]
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_mesh_not_initialized_is_an_explicit_error():
    with pytest.raises(RuntimeError, match="not initialized"):
        pm.get_tp_size()


def test_tp_only_mesh_groups():
    pm.init_parallel_mesh(MeshSpec(tp=4))
    assert pm.get_tp_size() == 4
    assert pm.get_tp_groups() == [[0, 1, 2, 3]]


def test_orthogonal_axes_partition_the_world():
    # tp innermost: rank = tp + T*(cp + C*(cfg + G*dp)).
    pm.init_parallel_mesh(MeshSpec(tp=2, cp=2))
    assert pm.get_tp_groups() == [[0, 1], [2, 3]]
    assert pm.get_cp_group() == [[0, 2], [1, 3]]


@pytest.mark.parametrize("axis", ["dp", "cfg", "cp", "tp"])
def test_every_axis_covers_every_replica_exactly_once(axis):
    # XLA rejects replica groups that do not cover every replica, so a trivial
    # axis must expand to [[0], [1], ...] rather than [[0]].
    spec = MeshSpec(dp=1, cfg=2, cp=2, tp=2)
    pm.init_parallel_mesh(spec)
    groups = pm.axis_replica_groups(axis)
    flat = sorted(r for g in groups for r in g)
    assert flat == list(range(spec.world_size))
    assert all(len(g) == spec.axis_size(axis) for g in groups)


def test_unknown_axis_rejected():
    pm.init_parallel_mesh(MeshSpec(tp=2))
    with pytest.raises(ValueError, match="unknown axis"):
        pm.axis_replica_groups("nope")


def test_mesh_reinit_with_a_different_spec_is_rejected():
    pm.init_parallel_mesh(MeshSpec(tp=4))
    pm.init_parallel_mesh(MeshSpec(tp=4))  # idempotent
    with pytest.raises(RuntimeError, match="already initialized"):
        pm.init_parallel_mesh(MeshSpec(tp=2, cp=2))


def test_flatten_roundtrip():
    groups = [[0, 1], [2, 3]]
    flat, size = C._flatten(groups)
    assert (flat, size) == ([0, 1, 2, 3], 2)
    assert C._unflatten(flat, size) == groups


def test_ragged_groups_rejected():
    # The custom-op boundary has no int[][] type, so groups cross it flattened;
    # ragged groups cannot be reconstructed.
    with pytest.raises(ValueError, match="ragged"):
        C._flatten([[0, 1], [2]])


def test_partial_replica_groups_rejected_with_a_legible_error():
    # Regression guard: [[0]] is contiguous from 0, so a contiguity-only check
    # lets it through — yet it is exactly what XLA's HLO verifier rejects
    # ("replica groups should contain 4 replicas, but found 1"). NxD accepts
    # partial groups, so this WILL be hit by ported Trainium code.
    pm.init_parallel_mesh(MeshSpec(tp=4))
    with pytest.raises(ValueError, match="partition all 4 replicas"):
        C._groups([[0]])


def test_groups_defaults_to_the_tp_axis():
    pm.init_parallel_mesh(MeshSpec(tp=2, cp=2))
    assert C._groups(None) == ([0, 1, 2, 3], 2)


def test_tp_degree_one_collectives_are_identity():
    # At tp=1 there is nothing to communicate, and no custom op (hence no
    # torch_xla import) should be reached.
    pm.init_parallel_mesh(MeshSpec(tp=1))
    sentinel = object()
    assert C.gather_tp_dim(sentinel, dim=0) is sentinel
    assert C.reduce_tp(sentinel) is sentinel
    assert C.scatter_tp_dim(sentinel, dim=0) is sentinel
    assert C.reduce_scatter_to_sequence_parallel_region(sentinel, dim=0) is sentinel


def test_scatter_rejects_indivisible_dimension():
    import torch

    pm.init_parallel_mesh(MeshSpec(tp=4))
    with pytest.raises(ValueError, match="cannot scatter"):
        C.scatter_tp_dim(torch.zeros(6, 2), dim=0)


def test_public_surface_matches_the_ops_contract():
    # difflet.ops.collectives dispatches by name into this module; a missing
    # name fails at first use on device, which is an expensive way to find out.
    required = {
        "gather_tp_dim",
        "reduce_tp",
        "scatter_tp_dim",
        "get_tp_size",
        "get_tp_rank",
        "scatter_to_sequence_parallel_region",
        "gather_from_sequence_parallel_region",
        "reduce_scatter_to_sequence_parallel_region",
    }
    assert required <= set(C.__all__)
    for name in required:
        assert callable(getattr(C, name))


# --------------------------------------------------------------------------
# NxD-shaped compatibility surface.
#
# Qwen-Image's transformer does `from difflet.ops import (...)` for names only
# the Trainium backend defined. That import resolves EVERY name eagerly, so a
# missing one breaks at import time even if the code path never runs.
# --------------------------------------------------------------------------

QWEN_IMAGE_OPS = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "SPMDRank",
    "attention",
    "gather_from_tensor_model_parallel_region_with_dim",
    "get_cp_group",
    "get_cp_rank_spmd",
    "get_tensor_model_parallel_size",
    "get_world_group",
    "init_parallel_mesh",
    "joint_ring_attention",
    "joint_ulysses_attention",
    "scatter_to_process_group_spmd",
]


def test_tpu_backend_covers_the_whole_trainium_ops_surface():
    # The ops surface is documented as "frozen v1, additive-only" so a second
    # backend can be added without touching model code. Any name Trainium
    # exports and TPU does not is a hole a model will fall through.
    import re
    from pathlib import Path

    def exports(backend):
        out = set()
        for path in Path(f"difflet/backends/{backend}/ops_impl").glob("*.py"):
            text = path.read_text()
            first = re.search(r"__all__\s*=\s*\[(.*?)\]", text, re.S)
            if first:
                out |= set(re.findall(r'"([^"]+)"', first.group(1)))
            for extra in re.finditer(r"__all__\s*\+=\s*\[(.*?)\]", text, re.S):
                out |= set(re.findall(r'"([^"]+)"', extra.group(1)))
        return out

    missing = exports("trainium") - exports("tpu")
    assert missing == set(), f"TPU backend is missing ops: {sorted(missing)}"


@pytest.mark.parametrize("name", QWEN_IMAGE_OPS)
def test_every_name_qwen_image_imports_resolves_on_tpu(name):
    from difflet.backends.tpu.ops_impl import attention as tpu_attention
    from difflet.backends.tpu.ops_impl import linear as tpu_linear

    found = any(
        hasattr(mod, name) for mod in (C, tpu_linear, tpu_attention)
    )
    assert found, f"{name} does not resolve on the TPU backend"


def test_world_group_reports_the_mesh_world_size():
    pm.init_parallel_mesh(MeshSpec(tp=2, cp=2))
    assert C.get_world_group().size() == 4
    assert C.get_tensor_model_parallel_size() == 2


@pytest.mark.parametrize(
    "rank,expected_cp,expected_cfg",
    # rank = tp + T*(cp + C*(cfg + G*dp)) with tp=2, cp=2, cfg=2 -> world 8
    [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 1, 0), (4, 0, 1), (7, 1, 1)],
)
def test_spmd_rank_coordinates_match_the_mesh_layout(rank, expected_cp, expected_cfg):
    pm.init_parallel_mesh(MeshSpec(tp=2, cp=2, cfg=2))
    assert C.get_cp_rank_spmd(rank) == expected_cp
    assert C.get_cfg_rank_spmd(rank) == expected_cfg


def test_spmd_coordinates_accept_tensors_too():
    # NxD passes a traced tensor here; model code written against that shape
    # must keep working.
    import torch

    pm.init_parallel_mesh(MeshSpec(tp=2, cp=2))
    got = C.get_cp_rank_spmd(torch.tensor(3))
    assert int(got) == 1


def test_scatter_to_process_group_takes_this_ranks_slice():
    import torch

    pm.init_parallel_mesh(MeshSpec(tp=4))
    full = torch.arange(8).reshape(8, 1)
    got = C.scatter_to_process_group_spmd(full, partition_dim=0, rank=2)
    assert got.flatten().tolist() == [4, 5]


def test_scatter_rejects_a_traced_rank_with_an_explanation():
    # TPU exports one artifact per rank, so the offset must be a constant.
    # Silently tracing this would produce a graph that is wrong for 3 of 4 ranks.
    import torch

    pm.init_parallel_mesh(MeshSpec(tp=4))
    with pytest.raises(TypeError, match="Python int known at export time"):
        C.scatter_to_process_group_spmd(
            torch.zeros(8, 1), partition_dim=0, rank=torch.tensor(2)
        )


def test_trivial_group_gather_is_identity():
    pm.init_parallel_mesh(MeshSpec(tp=1))
    sentinel = object()
    assert C.gather_from_tensor_model_parallel_region_with_dim(sentinel, 0) is sentinel


def test_mx_ops_refuse_rather_than_silently_dropping_quantization():
    from difflet.backends.tpu.ops_impl import mx

    for name in ("quantize_mx", "dequantize_mx", "linear_mx", "matmul_mx"):
        with pytest.raises(NotImplementedError, match="no TPU equivalent"):
            getattr(mx, name)()
