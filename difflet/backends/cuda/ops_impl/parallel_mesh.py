"""Orthogonal {dp, cfg, cp, tp} replica groups for the CUDA backend.

Same mesh math as TPU and Trainium (``difflet.pipeline.parallel_mesh``); the only
backend-specific facts are the world size and this process's rank, which come
from the ``torchrun`` environment (``WORLD_SIZE`` / ``RANK``) instead of an XLA
or NxD runtime. Without a launcher the world is what the config's factors ask for
(``mesh_spec_from_config``) and this process is rank 0.

At world size 1 no ``torch.distributed`` process group exists and the
group accessors return replica lists. At world size > 1 (one
process per GPU under ``torchrun``) ``init_parallel_mesh`` builds one
``torch.distributed`` group per replica list of every non-trivial cfg/cp/dp axis
— the process group itself (NCCL on GPUs) is initialised by
``CudaBackend.prepare_runtime`` — and ``get_cfg_group``/``get_cp_group`` return
this rank's group. Tensor parallel across processes is not supported yet.
"""

from __future__ import annotations

import os

import torch

from difflet.pipeline.parallel_mesh import AXES, MeshSpec

_MESH_SPEC: MeshSpec | None = None
# axis -> the torch.distributed group of this rank on that axis (world size > 1 only).
_AXIS_GROUPS: dict = {}

# Axes that get a torch.distributed group (Trainium's recipe); tp is refused.
_GROUP_AXES = ("cfg", "cp", "dp")
_TP_UNSUPPORTED = "tensor parallelism (tp > 1) is not supported on the CUDA backend yet"


def _launched_world_size() -> int | None:
    """The world size a launcher fixed — ``WORLD_SIZE`` when set, else an initialised
    process group's — or None in a plain single process."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return _distributed_world_size()


def _runtime_ordinal() -> int:
    return int(os.environ.get("RANK", "0"))


def mesh_spec_from_config(config) -> MeshSpec:
    """Derive the mesh from model-config flags plus the launcher's world size.

    Without a launcher (``WORLD_SIZE`` unset, no process group) the world is the
    product of the config's factors, ``tp × cfg × dp`` with cp=1: the CLI parent
    compiles — writes the manifest the ranks will load — before ``torchrun`` starts
    them. A world size a launcher *did* set must match the config.
    """
    tp = int(getattr(config, "tp_degree", 1) or 1)
    cfg = 2 if bool(getattr(config, "cfg_parallel_enabled", False)) else 1
    dp = int(getattr(config, "dp_degree", 1) or 1)
    denom = tp * cfg * dp
    launched = _launched_world_size()
    world = denom if launched is None else launched
    if world % denom != 0:
        raise ValueError(
            f"world_size {world} is not divisible by tp*cfg*dp = {denom} "
            f"(tp={tp}, cfg={cfg}, dp={dp})"
        )
    cp = world // denom if bool(getattr(config, "context_parallel_enabled", False)) else 1
    spec = MeshSpec(dp=dp, cfg=cfg, cp=cp, tp=tp)
    if spec.world_size != world:
        raise ValueError(f"mesh spec {spec} product {spec.world_size} != world_size {world}")
    return spec


def _distributed_world_size() -> int | None:
    """World size of the initialised default process group, or None without one."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    return torch.distributed.get_world_size()


def init_parallel_mesh(config) -> None:
    """Record the mesh once per process (idempotent for equal specs).

    With an initialised multi-process ``torch.distributed``, also build the axis
    groups. Every rank calls ``new_group`` for every ranks list in the same order
    (``torch.distributed`` requires it) and keeps the one containing its ``RANK``.
    """
    global _MESH_SPEC
    spec = config if isinstance(config, MeshSpec) else mesh_spec_from_config(config)
    if _MESH_SPEC is not None:
        if spec != _MESH_SPEC:
            raise RuntimeError(
                f"parallel mesh already initialized with {_MESH_SPEC}; "
                f"cannot re-initialize with {spec}"
            )
        return
    dist_world = _distributed_world_size()
    multi_process = dist_world is not None and spec.world_size > 1
    if multi_process:
        if spec.tp > 1:
            raise NotImplementedError(f"mesh {spec}: {_TP_UNSUPPORTED}")
        if spec.world_size != dist_world:
            raise ValueError(
                f"mesh spec {spec} has world size {spec.world_size} but the "
                f"torch.distributed process group has world size {dist_world}"
            )
    groups = {}
    if multi_process:
        rank = _runtime_ordinal()
        for axis in _GROUP_AXES:
            if spec.axis_size(axis) == 1:
                continue
            for ranks in spec.axis_groups(axis):
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    groups[axis] = group
    # Publish spec and groups together: a failed new_group leaves no half-built mesh.
    _AXIS_GROUPS.update(groups)
    _MESH_SPEC = spec


def _require_spec() -> MeshSpec:
    if _MESH_SPEC is None:
        from difflet.ops.collectives import ParallelMeshNotInitializedError

        raise ParallelMeshNotInitializedError(
            "CUDA parallel mesh is not initialized; call init_parallel_mesh() first"
        )
    return _MESH_SPEC


def get_mesh_spec() -> MeshSpec:
    return _require_spec()


def axis_replica_groups(axis: str) -> list[list[int]]:
    """Replica groups for ``axis``: always a full partition of the world."""
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    return _require_spec().axis_groups(axis)


def get_tp_groups() -> list[list[int]]:
    return axis_replica_groups("tp")


def _axis_group(axis: str):
    """This rank's ``torch.distributed`` group on ``axis`` when one was built, else
    the axis replica lists (world size 1, or a trivial axis)."""
    _require_spec()
    if axis in _AXIS_GROUPS:
        return _AXIS_GROUPS[axis]
    return axis_replica_groups(axis)


def get_cfg_group():
    return _axis_group("cfg")


def get_cp_group():
    return _axis_group("cp")


# No named accessor for the dp axis here, by design (see
# tests/unit/test_no_dp_parasites.py); it stays reachable via axis_replica_groups.


def get_cp_mesh() -> list[list[int]]:
    """Replica groups of the cp axis (for ring collectives)."""
    return axis_replica_groups("cp")


def get_tp_size() -> int:
    return _require_spec().tp


def get_tp_rank() -> int:
    return get_axis_rank("tp")


def get_axis_rank(axis: str) -> int:
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    spec = _require_spec()
    if spec.axis_size(axis) == 1:
        return 0
    return spec.axis_rank(_runtime_ordinal(), axis)


def destroy_parallel_mesh() -> None:
    """Reset module state and drop the axis groups.

    The default process group is left alone: whoever initialised it destroys it.
    """
    global _MESH_SPEC
    _MESH_SPEC = None
    _AXIS_GROUPS.clear()


__all__ = [
    "axis_replica_groups",
    "destroy_parallel_mesh",
    "get_axis_rank",
    "get_cfg_group",
    "get_cp_group",
    "get_cp_mesh",
    "get_mesh_spec",
    "get_tp_groups",
    "get_tp_rank",
    "get_tp_size",
    "init_parallel_mesh",
    "mesh_spec_from_config",
]
