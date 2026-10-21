"""Orthogonal {dp, cfg, cp, tp} replica groups for the CUDA backend.

Same mesh math as TPU and Trainium (``difflet.pipeline.parallel_mesh``); the only
backend-specific facts are the world size and this process's rank, which come
from the ``torchrun`` environment (``WORLD_SIZE`` / ``RANK``, defaulting to a
single process) instead of an XLA or NxD runtime. No ``torch.distributed``
process group is created here — Release 1 never needs one; a later change
adds NCCL groups built from these same replica lists.
"""

from __future__ import annotations

import os

from difflet.pipeline.parallel_mesh import AXES, MeshSpec

_MESH_SPEC: MeshSpec | None = None


def _runtime_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _runtime_ordinal() -> int:
    return int(os.environ.get("RANK", "0"))


def mesh_spec_from_config(config) -> MeshSpec:
    """Derive the mesh from model-config flags plus the launcher's world size."""
    world = _runtime_world_size()
    tp = int(getattr(config, "tp_degree", 1) or 1)
    cfg = 2 if bool(getattr(config, "cfg_parallel_enabled", False)) else 1
    dp = int(getattr(config, "dp_degree", 1) or 1)
    denom = tp * cfg * dp
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


def init_parallel_mesh(config) -> None:
    """Record the mesh once per process (idempotent for equal specs)."""
    global _MESH_SPEC
    spec = config if isinstance(config, MeshSpec) else mesh_spec_from_config(config)
    if _MESH_SPEC is not None:
        if spec != _MESH_SPEC:
            raise RuntimeError(
                f"parallel mesh already initialized with {_MESH_SPEC}; "
                f"cannot re-initialize with {spec}"
            )
        return
    _MESH_SPEC = spec


def _require_spec() -> MeshSpec:
    if _MESH_SPEC is None:
        raise RuntimeError("CUDA parallel mesh is not initialized; call init_parallel_mesh() first")
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


def get_cfg_group() -> list[list[int]]:
    return axis_replica_groups("cfg")


def get_cp_group() -> list[list[int]]:
    return axis_replica_groups("cp")


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
    """Reset module state (unit tests only)."""
    global _MESH_SPEC
    _MESH_SPEC = None


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
