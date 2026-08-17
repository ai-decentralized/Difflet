"""Orthogonal {dp, cfg, cp, tp} replica groups for the TPU backend.

The mesh *math* is shared with Trainium: ``difflet.pipeline.parallel_mesh``
is backend-agnostic, and its ``MeshSpec.axis_groups()`` already returns a
full partition of the world (every rank in exactly one group) — which is
precisely what XLA's HLO verifier demands of TPU replica groups.

Where this diverges from ``backends/trainium/core/parallel_mesh.py``:

* No ``torch.distributed`` process groups. ``torch_xla``'s collectives take
  the replica-group *lists* directly via ``groups=``, so an axis "group" here
  is just ``list[list[int]]``.
* TP is built here. Trainium omits it because NxD's ``parallel_state`` already
  owns a tensor-model-parallel group; on TPU nothing else owns it.

Phase 2a of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

from difflet.pipeline.parallel_mesh import AXES, MeshSpec

_MESH_SPEC: MeshSpec | None = None


def _runtime_world_size() -> int:
    import torch_xla.runtime as xr

    return int(xr.world_size())


def _runtime_ordinal() -> int:
    # torch_xla 2.9 removed xm.get_ordinal()/xm.xrt_world_size(); the
    # replacements live on torch_xla.runtime.
    import torch_xla.runtime as xr

    return int(xr.global_ordinal())


def mesh_spec_from_config(config) -> MeshSpec:
    """Derive the mesh from model-config flags plus the XLA runtime world size.

    Mirrors the Trainium derivation, but reads ``tp_degree`` off the config
    instead of NxD's ``parallel_state`` (which does not exist here).
    """
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
        raise ValueError(
            f"mesh spec {spec} product {spec.world_size} != world_size {world}"
        )
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
        raise RuntimeError(
            "TPU parallel mesh is not initialized; call init_parallel_mesh() first"
        )
    return _MESH_SPEC


def get_mesh_spec() -> MeshSpec:
    return _require_spec()


def axis_replica_groups(axis: str) -> list[list[int]]:
    """Replica groups for ``axis`` as XLA wants them.

    Always a full partition of the world, including when the axis is trivial
    (size 1) — XLA rejects a ``groups=`` list that does not cover every
    replica, so a trivial axis becomes ``[[0], [1], ...]``, not ``[[0]]``.
    """
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    return _require_spec().axis_groups(axis)


def get_tp_groups() -> list[list[int]]:
    return axis_replica_groups("tp")


def get_cfg_group() -> list[list[int]]:
    return axis_replica_groups("cfg")


def get_cp_group() -> list[list[int]]:
    return axis_replica_groups("cp")


# There is deliberately no named accessor for the dp axis here.
# tests/unit/test_no_dp_parasites.py enforces (by raw token search, so even
# naming it in a comment trips the lint) that no CFG/CP/TP collective rides
# the dp axis and that the sole named dp accessor in the tree is the Trainium
# manager's, which has no consumer. The dp axis stays reachable via
# axis_replica_groups("dp") if the DP feature ever lands.


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
        # Trivial axis: the coordinate is 0 by construction. Short-circuiting
        # keeps tp=1 code paths (and CPU-side model building) free of any
        # dependency on a live XLA runtime.
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
