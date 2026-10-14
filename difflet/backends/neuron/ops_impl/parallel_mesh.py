"""Parallel mesh state for the neuron (TorchNeuron) backend: tensor and CFG parallelism.

The rank layout is the Trainium and TPU one (``MeshSpec``): tp innermost, then cp, then cfg,
then dp, ``rank = tp + T * (cp + C * (cfg + G * dp))``. Each axis of size > 1 gets one process
subgroup per combination of the other coordinates (``MeshSpec.axis_groups``), created with
``dist.new_group`` by every rank in the same order, as torch.distributed requires; a rank keeps
the group it belongs to. When TP spans the whole world the TP group is ``dist.group.WORLD``
and no subgroup is made. Subgroups serve eager and ``torch.compile`` collectives alike on the
neuron process group (checked on device: all-reduce and all-gather over TP2 and CFG2
subgroups, exact in both modes). Context and data parallelism are not implemented yet.

Without an initialized process group the mesh is "spec-only": sizes are known,
so layers can be built with sharded shapes (on meta, for example), but anything
that needs a rank or a collective at tp > 1 raises.
"""

from __future__ import annotations

import torch.distributed as dist

from difflet.pipeline.parallel_mesh import MeshSpec

_MESH_SPEC: MeshSpec | None = None
_TP_GROUP: dist.ProcessGroup | None = None
#: This rank's process group per axis of size > 1 ("tp", "cfg").
_AXIS_GROUPS: dict[str, dist.ProcessGroup] = {}
_SUPPORTED_AXES = ("tp", "cfg")
_NOT_INITIALIZED = "neuron parallel mesh is not initialized; call init_parallel_mesh() first"


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


def mesh_spec_from_config(config) -> MeshSpec:
    """The mesh for a MeshSpec, a DiffletParallelConfig or a model config.

    A model config carries ``tp_degree``, ``cfg_parallel_enabled``,
    ``context_parallel_enabled`` and ``dp_degree``; as on Trainium and TPU, the
    cp degree is whatever the world leaves after tp * cfg * dp.
    """
    if isinstance(config, MeshSpec):
        return config
    spec = getattr(config, "mesh_spec", None)
    if isinstance(spec, MeshSpec):
        return spec
    tp = int(getattr(config, "tp_degree", 1) or 1)
    cfg = 2 if bool(getattr(config, "cfg_parallel_enabled", False)) else 1
    dp = int(getattr(config, "dp_degree", 1) or 1)
    cp = 1
    if bool(getattr(config, "context_parallel_enabled", False)):
        world = _world_size()
        denom = tp * cfg * dp
        if world % denom != 0:
            raise ValueError(
                f"world_size {world} is not divisible by tp*cfg*dp = {denom} "
                f"(tp={tp}, cfg={cfg}, dp={dp})"
            )
        cp = world // denom
    return MeshSpec(dp=dp, cfg=cfg, cp=cp, tp=tp)


def _check_supported(spec: MeshSpec) -> None:
    unsupported = [
        f"{axis}={spec.axis_size(axis)}" for axis in ("dp", "cp") if spec.axis_size(axis) > 1
    ]
    if unsupported:
        raise NotImplementedError(
            "the neuron backend supports tensor and CFG parallelism; unsupported: "
            + ", ".join(unsupported)
        )


def _bind_groups(spec: MeshSpec) -> dict[str, dist.ProcessGroup]:
    """This rank's group for every axis of size > 1; empty without a process group."""
    if not _dist_ready():
        return {}
    world = dist.get_world_size()
    if spec.world_size != world:
        raise ValueError(
            f"neuron parallel mesh {spec} needs {spec.world_size} ranks but the process "
            f"group has {world}"
        )
    if dist.get_backend() == "neuron":
        import torch_neuronx  # noqa: F401  registers the PrivateUse1 functional collectives
    rank = dist.get_rank()
    groups: dict[str, dist.ProcessGroup] = {}
    for axis in _SUPPORTED_AXES:
        size = spec.axis_size(axis)
        if size == 1:
            continue
        if size == world:
            groups[axis] = dist.group.WORLD
            continue
        for members in spec.axis_groups(axis):  # every rank creates every group, in order
            group = dist.new_group(members)
            if rank in members:
                groups[axis] = group
    return groups


def init_parallel_mesh(config) -> None:
    """Record the mesh once per process; idempotent for an equal spec."""
    global _MESH_SPEC, _TP_GROUP, _AXIS_GROUPS
    spec = mesh_spec_from_config(config)
    _check_supported(spec)
    if _MESH_SPEC is not None:
        if spec != _MESH_SPEC:
            raise RuntimeError(
                f"neuron parallel mesh already initialized with {_MESH_SPEC}; "
                f"cannot re-initialize with {spec}"
            )
        if not _AXIS_GROUPS and _dist_ready():
            # A spec-only mesh picks up the process groups once they exist.
            _AXIS_GROUPS = _bind_groups(spec)
            _TP_GROUP = _AXIS_GROUPS.get("tp")
        return
    groups = _bind_groups(spec)
    _MESH_SPEC = spec
    _AXIS_GROUPS = groups
    _TP_GROUP = groups.get("tp")


def is_mesh_initialized() -> bool:
    return _MESH_SPEC is not None


def get_mesh_spec() -> MeshSpec:
    if _MESH_SPEC is None:
        raise RuntimeError(_NOT_INITIALIZED)
    return _MESH_SPEC


def get_tp_size() -> int:
    if _MESH_SPEC is not None:
        return _MESH_SPEC.tp
    if _world_size() == 1:
        return 1
    raise RuntimeError(_NOT_INITIALIZED)


def get_tp_group() -> dist.ProcessGroup | None:
    """The TP process group, or None at tp == 1."""
    if get_tp_size() == 1:
        return None
    if _TP_GROUP is None:
        raise RuntimeError(
            f"neuron parallel mesh {_MESH_SPEC} was initialized without a process group "
            "(spec-only); initialize torch.distributed before init_parallel_mesh() to run "
            "collectives or query ranks at tp > 1"
        )
    return _TP_GROUP


def get_tp_rank() -> int:
    if get_tp_size() == 1:
        return 0
    return dist.get_rank(get_tp_group())


def get_axis_size(axis: str) -> int:
    return get_mesh_spec().axis_size(axis) if _MESH_SPEC is not None else 1


def get_axis_group(axis: str) -> dist.ProcessGroup:
    """This rank's group on ``axis`` (size > 1 only)."""
    spec = get_mesh_spec()
    if spec.axis_size(axis) == 1:
        raise RuntimeError(f"the {axis} axis of {spec} has size 1; it has no group")
    if axis not in _AXIS_GROUPS:
        raise RuntimeError(
            f"neuron parallel mesh {spec} has no {axis} process group (spec-only); initialize "
            "torch.distributed before init_parallel_mesh()"
        )
    return _AXIS_GROUPS[axis]


def get_axis_rank(axis: str) -> int:
    """This rank's coordinate on ``axis`` (0 at size 1 or without a process group)."""
    if _MESH_SPEC is None or _MESH_SPEC.axis_size(axis) == 1 or not _dist_ready():
        return 0
    return _MESH_SPEC.axis_rank(dist.get_rank(), axis)


def destroy_parallel_mesh() -> None:
    """Reset module state (tests only; does not destroy torch process groups)."""
    global _MESH_SPEC, _TP_GROUP, _AXIS_GROUPS
    _MESH_SPEC = None
    _TP_GROUP = None
    _AXIS_GROUPS = {}


__all__ = [
    "destroy_parallel_mesh",
    "get_axis_group",
    "get_axis_rank",
    "get_axis_size",
    "get_mesh_spec",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_size",
    "init_parallel_mesh",
    "is_mesh_initialized",
    "mesh_spec_from_config",
]
