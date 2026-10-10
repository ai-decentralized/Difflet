"""Tensor-parallel mesh state for the neuron (TorchNeuron) backend.

Phase 1 supports tensor parallelism only, and ``NeuronBackend.prepare_runtime``
requires the parallel config's world size to equal ``WORLD_SIZE``, so the TP
group is always the whole world: it is ``dist.group.WORLD`` and no subgroup is
created. torch_neuronx turns a whole-world group into replica groups without any
registration (``torch_neuronx/distributed/mesh_registry.py``,
``get_all_replica_groups``), so a compiled collective is identical on every rank.

Without an initialized process group the mesh is "spec-only": sizes are known,
so layers can be built with sharded shapes (on meta, for example), but anything
that needs a rank or a collective at tp > 1 raises.
"""

from __future__ import annotations

import torch.distributed as dist

from difflet.pipeline.parallel_mesh import MeshSpec

_MESH_SPEC: MeshSpec | None = None
_TP_GROUP: dist.ProcessGroup | None = None
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


def _check_tp_only(spec: MeshSpec) -> None:
    unsupported = [
        f"{axis}={spec.axis_size(axis)}" for axis in ("dp", "cfg", "cp") if spec.axis_size(axis) > 1
    ]
    if unsupported:
        raise NotImplementedError(
            "the neuron backend supports tensor parallelism only; unsupported: "
            + ", ".join(unsupported)
        )


def _bind_tp_group(spec: MeshSpec) -> dist.ProcessGroup | None:
    """The TP process group for ``spec``; None at tp == 1 or without a process group."""
    if not _dist_ready():
        return None
    world = dist.get_world_size()
    if spec.world_size != world:
        raise ValueError(
            f"neuron parallel mesh {spec} needs {spec.world_size} ranks but the process "
            f"group has {world}"
        )
    if dist.get_backend() == "neuron":
        import torch_neuronx  # noqa: F401  registers the PrivateUse1 functional collectives
    return dist.group.WORLD if spec.tp > 1 else None


def init_parallel_mesh(config) -> None:
    """Record the mesh once per process; idempotent for an equal spec."""
    global _MESH_SPEC, _TP_GROUP
    spec = mesh_spec_from_config(config)
    _check_tp_only(spec)
    if _MESH_SPEC is not None:
        if spec != _MESH_SPEC:
            raise RuntimeError(
                f"neuron parallel mesh already initialized with {_MESH_SPEC}; "
                f"cannot re-initialize with {spec}"
            )
        if _TP_GROUP is None:
            # A spec-only mesh picks up the process group once it exists.
            _TP_GROUP = _bind_tp_group(spec)
        return
    group = _bind_tp_group(spec)
    _MESH_SPEC = spec
    _TP_GROUP = group


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


def destroy_parallel_mesh() -> None:
    """Reset module state (tests only; does not destroy torch process groups)."""
    global _MESH_SPEC, _TP_GROUP
    _MESH_SPEC = None
    _TP_GROUP = None


__all__ = [
    "destroy_parallel_mesh",
    "get_mesh_spec",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_size",
    "init_parallel_mesh",
    "is_mesh_initialized",
    "mesh_spec_from_config",
]
