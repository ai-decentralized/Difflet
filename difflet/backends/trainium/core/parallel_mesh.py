"""Orthogonal {dp, cfg, cp, tp} process groups for the Trainium backend.

One singleton per process (same pattern as the NxDI-fork
``attention_process_groups.py``). ``init_parallel_mesh(config)`` derives the
MeshSpec from the model config plus NxD's already-initialized tp/world sizes,
then builds ONE torch.distributed group per NON-trivial axis with the full
axis mesh in ``xla_pg_options`` (that mesh becomes the collective's replica
groups under SPMD tracing). Trivial axes build no group at all — a
guidance-distilled model (cfg=1) never constructs a cfg group.

TP is NOT built here: with tp innermost, the mesh's tp axis coincides with
NxD's ``tensor_model_parallel_group``, which every TP layer already uses.

The dp axis group exists only when dp > 1 and intentionally has NO consumer
in any per-layer / per-step code path; it is reserved for the future DP
feature (replica-level request routing / load-time weight broadcast).
"""

from __future__ import annotations

import torch
import torch.distributed

from difflet.pipeline.parallel_mesh import MeshSpec

_MESH_SPEC: MeshSpec | None = None
_AXIS_GROUPS: dict[str, object] = {}


def _nxd_tp_size() -> int:
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_size,
    )

    return int(get_tensor_model_parallel_size())


def _nxd_world_group():
    from neuronx_distributed.parallel_layers.parallel_state import get_world_group

    return get_world_group()


def mesh_spec_from_config(config) -> MeshSpec:
    """Derive the mesh from model-config flags + NxD's tp/world sizes.

    cp is inferred as ``world / (tp * cfg * dp)`` when context parallelism is
    enabled, because the backbone configs carry only the boolean flag (the
    explicit degree never reached the backend pre-refactor either). The
    product == world_size invariant is asserted regardless.
    """
    tp = _nxd_tp_size()
    world = int(_nxd_world_group().size())
    cfg = 2 if bool(getattr(config, "cfg_parallel_enabled", False)) else 1
    dp = int(getattr(config, "dp_degree", 1))
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
    """Build the per-axis subgroups once per process (idempotent for equal specs)."""
    global _MESH_SPEC
    spec = mesh_spec_from_config(config)
    if _MESH_SPEC is not None:
        if spec != _MESH_SPEC:
            raise RuntimeError(
                f"parallel mesh already initialized with {_MESH_SPEC}; "
                f"cannot re-initialize with {spec}"
            )
        return
    _MESH_SPEC = spec
    for axis in ("cfg", "cp", "dp"):
        if spec.axis_size(axis) > 1:
            mesh = spec.axis_groups(axis)
            _AXIS_GROUPS[axis] = torch.distributed.new_group(
                mesh[0], pg_options={"xla_pg_options": {"mesh": mesh}}
            )


def _require_spec() -> MeshSpec:
    assert _MESH_SPEC is not None, "parallel mesh is not initialized"
    return _MESH_SPEC


def get_mesh_spec() -> MeshSpec:
    return _require_spec()


def _axis_group(axis: str):
    spec = _require_spec()
    assert axis in _AXIS_GROUPS, (
        f"{axis} axis is trivial ({axis}={spec.axis_size(axis)}); "
        f"no {axis} group exists"
    )
    return _AXIS_GROUPS[axis]


def get_cfg_group():
    return _axis_group("cfg")


def get_cp_group():
    return _axis_group("cp")


def get_dp_group():
    return _axis_group("dp")


def get_cp_mesh() -> list[list[int]]:
    """Replica groups of the cp axis (for ring collectives)."""
    return _require_spec().axis_groups("cp")


def get_cfg_rank_spmd(global_rank: torch.Tensor) -> torch.Tensor:
    """cfg coordinate of a traced global rank: (rank // (T*C)) % G."""
    spec = _require_spec()
    return torch.remainder(
        torch.div(global_rank, spec.tp * spec.cp, rounding_mode="floor"), spec.cfg
    ).to(torch.int32)


def get_cp_rank_spmd(global_rank: torch.Tensor) -> torch.Tensor:
    """cp coordinate of a traced global rank: (rank // T) % C."""
    spec = _require_spec()
    return torch.remainder(
        torch.div(global_rank, spec.tp, rounding_mode="floor"), spec.cp
    ).to(torch.int32)


def destroy_parallel_mesh() -> None:
    """Reset module state (unit tests only; does not destroy torch groups)."""
    global _MESH_SPEC
    _MESH_SPEC = None
    _AXIS_GROUPS.clear()
