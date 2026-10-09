"""Neuron backend runtime: TorchNeuron, the PyTorch-native Neuron device.

Execution model: one process per NeuronCore (torchrun MPMD), eager or
``torch.compile(backend="neuron")``, no ahead-of-time artifact. Each rank is
bound to one core through ``NEURON_RT_VISIBLE_CORES`` before the Neuron runtime
starts, and ranks communicate through the ``neuron`` process group.

Phase 1 supports tensor parallelism only; context, CFG, sequence and data
parallelism raise until their collectives are validated on this backend.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

from difflet.backends.base import BackendCapabilities, BackendRuntime

PROCESS_GROUP_BACKEND = "neuron"
VISIBLE_CORES_ENV = "NEURON_RT_VISIBLE_CORES"


class NeuronBackend(BackendRuntime):
    name = "neuron"
    capabilities = BackendCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=True,
    )

    def prepare_runtime(self, parallel: Any) -> None:
        if parallel is not None:
            check_parallel_supported(parallel)
            world_size = _env_int("WORLD_SIZE") or 1
            if parallel.world_size != world_size:
                raise ValueError(
                    f"parallel config needs {parallel.world_size} ranks but the process was "
                    f"launched with WORLD_SIZE={world_size}; launch with "
                    f"torchrun --nproc-per-node {parallel.world_size}"
                )
        bind_core()
        init_process_group()


def create_backend() -> NeuronBackend:
    return NeuronBackend()


def check_parallel_supported(parallel: Any) -> None:
    """Reject parallel modes that phase 1 of the neuron backend does not implement."""
    unsupported = []
    if parallel.cp_degree > 1:
        unsupported.append(f"cp_degree={parallel.cp_degree}")
    if parallel.cfg_parallel_enabled:
        unsupported.append("cfg_parallel_enabled")
    if parallel.sp_enabled:
        unsupported.append("sp_enabled")
    if parallel.dp_degree > 1:
        unsupported.append(f"dp_degree={parallel.dp_degree}")
    if unsupported:
        raise NotImplementedError(
            "the neuron backend supports tensor parallelism only; unsupported: "
            + ", ".join(unsupported)
        )


def parse_core_list(value: str | None) -> list[int]:
    """Parse ``NEURON_RT_VISIBLE_CORES`` ("2", "0-3", "0,2,3", "0-1,4-5")."""
    if not value or not value.strip():
        return []
    cores: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"invalid core range {part!r} in {VISIBLE_CORES_ENV}={value!r}")
            cores.extend(range(lo, hi + 1))
        else:
            cores.append(int(part))
    return cores


def bind_core(env: MutableMapping[str, str] | None = None) -> int | None:
    """Bind this process to one NeuronCore; must run before the Neuron runtime starts.

    Under torchrun, local rank r takes the r-th core of the cores visible to the
    launch (``NEURON_RT_VISIBLE_CORES``, or cores 0..N-1 when unset). A single
    process, or one already bound to exactly one core, is left unchanged.
    Returns the bound core, or None when nothing was bound.
    """
    env = os.environ if env is None else env
    local_world = _env_int("LOCAL_WORLD_SIZE", env)
    local_rank = _env_int("LOCAL_RANK", env)
    if local_world is None or local_world <= 1 or local_rank is None:
        return None
    visible = parse_core_list(env.get(VISIBLE_CORES_ENV))
    if len(visible) == 1:
        return visible[0]
    if visible and local_world > len(visible):
        raise ValueError(
            f"{local_world} local ranks but only {len(visible)} cores visible "
            f"({VISIBLE_CORES_ENV}={env[VISIBLE_CORES_ENV]!r})"
        )
    core = visible[local_rank] if visible else local_rank
    env[VISIBLE_CORES_ENV] = str(core)
    return core


def init_process_group() -> None:
    """Initialise the ``neuron`` process group for multi-rank launches; no-op for one rank."""
    if (_env_int("WORLD_SIZE") or 1) <= 1:
        return
    import torch.distributed as dist
    import torch_neuronx  # noqa: F401  registers the neuron device and process-group backend

    if dist.is_initialized():
        backend = dist.get_backend()
        if backend != PROCESS_GROUP_BACKEND:
            raise RuntimeError(
                f"process group already initialised with backend {backend!r}, "
                f"expected {PROCESS_GROUP_BACKEND!r}"
            )
        return
    dist.init_process_group(PROCESS_GROUP_BACKEND)


@contextmanager
def track_fallbacks() -> Iterator[list[str]]:
    """Collect the ops that silently fell back to CPU inside the block."""
    import torch
    import torch_neuronx

    torch_neuronx.clear_op_tracking()
    ops: list[str] = []
    yield ops
    torch.neuron.synchronize()
    ops.extend(torch_neuronx.get_fallback_ops())


def _env_int(name: str, env: MutableMapping[str, str] | None = None) -> int | None:
    value = (os.environ if env is None else env).get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
