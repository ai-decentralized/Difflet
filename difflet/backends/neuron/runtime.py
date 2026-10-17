"""Neuron backend runtime: TorchNeuron, the PyTorch-native Neuron device.

Execution model: one process per NeuronCore (torchrun MPMD), eager or
``torch.compile(backend="neuron")``, no ahead-of-time artifact. Each rank is
bound to one core through ``NEURON_RT_VISIBLE_CORES`` before the Neuron runtime
starts, and ranks communicate through the ``neuron`` process group.

Within one launch the backend runs tensor parallelism, with or without sequence
parallelism, CFG parallelism, and context parallelism in every cp_mode (gather_kv, ulysses,
ring); each model's application rejects what its own modeling does not run (Qwen-Image: SP and
CFG). Data parallelism is replicas, separate torchrun launches of the CLI's request router, so
``dp_degree > 1`` within one launch raises (``check_parallel_supported``).

``prepare_runtime`` first points torch-neuronx's persistent NEFF cache at the Difflet
compile cache (``compile.configure_compile_cache``): the runtime reads it once, when it
first compiles, so it must be set before any device work. A runtime already started by
earlier device work is refused rather than left on ``/tmp/neff_cache`` and every core.
The same call sets the HLO cache location (``TORCH_NEURONX_HLO_CACHE_DIR``), which this late
moves the HLO cache entries but not the lock directory torch-neuronx fixed when torch was
imported; the CLI launcher exports it to its ranks before they start (``compile``'s module
docstring).
"""

from __future__ import annotations

import datetime
import math
import os
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

from difflet import envs
from difflet.backends.base import BackendCapabilities, BackendRuntime
from difflet.backends.neuron.compile import configure_compile_cache

PROCESS_GROUP_BACKEND = "neuron"
VISIBLE_CORES_ENV = "NEURON_RT_VISIBLE_CORES"
PG_TIMEOUT_ENV = "DIFFLET_NEURON_PG_TIMEOUT_S"

# Set once prepare_runtime has configured this process; a later call (a second pipeline)
# finds the runtime it started and goes ahead.
_runtime_prepared = False


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
        global _runtime_prepared
        if not _runtime_prepared and _neuron_runtime_initialized():
            raise RuntimeError(
                "the Neuron runtime is already initialised in this process, so prepare_runtime "
                f"can no longer bind a NeuronCore ({VISIBLE_CORES_ENV}) or move the NEFF cache "
                "(TORCH_NEURONX_NEFF_CACHE_DIR) off /tmp/neff_cache; call "
                "get_backend('neuron').prepare_runtime(...) before anything touches the neuron "
                "device (a tensor .to('neuron'), a torch.neuron call, a prewarm)"
            )
        configure_compile_cache()
        bind_core()
        init_process_group()
        _runtime_prepared = True


def create_backend() -> NeuronBackend:
    return NeuronBackend()


def check_parallel_supported(parallel: Any) -> None:
    """Reject parallel modes the neuron backend does not implement (it runs TP with or without
    sequence parallelism, CFG, and CP in every cp_mode: gather_kv, ulysses, ring)."""
    unsupported = []
    if parallel.dp_degree > 1:
        unsupported.append(f"dp_degree={parallel.dp_degree}")
    if unsupported:
        raise NotImplementedError(
            "the neuron backend supports tensor, sequence, context and CFG parallelism; "
            "unsupported: "
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


def _neuron_runtime_initialized() -> bool:
    """Whether this process has started the Neuron runtime; never imports torch_neuronx."""
    torch_neuronx = sys.modules.get("torch_neuronx")
    return torch_neuronx is not None and bool(torch_neuronx.is_neuron_runtime_initialized())


def process_group_timeout() -> datetime.timedelta | None:
    """The ``neuron`` process group's collective timeout from ``DIFFLET_NEURON_PG_TIMEOUT_S``.

    The value is a positive number of seconds; unset or blank returns None, which leaves
    torch's default (30 min for non-NCCL groups). The neuron backend's watchdog fails a
    collective that waits longer than this, so a launch whose ranks can wait on a peer for
    more than 30 min (a cold compile of a large block graph) raises it.
    """
    raw = envs.DIFFLET_NEURON_PG_TIMEOUT_S
    if raw is None or not raw.strip():
        return None
    try:
        seconds = float(raw)
    except ValueError:
        seconds = math.nan
    if not (math.isfinite(seconds) and seconds > 0):
        raise ValueError(f"{PG_TIMEOUT_ENV}={raw!r} is not a positive number of seconds")
    return datetime.timedelta(seconds=seconds)


def init_process_group() -> None:
    """Initialise the ``neuron`` process group for multi-rank launches; no-op for one rank.

    ``DIFFLET_NEURON_PG_TIMEOUT_S`` sets the group's timeout (``process_group_timeout``);
    unset, ``timeout`` is not passed and torch's default applies.
    """
    if (_env_int("WORLD_SIZE") or 1) <= 1:
        return
    timeout = process_group_timeout()
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
    if timeout is None:
        dist.init_process_group(PROCESS_GROUP_BACKEND)
    else:
        dist.init_process_group(PROCESS_GROUP_BACKEND, timeout=timeout)


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
