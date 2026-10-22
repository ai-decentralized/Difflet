"""The one module on the TPU path that talks to a TPU runtime library.

Phase 1 of docs/plans/2026-10-22-torchtpu-migration.md. `torch_xla` is
deprecated and its repo is archived 2026-11-15, and TorchTPU's compilation
cache fixes the warm-start gap that `core/application_base.py` documents as
unfixable under torch_xla. Rather than edit the backend in place (which would
destroy the ability to A/B against the recorded `benchmark/v5e/` numbers) or
fork a second backend (which would mean forking the 1763 lines of
`models/*/tpu_application.py` where most of the call sites live), every TPU
call site routes through here and the runtime is chosen by one env var.

    DIFFLET_TPU_RUNTIME=torch_xla   (default, what ships today)
    DIFFLET_TPU_RUNTIME=torch_tpu   (Phase 2)

**Phase 1 changes no behavior.** Only the torch_xla implementations exist; the
torch_tpu ones raise, pointing at the phase that fills them in. The gate for
this phase is `benchmark.bench --backend tpu` reproducing the recorded numbers,
which it can only do if these wrappers are exactly what the call sites did
before — so each one documents the call it replaces.

Imports stay inside the functions, as they were at every call site before: the
TPU backend has to be importable on a host with no TPU runtime at all (the
CPU backend, unit tests on a laptop, `difflet plan`).

**Two torch_xla call sites are deliberately NOT routed through here**, both in
`difflet/models/flux/pipeline.py` (the `XLA_AVAILABLE` / `xm.mark_step()` pair
diffusers ships and this fork carried over). That file defines
`NeuronFluxPipeline`, which `difflet/models/flux/application.py` imports on
**Trainium**, and `requirements-neuron.lock` pins `torch-xla==2.9.0`, so
`is_torch_xla_available()` is true there and that `mark_step` is Trainium's,
not the TPU backend's. Routing it here would drag the production backend
through a TPU module for no gain. Leave it.
"""

from __future__ import annotations

import os

#: Selected once per process. Changing the env var mid-run does nothing, which
#: is deliberate: the runtime is process-global state and a mid-flight switch
#: would leave already-created tensors on the other one's device.
_RUNTIME: str | None = None

TORCH_XLA = "torch_xla"
TORCH_TPU = "torch_tpu"
_RUNTIMES = (TORCH_XLA, TORCH_TPU)


def runtime_name() -> str:
    """Which TPU runtime this process uses."""
    global _RUNTIME
    if _RUNTIME is None:
        chosen = os.environ.get("DIFFLET_TPU_RUNTIME", TORCH_XLA).strip() or TORCH_XLA
        if chosen not in _RUNTIMES:
            raise ValueError(
                f"DIFFLET_TPU_RUNTIME={chosen!r} is not one of {_RUNTIMES}"
            )
        _RUNTIME = chosen
    return _RUNTIME


def _not_yet(what: str):
    return NotImplementedError(
        f"{what} has no torch_tpu implementation yet — that is Phase 2 of "
        "docs/plans/2026-10-22-torchtpu-migration.md. Unset DIFFLET_TPU_RUNTIME "
        "to use torch_xla."
    )


# ------------------------------------------------------------------ device


def device():
    """This process's TPU device. Replaces ``torch_xla.device()``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla

        return torch_xla.device()
    raise _not_yet("device()")


def device_kind() -> str:
    """Chip generation (e.g. ``TPU v5 lite``), or ``unknown``.

    Replaces ``xm.xla_device_hw(xm.xla_device())``. Used for capacity
    decisions — v5e has 16 GB HBM per chip against v5p's ~95 GB, which is the
    difference between a model fitting at tp=1 and not.
    """
    try:
        if runtime_name() == TORCH_XLA:
            import torch_xla.core.xla_model as xm

            return str(xm.xla_device_hw(xm.xla_device()) or "unknown")
        raise _not_yet("device_kind()")
    except Exception:  # noqa: BLE001 — a probe must never break import
        return "unknown"


# -------------------------------------------------------------- topology


def rank() -> int:
    """Global ordinal of this process. Replaces ``xr.global_ordinal()``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.runtime as xr

        return int(xr.global_ordinal())
    raise _not_yet("rank()")


def world_size() -> int:
    """Replaces ``xr.world_size()``.

    torch_xla 2.9 removed ``xm.get_ordinal()``/``xm.xrt_world_size()``; the
    replacements live on ``torch_xla.runtime``.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.runtime as xr

        return int(xr.world_size())
    raise _not_yet("world_size()")


def initialize_replica(replica_rank: int, replica_count: int) -> None:
    """Give this process its own chip, before anything else touches one.

    Replaces ``torch_xla._internal.pjrt.initialize_multiprocess`` in
    ``serving/engines/resident_worker.py``, which reimplements what
    ``torch_xla.launch`` does internally because launch owns the process
    lifecycle and a resident worker has to outlive a single call.
    """
    if runtime_name() == TORCH_XLA:
        from torch_xla._internal import pjrt

        pjrt.initialize_multiprocess(replica_rank, replica_count)
        return
    raise _not_yet("initialize_replica()")


# ----------------------------------------------------------- synchronization


def step() -> None:
    """Flush the pending graph. Replaces ``xm.mark_step()``.

    Dispatches the accumulated work; it does **not** wait for it. Call sites
    that need the result on the host follow it with :func:`wait`.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        return
    raise _not_yet("step()")


def wait() -> None:
    """Block until queued device work finishes. Replaces ``xm.wait_device_ops()``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        xm.wait_device_ops()
        return
    raise _not_yet("wait()")


def sync() -> None:
    """``step()`` then ``wait()`` — the pair several call sites already used."""
    step()
    wait()


# ------------------------------------------------------------- collectives


def broadcast(tensors: list, *, src: int = 0) -> None:
    """Broadcast a list of device tensors in place from rank ``src``.

    Replaces ``xm.collective_broadcast(payload, root_ordinal=0)``, which every
    ``tpu_application`` uses to fan an encoded prompt out from the encoder rank
    together with a status flag.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        xm.collective_broadcast(tensors, root_ordinal=src)
        return
    raise _not_yet("broadcast()")


def all_gather(tensor, *, dim: int, groups: list[list[int]]):
    """Replaces ``xm.all_gather(..., groups=, pin_layout=False)``.

    ``pin_layout=False`` is load-bearing: pinned layouts lower to an
    ``mhlo.all_gather`` that the TPU compiler handles differently.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return xm.all_gather(tensor, dim=dim, groups=groups, pin_layout=False)
    raise _not_yet("all_gather()")


def all_reduce_sum(tensor, *, groups: list[list[int]]):
    """Replaces ``xm.all_reduce(xm.REDUCE_SUM, ..., groups=, pin_layout=False)``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return xm.all_reduce(xm.REDUCE_SUM, tensor, groups=groups, pin_layout=False)
    raise _not_yet("all_reduce_sum()")


def reduce_scatter_sum(tensor, *, dim: int, shard_count: int, groups: list[list[int]]):
    """Replaces ``xm.reduce_scatter(xm.REDUCE_SUM, ..., scale=1.0, ...)``."""
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return xm.reduce_scatter(
            xm.REDUCE_SUM,
            tensor,
            scale=1.0,
            scatter_dim=dim,
            shard_count=shard_count,
            groups=groups,
            pin_layout=False,
        )
    raise _not_yet("reduce_scatter_sum()")


# ------------------------------------------------------------------ memory


def memory_info(dev=None) -> dict:
    """HBM counters for ``dev``. Replaces ``xm.get_memory_info(device)``.

    Feeds the "resident HBM / chip" column in ``benchmark/v5e/RESULTS.md``,
    which is how a model is judged to fit: a v5e chip has 16 GB against a
    v5p's ~95 GB. Returns the runtime's own dict, so callers keep reading
    ``peak_bytes_used`` / ``bytes_used`` and tolerate either being absent.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.core.xla_model as xm

        return dict(xm.get_memory_info(dev if dev is not None else device()))
    raise _not_yet("memory_info()")


# --------------------------------------------------------------- precision


#: XLA's default matmul precision on TPU is NOT fp32: the MXU multiplies fp32
#: inputs at reduced precision unless told otherwise, measured at ~1e-2
#: absolute error per matmul on v5e against ~2e-6 at "highest". Compounded
#: through a DiT that makes the parity target unreachable, and the failure is
#: silent — every shape is right and nothing errors.
#:
#: Phase 0c measured the TorchTPU side: its `medium` default holds cosine far
#: inside the 0.9995 bar on a 4-block FLUX proxy at 2.42x the speed of
#: `highest`, so the Phase 2 default is expected to change. Until a real model
#: confirms it in Phase 4, this stays where torch_xla needs it.
DEFAULT_MATMUL_PRECISION = "highest"


def set_matmul_precision(precision: str = DEFAULT_MATMUL_PRECISION) -> str:
    """Set matmul precision process-wide, returning the previous value.

    Deliberately explicit rather than done at import time: it is process-global
    state, and silently mutating it on import would be worse than the bug it
    prevents.
    """
    if runtime_name() == TORCH_XLA:
        import torch_xla.backends as backends

        previous = backends.get_mat_mul_precision()
        backends.set_mat_mul_precision(precision)
        return previous
    raise _not_yet("set_matmul_precision()")


__all__ = [
    "DEFAULT_MATMUL_PRECISION",
    "TORCH_TPU",
    "TORCH_XLA",
    "all_gather",
    "all_reduce_sum",
    "broadcast",
    "device",
    "device_kind",
    "initialize_replica",
    "memory_info",
    "rank",
    "reduce_scatter_sum",
    "runtime_name",
    "set_matmul_precision",
    "step",
    "sync",
    "wait",
    "world_size",
]
