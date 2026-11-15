"""Opt-in structured timing events for ``difflet bench`` (design §8.1).

Set ``DIFFLET_BENCH_TIMING_DIR`` to a directory and every process of a ``difflet compile`` or
``difflet generate`` (the driver, each torchrun rank, each stage launch) appends JSON lines to
``<dir>/<role>-<pid>.jsonl``. An environment variable rather than a flag so it reaches every staged
subprocess unchanged, like ``DIFFLET_DUMP_LATENTS_DIR``. Unset, every function here returns at
its first line: the engine's code path is the one it has without bench.

Every event carries ``t_ns`` (``time.monotonic_ns()``, one clock for all processes on the host),
``pid``, ``rank`` and ``world``, then its own fields:

- ``process``   first event of a process: ``argv``, ``role``
- ``args``      the driver's final arguments, after validation and ``--mode`` resolution
- ``span``      ``name``, ``phase`` ``begin`` or ``end``, plus fields (``request``, ``stage``, ...)
- ``step``      a denoise iteration ended (after the scheduler step), ``index`` within the request
- ``forward``   one DiT forward, ``seconds`` as measured by the backend (synced on neuron)
- ``evidence``  what actually ran: exec mode, compiled blocks, unwarmed shapes, CPU fallbacks,
                phase seconds, compile-cache counters, device memory
- ``output``    a file the engine wrote: ``path``, ``kind``, ``sha256``

Nothing here may break a run: every write is wrapped, and a failure is reported once on stderr.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Iterator

TIMING_ENV = "DIFFLET_BENCH_TIMING_DIR"

_handle = None
_handle_pid: int | None = None
_failed = False


def enabled() -> bool:
    return bool(os.environ.get(TIMING_ENV))


def _rank_world() -> tuple[int, int]:
    # Never import torch here: the driver records events before it imports torch, and pulling it
    # in (seconds) would land inside the timed span.
    dist = sys.modules.get("torch.distributed")
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:  # noqa: BLE001 - a half-initialised process group
        pass
    try:
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 0, 1


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _open():
    global _handle, _handle_pid
    pid = os.getpid()
    if _handle is not None and _handle_pid == pid:
        return _handle
    directory = os.environ[TIMING_ENV]
    os.makedirs(directory, exist_ok=True)
    role = "rank" if "LOCAL_RANK" in os.environ else "proc"
    rank = os.environ.get("RANK", "0")
    path = os.path.join(directory, f"{role}{rank if role == 'rank' else ''}-{pid}.jsonl")
    _handle = open(path, "a", encoding="utf-8", buffering=1)  # line buffered: one write per event
    _handle_pid = pid
    first = {"argv": list(sys.argv), "role": role}
    _write("process", first)
    return _handle


def _write(kind: str, fields: dict[str, Any]) -> None:
    rank, world = _rank_world()
    record = {"event": kind, "t_ns": time.monotonic_ns(), "pid": os.getpid(), "rank": rank,
              "world": world}
    record.update(_jsonable(fields))
    _handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def event(kind: str, **fields: Any) -> None:
    """Append one event; a no-op when timing is off."""
    global _failed
    if _failed or not enabled():
        return
    try:
        _open()
        _write(kind, fields)
    except Exception as exc:  # noqa: BLE001 - instrumentation must never fail a run
        _failed = True
        print(f"[bench-timing] disabled after an error: {exc!r}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def span(name: str, **fields: Any) -> Iterator[None]:
    """``span`` begin/end events around a block (the end is written on error too)."""
    if not enabled():
        yield
        return
    event("span", name=name, phase="begin", **fields)
    ok = False
    try:
        yield
        ok = True
    finally:
        event("span", name=name, phase="end", ok=ok, **fields)


def step_hook(previous: Callable[[], None] | None = None) -> Callable[[], None]:
    """A ``step_hook`` (called once per denoise iteration) that records ``step`` events, counting
    from 0 per request (reset with :func:`new_request`), then calls ``previous``."""
    def hook() -> None:
        global _step_index
        event("step", index=_step_index)
        _step_index += 1
        if previous is not None:
            previous()
    return hook


def forward_hook(previous: Callable[[float], None] | None = None) -> Callable[[float], None]:
    """A ``forward_hook`` (called after each DiT forward with its seconds) that records it."""
    def hook(seconds: float) -> None:
        event("forward", seconds=float(seconds))
        if previous is not None:
            previous(seconds)
    return hook


_step_index = 0


def new_request() -> None:
    """Restart the step count (call at the start of every request)."""
    global _step_index
    _step_index = 0


def enable_compile_counters() -> None:
    """Turn on torch-neuronx's compilation-cache counters for this process (before the runtime
    is prepared, so the first compile is counted). A no-op when timing is off."""
    if not enabled():
        return
    try:
        from torch_neuronx import metrics

        metrics.set_enabled(True)
    except Exception:  # noqa: BLE001 - not every torch-neuronx build ships metrics
        pass


_COUNTERS = ("TotalCompilations", "PersistentHits", "PersistentMisses", "InMemoryHits",
             "InMemoryMisses")


def compile_counters() -> dict[str, int] | None:
    """``CompilationCache.*`` counters (None when unavailable or never enabled)."""
    try:
        from torch_neuronx import metrics

        return {name: int(metrics.get_counter_value(f"CompilationCache.{name}") or 0)
                for name in _COUNTERS}
    except Exception:  # noqa: BLE001
        return None


def device_memory() -> dict[str, float] | None:
    """This process's allocated and peak device memory in GiB (None off the neuron device)."""
    try:
        import torch_neuronx

        gib = float(1 << 30)
        return {"allocated_gib": torch_neuronx.memory_allocated() / gib,
                "max_allocated_gib": torch_neuronx.max_memory_allocated() / gib}
    except Exception:  # noqa: BLE001
        return None


def evidence(**fields: Any) -> None:
    """The ``evidence`` event, with the compile counters and device memory added.

    Those two are read only in a process that has already imported torch-neuronx (a rank): in the
    driver or a host stage, importing it and asking for the current device's memory could bring up
    the Neuron runtime there, which a measurement must not do. Safe to call from any process.
    """
    if not enabled():
        return
    on_device = "torch_neuronx" in sys.modules
    torch = sys.modules.get("torch")
    event("evidence", compile_counters=compile_counters() if on_device else None,
          device_memory=device_memory() if on_device else None,
          torch_threads=torch.get_num_threads() if torch is not None else None, **fields)


def file_sha256(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def output(path: str, kind: str) -> None:
    """Record a file the engine wrote, hashed after the timed work (the hash costs time)."""
    if enabled():
        event("output", path=os.path.abspath(path), kind=kind, sha256=file_sha256(path))
