"""Run a function on N CPU ranks joined by a gloo process group (not collected by pytest).

``run_ranks(target, *args)`` spawns one process per rank and calls
``target(rank, world_size, *args)`` in each, after ``init_process_group("gloo")``.
The target must live in an importable module, ``tests.unit.backends._neuron_workers``:
under ``--import-mode=importlib`` a function defined in a test module cannot be
unpickled in a spawned child. Results come back as a list ordered by rank, so they
must pickle; workers return plain Python data.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

_CHILD_ENV = {"DIFFLET_BACKEND": "neuron", "DIFFLET_DISABLE_PREWARM": "1"}
# torchrun variables would make bind_core / init_process_group treat the child as
# a torchrun rank (and bind a NeuronCore); gloo workers must never touch the device.
_CLEARED_ENV = (
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "TORCHELASTIC_RUN_ID",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NEURON_RT_VISIBLE_CORES",
)
_GRACE_S = 5.0


def _entry(target, rank, world_size, init_file, env, args, results):
    for name in _CLEARED_ENV:
        os.environ.pop(name, None)
    os.environ.update(env)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        import torch.distributed as dist

        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
        )
        try:
            value = target(rank, world_size, *args)
        finally:
            dist.destroy_process_group()
        results.put((rank, True, value))
    except BaseException:  # every failure, including SystemExit, goes back to the parent
        results.put((rank, False, traceback.format_exc()))


def run_ranks(
    target: Callable[..., Any],
    *args: Any,
    world_size: int = 4,
    timeout: float = 180.0,
    env: dict[str, str] | None = None,
) -> list[Any]:
    """Run ``target(rank, world_size, *args)`` on ``world_size`` gloo ranks; results by rank.

    Any rank error, a rank that exits without reporting, or the timeout kills the
    remaining ranks and raises AssertionError carrying every rank's traceback.
    """
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    child_env = {**_CHILD_ENV, **(env or {})}
    values: dict[int, Any] = {}
    failures: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="difflet_gloo_") as tmp:
        init_file = os.path.join(tmp, "rendezvous")
        procs = [
            ctx.Process(
                target=_entry,
                args=(target, rank, world_size, init_file, child_env, args, results),
                daemon=True,
            )
            for rank in range(world_size)
        ]
        for proc in procs:
            proc.start()
        deadline = time.monotonic() + timeout
        first_failure: float | None = None
        try:
            while len(values) + len(failures) < world_size:
                now = time.monotonic()
                if now > deadline or (first_failure and now > first_failure + _GRACE_S):
                    break
                try:
                    rank, ok, payload = results.get(timeout=0.5)
                except queue_mod.Empty:
                    for rank, proc in enumerate(procs):
                        reported = rank in values or rank in failures
                        if not reported and proc.exitcode not in (None, 0):
                            failures[rank] = f"exited with code {proc.exitcode} without reporting"
                            first_failure = first_failure or time.monotonic()
                    continue
                if ok:
                    values[rank] = payload
                else:
                    failures[rank] = payload
                    first_failure = first_failure or time.monotonic()
        finally:
            if not failures and len(values) == world_size:
                for proc in procs:  # let finished ranks exit cleanly before any kill
                    proc.join(timeout=30)
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
            for proc in procs:
                proc.join(timeout=10)
            results.close()
            results.join_thread()
    missing = [r for r in range(world_size) if r not in values and r not in failures]
    for rank in missing:
        failures[rank] = "killed: no result (timed out or waiting on a failed rank)"
    if failures:
        report = "\n".join(f"--- rank {r} ---\n{failures[r]}" for r in sorted(failures))
        raise AssertionError(f"{target.__name__} failed on ranks {sorted(failures)}:\n{report}")
    return [values[rank] for rank in range(world_size)]
