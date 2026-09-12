"""DP router: scatter (manifest) → spawn pinned workers → gather (markers).

Spec: docs/superpowers/specs/2026-07-06-dp-replication-routing-design.md §Router.
Workers are full difflet CLI invocations with dp=1 semantics; NEURON_RT_* are
plain-assigned (run_stage's setdefault must see the worker's range, not the
parent's).
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from difflet.cli.dp.claims import mark_failed, summarize
from difflet.cli.dp.requests_io import RequestSpec, write_manifest


def replica_core_ranges(dp: int, replica_cores: int) -> list[str]:
    ranges = []
    for w in range(dp):
        lo = w * replica_cores
        hi = lo + replica_cores - 1
        ranges.append(str(lo) if replica_cores == 1 else f"{lo}-{hi}")
    return ranges


_ROOT_COMM_PORTS_HANDED_OUT: set[int] = set()


def _free_localhost_port() -> int:
    """A currently-free TCP port on localhost, never one handed out earlier in
    this process (two back-to-back ``bind(0)`` calls can return the same port)."""
    for _ in range(64):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in _ROOT_COMM_PORTS_HANDED_OUT:
            _ROOT_COMM_PORTS_HANDED_OUT.add(port)
            return port
    raise RuntimeError("could not find a free localhost port for NEURON_RT_ROOT_COMM_ID")


def worker_env(
    base_env: Mapping[str, str],
    core_range: str,
    replica_cores: int,
    *,
    root_comm_id: str | None = None,
) -> dict[str, str]:
    env = dict(base_env)
    env["NEURON_RT_VISIBLE_CORES"] = core_range
    env["NEURON_RT_NUM_CORES"] = str(replica_cores)
    # Each replica is its own Neuron world and bootstraps its collectives
    # through a root socket at NEURON_RT_ROOT_COMM_ID. Left unset, torch_neuronx
    # pins it to localhost:62182 at import time (before libneuronxla's
    # free-port hook can run — the LTX-2 backend imports torch_neuronx at
    # module scope), so every worker shared one root and concurrent replica
    # bootstraps collided: "rank 1 of 2 ranks has already checked in", then
    # both replicas waited forever (ltx_2/dp2tp2 on device, 2026-09-07). A
    # value inherited from the parent is just as shared, so it is replaced too.
    env["NEURON_RT_ROOT_COMM_ID"] = root_comm_id or f"localhost:{_free_localhost_port()}"
    return env


def _die_with_parent() -> None:
    """``preexec_fn`` for worker processes: SIGKILL the worker when the router
    dies. The verification driver SIGKILLs a timed-out ``difflet generate``;
    without this the workers outlived it and kept all four NeuronCores
    (observed on device, 2026-09-07: two LTX-2 workers held cores 0-3 for hours
    after the router was gone)."""
    if not sys.platform.startswith("linux"):
        return
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    if os.getppid() == 1:  # the parent already died between fork and here
        os.kill(os.getpid(), signal.SIGKILL)


def worker_cli_args(args) -> list[str]:
    """Flags forwarded to worker CLI processes. NEVER --dp/--mode/--prompt/--output/--requests."""
    argv = ["--model-id", args.model_id]
    for flag, value in (
        ("--tp-degree", args.tp_degree),
        ("--cp-degree", args.cp_degree),
        ("--cp-mode", args.cp_mode),
        ("--attention-impl", getattr(args, "attention_impl", "megakernel")),
        ("--height", args.height),
        ("--width", args.width),
        ("--num-frames", args.num_frames),
        ("--steps", args.steps),
        ("--guidance-scale", args.guidance_scale),
        ("--seed", args.seed),
        ("--cache-dir", args.cache_dir),
        ("--revision", getattr(args, "revision", None)),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if getattr(args, "cfg_parallel", False):
        argv.append("--cfg-parallel")
    if getattr(args, "sp_enabled", False):
        argv.append("--sp")
    if getattr(args, "keep_work_dir", False):
        argv.append("--keep-work-dir")
    return argv


def _check_hbm(args) -> None:
    from difflet.cli.dp.hbm_check import assert_replica_fits
    from difflet.pipeline.path_resolver import resolve_model_path

    try:
        model_path = resolve_model_path(args.model_id, local_files_only=True)
    except OSError:
        print("[dp-router] weights not local; skipping HBM fit check", flush=True)
        return
    assert_replica_fits(model_path)


def run_router(
    args,
    requests: list[RequestSpec],
    *,
    replica_cores: int,
    worker_argv_prefix: list[str] | None = None,
) -> int:
    dp = int(args.dp or 1)
    schedule = args.dp_schedule
    work_dir = Path(args.work_dir or Path.home() / ".cache" / "difflet" / "work" / "dp")
    requests_dir = work_dir / "requests"

    if schedule == "round_robin":
        requests = [
            dataclasses.replace(req, assigned_worker=req.index % dp) for req in requests
        ]
    write_manifest(requests, requests_dir)
    _check_hbm(args)

    prefix = worker_argv_prefix or [sys.executable, "-m", "difflet.cli.main"]

    procs = []
    for w, core_range in enumerate(replica_core_ranges(dp, replica_cores)):
        argv = prefix + ["generate"] + worker_cli_args(args) + [
            "--requests-dir", str(requests_dir),
            "--worker-index", str(w),
            "--dp-schedule", schedule,
            "--work-dir", str(work_dir / f"worker_{w}"),
        ]
        env = worker_env(os.environ, core_range, replica_cores)
        print(f"[dp-router] worker {w}: cores {core_range}", flush=True)
        procs.append(subprocess.Popen(argv, env=env, preexec_fn=_die_with_parent))

    exit_codes = [p.wait() for p in procs]

    summary = summarize(requests_dir)
    for req in requests:
        if req.index in summary.done or req.index in summary.failed:
            continue
        crashed = (
            schedule == "round_robin"
            and req.assigned_worker is not None
            and exit_codes[req.assigned_worker] != 0
        )
        claim = requests_dir / f"req_{req.index:04d}.claim"
        if crashed or claim.exists():
            mark_failed(requests_dir, req.index, "worker crashed before finishing request")

    summary = summarize(requests_dir)
    print(
        f"[dp-router] done={len(summary.done)} failed={len(summary.failed)} "
        f"unfinished={len(summary.unfinished)}",
        flush=True,
    )
    for index, error in sorted(summary.failed.items()):
        print(f"[dp-router] request {index} FAILED: {error.splitlines()[0]}", flush=True)
    return 0 if (not summary.failed and not summary.unfinished) else 1
