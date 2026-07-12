"""DP router: scatter (manifest) → spawn pinned workers → gather (markers).

Spec: docs/superpowers/specs/2026-07-06-dp-replication-routing-design.md §Router.
Workers are full difflet CLI invocations with dp=1 semantics; NEURON_RT_* are
plain-assigned (run_stage's setdefault must see the worker's range, not the
parent's).
"""

from __future__ import annotations

import dataclasses
import os
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


def worker_env(base_env: Mapping[str, str], core_range: str, replica_cores: int) -> dict[str, str]:
    env = dict(base_env)
    env["NEURON_RT_VISIBLE_CORES"] = core_range
    env["NEURON_RT_NUM_CORES"] = str(replica_cores)
    return env


def worker_cli_args(args) -> list[str]:
    """Flags forwarded to worker CLI processes. NEVER --dp/--mode/--prompt/--output/--requests."""
    argv = ["--model-id", args.model_id]
    for flag, value in (
        ("--tp-degree", args.tp_degree),
        ("--cp-degree", args.cp_degree),
        ("--cp-mode", args.cp_mode),
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
        procs.append(subprocess.Popen(argv, env=env))

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
