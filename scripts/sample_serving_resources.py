#!/usr/bin/env python3
"""Sample one resident serving process tree and Neuron runtime resources.

The sampler writes JSON Lines so a long-running Trainium validation can retain
the raw ``neuron-monitor`` payload together with host RSS/PSS and an externally
managed phase label.  Update the phase file at lifecycle boundaries such as
``compile``, ``load``, ``startup_smoke``, and ``request_1``.
"""

from __future__ import annotations

import argparse
import json
import select
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True, help="Root serving PID")
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--neuron-monitor", default="neuron-monitor")
    return parser.parse_args()


def _read_int_fields(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return values
    for line in lines:
        name, separator, raw = line.partition(":")
        if not separator:
            continue
        tokens = raw.strip().split(maxsplit=1)
        if not tokens:
            continue
        token = tokens[0]
        try:
            values[name] = int(token)
        except ValueError:
            continue
    return values


def _process_tree(root_pid: int) -> list[int]:
    found: list[int] = []
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if not Path(f"/proc/{pid}").exists():
            continue
        found.append(pid)
        # A child is recorded under the Linux task/thread that created it, not
        # necessarily under the thread-group leader.  Python multiprocessing
        # may spawn the resident worker from a background thread, so reading
        # only ``task/<pid>/children`` can silently omit the largest process in
        # the serving tree.  Scan every live task and de-duplicate below.
        try:
            task_dirs = list(Path(f"/proc/{pid}/task").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for task_dir in task_dirs:
            try:
                children = (task_dir / "children").read_text(encoding="utf-8")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            pending.extend(int(value) for value in children.split())
    return sorted(found)


def _process_memory(pids: list[int]) -> dict[str, Any]:
    rss_kib = 0
    pss_kib = 0
    swap_kib = 0
    per_pid: list[dict[str, Any]] = []
    for pid in pids:
        status = _read_int_fields(Path(f"/proc/{pid}/status"))
        rollup = _read_int_fields(Path(f"/proc/{pid}/smaps_rollup"))
        pid_rss = status.get("VmRSS", 0)
        pid_pss = rollup.get("Pss", 0)
        pid_swap = status.get("VmSwap", 0)
        rss_kib += pid_rss
        pss_kib += pid_pss
        swap_kib += pid_swap
        try:
            command = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            command = ""
        per_pid.append(
            {
                "pid": pid,
                "command": command,
                "rss_kib": pid_rss,
                "pss_kib": pid_pss,
                "swap_kib": pid_swap,
            }
        )
    return {
        "process_count": len(pids),
        "rss_kib": rss_kib,
        "pss_kib": pss_kib,
        "swap_kib": swap_kib,
        "processes": per_pid,
    }


def _neuron_sample(command: str) -> dict[str, Any]:
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            [command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        readable, _, _ = select.select([process.stdout], [], [], 10.0)
        if not readable:
            return {"error": "neuron-monitor produced no sample within 10 seconds"}
        line = process.stdout.readline().strip()
        if not line:
            return {"error": "neuron-monitor returned an empty sample"}
        return json.loads(line)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


def _phase(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() or "unknown"
    except FileNotFoundError:
        return "unknown"


def main() -> int:
    args = _parse_args()
    if args.pid <= 0:
        raise SystemExit("--pid must be positive")
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")

    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing_samples = 0
    with args.output.open("a", encoding="utf-8") as output:
        while not stop:
            started = time.monotonic()
            pids = _process_tree(args.pid)
            if pids:
                missing_samples = 0
            else:
                missing_samples += 1
            record = {
                "schema": "difflet.serving_resource_sample.v1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": _phase(args.phase_file),
                "root_pid": args.pid,
                "host_memory": _read_int_fields(Path("/proc/meminfo")),
                "process_tree": _process_memory(pids),
                "neuron_monitor": _neuron_sample(args.neuron_monitor),
            }
            record["sample_duration_seconds"] = time.monotonic() - started
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            if missing_samples >= 3:
                break
            remaining = args.interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
