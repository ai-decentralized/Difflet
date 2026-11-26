"""Start and stop the fleet's ``difflet serve`` backends, each pinned to its own device.

``up`` starts one detached backend per entry, in config order, and waits for each port before
starting the next (six weight loads do not share the disk). Backends outlive the launcher and the
gateway: restarting the gateway never reloads a model. Design §6.3.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from difflet.bench import host
from difflet.serving.gateway.fleet import FleetConfig, FleetEntry

#: Seconds between checks of a starting backend's port.
POLL_S = 5.0
#: Set for every backend, after the inherited and the entry's own environment.
FIXED_ENV = {
    # the gateway strips Authorization, so a key inherited from the gateway's shell would 401
    "DIFFLET_API_KEY": "",
    # a missing snapshot fails loudly instead of downloading onto the (full) root disk
    "HF_HUB_OFFLINE": "1",
    "DIFFLET_BACKEND": "neuron",
    # torch-neuronx re-pins every rank to device 0's NUMA node otherwise (difflet/bench/host.py)
    "NEURON_NUMA_PIN": "0",
}


class LauncherError(RuntimeError):
    """A backend could not be started or stopped."""


def backend_env(entry: FleetEntry, base_env: Mapping[str, str], *, visible_cores: str,
                cc_dir: str | None) -> dict[str, str]:
    env = {**base_env, **entry.env, **FIXED_ENV, "NEURON_RT_VISIBLE_CORES": visible_cores}
    if cc_dir is not None:  # torch-neuronx spawns neuronx-cc by name
        env["PATH"] = os.pathsep.join(p for p in (cc_dir, env.get("PATH", "")) if p)
    return env


def backend_argv(entry: FleetEntry, *, python: str, cpus: str) -> list[str]:
    return ["taskset", "-c", cpus, python, "-m", "difflet.cli.main", "serve",
            "--model-id", entry.model_id, "--host", "127.0.0.1", "--port", str(entry.port),
            "--cache-dir", str(entry.cache_dir), *entry.serve_args]


def read_pid(pid_file: Path) -> int | None:
    """The backend's pid: the file's pid when that process is alive and runs in the pid file's
    directory (the launcher starts it there), so a stale file whose pid the kernel reused for
    another process is never signalled."""
    try:
        pid = int(pid_file.read_text().strip())
        alive_here = Path(os.readlink(f"/proc/{pid}/cwd")) == pid_file.parent.resolve()
    except (OSError, ValueError):
        return None
    return pid if alive_here else None


def up(config: FleetConfig, *, wait_s: float = 3600.0) -> None:
    fleet = {pid for e in config.models if (pid := read_pid(config.pid_file(e))) is not None}
    for entry in config.models:
        running = read_pid(config.pid_file(entry))
        if running is not None:
            print(f"{entry.model_id}: already running (pid {running})", flush=True)
            continue
        # A backend's ranks are in its session (started with start_new_session), so they are
        # the fleet's own too.
        holders = [h for h in host.device_holders([entry.device])[entry.device]
                   if h["pid"] not in fleet and _sid(h["pid"]) not in fleet]
        if holders:
            who = "; ".join(f"pid {h['pid']} {h['comm']} (cores "
                            f"{h['NEURON_RT_VISIBLE_CORES'] or '?'})" for h in holders[:3])
            raise LauncherError(f"device {entry.device} ({entry.model_id}) is in use: {who}")
        if _port_open(entry.port):
            raise LauncherError(f"port {entry.port} ({entry.model_id}) is already in use")
        cores = host.devices_to_cores([entry.device])
        node = host.topology()["devices"][entry.device].numa_node
        cpus = host.format_cpu_list(host.numa_cpus()[node])
        env = backend_env(entry, os.environ, visible_cores=cores,
                          cc_dir=host.neuronx_cc_dir(os.environ.get("PATH", "")))
        run_dir, log_file = config.run_dir(entry), config.log_file(entry)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(log_file, "ab") as log:
            proc = subprocess.Popen(backend_argv(entry, python=sys.executable, cpus=cpus),
                                    env=env, cwd=run_dir, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        config.pid_file(entry).write_text(f"{proc.pid}\n")
        fleet.add(proc.pid)
        print(f"{entry.model_id}: pid {proc.pid} on device {entry.device} (cores {cores}, "
              f"CPUs {cpus}); waiting for port {entry.port}, log {log_file}", flush=True)
        deadline = time.monotonic() + wait_s
        while not _port_open(entry.port):
            if proc.poll() is not None:
                raise LauncherError(
                    f"{entry.model_id} exited with status {proc.returncode} before opening port "
                    f"{entry.port}; last lines of {log_file}:\n{_tail(log_file)}")
            if time.monotonic() > deadline:
                raise LauncherError(
                    f"{entry.model_id} is not listening on port {entry.port} after {wait_s:.0f} s; "
                    f"it is still running as pid {proc.pid} (`difflet gateway down` stops it); "
                    f"last lines of {log_file}:\n{_tail(log_file)}")
            time.sleep(POLL_S)
        print(f"{entry.model_id}: listening on {entry.base_url}", flush=True)


def down(config: FleetConfig, *, grace_s: float = 60.0) -> None:
    """SIGTERM every backend's process group, SIGKILL the ones still alive after ``grace_s``."""
    pids = [pid for e in config.models if (pid := read_pid(config.pid_file(e))) is not None]
    for pid in pids:
        _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while any(_group_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(min(0.5, grace_s))
    for pid in pids:
        if _group_alive(pid):
            _signal_group(pid, signal.SIGKILL)
    for entry in config.models:
        config.pid_file(entry).unlink(missing_ok=True)


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _group_alive(pid: int) -> bool:
    # The group, not the leader: ranks that outlive the serve process still hold the device.
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sid(pid: int) -> int | None:
    try:
        return os.getsid(pid)
    except OSError:
        return None


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _tail(path: Path, lines: int = 40) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - 64 * 1024))
            text = f.read().decode(errors="replace")
    except OSError:
        return "(no log)"
    return "\n".join(text.splitlines()[-lines:])
