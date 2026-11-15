"""Run one ``difflet <command>`` child and measure it from the outside.

The child is ``taskset -c <cpus> <this python> -m difflet.cli.main <command> ...`` in a new session.
``taskset`` execs, so the pid bench waits on is the engine driver's, and the CPU affinity is
inherited by torchrun and every rank. ``os.wait4`` returns the tree's resource usage once every
level has reaped its children: ``ru_inblock`` is the bytes read from storage (page-cache hits are
not counted), the evidence that a cold run really read its weights.

Cleanup only touches processes bench started. torchrun puts each rank in a session of its own, so
signalling the driver's process group does not reach the ranks: a timeout sends SIGINT to the
group (torchrun forwards it to the ranks), waits, then SIGKILLs the group. Every launch carries a
unique ``DIFFLET_BENCH_LAUNCH_ID`` in its environment, inherited by every descendant; after the
driver exits, any process still carrying it (a rank that outlived torchrun and holds a NeuronCore)
is killed. Nothing is ever matched by name or port.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from difflet.bench.host import _environ_value, _proc_ids, format_cpu_list

LAUNCH_ID_ENV = "DIFFLET_BENCH_LAUNCH_ID"
#: How long torchrun gets to stop its ranks after SIGINT (the old adapter's STOP_GRACE_S).
STOP_GRACE_S = 90.0


@dataclass
class Launch:
    command: str
    argv: list[str]
    launch_id: str
    returncode: int
    wall_s: float
    #: time.monotonic_ns() at spawn and at reap: the clock the engine's timing events use too.
    started_ns: int
    ended_ns: int
    log: str
    fs_read_bytes: int
    max_rss_kib: int
    user_s: float
    sys_s: float
    timed_out: bool = False
    leftovers_killed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def command_line(command: str, argv: Sequence[str], cpus: Sequence[int]) -> list[str]:
    return ["taskset", "-c", format_cpu_list(cpus), sys.executable, "-m", "difflet.cli.main",
            command, *argv]


def run(command: str, argv: Sequence[str], *, env: Mapping[str, str], cpus: Sequence[int],
        log: Path, cwd: Path, timeout_s: float | None = None) -> Launch:
    """Run ``difflet <command> <argv>`` pinned to ``cpus`` and wait for it."""
    log.parent.mkdir(parents=True, exist_ok=True)
    full = command_line(command, argv, cpus)
    launch_id = uuid.uuid4().hex
    child_env = dict(env)
    child_env[LAUNCH_ID_ENV] = launch_id
    timed_out = threading.Event()
    with open(log, "w", encoding="utf-8") as out:
        out.write("$ " + " ".join(full) + "\n")
        out.flush()
        started = time.monotonic_ns()
        t0 = time.perf_counter()
        proc = subprocess.Popen(full, env=child_env, cwd=str(cwd), stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        timer = None
        if timeout_s:
            def _expire() -> None:
                timed_out.set()
                stop_group(proc.pid)
            timer = threading.Timer(timeout_s, _expire)
            timer.daemon = True
            timer.start()
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        except BaseException:
            # Ctrl-C or SIGTERM (raised as KeyboardInterrupt by the CLI): stop our group and our
            # tagged ranks. A second Ctrl-C must not cut the cleanup short and leave a rank
            # holding a NeuronCore, so SIGINT is ignored until it is done.
            main = threading.current_thread() is threading.main_thread()
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN) if main else None
            try:
                stop_group(proc.pid, grace_s=30.0, leader=proc)
                kill_tagged(launch_id)
            finally:
                if main:
                    signal.signal(signal.SIGINT, previous)
            raise
        finally:
            if timer:
                timer.cancel()
        wall = time.perf_counter() - t0
        ended = time.monotonic_ns()
        proc.returncode = os.waitstatus_to_exitcode(status)
    _signal_group(proc.pid, signal.SIGKILL)
    leftovers = kill_tagged(launch_id)
    return Launch(command=command, argv=list(argv), launch_id=launch_id,
                  returncode=proc.returncode, wall_s=wall, started_ns=started, ended_ns=ended,
                  log=str(log), fs_read_bytes=usage.ru_inblock * 512,
                  max_rss_kib=usage.ru_maxrss, user_s=usage.ru_utime, sys_s=usage.ru_stime,
                  timed_out=timed_out.is_set(), leftovers_killed=leftovers)


def _signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_group(pgid: int, grace_s: float = STOP_GRACE_S,
               leader: subprocess.Popen | None = None) -> None:
    """SIGINT the group (torchrun forwards it to its ranks), wait, then SIGKILL it.

    With ``leader`` (the driver bench is not waiting on), the leader is reaped as soon as it
    exits, so an exited group is not mistaken for a live one through its zombie leader."""
    if not _signal_group(pgid, signal.SIGINT):
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if leader is not None and leader.poll() is None:
            time.sleep(0.5)
            continue
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.5)
    _signal_group(pgid, signal.SIGKILL)


def tagged_processes(launch_id: str) -> list[int]:
    """Processes whose environment carries this launch's id (our own descendants only)."""
    me = os.getpid()
    return [pid for pid in _proc_ids()
            if pid != me and _environ_value(pid, LAUNCH_ID_ENV) == launch_id]


def kill_tagged(launch_id: str, wait_s: float = 30.0) -> int:
    """SIGKILL every process still carrying this launch's id and wait until they are gone (the
    next launch's idle check would otherwise see them holding the device)."""
    pids = tagged_processes(launch_id)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + wait_s
    while pids and time.monotonic() < deadline and tagged_processes(launch_id):
        time.sleep(0.2)
    return len(pids)
