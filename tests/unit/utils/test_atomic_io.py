"""difflet.utils.atomic_io: no truncated files at the final name, and one builder at a time."""

import errno
import fcntl
import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from difflet.utils import atomic_io
from difflet.utils.atomic_io import atomic_path, atomic_write_text, file_lock

REPO = Path(__file__).resolve().parents[3]  # the difflet the subprocess imports


def test_atomic_write_replaces_the_file(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text("old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"
    assert os.listdir(tmp_path) == ["manifest.json"]


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "model.pt"
    target.write_text("complete")
    with pytest.raises(RuntimeError):
        with atomic_path(target) as tmp:
            tmp.write_text("half")
            raise RuntimeError("killed mid-write")
    assert target.read_text() == "complete"
    assert os.listdir(tmp_path) == ["model.pt"]


_HOLDER = textwrap.dedent("""
    import sys, time
    from difflet.utils.atomic_io import file_lock
    with file_lock(sys.argv[1]):
        print("locked", flush=True)
        time.sleep(float(sys.argv[2]))
""")


def _start_holder(lock, seconds):
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(lock), str(seconds)],
                            stdout=subprocess.PIPE, text=True,
                            env={**os.environ, "PYTHONPATH": str(REPO)})
    assert proc.stdout.readline().strip() == "locked"
    return proc


def test_a_second_process_waits_and_says_who_holds_the_lock(tmp_path, caplog):
    lock = tmp_path / "key.lock"
    holder = _start_holder(lock, 1.0)
    start = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=atomic_io.__name__):
        with file_lock(lock, what="the transformer"):
            waited = time.monotonic() - start
    holder.wait(timeout=10)
    assert waited >= 0.5  # it blocked until the holder let go
    assert f"process {holder.pid} is building the transformer" in caplog.text


def test_a_killed_holder_does_not_leave_a_stale_lock(tmp_path):
    lock = tmp_path / "key.lock"
    holder = _start_holder(lock, 60)
    os.kill(holder.pid, signal.SIGKILL)
    holder.wait(timeout=10)
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if the lock were still held
    finally:
        os.close(fd)


def test_no_flock_support_means_no_lock_not_a_crash(tmp_path, monkeypatch, caplog):
    def no_flock(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(atomic_io.fcntl, "flock", no_flock)
    ran = []
    with caplog.at_level(logging.WARNING, logger=atomic_io.__name__):
        with file_lock(tmp_path / "key.lock"):
            ran.append(True)
    assert ran == [True]
    assert "continuing without a lock" in caplog.text


def test_a_write_removes_the_temp_files_left_by_killed_writers_only(tmp_path):
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True).stdout.strip()
    left = tmp_path / f".model.pt.tmp-{dead}-0badc0de"  # a writer killed mid-write
    busy = tmp_path / f".model.pt.tmp-{os.getpid()}-feedf00d"  # a writer still running
    other = tmp_path / f".config.json.tmp-{dead}-0badc0de"  # another target's: not ours to judge
    for p in (left, busy, other):
        p.write_text("partial")
    atomic_write_text(tmp_path / "model.pt", "full")
    assert (tmp_path / "model.pt").read_text() == "full"
    assert not left.exists() and busy.exists() and other.exists()
