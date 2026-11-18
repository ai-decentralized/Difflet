"""Crash- and concurrency-safe writes for compiled artifacts and caches.

``atomic_path`` writes to a temp file next to the target and moves it over the target only once
it is complete, so a process killed mid-write never leaves a truncated file at the final name.
``file_lock`` serialises processes that build the same artifact (an exclusive ``flock`` on a
sibling ``.lock`` file; the kernel releases it when the holder dies, so it never goes stale).
"""

from __future__ import annotations

import contextlib
import fcntl
import glob
import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def fsync_dir(directory: str | os.PathLike) -> None:
    """Make a rename in ``directory`` durable (best effort)."""
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's live process
        return True
    return True


def _remove_dead_writers_temps(path: Path) -> None:
    """Remove ``path``'s temp files whose writer is gone (killed before its ``finally`` ran)."""
    # shortcut: the pid is checked on this host only; fine for host-local caches, a cache shared
    # across hosts may lose another host's in-flight temp (its write then fails, never corrupts).
    for tmp in path.parent.glob(f".{glob.escape(path.name)}.tmp-*"):
        pid = tmp.name.rpartition(".tmp-")[2].partition("-")[0]
        if pid.isdigit() and not _pid_alive(int(pid)):
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()


@contextlib.contextmanager
def atomic_path(path: str | os.PathLike) -> Iterator[Path]:
    """Yield a temp path next to ``path``; when the block succeeds, move it over ``path``.

    On any error the temp file is removed and ``path`` keeps its previous content. A writer
    killed outright (SIGKILL, or SIGTERM with no handler) cannot remove its temp file; the next
    write to ``path`` does."""
    path = Path(path)
    _remove_dead_writers_temps(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    fsync_dir(path.parent)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())


def _holder(fd: int) -> str:
    with contextlib.suppress(OSError, ValueError):
        return os.pread(fd, 32, 0).decode().strip() or "?"
    return "?"


@contextlib.contextmanager
def file_lock(path: str | os.PathLike, *, what: str = "this artifact") -> Iterator[None]:
    """Hold an exclusive lock on ``path`` while the block runs; wait (and say so) if taken.

    Never use it around code that runs collectives on several ranks: a rank waiting here would
    stall the others in the collective. Filesystems without ``flock`` (some NFS/Lustre mounts)
    get a warning and no lock, which is the behaviour before locking existed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    locked = False
    try:
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.warning("waiting for %s: process %s is building %s", path, _holder(fd), what)
                fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            logger.warning("cannot lock %s (%s); continuing without a lock", path, exc)
        if locked:
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        yield
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
