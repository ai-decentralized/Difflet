"""Page-cache control without root, limited to the files a recipe reads.

``e2e_cold`` evicts exactly those files with ``posix_fadvise(POSIX_FADV_DONTNEED)`` instead of
dropping the whole host's cache (which would slow every other session down); ``e2e_warm`` reads them
through first. The kernel only drops clean, unmapped, unlocked pages, so eviction can fail
silently while another process maps the same weights: :func:`residency` (``mincore``) measures what
is actually cached before the launch, and the result records it instead of assuming.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
from pathlib import Path
from typing import Iterable

_CHUNK = 8 << 20


def files_under(paths: Iterable[Path]) -> list[Path]:
    """Every regular file under ``paths``, symlinks resolved (hub snapshots link into blobs/)."""
    seen: dict[str, Path] = {}
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        walk = [root] if root.is_file() else (p for p in root.rglob("*"))
        for path in walk:
            try:
                real = Path(os.path.realpath(path))
                if real.is_file():
                    seen.setdefault(str(real), real)
            except OSError:
                continue
    return sorted(seen.values())


def total_bytes(files: Iterable[Path]) -> int:
    return sum(f.stat().st_size for f in files)


def evict(files: Iterable[Path]) -> int:
    """Ask the kernel to drop ``files`` from the page cache; returns the bytes requested."""
    requested = 0
    for path in files:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            requested += os.fstat(fd).st_size
        finally:
            os.close(fd)
    return requested


def preread(files: Iterable[Path]) -> int:
    """Read ``files`` through so they are page-cached; returns the bytes read."""
    read = 0
    buf = bytearray(_CHUNK)
    view = memoryview(buf)
    for path in files:
        try:
            with open(path, "rb", buffering=0) as handle:
                while True:
                    n = handle.readinto(view)
                    if not n:
                        break
                    read += n
        except OSError:
            continue
    return read


_libc = None


def _mincore():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        _libc.mmap.restype = ctypes.c_void_p
        _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_long]
        _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_ubyte)]
    return _libc


def resident_bytes(path: Path) -> int:
    """Bytes of ``path`` currently in the page cache (``mincore`` on a read-only mapping)."""
    size = path.stat().st_size
    if size == 0:
        return 0
    libc = _mincore()
    page = mmap.PAGESIZE
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f"mmap failed for {path}")
        try:
            pages = (size + page - 1) // page
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore failed for {path}")
            import numpy as np  # a 50 GB checkpoint is ~12M pages: count them vectorised

            # Only the low bit is defined (mincore(2)); the others may be anything.
            resident = int((np.frombuffer(vec, dtype=np.uint8) & 1).sum())
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)
    return min(resident * page, size)


def residency(files: Iterable[Path]) -> tuple[int, int]:
    """(resident bytes, total bytes) over ``files``."""
    resident = total = 0
    for path in files:
        try:
            total += path.stat().st_size
            resident += resident_bytes(path)
        except OSError:
            continue
    return resident, total
