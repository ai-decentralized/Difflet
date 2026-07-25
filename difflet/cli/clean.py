"""Remove Neuron compiler scratch dumped into the working directory.

neuronx-cc writes several artifacts into the process cwd during device
compiles: per-kernel cache directories named with a 16-hex-char hash, a
``neuronxcc-<id>/`` work directory per compiler invocation, and a few loose
diagnostic files. They are all gitignored (see ``.gitignore``) but accumulate
across runs, so ``difflet clean`` sweeps them.

Only direct children of the target directory are considered, symlinks are
never followed, and a hash-named directory is only removed when its contents
look like compiler output — a same-named directory holding anything else is
reported and left alone.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# Loose diagnostic files neuronx-cc drops in cwd.
SCRATCH_FILES = (
    "log-neuron-cc.txt",
    "global_metric_store.json",
    "PostSPMDPassesExecutionDuration.txt",
)

# Per-kernel cache directories: 16 lowercase hex chars, no separators.
_HASH_DIR = re.compile(r"^[0-9a-f]{16}$")

# Per-invocation compiler work directories.
_WORK_DIR_PREFIX = "neuronxcc-"

# A hash-named directory is compiler scratch only if every entry looks like
# compiler output. Anything else means the name collided with real data.
_SCRATCH_ENTRY_SUFFIXES = (".json", ".neff", ".hlo", ".pb", ".txt", ".log", ".penguin", ".code")

# Completion/lock markers the NKI kernel cache writes alongside its payload.
_SCRATCH_ENTRY_NAMES = frozenset({".done", ".lock"})


def _is_scratch_entry(name: str) -> bool:
    if name in _SCRATCH_ENTRY_NAMES or "neuronxcc" in name:
        return True
    return name.endswith(_SCRATCH_ENTRY_SUFFIXES)


def _is_scratch_hash_dir(path: Path) -> bool:
    try:
        entries = list(path.iterdir())
    except OSError:
        return False
    return all(_is_scratch_entry(e.name) for e in entries)


def find_scratch(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (removable paths, skipped hash-named dirs) directly under ``root``."""
    removable: list[Path] = []
    skipped: list[Path] = []
    try:
        entries = sorted(root.iterdir())
    except OSError as exc:
        raise SystemExit(f"Error: cannot read {root}: {exc}")
    for entry in entries:
        if entry.is_symlink():
            continue
        if entry.is_dir():
            if entry.name.startswith(_WORK_DIR_PREFIX):
                removable.append(entry)
            elif _HASH_DIR.match(entry.name):
                (removable if _is_scratch_hash_dir(entry) else skipped).append(entry)
        elif entry.is_file() and entry.name in SCRATCH_FILES:
            removable.append(entry)
    return removable, skipped


def _size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _human(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"  # unreachable; keeps mypy happy


def run(args: argparse.Namespace) -> None:
    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Error: {root} is not a directory.")

    removable, skipped = find_scratch(root)
    for path in skipped:
        print(
            f"[difflet] skipping {path.name}/ — hash-named but holds non-compiler files",
            flush=True,
        )
    if not removable:
        print(f"[difflet] no Neuron compiler scratch in {root}", flush=True)
        return

    freed = 0
    verb = "would remove" if args.dry_run else "removed"
    for path in removable:
        size = _size(path)
        is_dir = path.is_dir()
        if not args.dry_run:
            if is_dir:
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        freed += size
        print(f"[difflet] {verb} {path.name}{'/' if is_dir else ''} ({_human(size)})", flush=True)
    print(f"[difflet] {len(removable)} item(s), {_human(freed)}", flush=True)
