"""CI sharding: split tests/unit's test files across n jobs by their recorded run time.

``CPU_GATE_SHARD=k/n`` (1 <= k <= n) makes ``tests/conftest.py`` collect only shard k's
files; unset or empty, everything is collected. Files are dealt longest first, each to the
shard with the least time so far (ties: the lowest shard; equal times: by path), from
``scripts/cpu-gate-durations.txt`` (``<seconds> <path>``, one line per file). A file it
does not list counts ``DEFAULT_SECONDS``. A file goes to one shard whole, so its module
fixtures and multi-rank tests behave as under the gate's ``--dist loadfile``.

Regenerate the durations from the gate logs of a sharded run (the gate passes
``--durations=0 --durations-min=0`` when sharded; CI uploads them as the
``cpu-gate-logs-*-shard*`` artifacts), every shard's log at once:

    python tests/_shard.py durations cpu-gate-unit-*.log > scripts/cpu-gate-durations.txt

``python tests/_shard.py files k/n`` prints shard k's files (``scripts/cpu-gate.sh`` uses it).
Loaded by ``tests/conftest.py`` by path, like ``_neuron_gate.py``.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "tests" / "unit"
DURATIONS = ROOT / "scripts" / "cpu-gate-durations.txt"
#: The weight of a test file the durations do not list yet (a new file, until regenerated).
DEFAULT_SECONDS = 1.0
#: One line of pytest's ``--durations`` report: "12.34s call     tests/unit/x.py::test_y[a]".
_DURATION_LINE = re.compile(r"^([0-9.]+)s (?:setup|call|teardown)\s+(tests/\S+?\.py)(?:::.*)?$")


def parse_spec(spec: str | None) -> tuple[int, int] | None:
    """``"k/n"`` as ``(k, n)``; ``None`` for no sharding (unset or empty)."""
    if not spec or not spec.strip():
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", spec.strip())
    if not match or not 1 <= int(match.group(1)) <= int(match.group(2)):
        raise ValueError(f"CPU_GATE_SHARD must be k/n with 1 <= k <= n, got {spec!r}")
    return int(match.group(1)), int(match.group(2))


def load_durations(path: Path = DURATIONS) -> dict[str, float]:
    if not path.is_file():
        return {}
    durations = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            seconds, name = line.split(maxsplit=1)
            durations[name] = float(seconds)
    return durations


def unit_test_files(root: Path = UNIT) -> list[str]:
    """Every ``test_*.py`` under ``root``, relative to the repository root."""
    return sorted(p.relative_to(ROOT).as_posix() for p in root.rglob("test_*.py"))


def assign(files: list[str], durations: dict[str, float], n: int) -> list[list[str]]:
    """Deal ``files`` into ``n`` shards, longest first, each to the least-loaded shard."""
    shards: list[list[str]] = [[] for _ in range(n)]
    loads = [0.0] * n
    weight = lambda name: durations.get(name, DEFAULT_SECONDS)  # noqa: E731
    for name in sorted(files, key=lambda f: (-weight(f), f)):
        index = loads.index(min(loads))
        shards[index].append(name)
        loads[index] += weight(name)
    return shards


def shard_files(spec: str | None) -> set[str] | None:
    """Shard k's test files for ``spec`` ``"k/n"``; ``None`` when not sharding."""
    parsed = parse_spec(spec)
    if parsed is None:
        return None
    k, n = parsed
    return set(assign(unit_test_files(), load_durations(), n)[k - 1])


def durations_from_logs(paths: list[str]) -> dict[str, float]:
    """Seconds per test file, summed over the setup/call/teardown lines of ``--durations``."""
    per_file: dict[str, float] = defaultdict(float)
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            match = _DURATION_LINE.match(line.strip())
            if match:
                per_file[match.group(2)] += float(match.group(1))
    return dict(per_file)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "files":
        print("\n".join(sorted(shard_files(argv[1]) or [])))
        return 0
    if len(argv) >= 2 and argv[0] == "durations":
        durations = durations_from_logs(argv[1:])
        if not durations:
            print("no --durations lines in the logs", file=sys.stderr)
            return 1
        print("# seconds per tests/unit file, from: python tests/_shard.py durations <logs>")
        for name in sorted(durations):
            print(f"{durations[name]:.1f} {name}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
