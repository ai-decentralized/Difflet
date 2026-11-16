"""Where a number came from: code, packages, SDK, hardware, environment (design §9).

Collected automatically for every result so another person can tell whether their run is the
same run: the git commit and whether the tree was dirty, every installed Python distribution, the
Neuron system packages, the instance type, Python, the caller's engine-relevant environment
(secrets excluded), bench's own argv. ``host_state`` is taken around every launch (memory, the NEFF
files present, throttling) so cache state is evidence, not assumption.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from difflet.bench import host as H

#: Packages whose versions decide what the engine compiles and runs, listed first in reports.
TOOLCHAIN = ("torch", "torch-neuronx", "neuronx-cc", "nki", "libneuronxla", "diffusers",
             "transformers", "numpy", "difflet-diffusion")


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 30.0) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_state(root: Path) -> dict[str, Any] | None:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if commit is None:
        return None
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root) or ""
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return {"commit": commit.strip(), "dirty": bool(status.strip()),
            "dirty_files": [ln[3:] for ln in status.splitlines()][:200],
            "branch": branch.strip() if branch else None}


def python_packages() -> dict[str, str]:
    out = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            out[name.lower()] = dist.version
    return dict(sorted(out.items()))


def toolchain(packages: Mapping[str, str]) -> dict[str, str | None]:
    return {name: packages.get(name) for name in TOOLCHAIN}


def neuron_system_packages() -> dict[str, str]:
    out = _run(["dpkg-query", "-W", "-f", "${Package} ${Version}\n", "aws-neuronx*"])
    rows = {}
    for line in (out or "").splitlines():
        name, _, version = line.partition(" ")
        if name and version:
            rows[name] = version
    return rows


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def collect(*, caller_env: Mapping[str, str]) -> dict[str, Any]:
    """Everything about this host and checkout that does not change between launches."""
    from difflet.bench.results import host_id

    packages = python_packages()
    try:
        instance = H.topology()["instance_type"]
    except H.HostError:
        instance = None
    return {
        "git": git_state(H.source_root()),
        "source_root": str(H.source_root()),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "toolchain": toolchain(packages),
        "packages": packages,
        "neuron_system_packages": neuron_system_packages(),
        "instance_type": instance,
        "host_id": host_id(),
        "kernel": platform.release(),
        "cpu_model": cpu_model(),
        "bench_argv": list(sys.argv),
        "caller_env": H.env_snapshot(caller_env),
        "paths": {"bench_root": str(H.bench_root()), "compile_cache": str(H.compile_cache_root()),
                  "hub_cache": str(H.hub_cache())},
    }


# ------------------------------------------------------------------------------- per launch


def meminfo() -> dict[str, float]:
    keep = {"MemTotal", "MemAvailable", "MemFree", "Cached", "Buffers"}
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in keep:
                out[f"{key}_gib"] = round(int(rest.split()[0]) / (1 << 20), 3)
    except OSError:
        pass
    return out


def neff_files(cache_root: Path) -> dict[str, int]:
    """Every ``*.neff`` under the compile cache's neuron dir, path -> size (new ones = compiled)."""
    base = Path(cache_root) / "neuron"
    if not base.exists():
        return {}
    return {str(p.relative_to(base)): p.stat().st_size for p in base.rglob("*.neff")}


def hlo_file_count(cache_root: Path) -> int:
    base = Path(cache_root) / "neuron" / "hlo"
    return sum(1 for p in base.rglob("*") if p.is_file()) if base.exists() else 0


def throttle(devices: list[int]) -> dict[str, str]:
    out = {}
    for d in devices:
        for path in sorted(Path(f"/sys/class/neuron_device/neuron{d}").glob(
                "neuron_core*/stats/hardware/ham_throttle/throttle")):
            try:
                out[f"neuron{d}/{path.parts[-5]}"] = path.read_text().strip()
            except OSError:
                continue
    return out


def host_state(cache_root: Path, devices: list[int]) -> dict[str, Any]:
    neffs = neff_files(cache_root)
    return {"meminfo": meminfo(), "neff_count": len(neffs), "neff_files": neffs,
            "hlo_files": hlo_file_count(cache_root), "throttle": throttle(devices)}


def new_neffs(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    return sorted(set(after.get("neff_files", {})) - set(before.get("neff_files", {})))


def cache_manifests(cache_root: Path, model_id: str) -> list[dict[str, Any]]:
    """The stage manifests under the compile cache whose inputs name ``model_id`` (what the
    compile cache keyed this recipe's artifacts on). FLUX and HV1.5 on neuron write none."""
    out = []
    root = Path(cache_root)
    if not root.exists():
        return out
    for path in root.glob("*/*/manifest.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        inputs = data.get("cache_inputs") or {}
        if inputs.get("model_id") == model_id:
            out.append({"path": str(path.relative_to(root)), "cache_key": data.get("cache_key"),
                        "cache_inputs": inputs})
    return out


def dir_bytes(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


# ------------------------------------------------------------------------------- check-env


def check_env(reference: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Differences between a result's recorded host and this one that can change the numbers."""
    diffs: list[str] = []
    ref, cur = reference.get("provenance") or reference, current
    for key in ("instance_type",):
        if ref.get(key) != cur.get(key):
            diffs.append(f"{key}: reference {ref.get(key)!r}, here {cur.get(key)!r}")
    if (ref.get("python") or {}).get("version") != (cur.get("python") or {}).get("version"):
        diffs.append(f"python: reference {(ref.get('python') or {}).get('version')!r}, "
                     f"here {(cur.get('python') or {}).get('version')!r}")
    ref_git, cur_git = ref.get("git") or {}, cur.get("git") or {}
    if ref_git.get("commit") != cur_git.get("commit"):
        diffs.append(f"git commit: reference {ref_git.get('commit')}, here {cur_git.get('commit')}")
    if cur_git.get("dirty"):
        diffs.append("the checkout here has uncommitted changes")
    for table in ("packages", "neuron_system_packages"):
        a, b = ref.get(table) or {}, cur.get(table) or {}
        for name in sorted(set(a) | set(b)):
            if a.get(name) != b.get(name):
                diffs.append(f"{table}: {name} reference {a.get(name)!r}, here {b.get(name)!r}")
    return diffs
