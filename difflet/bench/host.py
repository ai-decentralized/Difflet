"""The host side of a run: where things live, the engine's environment, devices, CPUs, idleness.

Design §10. Every host-specific path comes from three variables read here and nowhere else
(``DIFFLET_BENCH_ROOT``, ``HF_HUB_CACHE``, ``DIFFLET_COMPILE_CACHE``); the interpreter is the one
running bench. The engine's environment is the caller's minus every inherited engine knob, plus the
recipe's ``env``, the pinned cores and the timing directory, so two people's shells cannot make the
same recipe run differently. Before every measured process bench checks that the pinned devices and
CPUs are idle (the host is shared between sessions) and refuses otherwise.
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

#: Inherited variables that change what the engine runs or how fast; only a recipe may set them.
SCRUB_PREFIXES = ("DIFFLET_", "TORCH_NEURONX_", "NEURON_", "NEURONXCC_", "NKI_", "TORCHINDUCTOR_",
                  "TORCHDYNAMO_", "TORCH_COMPILE_", "XLA_")
SCRUB_NAMES = frozenset({"OMP_NUM_THREADS", "MKL_NUM_THREADS", "LD_PRELOAD", "TORCH_LOGS",
                         "COLLECTIVE_BUCKETSIZE_IN_MB", "TN_DEBUG_COPY", "UNSAFE_FP8FNCAST",
                         "BASE_COMPILE_WORK_DIR",
                         # torchrun owns these; an inherited one would misplace the ranks
                         "WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                         "MASTER_PORT"})
#: Set in every launch. HF_HUB_OFFLINE: bench never downloads, so a missing snapshot fails instead
#: of fetching. NEURON_NUMA_PIN=0: torch-neuronx 2.14's process-group init calls
#: pin_to_numa(local_rank), which reads the local rank as a core id, so every multi-rank stage is
#: pinned to device 0's NUMA node whatever device it runs on, overriding the lane's taskset (seen
#: 2026-11-15: device 8's ranks on CPUs 48-95,144-191). The lane's --cpus decide instead.
FORCED_ENV = {"HF_HUB_OFFLINE": "1", "NEURON_NUMA_PIN": "0"}
#: Variables recorded in a result's environment snapshot (secrets excluded, see is_secret).
RECORD_PREFIXES = ("DIFFLET_", "TORCH", "NEURON_", "XLA_", "HF_", "HUGGINGFACE_", "OMP_",
                   "MKL_", "PYTORCH_", "LD_PRELOAD", "PYTHONPATH", "TRANSFORMERS_", "DIFFUSERS_")
_SECRET = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|_KEY|KEY_|^KEY$|AUTH|PRIVATE|SESSION|COOKIE"
                     r"|^DIFFLET_S3_")

#: A pinned CPU busier than this over the sampling window makes the lane not idle.
CPU_BUSY_FRACTION = 0.20  # shortcut: initial value, calibrate on device data (design §17)
CPU_SAMPLE_SECONDS = 1.0


class HostError(RuntimeError):
    """The host cannot run this measurement (missing tool, busy device, bad CPU list)."""


# ----------------------------------------------------------------------------------- paths


def bench_root() -> Path:
    from difflet import envs

    return Path(envs.DIFFLET_BENCH_ROOT)


def compile_cache_root() -> Path:
    from difflet import envs

    return Path(envs.DIFFLET_COMPILE_CACHE)


def hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def snapshot_dir(model_id: str, revision: str) -> Path:
    """The local hub snapshot of ``model_id`` at ``revision``; bench never downloads."""
    path = hub_cache() / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not path.is_dir():
        raise HostError(f"{model_id}@{revision} is not in {hub_cache()}; download it first: "
                        f"difflet download --model-id {model_id} --revision {revision}")
    return path


def source_root() -> Path:
    """The directory holding the ``difflet`` package this bench imported (put on the child's path,
    so the engine runs the same code as bench)."""
    import difflet

    return Path(difflet.__file__).resolve().parents[1]


def neuronx_cc_dir(path: str) -> str | None:
    """The directory to put on PATH so torch-neuronx finds ``neuronx-cc`` (it spawns it by name,
    and every NEFF cache miss fails without it). None when ``path`` already has it."""
    import shutil
    from importlib import metadata

    if shutil.which("neuronx-cc", path=path):
        return None
    try:
        dist = metadata.distribution("neuronx-cc")
    except metadata.PackageNotFoundError:
        raise HostError("neuronx-cc is not installed in this environment") from None
    for entry in dist.files or ():
        if entry.name == "neuronx-cc" and entry.parent.name == "bin":
            found = Path(dist.locate_file(entry)).resolve()
            if found.is_file():
                return str(found.parent)
    raise HostError("neuronx-cc is installed but its console script was not found; put the "
                    "Neuron SDK venv's bin on PATH")


# ------------------------------------------------------------------------------- environment


def scrubbed(name: str) -> bool:
    return name in SCRUB_NAMES or name.startswith(SCRUB_PREFIXES)


def is_secret(name: str) -> bool:
    return bool(_SECRET.search(name.upper()))


def env_snapshot(env: Mapping[str, str]) -> dict[str, str]:
    """The engine-relevant part of ``env`` for a result; a secret's value is never recorded."""
    out = {}
    for name in sorted(env):
        if name.startswith(RECORD_PREFIXES):
            out[name] = "<redacted>" if is_secret(name) else env[name]
    return out


@dataclass
class ChildEnv:
    env: dict[str, str]
    #: name -> {"value", "source"}: how every variable bench or the recipe set got its value
    #: (a secret's value is recorded as "<redacted>").
    applied: dict[str, dict[str, str]]
    #: inherited names that were dropped.
    dropped: list[str]
    #: OMP_NUM_THREADS as set for the engine, or None: the engine's own default, under which
    #: torchrun gives every rank of a multi-rank stage 1 thread and a one-process stage gets
    #: torch's default (the CPU set's size). The ranks' actual torch thread counts are in the
    #: engine's evidence.
    threads: int | None


def child_env(*, caller: Mapping[str, str], env_force: Mapping[str, str],
              env_default: Mapping[str, str], cores: str, cpus: list[int],
              threads: int | None, extra: Mapping[str, str] = ()) -> ChildEnv:
    """The engine's environment for one launch.

    ``env_default`` keeps a caller's value (recorded as such); ``env_force`` always wins (recorded
    as overriding the caller when it differs). ``threads`` (a recipe's ``threads``) sets
    OMP_NUM_THREADS; unset, the engine keeps its own defaults and that is what gets recorded.
    """
    env = {k: v for k, v in caller.items() if not scrubbed(k)}
    dropped = sorted(k for k in caller if scrubbed(k))
    applied: dict[str, dict[str, str]] = {}
    for name, value in env_default.items():
        if name in caller:
            env[name] = caller[name]
            applied[name] = {"value": caller[name], "source": "caller (recipe default kept out)"}
        else:
            env[name] = value
            applied[name] = {"value": value, "source": "recipe env.default"}
    for name, value in env_force.items():
        source = "recipe env.force"
        if name in caller and caller[name] != value:
            source += " (overrode the caller's value)"
        env[name] = value
        applied[name] = {"value": value, "source": source}
    env["NEURON_RT_VISIBLE_CORES"] = cores
    applied["NEURON_RT_VISIBLE_CORES"] = {"value": cores, "source": "bench --devices"}
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
        applied["OMP_NUM_THREADS"] = {"value": str(threads), "source": "recipe threads"}
    path = os.pathsep.join(p for p in (str(source_root()), caller.get("PYTHONPATH", "")) if p)
    env["PYTHONPATH"] = path
    applied["PYTHONPATH"] = {"value": path, "source": "bench (the checkout bench runs from)"}
    for name, value in FORCED_ENV.items():
        env[name] = value
        applied[name] = {"value": value, "source": "bench"}
    cc_dir = neuronx_cc_dir(env.get("PATH", ""))
    if cc_dir:
        env["PATH"] = os.pathsep.join(p for p in (cc_dir, env.get("PATH", "")) if p)
        applied["PATH"] = {"value": env["PATH"], "source": "bench (prepended neuronx-cc's bin)"}
    for name, value in dict(extra).items():
        env[name] = value
        applied[name] = {"value": value, "source": "bench"}
    for name in applied:
        if name in dropped:
            dropped.remove(name)
        if is_secret(name):
            applied[name] = {**applied[name], "value": "<redacted>"}
    return ChildEnv(env=env, applied=applied, dropped=dropped, threads=threads)


# ------------------------------------------------------------------------------ CPU lists


def parse_cpu_list(spec: str) -> list[int]:
    """``"48-51,144"`` -> ``[48, 49, 50, 51, 144]`` (sorted, unique)."""
    cpus: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise HostError(f"bad CPU list {spec!r}")
        start, end = int(lo), int(hi) if sep else int(lo)
        if end < start:
            raise HostError(f"bad CPU range {part!r} in {spec!r}")
        cpus.update(range(start, end + 1))
    if not cpus:
        raise HostError("empty CPU list")
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """``[48, 49, 50, 144]`` -> ``"48-50,144"``."""
    out: list[str] = []
    run: list[int] = []
    for cpu in sorted(set(cpus)):
        if run and cpu != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = []
        run.append(cpu)
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def online_cpus() -> list[int]:
    return parse_cpu_list(Path("/sys/devices/system/cpu/online").read_text())


def numa_cpus() -> dict[int, list[int]]:
    nodes = {}
    for path in Path("/sys/devices/system/node").glob("node[0-9]*"):
        nodes[int(path.name[4:])] = parse_cpu_list((path / "cpulist").read_text())
    return nodes


# ---------------------------------------------------------------------------------- devices


@dataclass(frozen=True)
class Device:
    index: int
    cores: tuple[int, ...]
    numa_node: int | None
    cpu_affinity: str


@functools.lru_cache(maxsize=1)
def topology() -> dict[str, object]:
    """``neuron-ls --json-output``: the instance type and every device's cores and NUMA node.

    Used for topology only: neuron-ls does not reliably list another session's processes, so
    occupancy comes from /proc (:func:`device_holders`).
    """
    try:
        out = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True,
                             timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostError(f"neuron-ls --json-output failed: {exc}") from exc
    rows = json.loads(out)
    devices = {}
    instance = ""
    for row in rows:
        instance = instance or str(row.get("instance_type", ""))
        node = row.get("numa_node")
        devices[int(row["neuron_device"])] = Device(
            index=int(row["neuron_device"]),
            cores=tuple(int(c) for c in row.get("neuroncore_ids", [])),
            numa_node=int(node) if str(node).lstrip("-").isdigit() else None,
            cpu_affinity=str(row.get("cpu_affinity", "")))
    return {"instance_type": instance, "devices": devices}


def devices_to_cores(devices: Iterable[int]) -> str:
    """Device indices -> the ``NEURON_RT_VISIBLE_CORES`` value (from neuron-ls, not assumed)."""
    table = topology()["devices"]
    cores: list[int] = []
    for d in devices:
        if d not in table:
            raise HostError(f"no Neuron device {d} on this host (have {sorted(table)})")
        cores += table[d].cores
    return format_cpu_list(cores)


def _proc_ids() -> list[int]:
    return [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]


def _open_neuron_devices(pid: int) -> set[int]:
    found = set()
    try:
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            m = re.fullmatch(r"/dev/neuron(\d+)", target)
            if m:
                found.add(int(m.group(1)))
    except OSError:
        pass
    return found


def _environ_value(pid: int, name: str) -> str | None:
    try:
        data = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for item in data.split(b"\0"):
        key, _, value = item.partition(b"=")
        if key.decode(errors="replace") == name:
            return value.decode(errors="replace")
    return None


def _who(pid: int) -> dict[str, object]:
    """A process by its name and owner, never its command line: results are committed and
    another user's argv can carry a secret (an --api-key, a bearer token)."""
    try:
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        comm = "?"
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        uid = None
    return {"pid": pid, "comm": comm, "uid": uid}


_PF_KTHREAD = 0x00200000


def _is_kernel_thread(pid: int) -> bool:
    """Per-CPU kernel threads (migration/N, ksoftirqd/N, kworker/N:M) are pinned and always tick;
    they are not another workload. Real kernel load still shows in the CPU busy fraction."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()
        return bool(int(fields[6]) & _PF_KTHREAD)
    except (OSError, IndexError, ValueError):
        return False


def uses_device(device: int, open_devices: set[int], visible_cores: str | None,
                device0_cores: Iterable[int]) -> bool:
    """Whether a process with ``open_devices`` open is using ``device``.

    Every Neuron runtime process also opens ``/dev/neuron0``, so holding neuron0 alone does not
    mean device 0 is in use: it is, when the process holds no other device, or its
    ``NEURON_RT_VISIBLE_CORES`` overlaps device 0's cores, or that variable is unknown.
    """
    if device not in open_devices:
        return False
    if device != 0 or open_devices == {0} or not visible_cores:
        return True
    try:
        cores = set(parse_cpu_list(visible_cores))
    except HostError:
        return True
    return bool(cores & set(device0_cores))


def device_holders(devices: Iterable[int]) -> dict[int, list[dict[str, object]]]:
    """Other processes using each of ``devices``: pid, name, owner, pinned cores."""
    wanted = set(devices)
    out: dict[int, list[dict[str, object]]] = {d: [] for d in wanted}
    if not wanted:
        return out
    me = os.getpid()
    table = topology()["devices"]
    device0 = table[0].cores if 0 in table else ()
    for pid in _proc_ids():
        if pid == me:
            continue
        opened = _open_neuron_devices(pid)
        if not opened & wanted:
            continue
        cores = _environ_value(pid, "NEURON_RT_VISIBLE_CORES")
        for d in opened & wanted:
            if uses_device(d, opened, cores, device0):
                out[d].append({**_who(pid), "NEURON_RT_VISIBLE_CORES": cores})
    return out


# --------------------------------------------------------------------------------------- CPUs


def _cpu_times() -> dict[int, tuple[int, int]]:
    """cpu -> (busy jiffies, total jiffies) from /proc/stat."""
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if not re.match(r"cpu\d+ ", line):
            continue
        name, *fields = line.split()
        values = [int(v) for v in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
        total = sum(values[:8])  # guest time is already counted in user
        out[int(name[3:])] = (total - idle, total)
    return out


def busy_fraction(before: Mapping[int, tuple[int, int]],
                  after: Mapping[int, tuple[int, int]]) -> dict[int, float]:
    out = {}
    for cpu, (busy1, total1) in after.items():
        busy0, total0 = before.get(cpu, (busy1, total1))
        span = total1 - total0
        out[cpu] = (busy1 - busy0) / span if span > 0 else 0.0
    return out


def _allowed_cpus(pid: int) -> set[int] | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return set(parse_cpu_list(line.split(":", 1)[1]))
    except (OSError, HostError):
        return None
    return None


def _proc_cpu_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[-1].split()
    return int(fields[11]) + int(fields[12])  # utime + stime


# ------------------------------------------------------------------------------- idle check


@dataclass
class IdleReport:
    """What the pre-launch check saw; recorded in every result's provenance."""

    devices: list[int]
    cpus: list[int]
    device_holders: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    cpu_busy: dict[int, float] = field(default_factory=dict)
    pinned_conflicts: list[dict[str, object]] = field(default_factory=list)
    numa_warning: str = ""
    threshold: float = CPU_BUSY_FRACTION
    window_s: float = CPU_SAMPLE_SECONDS

    @property
    def busy_devices(self) -> dict[int, list[dict[str, object]]]:
        return {d: h for d, h in self.device_holders.items() if h}

    @property
    def busy_cpus(self) -> dict[int, float]:
        return {c: f for c, f in self.cpu_busy.items() if f > self.threshold}

    @property
    def idle(self) -> bool:
        return not (self.busy_devices or self.busy_cpus or self.pinned_conflicts)

    def reasons(self) -> list[str]:
        out = []
        for d, holders in sorted(self.busy_devices.items()):
            who = "; ".join(f"pid {h['pid']} {h['comm']} (uid {h['uid']}, cores "
                            f"{h['NEURON_RT_VISIBLE_CORES'] or '?'})" for h in holders[:3])
            out.append(f"device {d} is held by {len(holders)} other process(es): {who}")
        if self.busy_cpus:
            worst = sorted(self.busy_cpus.items(), key=lambda kv: -kv[1])[:8]
            out.append(f"{len(self.busy_cpus)} of the pinned CPUs are busier than "
                       f"{self.threshold:.0%}: " + ", ".join(f"{c}={f:.0%}" for c, f in worst))
        for c in self.pinned_conflicts[:3]:
            out.append(f"pid {c['pid']} {c['comm']} (uid {c['uid']}) is pinned to CPUs "
                       f"{c['allowed']} and ran during the sample")
        return out

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["idle"] = self.idle
        d["cpu_busy"] = {str(k): round(v, 4) for k, v in self.cpu_busy.items()}
        d["device_holders"] = {str(k): v for k, v in self.device_holders.items()}
        return d


def check_idle(devices: list[int], cpus: list[int], *,
               window_s: float = CPU_SAMPLE_SECONDS,
               threshold: float = CPU_BUSY_FRACTION) -> IdleReport:
    """Sample the pinned devices and CPUs; ``IdleReport.idle`` says whether the lane is clean."""
    report = IdleReport(devices=list(devices), cpus=list(cpus), threshold=threshold,
                        window_s=window_s)
    report.device_holders = device_holders(devices)
    mine = set(cpus)
    everything = set(online_cpus())
    pids = [p for p in _proc_ids() if p != os.getpid()]
    pinned = {}
    for pid in pids:
        allowed = _allowed_cpus(pid)
        if allowed and allowed != everything and allowed & mine and not _is_kernel_thread(pid):
            pinned[pid] = (allowed, _proc_cpu_ticks(pid))
    before = _cpu_times()
    time.sleep(window_s)
    after = _cpu_times()
    busy = busy_fraction(before, after)
    report.cpu_busy = {c: busy.get(c, 0.0) for c in cpus}
    for pid, (allowed, ticks0) in pinned.items():
        ticks1 = _proc_cpu_ticks(pid)
        if ticks0 is not None and ticks1 is not None and ticks1 > ticks0:
            report.pinned_conflicts.append({**_who(pid), "allowed": format_cpu_list(allowed)})
    if not devices:  # a CPU-only job (a quality reference): no device side to check
        return report
    table = topology()["devices"]
    nodes = numa_cpus()
    device_nodes = {table[d].numa_node for d in devices if d in table}
    if len(device_nodes) == 1 and None not in device_nodes:
        node = device_nodes.pop()
        if not mine <= set(nodes.get(node, [])):
            report.numa_warning = (f"devices {devices} sit on NUMA node {node} but CPUs "
                                   f"{format_cpu_list(cpus)} are not all on it")
    return report
