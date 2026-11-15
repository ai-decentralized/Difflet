"""Device memory per rank, sampled with ``neuron-monitor`` while a launch runs (design §7, memory).

The method is quant_bench's (``scripts/quant_bench/hbm_sampler.py`` on feat/mxfp8-video-models):
``neuron-monitor`` reports every runtime process once a second; a process whose memory sits on
exactly one logical core is a rank on that core, and its ``neuron_device`` bytes are its device
memory. Only the launch's own cores are kept. Samples carry the monitor process's arrival time on
the monotonic clock, so they line up with the engine's timing events, and the pid, so they can be
attributed to the stage that process ran. One-second sampling misses shorter peaks; the in-process
``max_memory_allocated`` from the engine's evidence is recorded beside it as a cross-check.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

CONFIG = {"period": "1s",
          "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "memory_used"}]}],
          "system_metrics": []}
GIB = float(1 << 30)


@dataclass
class Sample:
    t_ns: int
    pid: int
    core: int
    device_bytes: int


def parse_report(line: str, cores: set[int], t_ns: int) -> list[Sample]:
    """The samples in one ``neuron-monitor`` JSON line, kept to ``cores``."""
    try:
        report = json.loads(line)
    except ValueError:
        return []
    out = []
    for runtime in report.get("neuron_runtime_data") or []:
        used = ((runtime.get("report") or {}).get("memory_used") or {}).get(
            "neuron_runtime_used_bytes") or {}
        per_core = (used.get("usage_breakdown") or {}).get("neuroncore_memory_usage") or {}
        busy = [c for c, b in per_core.items() if isinstance(b, dict) and any(b.values())]
        if len(busy) != 1 or not str(busy[0]).isdigit():
            continue
        core = int(busy[0])
        if core in cores:
            out.append(Sample(t_ns=t_ns, pid=int(runtime.get("pid", 0)), core=core,
                              device_bytes=int(used.get("neuron_device", 0) or 0)))
    return out


@dataclass
class Sampler:
    """``neuron-monitor`` in the background for the duration of a ``with`` block."""

    cores: set[int]
    #: CPUs to run neuron-monitor on: off the lane's, so sampling does not load the measurement.
    cpus: list[int] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    error: str = ""
    _proc: subprocess.Popen | None = None
    _thread: threading.Thread | None = None
    _config: Path | None = None

    def __enter__(self) -> "Sampler":
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(CONFIG, handle)
        handle.close()
        self._config = Path(handle.name)
        command = ["neuron-monitor", "-c", str(self._config)]
        if self.cpus:
            from difflet.bench.host import format_cpu_list

            command = ["taskset", "-c", format_cpu_list(self.cpus), *command]
        try:
            self._proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError as exc:
            self.error = f"neuron-monitor could not start: {exc}"
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self.samples.extend(parse_report(line, self.cores, time.monotonic_ns()))

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._config is not None:
            self._config.unlink(missing_ok=True)


def summarize(samples: list[Sample], stage_of_pid: dict[int, str | None]) -> dict:
    """Peak device memory per core and per stage, in GiB."""
    per_core: dict[int, float] = {}
    per_stage: dict[str, float] = {}
    for s in samples:
        gib = s.device_bytes / GIB
        per_core[s.core] = max(per_core.get(s.core, 0.0), gib)
        stage = stage_of_pid.get(s.pid) or "unattributed"
        per_stage[stage] = max(per_stage.get(stage, 0.0), gib)
    return {"peak_gib_per_core": {str(c): round(v, 4) for c, v in sorted(per_core.items())},
            "peak_gib_per_stage": {k: round(v, 4) for k, v in sorted(per_stage.items())},
            "peak_gib": round(max(per_core.values()), 4) if per_core else None,
            "samples": len(samples)}
