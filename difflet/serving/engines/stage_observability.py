"""Per-stage wall time and resident device memory for the resident worker.

The engine already walks an ordered ``PipelineDefinition.stages`` tuple per request and
already owns the generic stage lifecycle logs. This module adds the two numbers those logs
were missing: how long each stage took on this request, and how much device memory each
stage's load left resident.

Two things are deliberately kept apart:

* **Timing** is entirely the engine's: it brackets every ``executor.execute`` call, so it
  works for any adapter without the adapter knowing it exists.
* **Memory attribution** needs a boundary only the adapter can draw, because only the adapter
  knows where one stage's weights stop and the next stage's begin inside its own
  ``create_loaded_runners``. An adapter marks those boundaries with ``record_stage_load``;
  the engine supplies the probe and reads the result. An adapter that marks nothing is not a
  failure — the report then says ``attribution=total_only`` and carries the whole delta.

Nothing here names a stage. Stage IDs come from the pipeline definition and from the adapter
that owns them, so the engine stays model-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

logger = logging.getLogger(__name__)

#: The role of a stage the pipeline definition does not declare, or that no pipeline was
#: supplied for. Never silently mapped onto a real role.
UNKNOWN_ROLE = "unknown"

MemoryScope = Literal["process", "host"]
LoadAttribution = Literal["per_stage", "total_only", "unavailable"]


@dataclass(frozen=True, slots=True)
class DeviceMemorySample:
    """Resident device bytes at one instant, and whose bytes they are."""

    resident_bytes: int
    scope: MemoryScope


class DeviceMemoryProbe(Protocol):
    name: str

    def sample(self) -> DeviceMemorySample | None: ...

    @property
    def unavailable_reason(self) -> str | None: ...


class NullDeviceMemoryProbe:
    """The probe for a backend that exposes no resident device memory at all."""

    def __init__(self, reason: str) -> None:
        self.name = "none"
        self._reason = reason

    def sample(self) -> DeviceMemorySample | None:
        return None

    @property
    def unavailable_reason(self) -> str | None:
        return self._reason


class CudaDeviceMemoryProbe:
    """``torch.cuda`` allocator bytes held by this process on its current device.

    ``memory_allocated`` and not ``mem_get_info``: the question is what this worker's stages
    hold, not what every process on the card holds together. Reserved-but-free blocks in the
    caching allocator are not a stage's residency, so they are excluded.
    """

    def __init__(self) -> None:
        self.name = "torch.cuda.memory_allocated"
        self._reason: str | None = None

    def sample(self) -> DeviceMemorySample | None:
        try:
            import torch

            if not torch.cuda.is_available():
                self._reason = "torch.cuda is not available in the worker process"
                return None
            return DeviceMemorySample(int(torch.cuda.memory_allocated()), "process")
        except BaseException as exc:  # pragma: no cover - defensive, never fails a load
            self._reason = f"{type(exc).__name__}: {exc}"
            return None

    @property
    def unavailable_reason(self) -> str | None:
        return self._reason


class NeuronDeviceMemoryProbe:
    """Device bytes reported by ``neuron-monitor`` for this process's Neuron runtime.

    ``neuron-monitor`` is the only resident-HBM source the Neuron runtime exposes to a
    process that is not holding the runtime's own handles, and it is the source the
    68.36 GiB figure in ``docs/design/qwen_trn2_topology`` came from. It is a subprocess that
    prints one JSON sample per period, so a sample costs well under a second against stage
    loads measured in tens of seconds.

    If the binary is missing, the sample does not arrive, or the payload does not carry the
    field this reads, the probe reports itself unavailable. It never estimates.
    """

    _FIELD = "neuron_runtime_data[].report.memory_used.neuron_runtime_used_bytes.neuron_device"

    def __init__(
        self,
        *,
        command: str = "neuron-monitor",
        timeout_s: float = 10.0,
        pid: int | None = None,
        read_sample: Callable[[], str] | None = None,
    ) -> None:
        self.name = "neuron-monitor"
        self._command = command
        self._timeout_s = float(timeout_s)
        self._pid = os.getpid() if pid is None else int(pid)
        self._read_sample = read_sample or self._read_sample_from_subprocess
        self._reason: str | None = None

    def sample(self) -> DeviceMemorySample | None:
        try:
            raw = self._read_sample()
        except BaseException as exc:
            self._reason = f"neuron-monitor unavailable ({type(exc).__name__}: {exc})"
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            self._reason = f"neuron-monitor returned unparseable JSON ({exc})"
            return None
        return self._resident_bytes(payload)

    def _resident_bytes(self, payload: Any) -> DeviceMemorySample | None:
        runtimes = payload.get("neuron_runtime_data") if isinstance(payload, dict) else None
        if not isinstance(runtimes, list) or not runtimes:
            self._reason = "neuron-monitor reported no Neuron runtimes"
            return None
        own: list[int] = []
        every: list[int] = []
        for runtime in runtimes:
            used = self._runtime_device_bytes(runtime)
            if used is None:
                continue
            every.append(used)
            if isinstance(runtime, dict) and runtime.get("pid") == self._pid:
                own.append(used)
        if own:
            return DeviceMemorySample(sum(own), "process")
        if every:
            # The runtime is there but is not attributed to this PID (a child process holds
            # it, or the payload omits pids). Report the host total and let the caller say
            # so, rather than silently presenting someone else's bytes as this worker's.
            return DeviceMemorySample(sum(every), "host")
        self._reason = f"neuron-monitor payload has no {self._FIELD}"
        return None

    @staticmethod
    def _runtime_device_bytes(runtime: Any) -> int | None:
        if not isinstance(runtime, dict):
            return None
        node: Any = runtime
        for key in ("report", "memory_used", "neuron_runtime_used_bytes", "neuron_device"):
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return int(node) if isinstance(node, (int, float)) else None

    def _read_sample_from_subprocess(self) -> str:
        process = subprocess.Popen(
            [self._command],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            assert process.stdout is not None
            readable, _, _ = select.select([process.stdout], [], [], self._timeout_s)
            if not readable:
                raise TimeoutError(f"no sample within {self._timeout_s:g}s")
            line = process.stdout.readline().strip()
            if not line:
                raise ValueError("empty sample")
            return line
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                    process.kill()
                    process.wait(timeout=2.0)

    @property
    def unavailable_reason(self) -> str | None:
        return self._reason


def resolve_backend_label() -> str:
    """The active platform's name, or ``"unknown"`` if it cannot be resolved here."""

    try:
        from difflet.platforms import current_platform

        return current_platform().name
    except BaseException:  # pragma: no cover - defensive
        return "unknown"


def select_device_memory_probe(backend: str | None = None) -> DeviceMemoryProbe:
    """The probe for the platform this worker runs on."""

    if backend is None:
        backend = resolve_backend_label()
    if backend == "unknown":
        return NullDeviceMemoryProbe("the active Difflet backend could not be resolved")
    from difflet.platforms import get_platform

    flow = get_platform(backend).capabilities.flow
    if flow == "gpu":
        return CudaDeviceMemoryProbe()
    if flow == "neuron":
        return NeuronDeviceMemoryProbe()
    return NullDeviceMemoryProbe(
        f"no resident device memory probe for backend {backend!r}; "
        "per-stage device memory is not attributed"
    )


@dataclass(frozen=True, slots=True)
class StageLoadRecord:
    """One load boundary. It usually covers one stage; an adapter whose stages come up in a
    single load names them all, so the bytes are reported jointly instead of being split by
    a guess."""

    stage_ids: tuple[str, ...]
    load_seconds: float
    device_bytes: int | None
    # ``role`` is the normalized vocabulary (prompt_encoder, denoiser, decoder, pipeline)
    # that is comparable across models; ``stage_id`` is not. Group on roles.
    roles: tuple[str, ...] = ()

    @property
    def stage_id(self) -> str:
        return "+".join(self.stage_ids)

    @property
    def role(self) -> str:
        return "+".join(self.roles) if self.roles else UNKNOWN_ROLE

    def as_event(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_ids": list(self.stage_ids),
            "role": self.role,
            "roles": list(self.roles),
            "load_seconds": round(self.load_seconds, 3),
            "device_bytes": self.device_bytes,
        }


@dataclass(frozen=True, slots=True)
class StageLoadReport:
    """What each stage's load left resident, and how trustworthy that split is."""

    backend: str
    probe: str
    attribution: LoadAttribution
    scope: MemoryScope | None
    stages: tuple[StageLoadRecord, ...]
    total_device_bytes: int | None
    reason: str | None = None
    # Stage IDs an adapter marked that the pipeline definition does not declare: a typo in
    # an adapter, surfaced rather than silently logged against a stage nobody runs.
    unknown_stage_ids: tuple[str, ...] = ()

    @property
    def attributed_device_bytes(self) -> int | None:
        attributed = [stage.device_bytes for stage in self.stages if stage.device_bytes is not None]
        return sum(attributed) if attributed else None

    @property
    def unattributed_device_bytes(self) -> int | None:
        attributed = self.attributed_device_bytes
        if self.total_device_bytes is None or attributed is None:
            return None
        return self.total_device_bytes - attributed

    def as_event(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "probe": self.probe,
            "attribution": self.attribution,
            "scope": self.scope,
            "stages": [stage.as_event() for stage in self.stages],
            "total_device_bytes": self.total_device_bytes,
            "attributed_device_bytes": self.attributed_device_bytes,
            "unattributed_device_bytes": self.unattributed_device_bytes,
            "reason": self.reason,
            "unknown_stage_ids": list(self.unknown_stage_ids),
        }


class StageLoadRecorder:
    """Collect per-stage load time and device memory deltas for one worker startup."""

    def __init__(
        self,
        probe: DeviceMemoryProbe,
        *,
        backend: str = "unknown",
        stage_roles: Mapping[str, str] | None = None,
    ) -> None:
        self.probe = probe
        self.backend = backend
        # stage_id -> role, from the pipeline definition the engine is about to run.
        self.stage_roles = dict(stage_roles or {})
        self._records: list[StageLoadRecord] = []
        self._unknown_stage_ids: list[str] = []
        self._baseline: DeviceMemorySample | None = None
        self._final: DeviceMemorySample | None = None
        self._scope: MemoryScope | None = None

    def begin(self) -> None:
        self._baseline = self._sample()

    def finish(self) -> None:
        self._final = self._sample()

    def _sample(self) -> DeviceMemorySample | None:
        sample = self.probe.sample()
        if sample is not None:
            # A probe that falls back to host scope at any point taints the whole report:
            # one host-scoped endpoint makes every delta a host delta.
            self._scope = "host" if self._scope == "host" or sample.scope == "host" else "process"
        return sample

    @contextmanager
    def record(self, *stage_ids: str) -> Iterator[None]:
        if not stage_ids:
            raise ValueError("a stage load boundary must name at least one stage")
        before = self._sample()
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            after = self._sample()
            delta = (
                after.resident_bytes - before.resident_bytes
                if before is not None and after is not None
                else None
            )
            roles = tuple(self._role(stage_id) for stage_id in stage_ids)
            self._records.append(StageLoadRecord(tuple(stage_ids), elapsed, delta, roles))

    def _role(self, stage_id: str) -> str:
        if not self.stage_roles:
            # No pipeline to check against (a bare recorder in a test): claim nothing.
            return UNKNOWN_ROLE
        try:
            return self.stage_roles[stage_id]
        except KeyError:
            if stage_id not in self._unknown_stage_ids:
                self._unknown_stage_ids.append(stage_id)
            return UNKNOWN_ROLE

    def report(self) -> StageLoadReport:
        total = (
            self._final.resident_bytes - self._baseline.resident_bytes
            if self._baseline is not None and self._final is not None
            else None
        )
        measured = any(record.device_bytes is not None for record in self._records)
        if measured:
            attribution: LoadAttribution = "per_stage"
            reason = None
        elif total is not None:
            attribution = "total_only"
            reason = (
                "no stage marked its load boundary; the total below covers every stage together"
                if not self._records
                else "stage load boundaries were marked but the probe returned no sample"
            )
        else:
            attribution = "unavailable"
            reason = self.probe.unavailable_reason or "the device memory probe returned no sample"
        return StageLoadReport(
            backend=self.backend,
            probe=self.probe.name,
            attribution=attribution,
            scope=self._scope,
            stages=tuple(self._records),
            total_device_bytes=total,
            reason=reason,
            unknown_stage_ids=tuple(self._unknown_stage_ids),
        )


_ACTIVE_RECORDER: ContextVar[StageLoadRecorder | None] = ContextVar(
    "difflet_stage_load_recorder", default=None
)


@contextmanager
def recording_stage_loads(recorder: StageLoadRecorder) -> Iterator[StageLoadRecorder]:
    """Make ``recorder`` the one ``record_stage_load`` writes to, for this context."""

    token = _ACTIVE_RECORDER.set(recorder)
    recorder.begin()
    try:
        yield recorder
    finally:
        recorder.finish()
        _ACTIVE_RECORDER.reset(token)


@contextmanager
def record_stage_load(*stage_ids: str) -> Iterator[None]:
    """Mark one load boundary inside an adapter's ``create_loaded_runners``.

    Name every stage the load brings up: one for a stage loaded on its own, several when one
    load covers several stages and splitting the bytes between them would be a guess.

    A no-op outside a worker startup (the CLI loads the same adapters), so an adapter can
    wrap its loads unconditionally.
    """

    recorder = _ACTIVE_RECORDER.get()
    if recorder is None:
        yield
        return
    with recorder.record(*stage_ids):
        yield


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage_id: str
    seconds: float
    role: str = UNKNOWN_ROLE


@dataclass(frozen=True, slots=True)
class RequestStageBreakdown:
    """Where one request's wall time went, stage by stage, in execution order."""

    request_id: str
    model: str
    outcome: str
    stages: tuple[StageTiming, ...]
    total_seconds: float

    @property
    def stage_seconds(self) -> dict[str, float]:
        return {stage.stage_id: stage.seconds for stage in self.stages}

    @property
    def role_seconds(self) -> dict[str, float]:
        """Seconds per role, summed — the comparable view across models, since two stages
        can share a role (HunyuanVideo's ``clip`` and ``llama`` are both prompt_encoder)."""

        totals: dict[str, float] = {}
        for stage in self.stages:
            totals[stage.role] = totals.get(stage.role, 0.0) + stage.seconds
        return totals

    def as_event(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "outcome": self.outcome,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "role": stage.role,
                    "seconds": round(stage.seconds, 3),
                }
                for stage in self.stages
            ],
            # The comparable view: two models with different stage IDs still line up here.
            "role_seconds": {
                role: round(seconds, 3) for role, seconds in self.role_seconds.items()
            },
            "total_seconds": round(self.total_seconds, 3),
        }

    def summary(self) -> str:
        """``text=1.204s(9.8%) generate=10.900s(89.0%)`` — the share is of total_seconds."""

        if not self.stages:
            return "none"
        total = self.total_seconds if self.total_seconds > 0 else None
        return " ".join(
            f"{stage.stage_id}={stage.seconds:.3f}s"
            + (f"({100.0 * stage.seconds / total:.1f}%)" if total else "")
            for stage in self.stages
        )


def format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value} ({value / (1024 ** 3):.2f} GiB)"


__all__ = [
    "UNKNOWN_ROLE",
    "CudaDeviceMemoryProbe",
    "DeviceMemoryProbe",
    "DeviceMemorySample",
    "NeuronDeviceMemoryProbe",
    "NullDeviceMemoryProbe",
    "RequestStageBreakdown",
    "StageLoadRecord",
    "StageLoadRecorder",
    "StageLoadReport",
    "StageTiming",
    "format_bytes",
    "record_stage_load",
    "resolve_backend_label",
    "recording_stage_loads",
    "select_device_memory_probe",
]
