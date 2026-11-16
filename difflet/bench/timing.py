"""Read the engine's timing events (``difflet/utils/bench_timing.py``) back into structure.

One launch writes one JSONL file per process (driver, every rank of every stage). All events use
``time.monotonic_ns()``, so files from different processes line up on one clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Span:
    name: str
    pid: int
    rank: int
    begin_ns: int
    end_ns: int | None
    ok: bool | None
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds(self) -> float | None:
        return None if self.end_ns is None else (self.end_ns - self.begin_ns) / 1e9

    @property
    def stage(self) -> str | None:
        return self.fields.get("stage")

    @property
    def request(self) -> Any:
        return self.fields.get("request")


@dataclass
class Timeline:
    events: list[dict[str, Any]]
    spans: list[Span]
    #: pid -> the process's first event (argv, role) plus its rank/world.
    processes: dict[int, dict[str, Any]]

    def spans_named(self, name: str, *, rank: int | None = None) -> list[Span]:
        return [s for s in self.spans if s.name == name and (rank is None or s.rank == rank)]

    def args(self) -> dict[str, Any] | None:
        """The driver's final arguments (the ``args`` event), or None."""
        for event in self.events:
            if event["event"] == "args":
                return event.get("args")
        return None

    def evidence(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == "evidence"]

    def stage_of(self, pid: int) -> str | None:
        """The stage a process ran (from its spans), e.g. a torchrun stage's tag or 'vae'."""
        for span in self.spans:
            if span.pid == pid and span.stage:
                return span.stage
        return None

    def steps(self, *, rank: int = 0) -> list[list[int]]:
        """Per request (in order) the step timestamps of ``rank``: one per denoise iteration."""
        out: list[list[int]] = []
        for request in sorted(self.spans_named("request", rank=rank), key=lambda s: s.begin_ns):
            end = request.end_ns if request.end_ns is not None else float("inf")
            out.append([e["t_ns"] for e in self.events
                        if e["event"] == "step" and e["pid"] == request.pid
                        and request.begin_ns <= e["t_ns"] <= end])
        return out

    def forwards(self, *, rank: int = 0) -> list[list[float]]:
        """Per request the DiT forward seconds of ``rank``."""
        out: list[list[float]] = []
        for request in sorted(self.spans_named("request", rank=rank), key=lambda s: s.begin_ns):
            end = request.end_ns if request.end_ns is not None else float("inf")
            out.append([float(e["seconds"]) for e in self.events
                        if e["event"] == "forward" and e["pid"] == request.pid
                        and request.begin_ns <= e["t_ns"] <= end])
        return out

    def ranks(self) -> list[int]:
        return sorted({s.rank for s in self.spans_named("request")})


def load(directory: Path) -> Timeline:
    events: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue  # a process killed mid-write leaves at most one torn last line
    events.sort(key=lambda e: e.get("t_ns", 0))
    return Timeline(events=events, spans=_pair_spans(events), processes=_processes(events))


def _processes(events: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for event in events:
        if event["event"] == "process":
            out.setdefault(event["pid"], dict(event))
    return out


def _pair_spans(events: Iterable[dict[str, Any]]) -> list[Span]:
    open_spans: dict[tuple, list[Span]] = {}
    out: list[Span] = []
    for event in events:
        if event["event"] != "span":
            continue
        fields = {k: v for k, v in event.items()
                  if k not in ("event", "t_ns", "pid", "rank", "world", "name", "phase", "ok")}
        key = (event["pid"], event["name"], json.dumps(fields, sort_keys=True))
        if event.get("phase") == "begin":
            span = Span(name=event["name"], pid=event["pid"], rank=event.get("rank", 0),
                        begin_ns=event["t_ns"], end_ns=None, ok=None, fields=fields)
            open_spans.setdefault(key, []).append(span)
            out.append(span)
        elif open_spans.get(key):
            span = open_spans[key].pop()
            span.end_ns = event["t_ns"]
            span.ok = event.get("ok")
    return out


def deltas(stamps: list[int]) -> list[float]:
    """Inter-step seconds; step 0 has no delta (it also carries any first-execution work)."""
    return [(b - a) / 1e9 for a, b in zip(stamps, stamps[1:])]


def stage_breakdown(timeline: Timeline) -> dict[str, float]:
    """Seconds by part of one generate, from rank 0 and the driver (design §7, e2e_warm).

    ``before denoise`` is a request's start to its denoise span (prompt encode, latent prep);
    ``after denoise`` is the denoise end to the request end (an in-process decode, if any).
    """
    out: dict[str, float] = {}

    def add(key: str, seconds: float | None) -> None:
        if seconds is not None:
            out[key] = out.get(key, 0.0) + seconds

    for span in timeline.spans:
        if span.rank != 0 or span.seconds is None:
            continue
        stage = span.stage or "?"
        if span.name in ("load", "save", "decode", "encode"):
            add(f"{stage}: {span.name}", span.seconds)
        elif span.name == "request":
            add(f"{stage}: request", span.seconds)
            denoise = [d for d in timeline.spans_named("denoise", rank=0)
                       if d.pid == span.pid and d.end_ns is not None and span.end_ns is not None
                       and span.begin_ns <= d.begin_ns and d.end_ns <= span.end_ns]
            if denoise:
                add(f"{stage}: before denoise", (denoise[0].begin_ns - span.begin_ns) / 1e9)
                add(f"{stage}: denoise", sum(d.seconds for d in denoise))
                add(f"{stage}: after denoise", (span.end_ns - denoise[-1].end_ns) / 1e9)
    for record in timeline.evidence():
        if record.get("rank", 0) != 0:
            continue
        warmup = warmup_seconds(record.get("phase_seconds") or {})
        if warmup is not None:
            add(f"{record.get('stage') or '?'}: load includes warmup forward", warmup)
    return {k: round(v, 6) for k, v in out.items()}


def warmup_seconds(phase_seconds: dict[str, Any]) -> float | None:
    """The warm-up forwards of every component (``warmup forward``, ``vae warmup forward``,
    ``dit warmup forward`` ...): where a compile, or the NEFF cache load, actually happens."""
    values = [float(v) for k, v in phase_seconds.items()
              if str(k).endswith("warmup forward") and isinstance(v, (int, float))]
    return sum(values) if values else None
