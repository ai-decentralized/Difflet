"""Request manifest I/O for the DP router (spec 2026-07-06 §Router)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

_OPTIONAL_FIELDS = ("negative_prompt", "guidance_scale", "steps")


@dataclass(frozen=True)
class RequestSpec:
    index: int
    prompt: str
    output: str
    seed: int = 42
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    steps: int | None = None
    assigned_worker: int | None = None


def load_requests_jsonl(path: str | Path) -> list[RequestSpec]:
    requests: list[RequestSpec] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {lineno}: invalid JSON: {exc}") from exc
        for field in ("prompt", "output"):
            if not data.get(field):
                raise ValueError(f"{path}: line {lineno}: missing required field {field!r}")
        kwargs = {k: data[k] for k in _OPTIONAL_FIELDS if k in data}
        requests.append(
            RequestSpec(
                index=len(requests),
                prompt=str(data["prompt"]),
                output=str(data["output"]),
                seed=int(data.get("seed", 42)),
                **kwargs,
            )
        )
    if not requests:
        raise ValueError(f"{path}: no requests found")
    outputs = [r.output for r in requests]
    dupes = {o for o in outputs if outputs.count(o) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate output paths: {sorted(dupes)}")
    return requests


def request_path(requests_dir: Path, index: int) -> Path:
    return Path(requests_dir) / f"req_{index:04d}.json"


def write_manifest(requests: list[RequestSpec], requests_dir: Path) -> None:
    requests_dir = Path(requests_dir)
    requests_dir.mkdir(parents=True, exist_ok=True)
    for req in requests:
        request_path(requests_dir, req.index).write_text(
            json.dumps(dataclasses.asdict(req)), encoding="utf-8"
        )


def read_manifest(requests_dir: Path) -> list[RequestSpec]:
    specs = []
    for p in sorted(Path(requests_dir).glob("req_*.json")):
        specs.append(RequestSpec(**json.loads(p.read_text(encoding="utf-8"))))
    return specs
