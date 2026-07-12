"""Filesystem claim protocol: the requests dir IS the queue (spec §Router)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from difflet.cli.dp.requests_io import RequestSpec, read_manifest


@dataclass
class RouterSummary:
    done: list[int]
    failed: dict[int, str]
    unfinished: list[int]


def _marker(requests_dir: Path, index: int, kind: str) -> Path:
    return Path(requests_dir) / f"req_{index:04d}.{kind}"


def _try_claim(requests_dir: Path, index: int, worker_index: int) -> bool:
    try:
        fd = os.open(_marker(requests_dir, index, "claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(str(worker_index))
    return True


def claim_next(requests_dir: Path, worker_index: int, schedule: str) -> RequestSpec | None:
    if schedule not in ("round_robin", "least_loaded"):
        raise ValueError(f"unknown schedule {schedule!r}")
    for req in read_manifest(requests_dir):
        if schedule == "round_robin" and req.assigned_worker != worker_index:
            continue
        if _try_claim(requests_dir, req.index, worker_index):
            return req
    return None


def claimed_by(requests_dir: Path, worker_index: int) -> list[RequestSpec]:
    mine = []
    for req in read_manifest(requests_dir):
        claim = _marker(requests_dir, req.index, "claim")
        if claim.exists() and claim.read_text(encoding="utf-8") == str(worker_index):
            mine.append(req)
    return mine


def mark_done(requests_dir: Path, index: int) -> None:
    _marker(requests_dir, index, "done").touch()


def mark_failed(requests_dir: Path, index: int, error: str) -> None:
    _marker(requests_dir, index, "failed").write_text(error, encoding="utf-8")


def is_failed(requests_dir: Path, index: int) -> bool:
    return _marker(requests_dir, index, "failed").exists()


def summarize(requests_dir: Path) -> RouterSummary:
    done, failed, unfinished = [], {}, []
    for req in read_manifest(requests_dir):
        if _marker(requests_dir, req.index, "done").exists():
            done.append(req.index)
        elif is_failed(requests_dir, req.index):
            failed[req.index] = _marker(requests_dir, req.index, "failed").read_text(
                encoding="utf-8"
            )
        else:
            unfinished.append(req.index)
    return RouterSummary(done=done, failed=failed, unfinished=unfinished)
