"""Request iteration for stages: legacy single-request or batch claim loop.

A worker's FIRST stage iterates claim_requests(args) (claims until the queue is
empty); every LATER stage iterates claimed_requests(args) (replays this
worker's claims, skipping ones an earlier stage failed). request_scope marks
.failed-and-continue in batch mode, re-raises in legacy mode, and marks .done
on success when final=True (the stage that writes the user-visible output).
"""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from difflet.cli.dp import claims
from difflet.cli.dp.requests_io import RequestSpec


def batch_mode(args) -> bool:
    return getattr(args, "requests_dir", None) is not None


def _legacy_request(args) -> RequestSpec:
    return RequestSpec(
        index=0,
        prompt=args.prompt,
        output=args.output,
        seed=int(getattr(args, "seed", 42)),
        guidance_scale=getattr(args, "guidance_scale", None),
        steps=getattr(args, "steps", None),
    )


def claim_requests(args) -> Iterator[RequestSpec]:
    if not batch_mode(args):
        yield _legacy_request(args)
        return
    while (
        req := claims.claim_next(args.requests_dir, int(args.worker_index), args.dp_schedule)
    ) is not None:
        yield req


def claimed_requests(args) -> Iterator[RequestSpec]:
    if not batch_mode(args):
        yield _legacy_request(args)
        return
    for req in claims.claimed_by(args.requests_dir, int(args.worker_index)):
        if not claims.is_failed(args.requests_dir, req.index):
            yield req


@contextmanager
def request_scope(args, request: RequestSpec, *, final: bool):
    if not batch_mode(args):
        yield
        return
    try:
        yield
    except Exception:
        claims.mark_failed(args.requests_dir, request.index, traceback.format_exc())
        print(f"[dp-worker] request {request.index} failed; continuing", flush=True)
    else:
        if final:
            claims.mark_done(args.requests_dir, request.index)


def work_file(args, request: RequestSpec | None, name: str) -> Path:
    base = Path(args.work_dir) / name
    if not batch_mode(args) or request is None:
        return base
    return base.with_name(f"{base.stem}_req{request.index:04d}{base.suffix}")


def effective(request: RequestSpec | None, args, field: str, default):
    if request is not None and getattr(request, field, None) is not None:
        return getattr(request, field)
    value = getattr(args, field, None)
    return value if value is not None else default
