"""N resident engines behind one engine: ``difflet serve --dp-degree N`` on CUDA.

Each replica is a complete :class:`ResidentWorkerServingEngine` over its own GPU (the worker
pins ``CUDA_VISIBLE_DEVICES`` from its plan's core id), with its own admission, run lock and
recovery; this wrapper only fans the lifecycle out and routes each request to the replica with
the fewest pending requests (ties go round-robin). It is not the TPU lockstep replica path
inside the resident worker: a dead replica here fails its own requests and makes the server
not ready, it does not hang the others. ``dp_degree=1`` never builds one of these.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from difflet.serving.types import DiffletGenerateRequest, GenerateOutput


class ReplicatedServingEngine:
    """The engine the API server drives when the model is resident on several GPUs."""

    def __init__(self, replicas: Sequence[Any]) -> None:
        if len(replicas) < 1:
            raise ValueError("ReplicatedServingEngine needs at least one replica")
        self.replicas = tuple(replicas)
        first = self.replicas[0]
        # What the API server and the factory read off an engine.
        self.runtime = getattr(first, "runtime", None)
        self.profile = getattr(first, "profile", None)
        self.config = getattr(first, "config", None)
        self._next = 0

    async def start(self) -> None:
        results = await asyncio.gather(
            *(replica.start() for replica in self.replicas), return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            # One replica failed to load: the others are up and would bill for nothing.
            await asyncio.gather(
                *(replica.shutdown() for replica in self.replicas), return_exceptions=True
            )
            raise failures[0]

    async def shutdown(self) -> None:
        results = await asyncio.gather(
            *(replica.shutdown() for replica in self.replicas), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    @property
    def ready(self) -> bool:
        return all(replica.ready for replica in self.replicas)

    @property
    def healthy(self) -> bool:
        return all(replica.healthy for replica in self.replicas)

    @property
    def pending_requests(self) -> int:
        return sum(int(replica.pending_requests) for replica in self.replicas)

    def _pick(self) -> Any:
        """The ready replica with the fewest pending requests, ties round-robin. A replica that
        is recovering or dead answers 503 at once and so never accumulates pending requests;
        ranking it would send it every other request while the live card idles. With no ready
        replica the caller still gets the resident engine's own 503."""
        n = len(self.replicas)
        order = [(self._next + i) % n for i in range(n)]
        candidates = [i for i in order if self.replicas[i].ready] or order
        index = min(candidates, key=lambda i: int(self.replicas[i].pending_requests))
        self._next = (index + 1) % n
        return self.replicas[index]

    async def generate(self, request: DiffletGenerateRequest) -> GenerateOutput:
        return await self._pick().generate(request)


__all__ = ["ReplicatedServingEngine"]
