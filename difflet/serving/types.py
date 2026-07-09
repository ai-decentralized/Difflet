"""Shared data contracts for Difflet serving."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from difflet.pipeline.parallel_config import DiffletParallelConfig

OutputModality = Literal["image", "video"]


@dataclass(frozen=True)
class ServingProfile:
    """The single loaded model/profile identity for a serving process."""

    model_id: str
    model_type: str
    height: int
    width: int
    num_frames: int | None
    parallel: DiffletParallelConfig
    cache_dir: str | None = None
    revision: str | None = None
    dtype: str = "bfloat16"
    output_modality: OutputModality = "image"
    output_mime_type: str = "image/png"

    @property
    def world_size(self) -> int:
        return self.parallel.world_size

    def shape_dict(self) -> dict[str, int | None]:
        return {"height": self.height, "width": self.width, "num_frames": self.num_frames}


@dataclass(frozen=True)
class DiffletGenerateRequest:
    request_id: str
    model: str
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    output_format: str = "png"


@dataclass(frozen=True)
class DiffletGenerateOutput:
    data: bytes
    mime_type: str
    output_format: str = "png"


@dataclass(frozen=True)
class DiffletStageSpec:
    stage_id: str
    role: str
    num_cores: int
    output_keys: tuple[str, ...] = ()
    final_output: bool = False


@dataclass(frozen=True)
class DiffletCompileSpec:
    stage_id: str
    artifact_path: str
    required: bool = True


@dataclass
class CancellationSignal:
    """Cooperative cancellation flag checked at model-safe points."""

    _event: asyncio.Event = field(default_factory=asyncio.Event)
    external_event: Any | None = None

    def cancel(self) -> None:
        self._event.set()
        if self.external_event is not None:
            self.external_event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or (
            self.external_event is not None and self.external_event.is_set()
        )

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError()


@dataclass(frozen=True)
class WorkerRequestContext:
    request_id: str
    deadline_monotonic: float
    cancellation: CancellationSignal

    @classmethod
    def with_timeout(
        cls,
        request_id: str,
        timeout_s: float,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> "WorkerRequestContext":
        return cls(
            request_id=request_id,
            deadline_monotonic=time.monotonic() + float(timeout_s),
            cancellation=cancellation or CancellationSignal(),
        )
