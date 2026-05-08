"""Parallelism configuration for Nova pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NovaParallelConfig:
    """Tensor, context, and CFG parallel settings.

    Context parallelism and CFG parallelism both consume a second data-parallel
    lane in the current NxDI Flux implementation, so they are mutually exclusive.
    """

    tp_degree: int = 1
    cp_enabled: bool = False
    cfg_parallel_enabled: bool = False

    def __post_init__(self) -> None:
        if self.tp_degree < 1:
            raise ValueError("tp_degree must be >= 1")
        if self.cp_enabled and self.cfg_parallel_enabled:
            raise ValueError("cp_enabled and cfg_parallel_enabled are mutually exclusive")

    @property
    def world_size(self) -> int:
        multiplier = 2 if self.cp_enabled or self.cfg_parallel_enabled else 1
        return self.tp_degree * multiplier

    def to_cache_dict(self) -> dict[str, object]:
        return asdict(self)
