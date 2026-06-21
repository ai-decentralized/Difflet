"""Common backend runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendCapabilities:
    requires_aot: bool
    single_process_multi_core: bool
    supports_torchrun_mpmd: bool


class BackendRuntime:
    """Small runtime surface used by DiffletPipeline.

    This is intentionally narrower than the eventual backend API. The first
    milestone only moves runtime selection and load-rank policy behind a
    backend boundary; model code can migrate to ``difflet.ops`` later.
    """

    name: str
    capabilities: BackendCapabilities

    def prepare_runtime(self, parallel: Any) -> None:
        """Initialize backend runtime state before application construction."""

    def resolve_load_rank_range(
        self,
        *,
        start_rank_id: int | None,
        local_ranks_size: int | None,
    ) -> tuple[int | None, int | None]:
        return start_rank_id, local_ranks_size
