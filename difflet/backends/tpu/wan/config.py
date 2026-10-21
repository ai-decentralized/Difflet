"""Geometry config for the Wan DiT on TPU.

The geometry, the latent-shape properties and ``from_pretrained`` live in the
backend-neutral ``difflet.models.wan.dit_config.WanDiTConfig``; this subclass
adds only what the TPU backend refuses, so that an unsupported parallel mode
fails here, with a message, instead of silently constructing a model that
ignores it.

Follows Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md, extended to
Wan.
"""

from __future__ import annotations

from dataclasses import dataclass

from difflet.models.wan.dit_config import WanDiTConfig


@dataclass
class TpuWanConfig(WanDiTConfig):
    """``WanDiTConfig`` plus the TPU backend's parallel-mode refusals."""

    def validate(self) -> None:
        super().validate()
        for mode, enabled in (
            ("context parallelism", self.context_parallel_enabled),
            ("cfg parallelism", self.cfg_parallel_enabled),
            ("sequence parallelism", self.sp_enabled),
        ):
            if enabled:
                raise NotImplementedError(
                    f"the TPU backend does not implement {mode} yet; run with "
                    f"tp only (see the plan's 'what is left')"
                )


__all__ = ["TpuWanConfig"]
