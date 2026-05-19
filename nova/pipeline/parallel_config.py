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


@dataclass(frozen=True)
class CandidateConfig:
    """Candidate (``N``) axis for the candidate-aware latent runtime.

    ``N`` is a *batch-like* axis, **not** a parallel axis: it never
    changes ``NovaParallelConfig.world_size`` and introduces no new
    collectives (cclog 43 §3.1). It is carried alongside
    ``NovaParallelConfig`` rather than merged into it so the
    parallel/communicator contract every model depends on is untouched.

    - ``max_candidates`` — the candidate batch the AOT artifact is
      traced for. Part of the compile-cache key (different ``max`` =
      different artifact).
    - ``active_candidates`` — runtime count, ``1 <= active <= max``.
      Smaller ``N`` is padding/masking *within* the ``max`` artifact,
      not a recompile, so it is **not** part of the cache key. Defaults
      to ``max_candidates``.

    The default ``max_candidates=1`` is exactly the pre-candidate
    behavior; a ``None`` / default config must leave the compile-cache
    key byte-identical to legacy (back-compat, enforced in
    ``compile_cache``).
    """

    max_candidates: int = 1
    active_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.active_candidates is None:
            object.__setattr__(self, "active_candidates", self.max_candidates)
        if not (1 <= self.active_candidates <= self.max_candidates):
            raise ValueError(
                "active_candidates must satisfy 1 <= active <= "
                f"max_candidates ({self.max_candidates}), got "
                f"{self.active_candidates}"
            )

    @property
    def is_trivial(self) -> bool:
        """True when this config is the legacy no-candidate-axis case."""

        return self.max_candidates == 1

    def world_size(self, parallel: NovaParallelConfig) -> int:
        """The candidate axis never changes ``world_size`` (cclog 43 §3.1)."""

        return parallel.world_size

    def to_cache_dict(self) -> dict[str, object]:
        """Only ``max_candidates`` drives artifact identity.

        ``active_candidates`` is a runtime occupancy knob within the
        traced ``max`` artifact and is deliberately excluded.
        """

        return {"max_candidates": self.max_candidates}
