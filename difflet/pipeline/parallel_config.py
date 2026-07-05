"""Parallelism configuration for Difflet pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from difflet.pipeline.parallel_mesh import MeshSpec


@dataclass(frozen=True)
class DiffletParallelConfig:
    """Tensor, context, CFG, and data parallel settings.

    Context parallelism is configured via ``cp_degree`` (1 = disabled) and
    ``cp_mode`` selects the CP attention strategy (``"gather_kv"`` default, or
    ``"ring"``). Context parallelism and CFG parallelism both consume extra
    data-parallel lanes, so they are mutually exclusive.

    ``sp_enabled`` turns on **Megatron-style sequence parallelism**: the
    otherwise-replicated norm / modulation / residual regions are sharded along
    the sequence axis across the *existing tensor-parallel group*, with the
    row-parallel all-reduce replaced by reduce-scatter and an all-gather added
    before each column-parallel projection. SP reuses the TP group and adds no
    new parallel axis, so ``world_size`` is unchanged. Because SP shards the
    sequence over the TP group while CP shards it over the data-parallel group,
    the two are mutually exclusive in this increment (``sp_enabled`` requires
    ``cp_degree == 1``).
    """

    tp_degree: int = 1
    cp_degree: int = 1
    cfg_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"
    sp_enabled: bool = False
    dp_degree: int = 1

    def __post_init__(self) -> None:
        if self.tp_degree < 1:
            raise ValueError("tp_degree must be >= 1")
        if self.cp_degree < 1:
            raise ValueError("cp_degree must be >= 1")
        if self.dp_degree < 1:
            raise ValueError("dp_degree must be >= 1")
        if self.cp_degree > 1 and self.cfg_parallel_enabled:
            raise ValueError("cp_degree > 1 and cfg_parallel_enabled are mutually exclusive")
        if self.cp_mode not in ("gather_kv", "ring"):
            raise ValueError(f"cp_mode must be one of {{'gather_kv', 'ring'}}, got {self.cp_mode!r}")
        if self.cp_mode == "ring" and self.cp_degree <= 1:
            raise ValueError("cp_mode='ring' requires cp_degree > 1")
        if self.sp_enabled and self.cp_degree > 1:
            raise ValueError("sp_enabled and cp_degree > 1 are mutually exclusive")

    @property
    def mesh_spec(self) -> MeshSpec:
        """The orthogonal {dp, cfg, cp, tp} device-mesh spec for this config."""
        return MeshSpec(
            dp=self.dp_degree,
            cfg=2 if self.cfg_parallel_enabled else 1,
            cp=self.cp_degree,
            tp=self.tp_degree,
        )

    @property
    def world_size(self) -> int:
        # Megatron-style SP reuses the tensor-parallel group; it does not add a
        # new world-size axis, so it never appears in this product.
        return self.mesh_spec.world_size

    def to_cache_dict(self) -> dict[str, object]:
        # Additive-only: omit cp_mode at its default, sp_enabled when off, and
        # dp_degree at 1 so a default config keeps a compile-cache key
        # byte-identical to every pre-cp_mode / pre-sp / pre-dp model cache.
        d = asdict(self)
        if self.cp_mode == "gather_kv":
            d.pop("cp_mode")
        if not self.sp_enabled:
            d.pop("sp_enabled")
        if self.dp_degree == 1:
            d.pop("dp_degree")
        return d


@dataclass(frozen=True)
class CandidateConfig:
    """Candidate (``N``) axis for the candidate-aware latent runtime.

    ``N`` is a *batch-like* axis, **not** a parallel axis: it never
    changes ``DiffletParallelConfig.world_size`` and introduces no new
    collectives (cclog 43 §3.1). It is carried alongside
    ``DiffletParallelConfig`` rather than merged into it so the
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

    def world_size(self, parallel: DiffletParallelConfig) -> int:
        """The candidate axis never changes ``world_size`` (cclog 43 §3.1)."""

        return parallel.world_size

    def to_cache_dict(self) -> dict[str, object]:
        """Only ``max_candidates`` drives artifact identity.

        ``active_candidates`` is a runtime occupancy knob within the
        traced ``max`` artifact and is deliberately excluded.
        """

        return {"max_candidates": self.max_candidates}
