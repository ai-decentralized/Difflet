"""Composable diffusion-cache runtime.

The package separates three concerns that the legacy TeaCache controller kept
in one state machine:

* policies decide *when* a transformer evaluation may be skipped;
* predictors decide *which value* replaces that evaluation; and
* :class:`CacheRunner` owns history, safety checks, and accounting.

Only real transformer outputs enter :class:`CacheHistory`. Predicted outputs
are retained in :class:`RuntimeObservation` because the scheduler consumes
them, but they are never allowed to become interpolation anchors.
"""

from difflet.pipeline.cache.policies import (
    CadencePolicy,
    ExplicitMaskPolicy,
    PeriodicAnchorPolicy,
    TeaCachePolicy,
)
from difflet.pipeline.cache.predictors import (
    LegacyResidualPredictor,
    TaylorSeerPredictor,
)
from difflet.pipeline.cache.recovery import (
    QualityRecoveryConfig,
    QualityRecoveryGuard,
)
from difflet.pipeline.cache.controller import (
    CachePlanController,
    CacheRuntimeController,
)
from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.spec import (
    CACHE_MASK_SCHEMA,
    CACHE_PLAN_SCHEMA,
    CacheCompatibility,
    CacheMask,
    CachePlan,
    CacheSpecError,
    ResolvedCacheConfig,
    build_policy,
    build_predictor,
    load_cache_mask,
    load_cache_plan,
    resolve_cache_config,
    resolve_cache_plan,
    validate_schedule_safety,
)
from difflet.pipeline.cache.types import (
    Anchor,
    CacheAnchor,
    CacheDecision,
    CacheHistory,
    CachePolicy,
    CachePredictor,
    CacheRecovery,
    CacheRunnerStats,
    CacheStepContext,
    Context,
    RuntimeObservation,
    RecoveryDecision,
    StepContext,
)

__all__ = [
    "CACHE_MASK_SCHEMA",
    "CACHE_PLAN_SCHEMA",
    "Anchor",
    "CacheAnchor",
    "CacheCompatibility",
    "CacheDecision",
    "CacheHistory",
    "CacheMask",
    "CachePlan",
    "CachePlanController",
    "CachePolicy",
    "CachePredictor",
    "CacheRecovery",
    "CacheRuntimeController",
    "CacheRunner",
    "CacheRunnerStats",
    "CacheSpecError",
    "CacheStepContext",
    "CadencePolicy",
    "Context",
    "ExplicitMaskPolicy",
    "LegacyResidualPredictor",
    "PeriodicAnchorPolicy",
    "QualityRecoveryConfig",
    "QualityRecoveryGuard",
    "RecoveryDecision",
    "ResolvedCacheConfig",
    "RuntimeObservation",
    "StepContext",
    "TaylorSeerPredictor",
    "TeaCachePolicy",
    "build_policy",
    "build_predictor",
    "load_cache_mask",
    "load_cache_plan",
    "resolve_cache_config",
    "resolve_cache_plan",
    "validate_schedule_safety",
]
