"""Composable, framework-neutral diffusion-cache runtime.

The package separates concerns that the legacy TeaCache controller kept in
one state machine:

* policies decide *when* a transformer evaluation may be skipped;
* predictors decide *which value* replaces that evaluation; and
* :class:`CacheRunner` owns history, safety checks, and accounting;
* :class:`CacheSession` owns one request's schedule and lifecycle; and
* measurement sinks observe real anchors without changing decisions; and
* :class:`TeaCacheControllerAdapter` only translates the existing loop API.

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
from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.session import CacheSession, ResolvedCacheSession
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
from difflet.pipeline.cache.measurements import (
    AnchorEstimateStatus,
    AnchorMeasurement,
    CacheMeasurementSink,
    InMemoryMeasurementSink,
    LatentUpdateMeasurement,
    measure_anchor_estimate,
    measure_latent_update,
)
from difflet.pipeline.cache.measurement_report import (
    CACHE_MEASUREMENT_SCHEMA,
    CACHE_MEASUREMENT_SCHEMA_REVISION,
    CacheMeasurementReport,
    load_cache_measurements,
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
from difflet.pipeline.cache.teacache_adapter import TeaCacheControllerAdapter

__all__ = [
    "CACHE_MASK_SCHEMA",
    "CACHE_PLAN_SCHEMA",
    "CACHE_MEASUREMENT_SCHEMA",
    "CACHE_MEASUREMENT_SCHEMA_REVISION",
    "Anchor",
    "AnchorEstimateStatus",
    "AnchorMeasurement",
    "CacheAnchor",
    "CacheCompatibility",
    "CacheDecision",
    "CacheHistory",
    "CacheMask",
    "CachePlan",
    "CachePolicy",
    "CachePredictor",
    "CacheRecovery",
    "CacheSession",
    "CacheRunner",
    "CacheRunnerStats",
    "CacheSpecError",
    "CacheStepContext",
    "CacheMeasurementReport",
    "CacheMeasurementSink",
    "CadencePolicy",
    "Context",
    "ExplicitMaskPolicy",
    "LegacyResidualPredictor",
    "InMemoryMeasurementSink",
    "LatentUpdateMeasurement",
    "PeriodicAnchorPolicy",
    "QualityRecoveryConfig",
    "QualityRecoveryGuard",
    "RecoveryDecision",
    "ResolvedCacheConfig",
    "RuntimeObservation",
    "ResolvedCacheSession",
    "StepContext",
    "TaylorSeerPredictor",
    "TeaCachePolicy",
    "TeaCacheControllerAdapter",
    "build_policy",
    "build_predictor",
    "load_cache_mask",
    "load_cache_plan",
    "load_cache_measurements",
    "measure_anchor_estimate",
    "measure_latent_update",
    "resolve_cache_config",
    "resolve_cache_plan",
    "validate_schedule_safety",
]
