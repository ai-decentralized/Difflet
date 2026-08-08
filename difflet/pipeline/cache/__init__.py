"""Composable, framework-neutral diffusion-cache runtime.

The package separates concerns that the legacy TeaCache controller kept in
one state machine:

* policies decide *when* a transformer evaluation may be skipped;
* predictors decide *which value* replaces that evaluation; and
* :class:`CacheRunner` owns history, safety checks, and accounting;
* :class:`CacheSession` owns one request's schedule and lifecycle;
* the anchor-error path exposes only the scalar needed for bounded braking; and
* :class:`TeaCacheControllerAdapter` only translates the existing loop API.

Only real transformer outputs enter :class:`CacheHistory`. Predicted outputs
are retained in :class:`RuntimeObservation` because the scheduler consumes
them, but they are never allowed to become interpolation anchors.
"""

from difflet.pipeline.cache.policies import (
    AdaptiveAnchorConfig,
    AdaptiveAnchorPolicy,
    CadencePolicy,
    ExplicitMaskPolicy,
    PhasedStaticPolicy,
    PeriodicAnchorPolicy,
    StaticPlusBrakeConfig,
    StaticPlusBrakePolicy,
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
from difflet.pipeline.cache.control_error import (
    AnchorEstimateStatus,
    AnchorErrorMeasurement,
    measure_anchor_error,
)
from difflet.pipeline.cache.profile import (
    PHASED_CANDIDATE_SCHEMA,
    PHASED_CANDIDATE_SCHEMA_REVISION,
    PROFILE_QUALIFICATION_SCHEMA,
    PROFILE_QUALIFICATION_SCHEMA_REVISION,
    CacheProfileError,
    PhasedCandidateArm,
    QualifiedCacheProfile,
    load_phased_candidate,
    load_phased_candidates,
    load_qualified_cache_profile,
    scheduler_config_sha256,
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
    "PHASED_CANDIDATE_SCHEMA",
    "PHASED_CANDIDATE_SCHEMA_REVISION",
    "PROFILE_QUALIFICATION_SCHEMA",
    "PROFILE_QUALIFICATION_SCHEMA_REVISION",
    "Anchor",
    "AnchorEstimateStatus",
    "AnchorErrorMeasurement",
    "AdaptiveAnchorConfig",
    "AdaptiveAnchorPolicy",
    "CacheAnchor",
    "CacheCompatibility",
    "CacheDecision",
    "CacheHistory",
    "CacheMask",
    "CachePlan",
    "CacheProfileError",
    "CachePolicy",
    "CachePredictor",
    "CacheRecovery",
    "CacheSession",
    "CacheRunner",
    "CacheRunnerStats",
    "CacheSpecError",
    "CacheStepContext",
    "CadencePolicy",
    "Context",
    "ExplicitMaskPolicy",
    "PhasedStaticPolicy",
    "PhasedCandidateArm",
    "LegacyResidualPredictor",
    "PeriodicAnchorPolicy",
    "QualityRecoveryConfig",
    "QualityRecoveryGuard",
    "QualifiedCacheProfile",
    "RecoveryDecision",
    "ResolvedCacheConfig",
    "RuntimeObservation",
    "ResolvedCacheSession",
    "StepContext",
    "StaticPlusBrakeConfig",
    "StaticPlusBrakePolicy",
    "TaylorSeerPredictor",
    "TeaCachePolicy",
    "TeaCacheControllerAdapter",
    "build_policy",
    "build_predictor",
    "load_cache_mask",
    "load_cache_plan",
    "load_phased_candidate",
    "load_phased_candidates",
    "load_qualified_cache_profile",
    "scheduler_config_sha256",
    "measure_anchor_error",
    "resolve_cache_config",
    "resolve_cache_plan",
    "validate_schedule_safety",
]
