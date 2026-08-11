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
    PhasedStaticPolicy,
    StaticPlusBrakeConfig,
    StaticPlusBrakePolicy,
    TeaCachePolicy,
)
from difflet.pipeline.cache.predictors import (
    CalibratedLinearPredictor,
    LegacyResidualPredictor,
    TaylorSeerPredictor,
)
from difflet.pipeline.cache.recovery import (
    QualityRecoveryConfig,
    QualityRecoveryGuard,
)
from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.session import CacheSession
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
    "PHASED_CANDIDATE_SCHEMA",
    "PHASED_CANDIDATE_SCHEMA_REVISION",
    "PROFILE_QUALIFICATION_SCHEMA",
    "PROFILE_QUALIFICATION_SCHEMA_REVISION",
    "Anchor",
    "AnchorEstimateStatus",
    "AnchorErrorMeasurement",
    "CacheAnchor",
    "CacheDecision",
    "CacheHistory",
    "CacheProfileError",
    "CachePolicy",
    "CachePredictor",
    "CalibratedLinearPredictor",
    "CacheRecovery",
    "CacheSession",
    "CacheRunner",
    "CacheRunnerStats",
    "CacheStepContext",
    "Context",
    "PhasedStaticPolicy",
    "PhasedCandidateArm",
    "LegacyResidualPredictor",
    "QualityRecoveryConfig",
    "QualityRecoveryGuard",
    "QualifiedCacheProfile",
    "RecoveryDecision",
    "RuntimeObservation",
    "StepContext",
    "StaticPlusBrakeConfig",
    "StaticPlusBrakePolicy",
    "TaylorSeerPredictor",
    "TeaCachePolicy",
    "TeaCacheControllerAdapter",
    "load_phased_candidate",
    "load_phased_candidates",
    "load_qualified_cache_profile",
    "scheduler_config_sha256",
    "measure_anchor_error",
]
