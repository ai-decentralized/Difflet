"""Compatibility imports for strict phase-aware cache profiles."""

from difflet.pipeline.cache.profile import (
    PHASED_CANDIDATE_SCHEMA,
    PHASED_CANDIDATE_SCHEMA_REVISION,
    PhasedCandidateArm,
    load_phased_candidate,
    load_phased_candidates,
)

__all__ = [
    "PHASED_CANDIDATE_SCHEMA",
    "PHASED_CANDIDATE_SCHEMA_REVISION",
    "PhasedCandidateArm",
    "load_phased_candidate",
    "load_phased_candidates",
]
