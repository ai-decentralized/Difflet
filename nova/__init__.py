"""Nova — Trainium3 diffusion inference framework.

A focused, lean inference framework for diffusion models on AWS Trainium3.
Forked from neuronx-distributed-inference (NxDI) for the diffusion path,
extended with a unified Pipeline API and automatic compile/load caching.

Public API:
    NovaPipeline        — unified entry point (NovaPipeline.from_pretrained(...))
    NovaParallelConfig  — TP / CP / CFG-parallel configuration
    CandidateConfig     — candidate (N) axis for the candidate-aware latent runtime
    current_backend     — current backend name (trainium / cuda / rocm)
    register_model      — decorator for registering new model entries
"""

# Lazy re-exports — concrete classes wired up after M1 lands the pipeline layer.
__all__ = [
    "NovaPipeline",
    "NovaParallelConfig",
    "CandidateConfig",
    "current_backend",
    "register_model",
]


def __getattr__(name: str):
    if name in ("NovaPipeline",):
        from nova.pipeline.nova_pipeline import NovaPipeline

        return NovaPipeline
    if name in ("NovaParallelConfig",):
        from nova.pipeline.parallel_config import NovaParallelConfig

        return NovaParallelConfig
    if name in ("CandidateConfig",):
        from nova.pipeline.parallel_config import CandidateConfig

        return CandidateConfig
    if name in ("current_backend",):
        from nova.backends import current_backend

        return current_backend
    if name in ("register_model",):
        from nova.registry import register_model

        return register_model
    raise AttributeError(f"module 'nova' has no attribute {name!r}")
