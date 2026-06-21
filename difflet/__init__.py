"""Difflet — Trainium3 diffusion inference framework.

A focused, lean inference framework for diffusion models on AWS Trainium3.
Forked from neuronx-distributed-inference (NxDI) for the diffusion path,
extended with a unified Pipeline API and automatic compile/load caching.

Public API:
    DiffletPipeline        — unified entry point (DiffletPipeline.from_pretrained(...))
    DiffletParallelConfig  — TP / CP / CFG-parallel configuration
    CandidateConfig     — candidate (N) axis for the candidate-aware latent runtime
    current_backend     — current backend name (trainium / cuda / rocm)
    register_model      — decorator for registering new model entries
"""

# Lazy re-exports — concrete classes wired up after M1 lands the pipeline layer.
__all__ = [
    "DiffletPipeline",
    "DiffletParallelConfig",
    "CandidateConfig",
    "current_backend",
    "register_model",
]


def __getattr__(name: str):
    if name in ("DiffletPipeline",):
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        return DiffletPipeline
    if name in ("DiffletParallelConfig",):
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        return DiffletParallelConfig
    if name in ("CandidateConfig",):
        from difflet.pipeline.parallel_config import CandidateConfig

        return CandidateConfig
    if name in ("current_backend",):
        from difflet.backends import current_backend

        return current_backend
    if name in ("register_model",):
        from difflet.registry import register_model

        return register_model
    raise AttributeError(f"module 'difflet' has no attribute {name!r}")
