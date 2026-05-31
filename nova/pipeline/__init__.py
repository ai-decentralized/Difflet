__all__ = [
    "CandidateConfig",
    "NovaParallelConfig",
    "NovaPipeline",
    "TeaCacheCalibration",
    "TeaCacheController",
]


def __getattr__(name: str):
    if name == "NovaPipeline":
        from nova.pipeline.nova_pipeline import NovaPipeline

        return NovaPipeline
    if name == "NovaParallelConfig":
        from nova.pipeline.parallel_config import NovaParallelConfig

        return NovaParallelConfig
    if name == "CandidateConfig":
        from nova.pipeline.parallel_config import CandidateConfig

        return CandidateConfig
    if name == "TeaCacheCalibration":
        from nova.pipeline.teacache import TeaCacheCalibration

        return TeaCacheCalibration
    if name == "TeaCacheController":
        from nova.pipeline.teacache import TeaCacheController

        return TeaCacheController
    raise AttributeError(f"module 'nova.pipeline' has no attribute {name!r}")
