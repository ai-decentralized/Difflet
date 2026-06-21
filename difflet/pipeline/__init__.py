__all__ = [
    "CandidateConfig",
    "DiffletParallelConfig",
    "DiffletPipeline",
    "TeaCacheCalibration",
    "TeaCacheController",
]


def __getattr__(name: str):
    if name == "DiffletPipeline":
        from difflet.pipeline.difflet_pipeline import DiffletPipeline

        return DiffletPipeline
    if name == "DiffletParallelConfig":
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        return DiffletParallelConfig
    if name == "CandidateConfig":
        from difflet.pipeline.parallel_config import CandidateConfig

        return CandidateConfig
    if name == "TeaCacheCalibration":
        from difflet.pipeline.teacache import TeaCacheCalibration

        return TeaCacheCalibration
    if name == "TeaCacheController":
        from difflet.pipeline.teacache import TeaCacheController

        return TeaCacheController
    raise AttributeError(f"module 'difflet.pipeline' has no attribute {name!r}")
