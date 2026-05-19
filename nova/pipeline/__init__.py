__all__ = ["NovaPipeline", "NovaParallelConfig", "CandidateConfig"]


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
    raise AttributeError(f"module 'nova.pipeline' has no attribute {name!r}")
