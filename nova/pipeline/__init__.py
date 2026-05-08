__all__ = ["NovaPipeline", "NovaParallelConfig"]


def __getattr__(name: str):
    if name == "NovaPipeline":
        from nova.pipeline.nova_pipeline import NovaPipeline

        return NovaPipeline
    if name == "NovaParallelConfig":
        from nova.pipeline.parallel_config import NovaParallelConfig

        return NovaParallelConfig
    raise AttributeError(f"module 'nova.pipeline' has no attribute {name!r}")
