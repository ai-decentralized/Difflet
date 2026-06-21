"""HunyuanVideo model entry package."""

__all__ = [
    "HunyuanVideoAttention",
    "HunyuanVideoSingleTransformerBlock",
    "HunyuanVideoTransformer3DModel",
    "HunyuanVideoTransformerBlock",
    "HunyuanVideoTransformerConfig",
    "dual_stream_attention",
]


def __getattr__(name: str):
    if name in __all__:
        from importlib import import_module

        module = import_module("difflet.models.hunyuan_video.modeling_hunyuan_video")
        return getattr(module, name)
    raise AttributeError(f"module 'difflet.models.hunyuan_video' has no attribute {name!r}")
