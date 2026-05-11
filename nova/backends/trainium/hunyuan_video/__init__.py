"""Trainium wrappers for HunyuanVideo components."""

from nova.backends.trainium.hunyuan_video.backbone import (
    HunyuanVideoBackboneInferenceConfig,
    ModelWrapperHunyuanVideoBackbone,
    NeuronHunyuanVideoBackboneApplication,
)

__all__ = [
    "HunyuanVideoBackboneInferenceConfig",
    "ModelWrapperHunyuanVideoBackbone",
    "NeuronHunyuanVideoBackboneApplication",
]
