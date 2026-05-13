"""Trainium wrappers for HunyuanVideo components."""

from nova.backends.trainium.hunyuan_video.backbone import (
    HunyuanVideoBackboneInferenceConfig,
    ModelWrapperHunyuanVideoBackbone,
    NeuronHunyuanVideoBackboneApplication,
)
from nova.backends.trainium.hunyuan_video.backbone15 import (
    HunyuanVideo15BackboneInferenceConfig,
    ModelWrapperHunyuanVideo15Backbone,
    NeuronHunyuanVideo15BackboneApplication,
)
from nova.backends.trainium.hunyuan_video.segmented15 import (
    HunyuanVideo15SegmentedTransformerApplication,
)
from nova.backends.trainium.hunyuan_video.vae15 import (
    HunyuanVideo15VAEDecoderInferenceConfig,
    ModelWrapperHunyuanVideo15VAEDecoder,
    NeuronHunyuanVideo15VAEDecoderApplication,
)

__all__ = [
    "HunyuanVideo15BackboneInferenceConfig",
    "HunyuanVideoBackboneInferenceConfig",
    "ModelWrapperHunyuanVideo15Backbone",
    "ModelWrapperHunyuanVideoBackbone",
    "NeuronHunyuanVideo15BackboneApplication",
    "NeuronHunyuanVideoBackboneApplication",
    "HunyuanVideo15SegmentedTransformerApplication",
    "HunyuanVideo15VAEDecoderInferenceConfig",
    "ModelWrapperHunyuanVideo15VAEDecoder",
    "NeuronHunyuanVideo15VAEDecoderApplication",
]
