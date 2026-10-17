"""Trainium (AoT) application, inference config and model wrapper for Flux's VAE.

Moved from ``difflet.models.flux.vae.modeling_vae`` unchanged, so the model module is backend-neutral; that module re-exports
these names lazily for existing imports.
"""

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F, init
from diffusers.models.autoencoders.vae import Decoder, DecoderTiny
from difflet.backends.trainium.core.bucketing import (
    ShapeBucketedInputGenerator,
    canonicalize_shapes,
)
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from typing import List, Tuple
from difflet.models.flux.vae.modeling_vae import (  # noqa: F401
    PatchedGroupNorm,
    get_decoder_config,
    get_vae_scale_factor,
)


class VAEDecoderInferenceConfig(InferenceConfig):
    def __init__(self, *args, model_cls=Decoder, **kwargs):
        self._decoder_cls = model_cls
        # Pre-populate decoder_config BEFORE super().__init__() so that
        # vae_scale_factor (which reads self.decoder_config) does not fail
        # when load_config accesses it during attribute validation.
        self.decoder_config = {}
        super().__init__(*args, **kwargs)
        # Bucket shape set ((h, w) entries), largest first; pin height/width
        # to the priority shape so decoder_config below describes it.
        shapes = getattr(self, "compile_shapes", None)
        if shapes:
            self.compile_shapes = canonicalize_shapes(shapes)
            self.height = self.compile_shapes[0][0]
            self.width = self.compile_shapes[0][1]
        # Now that load_config has populated all config.json keys as
        # attributes, rebuild decoder_config from the loaded values.
        self.decoder_config = get_decoder_config(
            self._decoder_cls,
            {k: getattr(self, k) for k in dir(self) if not k.startswith("_")},
            getattr(self, "height", 1024),
            getattr(self, "width", 1024),
        )

    def get_required_attributes(self) -> List[str]:
        return [
            "height",
            "width",
        ]

    @property
    def vae_scale_factor(self):
        return get_vae_scale_factor(self._decoder_cls, self.decoder_config)


class ModelWrapperVAEDecoder(ShapeBucketedInputGenerator, ModelWrapper):

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        model_init_kwargs={},
    ) -> None:
        super().__init__(
            config, model_cls, tag, compiler_args, priority_model_idx, model_init_kwargs
        )
        self.bucket_config = None  # Set to None if you don't have bucketing

    def example_inputs_for_shape(self, shape) -> Tuple[torch.Tensor, ...]:
        height, width, _frames = shape
        in_channels = self.config.decoder_config.get("in_channels",
                          getattr(self.config, "latent_channels", 16))
        model_inputs = torch.rand(
            [
                1,
                in_channels,
                int(height) // self.config.vae_scale_factor,
                int(width) // self.config.vae_scale_factor,
            ],
            dtype=self.config.neuron_config.torch_dtype,
        )
        return (model_inputs,)

    def get_model_instance(self):
        # Create the model instance
        is_tiny = self.model_cls is DecoderTiny

        def _create_model():
            # DecoderTiny has no GroupNorm — skip the PatchedGroupNorm patch.
            if not is_tiny and self.config.neuron_config.torch_dtype == torch.bfloat16:
                torch.nn.GroupNorm = PatchedGroupNorm
            model = self.model_cls(**self.config.decoder_config)
            model = model.to(self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        model_instance = BaseModelInstance(module_cls=_create_model, input_output_aliases={})

        return model_instance

    def forward(self, *args):
        """
        Override ModelWrapper.forward().
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )
        output = self._forward(*args)

        return output


class NeuronVAEDecoderApplication(NeuronApplicationBase):

    _model_cls = Decoder

    def __init__(self, *args, model_cls=None, **kwargs):
        if model_cls is not None:
            self._model_cls = model_cls
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
        )

        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def get_model_wrapper_cls(self):
        return ModelWrapperVAEDecoder

    def forward(self, model_inputs):
        return self.models[0](model_inputs)

    def get_compiler_args(self):
        # DecoderTiny is a pure conv stack (no attention, no GroupNorm).
        # --model-type=unet-inference still gives the best conv fusion.
        compiler_args = "--model-type=unet-inference -O1 --auto-cast=none"
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        new_load = {
            key.replace("decoder.", ""): state_dict[key]
            .to(config.neuron_config.torch_dtype)
            .clone()
            .detach()
            .contiguous()
            for key in list(state_dict.keys())
        }

        state_dict.update(new_load)
        return state_dict
