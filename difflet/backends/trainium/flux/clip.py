"""Trainium (AoT) application, inference config and model wrapper for Flux's CLIP.

Moved from ``difflet.models.flux.clip.modeling_clip`` unchanged, so the model module is backend-neutral; that module re-exports
these names lazily for existing imports.
"""

import os
from typing import List, Tuple
import torch
from difflet.ops import (
    hardware,
)
from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.flux.clip.modeling_clip import (  # noqa: F401
    NeuronCLIPTextModel,
    _HARDWARE,
)


class CLIPInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_required_attributes(self) -> List[str]:
        # To validate if the config.json include all the configs we need in model.
        # Need to manually add what's required in below list
        return [
            "_name_or_path",
            "architectures",
            "attention_dropout",
            "bos_token_id",
            "dropout",
            "eos_token_id",
            "hidden_act",
            "hidden_size",
            "initializer_factor",
            "initializer_range",
            "intermediate_size",
            "layer_norm_eps",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_hidden_layers",
            "pad_token_id",
            "projection_dim",
            "transformers_version",
            "vocab_size",
        ]


class ModelWrapperClip(ModelWrapper):
    """
    Each encoder model should have their own ModelWrapper.
    """

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

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        """
        Override ModelWrapper.input_generator().
        Generate a list of valid sample inputs containing one input list for each bucket.
        Different model may have a different set of input args.

        Returns:
            inputs (List[Tuple[torch.Tensor]]): Example input args for every bucket.
        """
        model_inputs = torch.zeros(1, self.config.max_position_embeddings).to(dtype=torch.int64)
        inputs = [(model_inputs,)]
        return inputs

    def get_model_instance(self):
        # Create the model instance

        def _create_model():
            # Return a partial function that creates the model and calls eval() on the resulting instance
            model = self.model_cls(self.config)  # Instantiate the model with the given config
            if self.config.neuron_config.torch_dtype == torch.bfloat16:
                model.bfloat16()
            model.eval()  # Set the model to eval mode
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


# Output wrapper for CLIP
class CLIPEncoderOutput:
    def __init__(self, dictionary):
        self.pooler_output = dictionary["pooler_output"]


class NeuronClipApplication(NeuronApplicationBase):

    _model_cls = NeuronCLIPTextModel

    def __init__(self, *args, **kwargs):
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
        return ModelWrapperClip

    def forward(self, pixel_values, output_hidden_states=False):
        return CLIPEncoderOutput(self.models[0](pixel_values))

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer -O1"
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=4'"
        )
        compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if _HARDWARE == hardware.TRN2:
            os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        new_load = {
            key.replace("text_model.", "neuron_text_encoder."): state_dict[key]
            .clone()
            .detach()
            .contiguous()
            for key in list(state_dict.keys())
        }
        state_dict.update(new_load)
        return state_dict
