"""Trainium (AoT) application, inference config and model wrapper for Flux's T5.

Moved from ``difflet.models.flux.t5.modeling_t5`` unchanged, so the model module is backend-neutral; that module re-exports
these names lazily for existing imports.
"""

import copy
import math
import os
import logging
from typing import List, Optional, Tuple, Union
import torch
from difflet.ops import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    get_platform_target,
    hardware,
)
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)
from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.flux.t5.modeling_t5 import (  # noqa: F401
    NeuronT5EncoderModel,
    _HARDWARE,
)


class T5InferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_required_attributes(self) -> List[str]:
        return [
            "vocab_size",
            "d_model",
            "d_kv",
            "d_ff",
            "num_layers",
            "num_decoder_layers",
            "num_heads",
            "relative_attention_num_buckets",
            "relative_attention_max_distance",
            "dropout_rate",
            "layer_norm_epsilon",
            "initializer_factor",
            "feed_forward_proj",
            "is_encoder_decoder",
            "use_cache",
            "pad_token_id",
            "eos_token_id",
            "classifier_dropout",
        ]


class ModelWrapperT5(ModelWrapper):

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
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        model_inputs = torch.zeros((1, 512), dtype=torch.int64)
        inputs = [(model_inputs,)]
        return inputs

    def get_model_instance(self):
        # Create the model instance

        def _create_model():
            model = self.model_cls(self.config)
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


class NeuronT5Application(NeuronApplicationBase):
    """
    Application class of the encoder model(s).
    """

    _model_cls = NeuronT5EncoderModel

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
        return ModelWrapperT5

    def forward(self, model_inputs, output_hidden_states=False):
        return [self.models[0](model_inputs)["last_hidden_state"].to(self.dtype)]

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer -O1"
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=4'"
        )

        compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if _HARDWARE == hardware.TRN2:
            os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = str(self.config.neuron_config.logical_nc_config)
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        for k, v in state_dict.items():
            state_dict[k] = v.clone().detach().contiguous()
        return state_dict
