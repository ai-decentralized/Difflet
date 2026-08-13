"""Experimental K=2 resident cache-state graph for FLUX H1a.

This module is a mechanism and graph-boundary spike, not the final resident
denoise data plane.  The graph keeps two full, post-gather BF16 anchors in
aliased HBM parameters and accepts a scalar action on every invocation:

* ``RESET`` clears both anchors;
* ``ANCHOR`` shifts the ring and stores ``candidate``; and
* ``PREDICT`` returns the FP32-accumulated two-anchor linear combination.

The static signature still contains a full ``candidate`` tensor and returns a
full prediction.  Hardware profiling must therefore decide whether this graph
actually removes tensor traffic; alias correctness alone is not a system
speed claim.
"""

from __future__ import annotations

import os

import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.model_wrapper import ModelWrapper
from difflet.backends.trainium.flux.resident_cache_state_model import (
    ANCHOR_ACTION,
    PREDICT_ACTION,
    RESET_ACTION,
    FluxResidentCacheStateModel,
)
from difflet.models.flux.modeling_flux import FluxBackboneInferenceConfig

class _FluxResidentStateInstance(BaseModelInstance):
    def __init__(self, module_builder) -> None:
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        del bucket_rank, kwargs
        # Outputs 2 and 3 are written back to the persistent runtime parameters.
        return self.module, {self.module.anchor0: 2, self.module.anchor1: 3}


def _seq_len(config: FluxBackboneInferenceConfig) -> int:
    return int(config.height) * int(config.width) // (
        (2 * int(config.vae_scale_factor)) ** 2
    )


class ModelWrapperFluxResidentCacheState(ModelWrapper):
    def __init__(
        self,
        config,
        model_cls,
        tag="",
        compiler_args=None,
        priority_model_idx=None,
        model_init_kwargs=None,
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            model_init_kwargs or {},
        )
        self.bucket_config = None

    def input_generator(self):
        dtype = self.config.neuron_config.torch_dtype
        shape = (1, _seq_len(self.config), int(self.config.in_channels))
        return [
            (
                torch.zeros(shape, dtype=dtype),
                torch.tensor([0.0, 1.0], dtype=torch.float32),
                torch.tensor([RESET_ACTION], dtype=torch.int32),
            )
        ]

    def get_model_instance(self):
        config = self.config
        dtype = config.neuron_config.torch_dtype

        def _create():
            return FluxResidentCacheStateModel(
                seq_len=_seq_len(config),
                channels=int(config.in_channels),
                dtype=dtype,
            ).eval()

        return _FluxResidentStateInstance(module_builder=_create)

    def forward(self, candidate, coefficients, action):
        if self.model is None:
            raise RuntimeError("Forward called before load.")
        candidate = candidate.to(dtype=self.config.neuron_config.torch_dtype)
        coefficients = coefficients.to(dtype=torch.float32)
        action = action.to(dtype=torch.int32)
        return self._forward(candidate, coefficients, action)


class NeuronFluxResidentCacheStateApplication(NeuronApplicationBase):
    """Standalone H1a graph with no FLUX model weights."""

    _model_cls = FluxResidentCacheStateModel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_wrapper = ModelWrapperFluxResidentCacheState
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return FluxBackboneInferenceConfig

    @classmethod
    def get_state_dict(cls, model_name_or_path, config):
        del model_name_or_path, config
        # Aliased anchors are runtime state, not checkpoint weights.
        return {}

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        del state_dict, config
        return {}

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        del state_dict

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"


__all__ = [
    "ANCHOR_ACTION",
    "PREDICT_ACTION",
    "RESET_ACTION",
    "FluxResidentCacheStateModel",
    "NeuronFluxResidentCacheStateApplication",
]
