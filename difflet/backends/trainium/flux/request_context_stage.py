"""TP-ranked immutable request-context staging application for FLUX H1d."""

from __future__ import annotations

import os

import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.model_wrapper import ModelWrapper
from difflet.backends.trainium.flux.request_context_stage_model import (
    FluxRequestContextStageModel,
)
from difflet.models.flux.modeling_flux import FluxBackboneInferenceConfig


def _image_seq_len(config: FluxBackboneInferenceConfig) -> int:
    return int(config.height) * int(config.width) // (
        (2 * int(config.vae_scale_factor)) ** 2
    )


def _batch_size(config: FluxBackboneInferenceConfig) -> int:
    return 2 if bool(config.cfg_parallel_enabled) else 1


class _RequestContextStageWrapper(ModelWrapper):
    def __init__(self, *args, slot_token_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bucket_config = None
        self.slot_token_size = int(slot_token_size)

    def input_generator(self):
        config = self.config
        dtype = config.neuron_config.torch_dtype
        batch = _batch_size(config)
        text_seq_len = 512
        return [
            (
                torch.zeros(
                    (batch, text_seq_len, int(config.joint_attention_dim)),
                    dtype=dtype,
                ),
                torch.zeros(
                    (batch, int(config.pooled_projection_dim)), dtype=dtype
                ),
                torch.zeros((batch,), dtype=dtype)
                if bool(config.guidance_embeds)
                else torch.empty((0,), dtype=dtype),
                torch.zeros(
                    (
                        _image_seq_len(config) + text_seq_len,
                        int(config.attention_head_dim),
                        2,
                    ),
                    dtype=dtype,
                ),
                torch.zeros((self.slot_token_size,), dtype=torch.int32),
            )
        ]

    def get_model_instance(self):
        return BaseModelInstance(
            module_cls=lambda: FluxRequestContextStageModel().eval(),
            input_output_aliases={},
        )


class NeuronFluxRequestContextStageApplication(NeuronApplicationBase):
    """One-shot host-to-device staging for request-lifetime FLUX context."""

    _model_cls = FluxRequestContextStageModel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for slot in range(2):
            self.models.append(
                _RequestContextStageWrapper(
                    config=self.config,
                    model_cls=self._model_cls,
                    tag=f"request_context_stage_slot_{slot}",
                    compiler_args=self.get_compiler_args(),
                    priority_model_idx=0 if slot == 0 else None,
                    model_init_kwargs={},
                    slot_token_size=slot + 1,
                )
            )
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return FluxBackboneInferenceConfig

    @classmethod
    def get_state_dict(cls, model_name_or_path, config):
        del model_name_or_path, config
        return {}

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        del state_dict, config
        return {}

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        del state_dict

    def forward(self, *model_inputs, **kwargs):
        del kwargs
        return self.traced_model(*model_inputs)

    def ranked_forward(self, slot: int, *inputs: torch.Tensor):
        if not self.is_loaded_to_neuron or self.traced_model is None:
            raise RuntimeError("Application must be loaded before ranked execution.")
        if slot not in (0, 1):
            raise ValueError(f"H1d proof artifact has two slots; got slot={slot}")
        slot_token = torch.zeros((slot + 1,), dtype=torch.int32)
        ranked_inputs = [
            [*inputs, slot_token]
            for _ in range(int(self.config.neuron_config.local_ranks_size))
        ]
        return self.traced_model.nxd_model.forward_ranked(ranked_inputs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"


__all__ = ["NeuronFluxRequestContextStageApplication"]
