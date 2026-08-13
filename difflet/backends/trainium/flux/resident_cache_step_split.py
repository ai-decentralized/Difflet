"""Split-NEFF H1c cache step preserving the BF16 predictor boundary."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.model_wrapper import ModelWrapper
from difflet.backends.trainium.flux.resident_cache_step_model import (
    ResidentCacheAnchorStepModel,
    ResidentCacheFinalizeModel,
    ResidentCacheInitializeModel,
    ResidentCachePredictOnlyModel,
    ResidentCacheSchedulerStepModel,
)
from difflet.models.flux.modeling_flux import FluxBackboneInferenceConfig


def _seq_len(config: FluxBackboneInferenceConfig) -> int:
    return int(config.height) * int(config.width) // (
        (2 * int(config.vae_scale_factor)) ** 2
    )


class _SplitCacheStepInstance(BaseModelInstance):
    def __init__(self, module_builder: Callable[[], torch.nn.Module]) -> None:
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        del bucket_rank, kwargs
        return self.module, {
            self.module.anchor0: 2,
            self.module.anchor1: 3,
            self.module.latent: 4,
        }


class _SplitCacheStepWrapper(ModelWrapper):
    def __init__(self, *args, example_inputs, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._example_inputs = example_inputs
        self.bucket_config = None

    def input_generator(self):
        return [self._example_inputs(self.config)]

    def get_model_instance(self):
        config = self.config
        model_cls = self.model_cls

        def _create():
            return model_cls(
                seq_len=_seq_len(config),
                channels=int(config.in_channels),
                dtype=config.neuron_config.torch_dtype,
            ).eval()

        return _SplitCacheStepInstance(_create)


def _shape(config):
    return (1, _seq_len(config), int(config.in_channels))


def _initialize_inputs(config):
    return (
        torch.zeros(_shape(config), dtype=config.neuron_config.torch_dtype),
        torch.zeros((4,), dtype=torch.int32),
    )


def _anchor_inputs(config):
    # Two scalar slots make this signature distinct from scheduler_step.  Only
    # the first contains delta sigma; the second is a reserved control slot.
    return (
        torch.zeros(_shape(config), dtype=config.neuron_config.torch_dtype),
        torch.zeros((2,), dtype=torch.float32),
    )


def _predict_inputs(config):
    del config
    return (torch.tensor([0.0, 1.0], dtype=torch.float32),)


def _scheduler_inputs(config):
    return (
        torch.zeros(_shape(config), dtype=config.neuron_config.torch_dtype),
        torch.zeros((1,), dtype=torch.float32),
    )


def _finalize_inputs(config):
    del config
    return (torch.zeros((3,), dtype=torch.int32),)


class NeuronFluxResidentCacheStepSplitApplication(NeuronApplicationBase):
    _model_cls = ResidentCacheInitializeModel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        compiler_args = self.get_compiler_args()
        definitions = (
            ("cache_initialize", ResidentCacheInitializeModel, _initialize_inputs),
            ("cache_anchor_step", ResidentCacheAnchorStepModel, _anchor_inputs),
            ("cache_predict", ResidentCachePredictOnlyModel, _predict_inputs),
            ("cache_scheduler_step", ResidentCacheSchedulerStepModel, _scheduler_inputs),
            ("cache_finalize", ResidentCacheFinalizeModel, _finalize_inputs),
        )
        for priority, (tag, model_cls, input_factory) in enumerate(definitions):
            self.models.append(
                _SplitCacheStepWrapper(
                    config=self.config,
                    model_cls=model_cls,
                    tag=tag,
                    compiler_args=compiler_args,
                    priority_model_idx=0 if priority == 0 else None,
                    model_init_kwargs={},
                    example_inputs=input_factory,
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

    def ranked_forward(self, *inputs: torch.Tensor):
        if not self.is_loaded_to_neuron or self.traced_model is None:
            raise RuntimeError("Application must be loaded before ranked execution.")
        ranked_inputs = [
            [value for value in inputs]
            for _ in range(int(self.config.neuron_config.local_ranks_size))
        ]
        return self.traced_model.nxd_model.forward_ranked(ranked_inputs)

    def ranked_scheduler(self, ranked_prediction, delta_sigma: torch.Tensor):
        if not self.is_loaded_to_neuron or self.traced_model is None:
            raise RuntimeError("Application must be loaded before ranked execution.")
        ranked_inputs = [
            [rank_outputs[0], delta_sigma] for rank_outputs in ranked_prediction
        ]
        return self.traced_model.nxd_model.forward_ranked(ranked_inputs)

    def ranked_anchor(self, ranked_noise, delta_packet: torch.Tensor):
        """Consume a real backbone output without materializing it on host."""
        if not self.is_loaded_to_neuron or self.traced_model is None:
            raise RuntimeError("Application must be loaded before ranked execution.")
        ranked_inputs = [
            [rank_outputs[0], delta_packet] for rank_outputs in ranked_noise
        ]
        return self.traced_model.nxd_model.forward_ranked(ranked_inputs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"


__all__ = ["NeuronFluxResidentCacheStepSplitApplication"]
