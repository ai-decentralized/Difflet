"""H1g cache-step application with a BF16 XLA While/scan carry."""

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
    ResidentCacheSkipSegmentScanModel,
)
from difflet.models.flux.modeling_flux import FluxBackboneInferenceConfig


def _seq_len(config: FluxBackboneInferenceConfig) -> int:
    return int(config.height) * int(config.width) // (
        (2 * int(config.vae_scale_factor)) ** 2
    )


class _ScanCacheStepInstance(BaseModelInstance):
    def __init__(self, module_builder: Callable[[], torch.nn.Module]) -> None:
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        del bucket_rank, kwargs
        return self.module, {
            self.module.anchor0: 2,
            self.module.anchor1: 3,
            self.module.latent: 4,
        }


class _ScanCacheStepWrapper(ModelWrapper):
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

        return _ScanCacheStepInstance(_create)


def _shape(config):
    return (1, _seq_len(config), int(config.in_channels))


def _initialize_inputs(config):
    return (
        torch.zeros(_shape(config), dtype=config.neuron_config.torch_dtype),
        torch.zeros((2,), dtype=torch.int32),
    )


def _anchor_inputs(config):
    return (
        torch.zeros(_shape(config), dtype=config.neuron_config.torch_dtype),
        torch.zeros((1,), dtype=torch.float32),
    )


def _segment_inputs(config):
    del config
    steps = ResidentCacheSkipSegmentScanModel.max_segment_steps
    return (
        torch.zeros((steps, 2), dtype=torch.float32),
        torch.zeros((steps, 1), dtype=torch.float32),
    )


def _finalize_inputs(config):
    del config
    return (torch.zeros((3,), dtype=torch.int32),)


class NeuronFluxResidentCacheStepScanApplication(NeuronApplicationBase):
    """One XLA While invocation per contiguous A12 skip segment."""

    _model_cls = ResidentCacheInitializeModel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        compiler_args = self.get_compiler_args()
        definitions = (
            ("cache_initialize", ResidentCacheInitializeModel, _initialize_inputs),
            ("cache_anchor_step", ResidentCacheAnchorStepModel, _anchor_inputs),
            (
                "cache_skip_segment_scan",
                ResidentCacheSkipSegmentScanModel,
                _segment_inputs,
            ),
            ("cache_finalize", ResidentCacheFinalizeModel, _finalize_inputs),
        )
        for priority, (tag, model_cls, input_factory) in enumerate(definitions):
            self.models.append(
                _ScanCacheStepWrapper(
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

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"


__all__ = ["NeuronFluxResidentCacheStepScanApplication"]
