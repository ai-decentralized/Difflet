"""Truncated FLUX teacher for real single-block residual cache diagnostics.

The AOT graph executes the embedding prefix and all double-stream blocks, then
stops at the MLP branch of the first single-stream block.  It returns an
arithmetic-free sample of that block's explicit residual input plus the true
MLP contribution.  This is an offline teacher only; a serving split-cache graph
would already have the block input and would retain only the 1.5 KiB sample.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.layers.embeddings import FluxPosEmbed
from difflet.models.flux.modeling_flux import (
    FluxBackboneInferenceConfig,
    NeuronFluxBackboneApplication,
    NeuronFluxTransformer2DModel,
)
from difflet.ops import reduce_from_tensor_model_parallel_region
from difflet.pipeline.cache.component_signal import (
    residual_input_samples,
    residual_input_sketch,
)


class _FluxSingleBlockResidualTraceModule(nn.Module):
    """Execute the real backbone prefix and expose single-block-0 MLP state."""

    def __init__(self, config: FluxBackboneInferenceConfig) -> None:
        super().__init__()
        if bool(
            getattr(config, "cfg_parallel_enabled", False)
            or getattr(config, "context_parallel_enabled", False)
            or getattr(config, "sp_enabled", False)
        ):
            raise ValueError("single-block residual probe currently supports TP-only execution")
        self.transformer = NeuronFluxTransformer2DModel(config)
        divisor = 2 * int(config.vae_scale_factor)
        self.image_height = int(config.height) // divisor
        self.image_width = int(config.width) // divisor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transformer = self.transformer
        image_states = transformer.x_embedder(hidden_states)
        text_states = transformer.context_embedder(encoder_hidden_states)
        timestep = timestep * 1000
        if transformer.config.guidance_embeds:
            guidance = guidance * 1000
            temb = transformer.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = transformer.time_text_embed(timestep, pooled_projections)

        for block in transformer.transformer_blocks:
            text_states, image_states = block(
                hidden_states=image_states,
                encoder_hidden_states=text_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        joint_states = torch.cat((text_states, image_states), dim=1)
        block = transformer.single_transformer_blocks[0]
        normalized, _ = block.norm(joint_states, emb=temb)
        raw_samples = residual_input_samples(
            joint_states,
            text_token_count=512,
            image_height=self.image_height,
            image_width=self.image_width,
            text_regions=8,
            image_region_rows=4,
            image_region_columns=4,
            channel_groups=32,
        )
        raw_moments = residual_input_sketch(
            joint_states,
            text_token_count=512,
            image_height=self.image_height,
            image_width=self.image_width,
            text_regions=8,
            image_region_rows=4,
            image_region_columns=4,
            channel_groups=32,
        )
        normalized_samples = residual_input_samples(
            normalized,
            text_token_count=512,
            image_height=self.image_height,
            image_width=self.image_width,
            text_regions=8,
            image_region_rows=4,
            image_region_columns=4,
            channel_groups=32,
        )
        mlp_hidden = block.act_mlp(block.proj_mlp(normalized))
        mlp_partial = block.proj_out_mlp(mlp_hidden)
        mlp_global = reduce_from_tensor_model_parallel_region(
            mlp_partial,
            process_group=block.proj_out_mlp.tensor_parallel_group,
        )
        return raw_samples, raw_moments, normalized_samples, mlp_global


class FluxSingleBlockResidualProbeModel(nn.Module):
    def __init__(self, config: FluxBackboneInferenceConfig) -> None:
        super().__init__()
        self.trace_module = _FluxSingleBlockResidualTraceModule(config)

    def forward(
        self, *args: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.trace_module(*args)


class ModelWrapperFluxSingleBlockResidualProbe(ModelWrapper):
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
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56))

    def input_generator(self):
        config = self.config
        image_count = (
            int(config.height) * int(config.width) // ((2 * int(config.vae_scale_factor)) ** 2)
        )
        dtype = config.neuron_config.torch_dtype
        return [
            (
                torch.randn([1, image_count, config.in_channels], dtype=dtype),
                torch.randn([1, 512, config.joint_attention_dim], dtype=dtype),
                torch.randn([1, config.pooled_projection_dim], dtype=dtype),
                torch.randn([1], dtype=dtype),
                torch.randn([1], dtype=dtype)
                if config.guidance_embeds
                else torch.tensor([], dtype=dtype),
                torch.randn([image_count + 512, config.attention_head_dim, 2], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        config = self.config

        def create_model():
            return (
                FluxSingleBlockResidualProbeModel(config)
                .to(dtype=config.neuron_config.torch_dtype)
                .eval()
            )

        return BaseModelInstance(module_cls=create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        img_ids,
        txt_ids,
        guidance,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before single-block residual probe load")
        dtype = self.config.neuron_config.torch_dtype
        guidance = torch.tensor([], dtype=dtype) if guidance is None else guidance.to(dtype)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        rotary = torch.stack(self.pos_embed(ids), dim=2).to(dtype=dtype)
        return self._forward(
            hidden_states.to(dtype),
            encoder_hidden_states.to(dtype),
            pooled_projections.to(dtype),
            timestep.to(dtype),
            guidance,
            rotary,
        )


class NeuronFluxSingleBlockResidualProbeApplication(NeuronApplicationBase):
    _model_cls = FluxSingleBlockResidualProbeModel

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ModelWrapperFluxSingleBlockResidualProbe(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype
        self.single_block_residual_probe = True

    @classmethod
    def get_config_cls(cls):
        return FluxBackboneInferenceConfig

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return "--model-type=transformer -O1 --auto-cast=none"

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        converted = NeuronFluxBackboneApplication.convert_hf_to_neuron_state_dict(
            dict(state_dict), config
        )
        return {f"trace_module.transformer.{key}": value for key, value in converted.items()}

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        del state_dict


__all__ = [
    "FluxSingleBlockResidualProbeModel",
    "NeuronFluxSingleBlockResidualProbeApplication",
]
