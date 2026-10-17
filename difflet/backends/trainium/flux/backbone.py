"""Trainium (AoT) application, inference config and model wrapper for the Flux DiT backbone.

Moved from ``difflet.models.flux.modeling_flux`` unchanged, so the model module is backend-neutral;
``modeling_flux`` re-exports these names lazily for existing imports.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import List, Tuple

import torch

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.bucketing import ShapeBucketedInputGenerator
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.layers.embeddings import FluxPosEmbed
from difflet.models.flux.modeling_flux import _HARDWARE, NeuronFluxTransformer2DModel
from difflet.ops import hardware

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FluxBackboneInferenceConfig(InferenceConfig):
    def __init__(self, *args, cfg_parallel_enabled: bool = False, context_parallel_enabled: bool = False, cp_mode: str = "gather_kv", sp_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_parallel_enabled = cfg_parallel_enabled
        self.context_parallel_enabled = context_parallel_enabled
        self.cp_mode = cp_mode
        # Megatron-style sequence parallelism reuses the TP group.
        self.sp_enabled = sp_enabled
        # Bucket shape set (image model: (h, w) entries). Largest-first; pin the
        # single height/width convention to the priority shape.
        shapes = getattr(self, "compile_shapes", None)
        if shapes:
            from difflet.backends.trainium.core.bucketing import canonicalize_shapes

            self.compile_shapes = canonicalize_shapes(shapes)
            self.height = self.compile_shapes[0][0]
            self.width = self.compile_shapes[0][1]
        for height, width, _frames in (getattr(self, "compile_shapes", None) or ()):
            if int(height) % 16 or int(width) % 16:
                raise ValueError(
                    f"Flux compile shapes must be divisible by 16; got {height}x{width}."
                )

        # Validate mutual exclusivity
        if self.cfg_parallel_enabled and self.context_parallel_enabled:
            raise ValueError(
                "cfg_parallel_enabled and context_parallel_enabled are mutually exclusive. "
                "Only one can be True at a time."
            )
        if self.sp_enabled and self.context_parallel_enabled:
            raise ValueError(
                "sp_enabled and context_parallel_enabled are mutually exclusive "
                "(both shard the sequence dimension)."
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "attention_head_dim",
            "guidance_embeds",
            "in_channels",
            "joint_attention_dim",
            "num_attention_heads",
            "num_layers",
            "num_single_layers",
            "patch_size",
            "pooled_projection_dim",
            "height",
            "width",
        ]


class ModelWrapperFluxBackbone(ShapeBucketedInputGenerator, ModelWrapper):

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
        self.image_rotary_emb = None
        self.cache_image_rotary_emb = False
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56))

    def example_inputs_for_shape(self, shape) -> Tuple[torch.Tensor, ...]:
        height, width, _frames = shape
        joint_attention_dim = self.config.joint_attention_dim
        in_channels = self.config.in_channels
        pooled_projection_dim = self.config.pooled_projection_dim
        attention_head_dim = self.config.attention_head_dim
        # image_rotary_emb is a per-request HOST-computed input (see forward):
        # its example must track num_patches per bucket shape.
        num_patches = int(height) * int(width) // ((2 * self.config.vae_scale_factor) ** 2)

        batch_size = 2 if self.config.cfg_parallel_enabled else 1

        return (
            torch.randn(
                [batch_size, num_patches, in_channels], dtype=self.config.neuron_config.torch_dtype
            ),
            torch.randn([batch_size, 512, joint_attention_dim], dtype=self.config.neuron_config.torch_dtype),
            torch.randn([batch_size, pooled_projection_dim], dtype=self.config.neuron_config.torch_dtype),
            torch.randn([batch_size], dtype=self.config.neuron_config.torch_dtype),
            torch.randn([batch_size], dtype=self.config.neuron_config.torch_dtype)
            if self.config.guidance_embeds else torch.tensor([], dtype=self.config.neuron_config.torch_dtype),
            torch.randn([num_patches + 512, attention_head_dim, 2], dtype=self.config.neuron_config.torch_dtype),
        )

    def get_model_instance(self):
        # Create the model instance
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        model_instance = BaseModelInstance(module_cls=_create_model, input_output_aliases={})

        return model_instance

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        pooled_projections=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=False,
    ):
        """
        Override ModelWrapper.forward().
        """
        if self.model is None:
            raise RuntimeError(
                "Forward called before load. Run load() or load_state_dict() making calling forward"
            )
        # Convert timestep to correct dtype before passing to compiled model
        if timestep is not None:
            timestep = timestep.to(self.config.neuron_config.torch_dtype)

        # Handle guidance: convert to tensor and ensure correct dtype
        if guidance is None:
            guidance = torch.tensor([], dtype=self.config.neuron_config.torch_dtype)
        elif not isinstance(guidance, torch.Tensor):
            guidance = torch.tensor([guidance], dtype=self.config.neuron_config.torch_dtype)
        else:
            # If guidance is already a tensor, convert to correct dtype
            guidance = guidance.to(self.config.neuron_config.torch_dtype)

        # Calculate image_rotary_embedding before the denoising loop
        image_rotary_emb = self.image_rotary_emb
        if image_rotary_emb is None:
            if txt_ids.ndim == 3:
                logger.warning(
                    "Passing `txt_ids` 3d torch.Tensor is deprecated."
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                logger.warning(
                    "Passing `img_ids` 3d torch.Tensor is deprecated."
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                img_ids = img_ids[0]
            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = torch.stack(self.pos_embed(ids), dim=2).to(dtype=self.config.neuron_config.torch_dtype)

        if self.cache_image_rotary_emb:
            self.image_rotary_emb = image_rotary_emb

        output = self._forward(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            guidance,
            image_rotary_emb,
        )
        return output


class NeuronFluxBackboneApplication(NeuronApplicationBase):
    """
    Application class of the encoder model(s).
    """

    _model_cls = NeuronFluxTransformer2DModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_parallel_enabled = getattr(self.config, 'context_parallel_enabled', False)
        self.model_wrapper = self.get_model_wrapper_cls()

        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    def get_model_wrapper_cls(self):
        return ModelWrapperFluxBackbone

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    @contextmanager
    def cache_context(self, name: str):
        """ Cache context for matching huggingface FluxPipeline implementation. Currently a no-op."""
        yield

    @contextmanager
    def image_rotary_emb_cache_context(self):
        self.model.cache_image_rotary_emb = True
        self.model.image_rotary_emb = None
        yield
        self.model.cache_image_rotary_emb = False
        self.model.image_rotary_emb = None

    def get_compiler_args(self):
        compiler_args = "--model-type=transformer -O1"
        if self.context_parallel_enabled and _HARDWARE == hardware.TRN1:
            compiler_args = "--model-type=transformer -O2"
        else:
            compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap'"

        compiler_args += " --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'"

        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        if _HARDWARE == hardware.TRN2:
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
        return compiler_args

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        state_dict["global_rank.rank"] = torch.arange(
            0, config.neuron_config.world_size, dtype=torch.int32
        )
        inner_dim = config.num_attention_heads * config.attention_head_dim
        for i in range(config.num_single_layers):
            state_dict[f"single_transformer_blocks.{i}.proj_out_attn.weight"] = state_dict[
                f"single_transformer_blocks.{i}.proj_out.weight"
            ][:, :inner_dim].contiguous()
            state_dict[f"single_transformer_blocks.{i}.proj_out_attn.bias"] = (
                state_dict[f"single_transformer_blocks.{i}.proj_out.bias"]
                .clone()
                .detach()
                .contiguous()
            )
            state_dict[f"single_transformer_blocks.{i}.proj_out_mlp.weight"] = state_dict[
                f"single_transformer_blocks.{i}.proj_out.weight"
            ][:, inner_dim:].contiguous()
        for k, v in state_dict.items():
            state_dict[k] = v.clone().detach().contiguous()
        return state_dict
