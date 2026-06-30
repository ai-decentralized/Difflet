"""Trainium application wrapper for the HunyuanVideo DiT backbone."""

from __future__ import annotations

import os
from typing import List

import torch

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformer3DModel


class HunyuanVideoBackboneInferenceConfig(InferenceConfig):
    """Inference config for the HunyuanVideo transformer component."""

    def add_derived_config(self):
        super().add_derived_config()
        if getattr(self, "out_channels", None) is None:
            self.out_channels = self.in_channels
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = 256
        if not hasattr(self, "image_condition_type"):
            self.image_condition_type = None
        if not hasattr(self, "context_parallel_enabled"):
            self.context_parallel_enabled = False
        if not hasattr(self, "cp_mode"):
            self.cp_mode = "gather_kv"
        if not hasattr(self, "sp_enabled"):
            self.sp_enabled = False

    def get_required_attributes(self) -> List[str]:
        return [
            "in_channels",
            "out_channels",
            "num_attention_heads",
            "attention_head_dim",
            "num_layers",
            "num_single_layers",
            "num_refiner_layers",
            "mlp_ratio",
            "patch_size",
            "patch_size_t",
            "qk_norm",
            "guidance_embeds",
            "text_embed_dim",
            "pooled_projection_dim",
            "rope_theta",
            "rope_axes_dim",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def latent_height(self) -> int:
        return int(self.height) // 8

    @property
    def latent_width(self) -> int:
        return int(self.width) // 8

    @property
    def latent_frames(self) -> int:
        return (int(self.num_frames) - 1) // 4 + 1

    def validate_config(self):
        super().validate_config()
        if isinstance(self.rope_axes_dim, list):
            self.rope_axes_dim = tuple(self.rope_axes_dim)
        if self.qk_norm != "rms_norm":
            raise NotImplementedError(
                "HunyuanVideo M3 currently supports only qk_norm='rms_norm'."
            )
        image_condition_type = getattr(self, "image_condition_type", None)
        if image_condition_type == "token_replace":
            raise NotImplementedError("HunyuanVideo M3 does not support token_replace.")
        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError("HunyuanVideo compile height/width must be divisible by 8.")
        if self.latent_height % self.patch_size != 0 or self.latent_width % self.patch_size != 0:
            raise ValueError("HunyuanVideo latent height/width must be divisible by patch_size.")
        if self.latent_frames % self.patch_size_t != 0:
            raise ValueError("HunyuanVideo latent frame count must be divisible by patch_size_t.")


class ModelWrapperHunyuanVideoBackbone(ModelWrapper):
    """ModelBuilder wrapper for HunyuanVideo DiT compile inputs."""

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag: str = "",
        compiler_args: str | None = None,
        priority_model_idx: int | None = None,
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

    def input_generator(self) -> list[tuple[torch.Tensor, ...]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", 256))

        return [
            (
                torch.randn(
                    [
                        batch_size,
                        self.config.in_channels,
                        self.config.latent_frames,
                        self.config.latent_height,
                        self.config.latent_width,
                    ],
                    dtype=dtype,
                ),
                torch.ones([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len, self.config.text_embed_dim], dtype=dtype),
                torch.ones([batch_size, text_seq_len], dtype=torch.int64),
                torch.randn([batch_size, self.config.pooled_projection_dim], dtype=dtype),
                torch.ones([batch_size], dtype=dtype),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            guidance,
        )


class NeuronHunyuanVideoBackboneApplication(NeuronApplicationBase):
    """Compile/load wrapper for HunyuanVideoTransformer3DModel."""

    _model_cls = HunyuanVideoTransformer3DModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideoBackboneInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperHunyuanVideoBackbone

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        compiler_args = (
            "--model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap' "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        # The model-root ``global_rank`` SPMDRank is created whenever any
        # sequence/batch-sharding mode is on (CP, CFG, or Megatron-SP); its
        # ``.rank`` buffer must hold ``arange(world_size)`` so each rank reads its
        # own rank. For SP-only, world_size == tp_degree, so the world-group rank
        # is the TP rank used by the entry sequence scatter. Without this the
        # buffer is all-zeros and every rank acts as rank 0 → wrong output.
        if (
            getattr(config, "context_parallel_enabled", False)
            or getattr(config, "cfg_parallel_enabled", False)
            or getattr(config, "sp_enabled", False)
        ):
            out = dict(state_dict)
            world_size = config.neuron_config.world_size
            out["global_rank.rank"] = torch.arange(0, world_size, dtype=torch.int32)
            return out
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
