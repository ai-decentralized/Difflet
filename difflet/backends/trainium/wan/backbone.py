"""Trainium application wrapper for the Wan DiT backbone."""

from __future__ import annotations

import os
from typing import List, Tuple

import torch

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.bucketing import (
    CompileShape,
    ShapeBucketedInputGenerator,
    canonicalize_shapes,
    resolve_compile_shapes,
)
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.wan.modeling_wan import WanTransformer3DModel


class WanBackboneInferenceConfig(InferenceConfig):
    """Inference config for the Wan transformer component."""

    def add_derived_config(self):
        super().add_derived_config()
        if not hasattr(self, "rope_theta"):
            self.rope_theta = 10000.0
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = 512
        if getattr(self, "out_channels", None) is None:
            self.out_channels = self.in_channels
        if not hasattr(self, "context_parallel_enabled"):
            self.context_parallel_enabled = False
        if not hasattr(self, "cp_mode"):
            self.cp_mode = "gather_kv"
        if not hasattr(self, "cfg_parallel_enabled"):
            self.cfg_parallel_enabled = False
        if not hasattr(self, "sp_enabled"):
            self.sp_enabled = False
        # Bucket shape set. NOTE: for the Wan backbone, the frames component of
        # each compile shape is the LATENT frame count (matching the semantics
        # of config.num_frames, which the application layer pre-converts).
        shapes = getattr(self, "compile_shapes", None)
        if shapes:
            self.compile_shapes = canonicalize_shapes(shapes)
            self.height, self.width, self.num_frames = self.compile_shapes[0]

    def get_required_attributes(self) -> List[str]:
        return [
            "patch_size",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "cross_attn_norm",
            "qk_norm",
            "rope_max_seq_len",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    def validate_config(self):
        super().validate_config()
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if self.qk_norm != "rms_norm_across_heads":
            raise NotImplementedError(
                "Wan T2V spike currently supports only "
                "qk_norm='rms_norm_across_heads'."
            )
        unsupported = {
            "image_dim": getattr(self, "image_dim", None),
            "added_kv_proj_dim": getattr(self, "added_kv_proj_dim", None),
            "pos_embed_seq_len": getattr(self, "pos_embed_seq_len", None),
        }
        active = {name: value for name, value in unsupported.items() if value is not None}
        if active:
            raise NotImplementedError(
                "Wan T2V spike does not support I2V/added-kv config fields: "
                + ", ".join(f"{name}={value!r}" for name, value in active.items())
            )
        for height, width, num_frames in resolve_compile_shapes(self):
            if num_frames is None:
                raise ValueError("Wan compile shapes must include a frame count.")
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(
                    f"Wan compile height/width must be divisible by 8; got {height}x{width}."
                )


class ModelWrapperWanBackbone(ShapeBucketedInputGenerator, ModelWrapper):
    """ModelBuilder wrapper for Wan DiT compile inputs.

    One bucket per entry in ``config.compile_shapes`` (largest first); shape
    frames are LATENT frames. Only the latent tensor varies per bucket.
    """

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

    def example_inputs_for_shape(
        self, shape: CompileShape
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width, latent_frames = shape
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", 512))

        return (
            torch.randn(
                [
                    batch_size,
                    self.config.in_channels,
                    int(latent_frames),
                    int(height) // 8,
                    int(width) // 8,
                ],
                dtype=dtype,
            ),
            torch.randn([batch_size], dtype=dtype),
            torch.randn([batch_size, text_seq_len, self.config.text_dim], dtype=dtype),
        )

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(hidden_states, timestep, encoder_hidden_states)


class NeuronWanBackboneApplication(NeuronApplicationBase):
    """Compile/load wrapper for WanTransformer3DModel."""

    _model_cls = WanTransformer3DModel

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
        return WanBackboneInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperWanBackbone

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
        from difflet.models.wan.checkpoint import convert_backbone_state_dict

        return convert_backbone_state_dict(state_dict, config=config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
