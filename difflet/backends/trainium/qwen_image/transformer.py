"""Trainium wrapper for the Qwen-Image DiT transformer.

The modeling itself is backend-neutral and lives in
``difflet/models/qwen_image/modeling_qwen_image.py``; it is re-exported below
so existing importers (the application, the TeaCache probe, the tests) keep
working unchanged. What stays here is what is actually Trainium: the
``InferenceConfig`` subclass, the ``ModelWrapper``, and the
``NeuronApplicationBase`` subclass.
"""

from __future__ import annotations

import os
from typing import List

import torch

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.bucketing import ShapeBucketedInputGenerator
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.qwen_image.modeling_qwen_image import (  # noqa: F401 — re-exported
    _QwenImageTrainiumAttnProcessor,
    _QwenImageTransformerTraceModule,
    _StaticQwenImageRealRope,
    _ZeroLikeModule,
    _ZeroQwenImageAttention,
    _apply_qwen_block_diagnostics,
    _apply_qwen_rope_real,
    _column_parallel_like,
    _env_flag,
    _qwen_complex_rope_to_real,
    _qwen_rope_as_real,
    _replace_qwen_linears_for_tp,
    _row_parallel_like,
    _safe_tensor_parallel_size,
)
from difflet.ops import SPMDRank



class QwenImageTransformerInferenceConfig(InferenceConfig):
    """Inference config for the Qwen-Image transformer component."""

    def add_derived_config(self):
        super().add_derived_config()
        if getattr(self, "out_channels", None) is None:
            self.out_channels = self.in_channels // 4
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = 1024
        if not hasattr(self, "vae_scale_factor"):
            self.vae_scale_factor = 8
        if not hasattr(self, "context_parallel_enabled"):
            self.context_parallel_enabled = False
        if not hasattr(self, "cp_mode"):
            self.cp_mode = "gather_kv"
        # Bucket shape set (image model: (h, w) entries), largest first; pin
        # the single height/width convention to the priority shape.
        shapes = getattr(self, "compile_shapes", None)
        if shapes:
            from difflet.backends.trainium.core.bucketing import canonicalize_shapes

            self.compile_shapes = canonicalize_shapes(shapes)
            self.height = self.compile_shapes[0][0]
            self.width = self.compile_shapes[0][1]

    def get_required_attributes(self) -> List[str]:
        return [
            "patch_size",
            "in_channels",
            "out_channels",
            "num_layers",
            "attention_head_dim",
            "num_attention_heads",
            "joint_attention_dim",
            "guidance_embeds",
            "axes_dims_rope",
            "height",
            "width",
        ]

    @property
    def latent_height(self) -> int:
        return 2 * (int(self.height) // (int(self.vae_scale_factor) * 2))

    @property
    def latent_width(self) -> int:
        return 2 * (int(self.width) // (int(self.vae_scale_factor) * 2))

    @property
    def packed_height(self) -> int:
        return self.latent_height // int(self.patch_size)

    @property
    def packed_width(self) -> int:
        return self.latent_width // int(self.patch_size)

    @property
    def image_seq_len(self) -> int:
        return self.packed_height * self.packed_width

    def validate_config(self):
        super().validate_config()
        if isinstance(self.axes_dims_rope, list):
            self.axes_dims_rope = tuple(self.axes_dims_rope)
        if sum(int(dim) for dim in self.axes_dims_rope) != int(self.attention_head_dim):
            raise ValueError("Qwen-Image axes_dims_rope must sum to attention_head_dim.")
        if any(int(dim) % 2 != 0 for dim in self.axes_dims_rope):
            raise ValueError("Qwen-Image axes_dims_rope entries must be even.")
        if int(self.patch_size) != 2:
            raise NotImplementedError("Qwen-Image M4a currently supports patch_size=2.")
        from difflet.backends.trainium.core.bucketing import resolve_compile_shapes

        vsf2 = int(self.vae_scale_factor) * 2
        for height, width, _frames in resolve_compile_shapes(self):
            if int(height) % vsf2 != 0:
                raise ValueError(
                    f"Qwen-Image compile height must be divisible by 16; got {height}."
                )
            if int(width) % vsf2 != 0:
                raise ValueError(
                    f"Qwen-Image compile width must be divisible by 16; got {width}."
                )
            latent_height = 2 * (int(height) // vsf2)
            latent_width = 2 * (int(width) // vsf2)
            if latent_height % int(self.patch_size) != 0:
                raise ValueError("Qwen-Image latent height must be divisible by patch_size.")
            if latent_width % int(self.patch_size) != 0:
                raise ValueError("Qwen-Image latent width must be divisible by patch_size.")
        if getattr(self, "use_additional_t_cond", False):
            raise NotImplementedError("Qwen-Image M4a does not support additional_t_cond yet.")




class ModelWrapperQwenImageTransformer(ShapeBucketedInputGenerator, ModelWrapper):
    """ModelBuilder wrapper for Qwen-Image transformer compile inputs.

    One bucket per (h, w) in ``config.compile_shapes``; only the packed image
    sequence length varies per bucket.
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

    def example_inputs_for_shape(self, shape) -> tuple[torch.Tensor, ...]:
        height, width, _frames = shape
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", 1024))
        vsf2 = int(self.config.vae_scale_factor) * 2
        patch = int(self.config.patch_size)
        image_seq_len = ((2 * (int(height) // vsf2)) // patch) * (
            (2 * (int(width) // vsf2)) // patch
        )

        return (
            torch.randn(
                [batch_size, image_seq_len, self.config.in_channels],
                dtype=dtype,
            ),
            torch.ones([batch_size], dtype=dtype),
            torch.randn(
                [batch_size, text_seq_len, self.config.joint_attention_dim],
                dtype=dtype,
            ),
            torch.ones([batch_size, text_seq_len], dtype=torch.bool),
            torch.ones([batch_size], dtype=dtype),
        )

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
        encoder_hidden_states_mask,
        guidance,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            guidance,
        )


class NeuronQwenImageTransformerApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``QwenImageTransformer2DModel``."""

    _model_cls = _QwenImageTransformerTraceModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag="QwenImageTransformer2DModel",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return QwenImageTransformerInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperQwenImageTransformer

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
        out = {
            key if key.startswith("transformer.") else f"transformer.{key}": value
            for key, value in state_dict.items()
        }
        # The SPMDRank buffer sits on the trace module (not under transformer.),
        # so its key is un-prefixed. arange(world_size) lets each rank read its id.
        if getattr(config, "context_parallel_enabled", False):
            world_size = config.neuron_config.world_size
            out["global_rank.rank"] = torch.arange(0, world_size, dtype=torch.int32)
        return out

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
