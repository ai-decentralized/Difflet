"""Trainium application wrapper for the Wan VAE decoder."""

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
from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel


class WanVAEDecoderInferenceConfig(InferenceConfig):
    """Inference config for decoder-only Wan VAE compile."""

    def add_derived_config(self):
        super().add_derived_config()
        if not hasattr(self, "decoder_base_dim"):
            self.decoder_base_dim = getattr(self, "base_dim", 96)
        if not hasattr(self, "out_channels"):
            self.out_channels = 3
        if not hasattr(self, "scale_factor_temporal"):
            self.scale_factor_temporal = 4
        if not hasattr(self, "scale_factor_spatial"):
            self.scale_factor_spatial = 8
        # Bucket shape set (frames component = PIXEL frames, matching
        # config.num_frames semantics for the Wan VAE decoder).
        shapes = getattr(self, "compile_shapes", None)
        if shapes:
            self.compile_shapes = canonicalize_shapes(shapes)
            self.height, self.width, self.num_frames = self.compile_shapes[0]

    def get_required_attributes(self) -> List[str]:
        return [
            "base_dim",
            "z_dim",
            "dim_mult",
            "num_res_blocks",
            "attn_scales",
            "temperal_downsample",
            "dropout",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.scale_factor_spatial)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.scale_factor_spatial)

    @property
    def latent_frames(self) -> int:
        return (int(self.num_frames) - 1) // int(self.scale_factor_temporal) + 1

    def validate_config(self):
        super().validate_config()
        for height, width, _num_frames in resolve_compile_shapes(self):
            if int(height) % int(self.scale_factor_spatial) != 0:
                raise ValueError(
                    f"Wan VAE height must be divisible by spatial scale factor; got {height}."
                )
            if int(width) % int(self.scale_factor_spatial) != 0:
                raise ValueError(
                    f"Wan VAE width must be divisible by spatial scale factor; got {width}."
                )
        if int(self.scale_factor_temporal) != 4:
            raise NotImplementedError("Wan VAE spike expects temporal scale factor 4.")
        if getattr(self, "is_residual", False):
            raise NotImplementedError("Wan VAE spike supports only is_residual=False.")
        if getattr(self, "patch_size", None) is not None:
            raise NotImplementedError("Wan VAE spike does not support patchified VAE.")


class ModelWrapperWanVAEDecoder(ShapeBucketedInputGenerator, ModelWrapper):
    """ModelBuilder wrapper for Wan VAE decoder compile inputs.

    Decodes the full latent in one shot, so each compile shape genuinely
    becomes its own bucket NEFF (unlike the tiled HunyuanVideo decoder).
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

    def example_inputs_for_shape(self, shape: CompileShape) -> Tuple[torch.Tensor, ...]:
        height, width, num_frames = shape
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        return (
            torch.randn(
                [
                    batch_size,
                    int(self.config.z_dim),
                    (int(num_frames) - 1) // int(self.config.scale_factor_temporal) + 1,
                    int(height) // int(self.config.scale_factor_spatial),
                    int(width) // int(self.config.scale_factor_spatial),
                ],
                dtype=dtype,
            ),
        )

    def get_model_instance(self):
        def _create_model():
            cfg = WanVAEDecoderConfig(
                base_dim=int(self.config.base_dim),
                decoder_base_dim=int(self.config.decoder_base_dim),
                z_dim=int(self.config.z_dim),
                dim_mult=list(self.config.dim_mult),
                num_res_blocks=int(self.config.num_res_blocks),
                attn_scales=list(self.config.attn_scales),
                temperal_downsample=list(self.config.temperal_downsample),
                dropout=float(self.config.dropout),
                latents_mean=list(getattr(self.config, "latents_mean", [])),
                latents_std=list(getattr(self.config, "latents_std", [])),
                is_residual=bool(getattr(self.config, "is_residual", False)),
                out_channels=int(getattr(self.config, "out_channels", 3)),
                patch_size=getattr(self.config, "patch_size", None),
                scale_factor_temporal=int(getattr(self.config, "scale_factor_temporal", 4)),
                scale_factor_spatial=int(getattr(self.config, "scale_factor_spatial", 8)),
            )
            model = self.model_cls(cfg)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, latents):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(latents)


class NeuronWanVAEDecoderApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``WanVAEDecoderModel``."""

    _model_cls = WanVAEDecoderModel

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
        return WanVAEDecoderInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperWanVAEDecoder

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        compiler_args = (
            "--model-type=unet-inference -O1 "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        from difflet.models.wan.checkpoint import convert_vae_decoder_state_dict

        return convert_vae_decoder_state_dict(state_dict, config=config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass
