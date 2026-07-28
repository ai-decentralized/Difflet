"""Trainium application wrapper for the HunyuanVideo VAE decoder."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn.functional as F

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.hunyuan_video.vae.modeling_vae import (
    HunyuanVideoVAEDecoderConfig,
    HunyuanVideoVAEDecoderModel,
)


class HunyuanVideoVAEDecoderInferenceConfig(InferenceConfig):
    """Inference config for decoder-only HunyuanVideo VAE tile compile."""

    def add_derived_config(self):
        super().add_derived_config()
        self.tile_sample_min_height = int(getattr(self, "tile_sample_min_height", 256))
        self.tile_sample_min_width = int(getattr(self, "tile_sample_min_width", 256))
        self.tile_sample_min_num_frames = int(getattr(self, "tile_sample_min_num_frames", 16))
        self.tile_sample_stride_height = int(getattr(self, "tile_sample_stride_height", 192))
        self.tile_sample_stride_width = int(getattr(self, "tile_sample_stride_width", 192))
        self.tile_sample_stride_num_frames = int(getattr(self, "tile_sample_stride_num_frames", 12))
        if not hasattr(self, "scaling_factor"):
            self.scaling_factor = 0.476986

    def get_required_attributes(self) -> List[str]:
        return [
            "out_channels",
            "latent_channels",
            "up_block_types",
            "block_out_channels",
            "layers_per_block",
            "act_fn",
            "norm_num_groups",
            "scaling_factor",
            "spatial_compression_ratio",
            "temporal_compression_ratio",
            "mid_block_add_attention",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.spatial_compression_ratio)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.spatial_compression_ratio)

    @property
    def latent_frames(self) -> int:
        return (int(self.num_frames) - 1) // int(self.temporal_compression_ratio) + 1

    @property
    def tile_latent_height(self) -> int:
        return int(self.tile_sample_min_height) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_width(self) -> int:
        return int(self.tile_sample_min_width) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_frames(self) -> int:
        return int(self.tile_sample_min_num_frames) // int(self.temporal_compression_ratio) + 1

    @property
    def tile_latent_stride_height(self) -> int:
        return int(self.tile_sample_stride_height) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_stride_width(self) -> int:
        return int(self.tile_sample_stride_width) // int(self.spatial_compression_ratio)

    @property
    def tile_latent_stride_num_frames(self) -> int:
        return int(self.tile_sample_stride_num_frames) // int(self.temporal_compression_ratio)

    def validate_config(self):
        super().validate_config()
        if int(self.height) % int(self.spatial_compression_ratio) != 0:
            raise ValueError("HunyuanVideo VAE height must be divisible by spatial compression ratio.")
        if int(self.width) % int(self.spatial_compression_ratio) != 0:
            raise ValueError("HunyuanVideo VAE width must be divisible by spatial compression ratio.")
        if int(self.spatial_compression_ratio) != 8:
            raise NotImplementedError("HunyuanVideo VAE spike expects spatial compression ratio 8.")
        if int(self.temporal_compression_ratio) != 4:
            raise NotImplementedError("HunyuanVideo VAE spike expects temporal compression ratio 4.")
        if int(self.tile_latent_stride_num_frames) <= 0:
            raise ValueError("HunyuanVideo VAE tile frame stride must be positive.")


class ModelWrapperHunyuanVideoVAEDecoder(ModelWrapper):
    """ModelBuilder wrapper for HunyuanVideo VAE tile decoder compile inputs."""

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
            config=config,
            model_cls=model_cls,
            tag=tag,
            compiler_args=compiler_args,
            priority_model_idx=priority_model_idx,
            model_init_kwargs=model_init_kwargs or {},
        )
        self.bucket_config = None

    def input_generator(self) -> List[Tuple[torch.Tensor]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        return [
            (
                torch.randn(
                    [
                        batch_size,
                        int(self.config.latent_channels),
                        int(self.config.tile_latent_frames),
                        int(self.config.tile_latent_height),
                        int(self.config.tile_latent_width),
                    ],
                    dtype=dtype,
                ),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            cfg = HunyuanVideoVAEDecoderConfig(
                out_channels=int(self.config.out_channels),
                latent_channels=int(self.config.latent_channels),
                up_block_types=tuple(self.config.up_block_types),
                block_out_channels=tuple(self.config.block_out_channels),
                layers_per_block=int(self.config.layers_per_block),
                act_fn=str(self.config.act_fn),
                norm_num_groups=int(self.config.norm_num_groups),
                scaling_factor=float(self.config.scaling_factor),
                spatial_compression_ratio=int(self.config.spatial_compression_ratio),
                temporal_compression_ratio=int(self.config.temporal_compression_ratio),
                mid_block_add_attention=bool(self.config.mid_block_add_attention),
                tile_sample_min_height=int(self.config.tile_sample_min_height),
                tile_sample_min_width=int(self.config.tile_sample_min_width),
                tile_sample_min_num_frames=int(self.config.tile_sample_min_num_frames),
                tile_sample_stride_height=int(self.config.tile_sample_stride_height),
                tile_sample_stride_width=int(self.config.tile_sample_stride_width),
                tile_sample_stride_num_frames=int(self.config.tile_sample_stride_num_frames),
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


class NeuronHunyuanVideoVAEDecoderApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``HunyuanVideoVAEDecoderModel``.

    The decoder compiles to a single NEFF. It previously had to be split into
    16 NEFFs at every ``GroupNorm -> causal-Conv3d`` boundary; that split was
    working around bf16 GroupNorm statistics, which ``_Fp32GroupNorm`` in
    ``models/hunyuan_video/vae/modeling_vae.py`` now handles directly.
    """

    _model_cls = HunyuanVideoVAEDecoderModel

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
        self.config_obj = SimpleNamespace(scaling_factor=float(self.config.scaling_factor))

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideoVAEDecoderInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperHunyuanVideoVAEDecoder

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        decoded = self._temporal_tiled_decode(latents)
        if not return_dict:
            return (decoded,)
        return SimpleNamespace(sample=decoded)

    def _decode_tile(self, latents: torch.Tensor) -> torch.Tensor:
        orig_t, orig_h, orig_w = latents.shape[-3:]
        target = (
            int(self.config.tile_latent_frames),
            int(self.config.tile_latent_height),
            int(self.config.tile_latent_width),
        )
        pad_t = target[0] - orig_t
        pad_h = target[1] - orig_h
        pad_w = target[2] - orig_w
        if pad_t < 0 or pad_h < 0 or pad_w < 0:
            raise ValueError(
                "HunyuanVideo VAE tile exceeds compiled tile shape: "
                f"got {(orig_t, orig_h, orig_w)}, expected <= {target}."
            )
        if pad_t or pad_h or pad_w:
            latents = F.pad(latents, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")

        decoded = self.forward(latents.to(dtype=self.dtype))
        if isinstance(decoded, (tuple, list)):
            decoded = decoded[0]
        sample_frames = (orig_t - 1) * int(self.config.temporal_compression_ratio) + 1
        sample_height = orig_h * int(self.config.spatial_compression_ratio)
        sample_width = orig_w * int(self.config.spatial_compression_ratio)
        return decoded[:, :, :sample_frames, :sample_height, :sample_width]

    def _spatial_tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        _, _, _, height, width = latents.shape
        sample_height = height * int(self.config.spatial_compression_ratio)
        sample_width = width * int(self.config.spatial_compression_ratio)
        tile_h = int(self.config.tile_latent_height)
        tile_w = int(self.config.tile_latent_width)
        stride_h = int(self.config.tile_latent_stride_height)
        stride_w = int(self.config.tile_latent_stride_width)
        blend_h = int(self.config.tile_sample_min_height) - int(self.config.tile_sample_stride_height)
        blend_w = int(self.config.tile_sample_min_width) - int(self.config.tile_sample_stride_width)

        rows = []
        for i in range(0, height, stride_h):
            row = []
            for j in range(0, width, stride_w):
                tile = latents[:, :, :, i : i + tile_h, j : j + tile_w]
                row.append(self._decode_tile(tile))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = _blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = _blend_h(row[j - 1], tile, blend_w)
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : int(self.config.tile_sample_stride_height),
                        : int(self.config.tile_sample_stride_width),
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

    def _temporal_tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = latents.shape
        num_sample_frames = (num_frames - 1) * int(self.config.temporal_compression_ratio) + 1
        tile_frames = int(self.config.tile_latent_frames)
        stride_frames = int(self.config.tile_latent_stride_num_frames)
        blend_frames = int(self.config.tile_sample_min_num_frames) - int(
            self.config.tile_sample_stride_num_frames
        )

        row = []
        for i in range(0, num_frames, stride_frames):
            tile = latents[:, :, i : i + tile_frames, :, :]
            if height > int(self.config.tile_latent_height) or width > int(self.config.tile_latent_width):
                decoded = self._spatial_tiled_decode(tile)
            else:
                decoded = self._decode_tile(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = _blend_t(row[i - 1], tile, blend_frames)
                result_row.append(tile[:, :, : int(self.config.tile_sample_stride_num_frames), :, :])
            else:
                result_row.append(tile[:, :, : int(self.config.tile_sample_stride_num_frames) + 1, :, :])
        return torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

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
        from difflet.models.hunyuan_video.checkpoint import convert_vae_decoder_state_dict

        return convert_vae_decoder_state_dict(state_dict, config=config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


def _blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-2], b.shape[-2], int(blend_extent))
    for y in range(blend_extent):
        b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
            :, :, :, y, :
        ] * (y / blend_extent)
    return b


def _blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-1], b.shape[-1], int(blend_extent))
    for x in range(blend_extent):
        b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
            :, :, :, :, x
        ] * (x / blend_extent)
    return b


def _blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-3], b.shape[-3], int(blend_extent))
    for x in range(blend_extent):
        b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[
            :, :, x, :, :
        ] * (x / blend_extent)
    return b
