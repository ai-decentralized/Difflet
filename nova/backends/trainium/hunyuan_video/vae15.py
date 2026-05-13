"""Trainium decoder wrapper for the HunyuanVideo 1.5 VAE."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig
from nova.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper


class HunyuanVideo15VAEDecoderInferenceConfig(InferenceConfig):
    """Inference config for decoder-only HunyuanVideo 1.5 VAE tile compile."""

    def add_derived_config(self):
        super().add_derived_config()
        self.tile_sample_min_height = int(getattr(self, "tile_sample_min_height", 256))
        self.tile_sample_min_width = int(getattr(self, "tile_sample_min_width", 256))
        self.tile_overlap_factor = float(getattr(self, "tile_overlap_factor", 0.25))
        self.tile_latent_min_height = int(
            getattr(
                self,
                "tile_latent_min_height",
                self.tile_sample_min_height // int(self.spatial_compression_ratio),
            )
        )
        self.tile_latent_min_width = int(
            getattr(
                self,
                "tile_latent_min_width",
                self.tile_sample_min_width // int(self.spatial_compression_ratio),
            )
        )

    def get_required_attributes(self) -> List[str]:
        return [
            "out_channels",
            "latent_channels",
            "block_out_channels",
            "layers_per_block",
            "scaling_factor",
            "spatial_compression_ratio",
            "temporal_compression_ratio",
            "upsample_match_channel",
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
    def tile_latent_stride_height(self) -> int:
        return max(1, int(self.tile_latent_min_height * (1.0 - float(self.tile_overlap_factor))))

    @property
    def tile_latent_stride_width(self) -> int:
        return max(1, int(self.tile_latent_min_width * (1.0 - float(self.tile_overlap_factor))))

    def validate_config(self):
        super().validate_config()
        if int(self.height) % int(self.spatial_compression_ratio) != 0:
            raise ValueError("HunyuanVideo 1.5 VAE height must be divisible by spatial ratio.")
        if int(self.width) % int(self.spatial_compression_ratio) != 0:
            raise ValueError("HunyuanVideo 1.5 VAE width must be divisible by spatial ratio.")
        if int(self.spatial_compression_ratio) != 16:
            raise NotImplementedError("HunyuanVideo 1.5 VAE expects spatial compression ratio 16.")
        if int(self.temporal_compression_ratio) != 4:
            raise NotImplementedError("HunyuanVideo 1.5 VAE expects temporal compression ratio 4.")


def _repeat_channels(x: torch.Tensor, repeats: int) -> torch.Tensor:
    repeats = int(repeats)
    if repeats == 1:
        return x
    return torch.cat([x] * repeats, dim=1)


def apply_hunyuan15_vae_repeat_workaround() -> None:
    """Avoid the XLA ``repeat_interleave`` channel-repeat lowering blocker."""

    from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
        HunyuanVideo15Upsample,
    )

    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = self._dcae_upsample_rearrange(h_first, r1=1, r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:, :, :]
            h_next = self._dcae_upsample_rearrange(h_next, r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            x_first = x[:, :, :1, :, :]
            x_first = self._dcae_upsample_rearrange(x_first, r1=1, r2=2, r3=2)
            x_first = _repeat_channels(x_first, self.repeats // 2)

            x_next = x[:, :, 1:, :, :]
            x_next = self._dcae_upsample_rearrange(x_next, r1=r1, r2=2, r3=2)
            x_next = _repeat_channels(x_next, self.repeats)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = _repeat_channels(x, self.repeats)
            shortcut = self._dcae_upsample_rearrange(shortcut, r1=r1, r2=2, r3=2)
        return h + shortcut

    HunyuanVideo15Upsample.forward = patched_forward


class HunyuanVideo15VAEDecoderModel(nn.Module):
    """Decoder-only HunyuanVideo 1.5 VAE module with HF-compatible key names."""

    def __init__(self, config: HunyuanVideo15VAEDecoderInferenceConfig) -> None:
        super().__init__()
        apply_hunyuan15_vae_repeat_workaround()
        from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
            HunyuanVideo15Decoder3D,
        )

        self.decoder = HunyuanVideo15Decoder3D(
            in_channels=int(config.latent_channels),
            out_channels=int(config.out_channels),
            block_out_channels=tuple(reversed(tuple(config.block_out_channels))),
            layers_per_block=int(config.layers_per_block),
            spatial_compression_ratio=int(config.spatial_compression_ratio),
            temporal_compression_ratio=int(config.temporal_compression_ratio),
            upsample_match_channel=bool(config.upsample_match_channel),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)


class ModelWrapperHunyuanVideo15VAEDecoder(ModelWrapper):
    """ModelBuilder wrapper for HunyuanVideo 1.5 VAE decoder compile inputs."""

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
                        int(self.config.latent_frames),
                        int(self.config.tile_latent_min_height),
                        int(self.config.tile_latent_min_width),
                    ],
                    dtype=dtype,
                ),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, latents):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(latents)


class NeuronHunyuanVideo15VAEDecoderApplication(NeuronApplicationBase):
    """Direct-trace wrapper for the HunyuanVideo 1.5 VAE decoder tile."""

    _model_cls = HunyuanVideo15VAEDecoderModel

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

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideo15VAEDecoderInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperHunyuanVideo15VAEDecoder

    def compile(
        self,
        compiled_model_path,
        debug=False,
        pre_shard_weights_hook=None,
        dry_run=False,
        disable_fail_fast=False,
    ):
        del debug, pre_shard_weights_hook, disable_fail_fast
        compiled_path = Path(compiled_model_path)
        compiled_path.mkdir(parents=True, exist_ok=True)
        self.config.save(compiled_path)
        if dry_run:
            return

        import torch_neuronx

        model = self._model_cls(self.config).to(dtype=self.dtype).eval()
        state_dict = self.checkpoint_loader_fn()
        model.load_state_dict(state_dict, strict=False)
        example = self.models[0].input_generator()[0][0]
        with torch.no_grad():
            traced = torch_neuronx.trace(
                model,
                example,
                compiler_args=self.get_compiler_args(),
            )
        torch.jit.save(traced, compiled_path / "model.pt")

    def load(
        self,
        compiled_model_path,
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    ):
        del start_rank_id, local_ranks_size
        compiled_path = Path(compiled_model_path)
        self.traced_model = torch.jit.load(compiled_path / "model.pt")
        self.models[0].model = self.traced_model
        self.is_loaded_to_neuron = True
        if not self.config.neuron_config.skip_warmup and not skip_warmup:
            example = self.models[0].input_generator()[0]
            self.models[0](*example)

    def forward(self, *model_inputs, **kwargs):
        if kwargs:
            raise TypeError("HunyuanVideo 1.5 VAE decoder accepts positional tensors only.")
        if len(model_inputs) != 1:
            raise TypeError("HunyuanVideo 1.5 VAE decoder expects one latent tensor.")
        return self.models[0](*model_inputs)

    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        decoded = self._spatial_tiled_decode(latents)
        if not return_dict:
            return (decoded,)
        return SimpleNamespace(sample=decoded)

    def _decode_tile(self, latents: torch.Tensor) -> torch.Tensor:
        orig_t, orig_h, orig_w = latents.shape[-3:]
        target_h = int(self.config.tile_latent_min_height)
        target_w = int(self.config.tile_latent_min_width)
        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                "HunyuanVideo 1.5 VAE tile exceeds compiled tile shape: "
                f"got {(orig_t, orig_h, orig_w)}, expected <= "
                f"{(int(self.config.latent_frames), target_h, target_w)}."
            )
        if pad_h or pad_w:
            latents = F.pad(latents, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")

        decoded = self.forward(latents.to(dtype=self.dtype))
        sample_frames = (orig_t - 1) * int(self.config.temporal_compression_ratio) + 1
        sample_height = orig_h * int(self.config.spatial_compression_ratio)
        sample_width = orig_w * int(self.config.spatial_compression_ratio)
        return decoded[:, :, :sample_frames, :sample_height, :sample_width]

    def _spatial_tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        _, _, _, height, width = latents.shape
        if height <= int(self.config.tile_latent_min_height) and width <= int(
            self.config.tile_latent_min_width
        ):
            return self._decode_tile(latents)

        sample_height = height * int(self.config.spatial_compression_ratio)
        sample_width = width * int(self.config.spatial_compression_ratio)
        tile_h = int(self.config.tile_latent_min_height)
        tile_w = int(self.config.tile_latent_min_width)
        stride_h = int(self.config.tile_latent_stride_height)
        stride_w = int(self.config.tile_latent_stride_width)
        blend_h = int(self.config.tile_sample_min_height * float(self.config.tile_overlap_factor))
        blend_w = int(self.config.tile_sample_min_width * float(self.config.tile_overlap_factor))
        row_limit_h = int(self.config.tile_sample_min_height) - blend_h
        row_limit_w = int(self.config.tile_sample_min_width) - blend_w

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
                result_row.append(tile[:, :, :, :row_limit_h, :row_limit_w])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)[:, :, :, :sample_height, :sample_width]

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
        del config
        return {key: value for key, value in state_dict.items() if key.startswith("decoder.")}

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
