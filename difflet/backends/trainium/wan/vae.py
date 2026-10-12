"""Trainium application wrapper for the Wan VAE decoder."""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from typing import List, Tuple

import torch

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.bucketing import (
    CompileShape,
    ShapeBucketedInputGenerator,
    canonicalize_shapes,
    dedupe_example_inputs,
    resolve_compile_shapes,
)
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.wan.vae.modeling_vae import (
    WanVAEDecoderConfig,
    WanVAEDecoderModel,
)
from .vae_model import NeuronWanVAEDecoderStepModel


def _tile_grid(height, width):
    # Match Omni's 256-pixel tiles and 192-pixel stride (latent scale is 8).
    return [(i, j, min(32, height - i), min(32, width - j))
            for i in range(0, height, 24) for j in range(0, width, 24)]


def _blend_tile_edge(previous, current, extent, dim):
    """Vectorize Diffusers 0.38's Wan blend_v/blend_h without changing rounding."""
    extent = min(previous.shape[dim], current.shape[dim], extent)
    if not extent:
        return current
    shape = [1] * current.ndim
    shape[dim] = extent
    # Python's original scalar coefficients use double precision, then cast
    # separately at each multiply. Compute both before casting (also for BF16).
    fraction = torch.arange(extent, dtype=torch.float64, device=current.device) / extent
    before = (1 - fraction).to(previous.dtype).reshape(shape)
    after = fraction.to(current.dtype).reshape(shape)
    edge = current.narrow(dim, 0, extent)
    edge.copy_(previous.narrow(dim, previous.shape[dim] - extent, extent) * before + edge * after)
    return current


class _StatefulVAEInstance(BaseModelInstance):
    def get(self, bucket_rank, **kwargs):
        return self.module, {state: index + 1
                             for index, state in enumerate(self.module.decode_cache)}


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

    Images retain their single-frame graph. Videos use first, second and
    alternating steady-state graphs per tile shape, with device history.
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
            model_init_kwargs=model_init_kwargs or {},
        )
        self.bucket_config = None
        self.decode_step = (model_init_kwargs or {}).get("step")
        self.tile_shape = (model_init_kwargs or {}).get("tile_shape")

    def input_generator(self):
        if self.decode_step is not None:
            return [(torch.zeros(
                int(getattr(self.config.neuron_config, "batch_size", 1)),
                int(self.config.z_dim), 1, *self.tile_shape,
                dtype=self.config.neuron_config.torch_dtype,
            ), torch.ones(self.decode_step + 1, dtype=torch.bool))]
        return dedupe_example_inputs(
            [
                self.example_inputs_for_shape(shape)
                for shape in resolve_compile_shapes(self.config)
            ]
        )

    def example_inputs_for_shape(self, shape: CompileShape) -> Tuple[torch.Tensor, ...]:
        height, width, num_frames = shape
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        latent_shape = [
            batch_size,
            int(self.config.z_dim),
            (int(num_frames) - 1) // int(self.config.scale_factor_temporal) + 1,
            int(height) // int(self.config.scale_factor_spatial),
            int(width) // int(self.config.scale_factor_spatial),
        ]
        latents = torch.zeros(latent_shape, dtype=dtype)
        return (latents,)

    def decoder_config(self):
        return WanVAEDecoderConfig(
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

    def get_model_instance(self):
        def _create_model():
            kwargs = self.model_init_kwargs
            model = self.model_cls(self.decoder_config(), **kwargs)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        instance_cls = _StatefulVAEInstance if self.decode_step is not None else BaseModelInstance
        return instance_cls(module_cls=_create_model, input_output_aliases={})

    def forward(self, latents, *history):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        if self.decode_step is not None:
            # Call the existing scripted shape router directly, as in warmup.
            return self.model.nxd_model.forward([latents, *history])
        return self._forward(latents, *history)


class NeuronWanVAEDecoderApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``WanVAEDecoderModel``."""

    _model_cls = WanVAEDecoderModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()
        max_latent_frames = max(
            (int(shape[2]) - 1) // int(self.config.scale_factor_temporal) + 1
            for shape in resolve_compile_shapes(self.config)
        )
        self.temporal_decode = max_latent_frames > 1
        self._decode_lock = threading.Lock()
        if self.temporal_decode:
            tile_shapes = sorted({
                (h, w) for height, width, _ in resolve_compile_shapes(self.config)
                for _, _, h, w in _tile_grid(height // 8, width // 8)
            }, reverse=True)
            maximum = (max(h for h, _ in tile_shapes), max(w for _, w in tile_shapes))
            for tile_shape in tile_shapes:
                for step in range(min(4, max_latent_frames)):
                    self.models.append(self.model_wrapper(
                        config=self.config, model_cls=NeuronWanVAEDecoderStepModel,
                        tag=f"WanVAETile_{tile_shape[0]}x{tile_shape[1]}_{step}",
                        compiler_args=self.get_compiler_args(), priority_model_idx=None,
                        model_init_kwargs={"step": step, "tile_shape": tile_shape,
                                           "max_tile_shape": maximum,
                                           "batch_size": int(self.neuron_config.batch_size),
                                           "logical_nc_config": self.neuron_config.logical_nc_config},
                    ))
            self.model = self.models[0]
        else:
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

    def forward(self, latents):
        if not self.temporal_decode:
            return self.models[0](latents)
        # A loaded NxD instance owns one cache per rank. Serialize requests so
        # their temporal histories cannot interleave; phase zero resets storage.
        with self._decode_lock:
            return self._forward_tiled(latents)

    def _forward_tiled(self, latents):
        shape = (
            int(latents.shape[3]) * int(self.config.scale_factor_spatial),
            int(latents.shape[4]) * int(self.config.scale_factor_spatial),
            (int(latents.shape[2]) - 1) * int(self.config.scale_factor_temporal) + 1,
        )
        if shape not in resolve_compile_shapes(self.config):
            raise ValueError(f"Wan VAE shape {shape} is not in the compiled shape set.")
        if int(latents.shape[0]) != int(self.neuron_config.batch_size):
            raise ValueError("Wan VAE batch size differs from the compiled batch size.")
        tiles_by_shape = defaultdict(list)
        for i, j, h, w in _tile_grid(latents.shape[3], latents.shape[4]):
            tiles_by_shape[h, w].append((i, j))
        decoded = {}
        ranks = int(self.neuron_config.local_ranks_size)
        router = self.traced_model.nxd_model
        for (h, w), positions in tiles_by_shape.items():
            for offset in range(0, len(positions), ranks):
                active = positions[offset:offset + ranks]
                # All ranks execute the same graph, each on a different tile.
                # Idle ranks repeat the final tile; their output is discarded.
                assigned = active + [active[-1]] * (ranks - len(active))
                chunks = [[] for _ in active]
                for frame in range(latents.shape[2]):
                    phase = frame if frame < 2 else 2 + frame % 2
                    marker = torch.ones(phase + 1, dtype=torch.bool)
                    inputs = [[latents[:, :, frame:frame + 1, i:i + h, j:j + w].contiguous(), marker]
                              for i, j in assigned]
                    outputs = router.forward_ranked(inputs)
                    for rank in range(len(active)):
                        chunks[rank].append(outputs[rank][0].cpu())
                for position, parts in zip(active, chunks):
                    decoded[position] = torch.cat(parts, dim=2)
        # Preserve Diffusers' overlap blending and order. Tile edges keep their true
        # shapes, so spatial attention never sees artificial padded pixels.
        rows = []
        for i in range(0, latents.shape[3], 24):
            row = []
            for j in range(0, latents.shape[4], 24):
                tile = decoded[i, j]
                if i:
                    tile = _blend_tile_edge(decoded[i - 24, j], tile, 64, -2)
                if j:
                    tile = _blend_tile_edge(decoded[i, j - 24], tile, 64, -1)
                decoded[i, j] = tile
                row.append(tile[:, :, :, :192, :192])
            rows.append(torch.cat(row, dim=-1))
        return torch.cat(rows, dim=-2)[:, :, :, :shape[0], :shape[1]].clamp(-1, 1)

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
