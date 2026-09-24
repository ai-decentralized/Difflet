"""Chunked Wan VAE decoder application: two small graphs instead of one large one.

``NeuronWanVAEDecoderApplication`` compiles the whole temporal loop as one graph,
which neuronx-cc rejects past 3 latent frames at 480x832 (NCC_EBVF030, limit
5,000,000 instructions; 6,675,705 at 4 latent frames, 39,093,968 at the 21 behind
81 output frames). This application compiles the loop body for a fixed chunk and
calls it repeatedly, which ``difflet/models/wan/vae/chunked.py`` shows is
bit-identical because the body was already per-latent-frame.

Two graphs, not one: the first chunk starts from an empty cache and its
upsample3d slot briefly holds the ``"Rep"`` sentinel, while every later chunk
reads settled fixed-shape tensors. Splitting them keeps both graphs branch-free.

The cache never leaves the device. At 480x832 its 32 written slots hold 1,889 MB
in bf16, so returning them to the host each chunk would move ~13 GB per decode --
more than the host decode this replaces. Each slot is an aliased Parameter,
updated in place, the way the TeaCache probe holds ``prev_mod``.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.bucketing import (
    CompileShape,
    ShapeBucketedInputGenerator,
)
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import ModelWrapper
from difflet.backends.trainium.wan.vae import WanVAEDecoderInferenceConfig
from difflet.backends.trainium.wan.vae_chunked import (
    StatefulChunkedWanVAEDecoder,
    cache_shapes_for,
)
from difflet.models.wan.vae.chunked import DEFAULT_CHUNK_LATENT_FRAMES
from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel


def _decoder_config(config: InferenceConfig) -> WanVAEDecoderConfig:
    return WanVAEDecoderConfig(
        base_dim=int(config.base_dim),
        decoder_base_dim=int(config.decoder_base_dim),
        z_dim=int(config.z_dim),
        dim_mult=list(config.dim_mult),
        num_res_blocks=int(config.num_res_blocks),
        attn_scales=list(config.attn_scales),
        temperal_downsample=list(config.temperal_downsample),
        dropout=float(config.dropout),
        latents_mean=list(getattr(config, "latents_mean", [])),
        latents_std=list(getattr(config, "latents_std", [])),
        is_residual=bool(getattr(config, "is_residual", False)),
        out_channels=int(getattr(config, "out_channels", 3)),
        patch_size=getattr(config, "patch_size", None),
        scale_factor_temporal=int(getattr(config, "scale_factor_temporal", 4)),
        scale_factor_spatial=int(getattr(config, "scale_factor_spatial", 8)),
    )


class ChunkedVAEInstance(BaseModelInstance):
    """Alias every cache Parameter onto its output so updates stay in HBM."""

    def __init__(self, module_builder):
        super().__init__(module_builder, input_output_aliases={})

    def get(self, bucket_rank, **kwargs):
        module = self.module
        # Output 0 is the chunk's pixels; the cache slots follow in slot order.
        # The first-chunk module holds no Parameters (it reads no prior cache), so
        # this yields an empty alias map for it and the real mapping for the
        # later-chunk module, whose Parameters are read and rewritten each call.
        return module, {param: 1 + i for i, param in enumerate(module.cache)}


class ChunkedVAEWrapper(ShapeBucketedInputGenerator, ModelWrapper):
    """One compiled chunk graph -- first chunk or later chunk.

    ``ShapeBucketedInputGenerator`` comes first so its ``input_generator`` wins:
    ``ModelWrapper``'s default builds LLM-shaped inputs and calls
    ``prepare_sampling_params``, which is None here and fails with
    "'NoneType' object is not callable". The VAE takes a latent tensor, nothing
    else, so the shape-bucketed generator is the right one.
    """

    def __init__(self, *args, first: bool, cache_shapes, chunk: int, **kwargs):
        self._first = bool(first)
        self._cache_shapes = cache_shapes
        self._chunk = int(chunk)
        super().__init__(*args, **kwargs)
        self.bucket_config = None

    def example_inputs_for_shape(self, shape: CompileShape) -> Tuple[torch.Tensor, ...]:
        height, width, _num_frames = shape
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        # The graph's frame extent is the chunk, not the request's frame count:
        # that is the whole point of chunking.
        return (
            torch.randn(
                [
                    batch_size,
                    int(self.config.z_dim),
                    self._chunk,
                    int(height) // int(self.config.scale_factor_spatial),
                    int(width) // int(self.config.scale_factor_spatial),
                ],
                dtype=dtype,
            ),
        )

    def get_model_instance(self):
        def _create_model():
            decoder = WanVAEDecoderModel(_decoder_config(self.config))
            decoder = decoder.to(dtype=self.config.neuron_config.torch_dtype).eval()
            model = StatefulChunkedWanVAEDecoder(
                decoder,
                self._cache_shapes,
                first=self._first,
                dtype=self.config.neuron_config.torch_dtype,
            )
            return model.eval()

        return ChunkedVAEInstance(_create_model)

    def forward(self, z_chunk):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(z_chunk)


class NeuronWanVAEDecoderChunkedApplication(NeuronApplicationBase):
    """Decode a full clip by driving the chunk graphs in sequence."""

    _model_cls = WanVAEDecoderModel
    # The cache slots are graph state: allocated at load, never read from the
    # checkpoint, exactly like the TeaCache probe's prev_mod.
    state_tensor_names = frozenset()

    def __init__(
        self,
        *args,
        chunk: int = DEFAULT_CHUNK_LATENT_FRAMES,
        later_chunk: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chunk = int(chunk)
        if self.chunk < DEFAULT_CHUNK_LATENT_FRAMES:
            raise ValueError(
                f"chunk must be at least {DEFAULT_CHUNK_LATENT_FRAMES} latent frames so the "
                f"first chunk settles the cache; got {self.chunk}."
            )
        # The later chunk may be smaller than the first. Only the FIRST chunk has
        # to span the three cache regimes (empty, "Rep" sentinel, settled); every
        # later chunk reads settled tensors, so one latent frame would do. Smaller
        # matters because the later graph also takes 19 cache tensors as inputs,
        # which cost instructions the first graph does not pay: at 3 latent frames
        # it needed 5,426,220 against the 5,000,000 ceiling (NCC_EBVF030), 8.5%
        # over, while the first chunk compiled.
        self.later_chunk_size = int(later_chunk) if later_chunk else self.chunk
        if self.later_chunk_size < 1:
            raise ValueError(f"later_chunk must be >= 1; got {self.later_chunk_size}")
        self.dtype = self.config.neuron_config.torch_dtype

        cache_shapes = self._cache_shapes()
        common = dict(
            config=self.config,
            model_cls=self._model_cls,
            compiler_args=self.get_compiler_args(),
            cache_shapes=cache_shapes,
        )
        self.first_chunk = ChunkedVAEWrapper(
            tag="WanVAEDecoderFirstChunk",
            first=True,
            priority_model_idx=0,
            chunk=self.chunk,
            **common,
        )
        self.later_chunk = ChunkedVAEWrapper(
            tag="WanVAEDecoderLaterChunk",
            first=False,
            priority_model_idx=None,
            chunk=self.later_chunk_size,
            **common,
        )
        self.models.extend([self.first_chunk, self.later_chunk])

    def _cache_shapes(self):
        """Settle the slot shapes once, on CPU, from the real decoder structure."""
        decoder = WanVAEDecoderModel(_decoder_config(self.config)).eval()
        return cache_shapes_for(
            decoder,
            int(self.config.height) // int(self.config.scale_factor_spatial),
            int(self.config.width) // int(self.config.scale_factor_spatial),
            z_dim=int(self.config.z_dim),
            chunk=self.chunk,
            batch_size=int(getattr(self.config.neuron_config, "batch_size", 1)),
        )

    @classmethod
    def get_config_cls(cls):
        return WanVAEDecoderInferenceConfig

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return (
            "--model-type=unet-inference -O1 "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode all latent frames, chunk by chunk, clamping once at the end.

        The cache advances on device between calls, so nothing but the chunk's
        pixels crosses back to the host.
        """
        total = int(latents.shape[2])
        remainder = total - self.chunk
        if remainder < 0 or remainder % self.later_chunk_size:
            raise ValueError(
                f"latent frames ({total}) must be the first chunk ({self.chunk}) plus a "
                f"whole number of later chunks ({self.later_chunk_size}); "
                f"{total} - {self.chunk} = {remainder} is not divisible."
            )

        # The first chunk returns its cache as ordinary outputs, because it holds
        # no Parameters to alias. Seeding the later graph's Parameters with them
        # is the one host round trip in the decode; from then on the aliases keep
        # the cache on device across the remaining chunks.
        first = self.first_chunk(latents[:, :, : self.chunk, :, :])
        pieces = [first[0]]
        self._seed_later_cache(first[1:])

        for start in range(self.chunk, total, self.later_chunk_size):
            pieces.append(
                self.later_chunk(latents[:, :, start : start + self.later_chunk_size, :, :])[0]
            )
        return torch.clamp(torch.cat(pieces, dim=2), min=-1.0, max=1.0)

    def _seed_later_cache(self, cache) -> None:
        """Copy the first chunk's cache into the later graph's aliased state."""
        module = getattr(self.later_chunk, "model", None)
        params = getattr(module, "cache", None) if module is not None else None
        if params is None:
            raise RuntimeError("later-chunk graph is not loaded; call load() first")
        if len(params) != len(cache):
            raise RuntimeError(
                f"first chunk returned {len(cache)} cache slots but the later graph "
                f"declares {len(params)}"
            )
        with torch.no_grad():
            for param, value in zip(params, cache):
                param.copy_(value)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        from difflet.models.wan.checkpoint import convert_vae_decoder_state_dict

        # The chunk modules wrap the decoder, so the checkpoint keys they expect
        # carry that prefix; the cache Parameters are graph state and are absent
        # from the checkpoint by construction.
        converted = convert_vae_decoder_state_dict(state_dict, config=config)
        return {f"decoder.{k}": v for k, v in converted.items()}

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


__all__ = ["NeuronWanVAEDecoderChunkedApplication", "ChunkedVAEWrapper", "ChunkedVAEInstance"]
