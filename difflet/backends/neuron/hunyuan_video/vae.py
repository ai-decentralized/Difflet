"""Neuron application for HunyuanVideo's VAE decoder: diffusers' AutoencoderKLHunyuanVideo.

Decodes as HunyuanVideoPipeline does, ``vae.decode(latents / scaling_factor)``, on one core (the
separate VAE stage), eager by default (``exec_mode``, below), in fp32 by default (the checkpoint is fp32, and Wan and Flux decode in
fp32 too); bf16 is accepted. The encoder is dropped after loading.

Tiling is always on, in space and in time: diffusers decodes ``tile_frames``-frame windows of the
clip one after another (blending the overlap) and splits each window into ``tile_size``-pixel
tiles, so a tile's activations do not grow with the clip length (HunyuanVideo 1.5's autoencoder
tiles space only, which is why its long clips ran out of memory). Both strides are 3/4 of the tile
(25% overlap, diffusers' default ratio; its defaults are 256 pixels and 16 frames). diffusers walks
the latent by the stride in latent units and crops the decoded tiles by the stride in sample
units, so both must be whole latent units: ``tile_size`` a multiple of 32 pixels and
``tile_frames`` a multiple of 16 frames at the 8x spatial and 4x temporal compression; anything
else is refused rather than stitched off by a fraction of a latent.

``exec_mode="compile"`` compiles the decoder with ``torch.compile(backend="neuron")`` region by
region (``compile.compile_region``: each resnet, the attention, each upsampler; neuronx-cc
refuses the whole decoder as one graph at the 256-pixel tile). Eager, the device runs the tiled
decode op by op: about 3,751 small per-op NEFFs at 320x512x61, one set per window shape.
Compiled, each region at each input shape is one graph, so the tiling decides how many there
are:

* ``tiling="fixed"`` (the default when compiled): ``models/hunyuan_video/vae/tiling.py`` walks
  diffusers' windows but pads each to the full tile and decodes it with
  ``HunyuanVideoVAEValidTileDecoder``, which confines GroupNorm, attention and the convolution
  borders to the window's valid extent: one set of graphs for every window, the same output as
  diffusers' tiled decode. Blending runs on the host. The warm-up (``load(skip_warmup=False)``)
  decodes one full tile, which builds or loads every graph.
* ``tiling="diffusers"`` (the default when eager): diffusers' own tiled decode; compiled, every
  region builds a graph per window shape (12 at 320x512x61), so Dynamo's recompile limit is
  raised (``DIFFUSERS_TILING_RECOMPILE_LIMIT``). The control for the fixed tiling.

The config is read from ``config.json`` at construction, so ``latent_shape`` and
``scaling_factor`` are known before the weights load.
"""

from __future__ import annotations

import json
import os
import time

import torch


class NeuronHunyuanVideoVaeDecoderApplication:
    """``load(skip_warmup=False)``, then ``decode(latents) -> video`` ([B, 3, F, H, W], in [-1, 1]).

    ``model_path`` is the VAE's directory (``<model>/vae``); ``height``, ``width`` and
    ``num_frames`` are the sample shape the warm-up decodes. ``exec_mode`` is ``"eager"`` or
    ``"compile"``; ``tiling`` is ``"diffusers"`` or ``"fixed"`` (default: fixed when compiled,
    diffusers when eager); ``compile_backend`` is the ``torch.compile`` backend (``"neuron"``;
    the CPU tests pass ``"aot_eager"``). With fixed tiling ``decode`` returns the clip on the host.
    """

    EXEC_MODES = ("eager", "compile")
    TILINGS = ("diffusers", "fixed")
    #: Dynamo's recompile limit for the compiled diffusers-tiling control. Its resnets share one
    #: ``forward``: 12 window shapes at 320x512x61 (frames 5, 4, 1 x two heights x two widths)
    #: times 6 resnet shapes is 72 entries, past the default of 8.
    DIFFUSERS_TILING_RECOMPILE_LIMIT = 128
    #: The same for the fixed tiling: its regions of one class share a ``forward``, and the
    #: convolution region takes 9 input and weight shapes at the 5x32x32 latent tile.
    FIXED_TILING_RECOMPILE_LIMIT = 32

    def __init__(self, *, model_path, height: int = 320, width: int = 512, num_frames: int = 61,
                 tile_size: int = 256, tile_frames: int = 16, dtype=torch.float32,
                 device="neuron", exec_mode: str = "eager", tiling: str | None = None,
                 compile_backend="neuron") -> None:
        self.model_path = str(model_path)
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        self.tile_size, self.tile_frames = int(tile_size), int(tile_frames)
        self.dtype, self.device = dtype, torch.device(device)
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"the HunyuanVideo VAE decoder runs in bf16 or fp32, got {self.dtype}")
        if exec_mode not in self.EXEC_MODES:
            raise ValueError(f"HunyuanVideo VAE exec_mode={exec_mode!r}: expected one of "
                             f"{', '.join(self.EXEC_MODES)}")
        if tiling is None:
            tiling = "fixed" if exec_mode == "compile" else "diffusers"
        if tiling not in self.TILINGS:
            raise ValueError(f"HunyuanVideo VAE tiling={tiling!r}: expected one of "
                             f"{', '.join(self.TILINGS)}")
        self.exec_mode, self.tiling, self.compile_backend = exec_mode, tiling, compile_backend
        with open(os.path.join(self.model_path, "config.json"), encoding="utf-8") as f:
            self.vae_config = json.load(f)
        self._check_grid()
        self.vae = None
        self.tile_decoder = None  # tiling="fixed": HunyuanVideoVAEValidTileDecoder
        self._valid_tensors: dict[tuple[int, int, int], torch.Tensor] = {}
        self.phase_seconds: dict[str, float] = {}

    @property
    def _compression(self) -> tuple[int, int]:
        cfg = self.vae_config
        return int(cfg.get("spatial_compression_ratio", 8)), int(cfg.get("temporal_compression_ratio", 4))

    def _check_grid(self) -> None:
        s, t = self._compression
        for name, n in (("height", self.height), ("width", self.width)):
            if n % s:
                raise ValueError(f"HunyuanVideo VAE {name}={n} must be a multiple of {s}")
        if self.num_frames % t != 1:
            raise ValueError(f"HunyuanVideo VAE num_frames={self.num_frames} must be {t}k+1")
        for name, n, step in (("tile_size", self.tile_size, s), ("tile_frames", self.tile_frames, t)):
            if n % step or (n * 3 // 4) % step:
                raise ValueError(
                    f"HunyuanVideo VAE {name}={n}: the tile and its stride ({n * 3 // 4}) must be "
                    f"whole latent units of {step} (a multiple of {4 * step})"
                )

    @property
    def scaling_factor(self) -> float:
        return float(self.vae_config.get("scaling_factor", 0.476986))

    @property
    def latent_shape(self) -> tuple[int, ...]:
        s, t = self._compression
        channels = int(self.vae_config.get("latent_channels", 16))
        return (1, channels, (self.num_frames - 1) // t + 1, self.height // s, self.width // s)

    def load(self, skip_warmup: bool = False) -> None:
        from diffusers import AutoencoderKLHunyuanVideo

        t = time.perf_counter()
        vae = AutoencoderKLHunyuanVideo.from_pretrained(self.model_path, torch_dtype=self.dtype).eval()
        vae.encoder = None  # text-to-video only decodes
        stride, stride_frames = self.tile_size * 3 // 4, self.tile_frames * 3 // 4
        vae.enable_tiling(tile_sample_min_height=self.tile_size, tile_sample_min_width=self.tile_size,
                          tile_sample_min_num_frames=self.tile_frames,
                          tile_sample_stride_height=stride, tile_sample_stride_width=stride,
                          tile_sample_stride_num_frames=stride_frames)
        if self.tiling == "fixed":
            self.tile_decoder = self._build_tile_decoder(vae)
        else:
            self.vae = vae.to(self.device).requires_grad_(False)
            if self.exec_mode == "compile":
                import torch._dynamo as dynamo

                from difflet.backends.neuron.compile import compile_blocks

                dynamo.config.recompile_limit = max(dynamo.config.recompile_limit,
                                                    self.DIFFUSERS_TILING_RECOMPILE_LIMIT)
                compile_blocks(self.vae.decoder, block_attrs=_diffusers_regions(self.vae.decoder),
                               backend=self.compile_backend)
        self.phase_seconds["load"] = time.perf_counter() - t
        if not skip_warmup:  # compiles the kernels or graphs for this shape (once; cached)
            t = time.perf_counter()
            if self.tiling == "fixed":
                self._warmup_tile()
            else:
                self.decode(torch.zeros(self.latent_shape, dtype=self.dtype))
            self.phase_seconds["warmup decode"] = time.perf_counter() - t

    @torch.no_grad()  # as decode: a graph built with grad enabled is rebuilt under no_grad
    def _warmup_tile(self) -> None:
        """The fixed tiling's warm-up: one full tile of zeros. Every window is padded to this
        shape and its extent is a tensor input, so the first tile builds (or loads) every
        region's graph and the rest of a decode builds none; decoding the whole clip as well
        would only spend another decode (about 16 s in bf16 at 320x512x61 on trn2)."""
        s, t = self._compression
        channels = int(self.vae_config.get("latent_channels", 16))
        tile = (self.tile_frames // t + 1, self.tile_size // s, self.tile_size // s)
        self._decode_tile(torch.zeros((1, channels, *tile), dtype=self.dtype), tile)

    def _build_tile_decoder(self, vae):
        """diffusers' decoder weights in ``HunyuanVideoVAEValidTileDecoder``, on the device,
        compiled when ``exec_mode="compile"``; diffusers' model itself is not kept."""
        from difflet.models.hunyuan_video.checkpoint.vae import convert_vae_decoder_state_dict
        from difflet.models.hunyuan_video.vae.modeling_vae import (
            HunyuanVideoVAEDecoderConfig,
            HunyuanVideoVAEDecoderModel,
            HunyuanVideoVAEValidTileDecoder,
        )

        model = HunyuanVideoVAEDecoderModel(HunyuanVideoVAEDecoderConfig.from_diffusers_dict(
            dict(self.vae_config)))
        model.load_state_dict(convert_vae_decoder_state_dict(vae.state_dict()))
        decoder = HunyuanVideoVAEValidTileDecoder(model).to(self.device, self.dtype).eval()
        decoder.requires_grad_(False)
        if self.exec_mode == "compile":
            import torch._dynamo as dynamo

            from difflet.backends.neuron.compile import compile_region

            dynamo.config.recompile_limit = max(dynamo.config.recompile_limit,
                                                self.FIXED_TILING_RECOMPILE_LIMIT)
            for region in decoder.regions():  # not the whole decoder: see its docstring
                compile_region(region, backend=self.compile_backend)
        return decoder

    def _decode_tile(self, padded: torch.Tensor, valid: tuple[int, int, int]) -> torch.Tensor:
        """One padded window on the device; the decoded tile comes back to the host to blend."""
        extent = self._valid_tensors.get(valid)
        if extent is None:
            extent = torch.tensor(valid, dtype=torch.int32).to(self.device)
            self._valid_tensors[valid] = extent
        return self.tile_decoder(padded.to(self.device), extent).cpu()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.tiling == "fixed":
            from difflet.models.hunyuan_video.vae.tiling import fixed_tile_decode

            s, t = self._compression
            z = latents.to("cpu", self.dtype) / self.scaling_factor
            return fixed_tile_decode(
                z, self._decode_tile, tile_sample_size=self.tile_size,
                tile_sample_frames=self.tile_frames, stride_sample_size=self.tile_size * 3 // 4,
                stride_sample_frames=self.tile_frames * 3 // 4, spatial_compression=s,
                temporal_compression=t)
        z = latents.to(self.device, self.dtype) / self.scaling_factor
        return self.vae.decode(z, return_dict=False)[0]


def _diffusers_regions(decoder) -> list[str]:
    """The block lists of diffusers' ``HunyuanVideoDecoder3D`` to compile one block at a time:
    the mid block's resnets and attention, each up block's resnets and upsampler. ``conv_in``,
    the causal attention mask and the output head stay eager."""
    names = ["mid_block.resnets", "mid_block.attentions"]
    for i, block in enumerate(decoder.up_blocks):
        names.append(f"up_blocks.{i}.resnets")
        if block.upsamplers is not None:
            names.append(f"up_blocks.{i}.upsamplers")
    return names


__all__ = ["NeuronHunyuanVideoVaeDecoderApplication"]
