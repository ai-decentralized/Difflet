"""Neuron application for HunyuanVideo 1.5's VAE decoder: diffusers' AutoencoderKLHunyuanVideo15.

Decodes as HunyuanVideo15Pipeline does, ``vae.decode(latents / scaling_factor)``, eager on one
core (the separate VAE stage), in bf16 by default (on the device bf16 matched the CPU's tiled fp32
decode to 4.3e-3; fp32 to 3.3e-7). The encoder is dropped after loading.

Tiling is always on: diffusers tiles space only, so a tile's activations grow with the clip length
(at 61 frames, 256 x 256-pixel fp32 tiles needed 2 GB tensors). The tile size is in sample pixels
(25% overlap, diffusers' default); smaller tiles fit longer clips but blend more seams (against the
untiled decode, 5 frames: 1.2e-2 at 256 pixels, 3.2e-2 at 128). The default is 128: at 61 frames 256
pixels runs out of memory even in bf16, 128 fits (44.7 s, 6.3 GiB peak); temporal chunking would
lift this. Both of diffusers' tile attributes
are set: its tiling loop reads the latent-unit sizes, which ``enable_tiling`` does not derive from
the sample-pixel ones.

With a process ``group`` of several ranks, the tiles are decoded in parallel
(``parallel_tiled_decode``): rank r decodes tiles r, r + N, ... of diffusers' tiling loop, the
decoded tiles are gathered to group rank 0 over a CPU group, and rank 0 runs diffusers' own
``tiled_decode`` with its decoder replaced by a replay of those tiles, so the cuts, the blending and
its order are diffusers' code and the frames are bit-identical to the single-rank decode.
"""

from __future__ import annotations

import time

import torch


class NeuronHunyuanVideo15VaeDecoderApplication:
    """``load(skip_warmup=False)``, then ``decode(latents) -> video`` ([B, 3, F, H, W], in [-1, 1])."""

    def __init__(self, *, model_path, height: int = 480, width: int = 848, num_frames: int = 61,
                 tile_size: int = 128, dtype=torch.bfloat16, device="neuron", group=None) -> None:
        self.model_path = str(model_path)
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        self.tile_size, self.dtype, self.device = int(tile_size), dtype, torch.device(device)
        self.group = group  # a CPU process group: decode the tiles in parallel over its ranks
        self.vae = None
        self.phase_seconds: dict[str, float] = {}

    @property
    def scaling_factor(self) -> float:
        return float(self.vae.config.scaling_factor)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        cfg = self.vae.config
        s, t = int(cfg.spatial_compression_ratio), int(cfg.temporal_compression_ratio)
        return (1, int(cfg.latent_channels), (self.num_frames - 1) // t + 1, self.height // s, self.width // s)

    def load(self, skip_warmup: bool = False) -> None:
        from diffusers import AutoencoderKLHunyuanVideo15

        t = time.perf_counter()
        vae = AutoencoderKLHunyuanVideo15.from_pretrained(self.model_path, torch_dtype=self.dtype).eval()
        vae.encoder = None  # text-to-video only decodes
        s = int(vae.config.spatial_compression_ratio)
        vae.enable_tiling()
        vae.tile_sample_min_height = vae.tile_sample_min_width = self.tile_size
        vae.tile_latent_min_height = vae.tile_latent_min_width = self.tile_size // s
        self.vae = vae.to(self.device).requires_grad_(False)
        self.phase_seconds["load"] = time.perf_counter() - t
        if not skip_warmup:  # compiles the eager kernels for this shape (once; cached)
            t = time.perf_counter()
            self.decode(torch.zeros(self.latent_shape, dtype=self.dtype))
            self.phase_seconds["warmup decode"] = time.perf_counter() - t

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor | None:
        """The video on the device; with a group, on group rank 0 only (``None`` elsewhere)."""
        z = latents.to(self.device, self.dtype) / self.scaling_factor
        if self.group is not None:
            return parallel_tiled_decode(self.vae, z, self.group)
        return self.vae.decode(z, return_dict=False)[0]


class _ReplayDecoder(torch.nn.Module):
    """Stands in for the VAE's decoder: returns precomputed tiles in call order."""

    def __init__(self, tiles) -> None:
        super().__init__()
        self._tiles = iter(tiles)

    def forward(self, tile: torch.Tensor) -> torch.Tensor:
        return next(self._tiles)


def _tile_origins(vae, z: torch.Tensor) -> list[tuple[int, int]]:
    """The latent tile origins, in the order of diffusers' ``tiled_decode`` loop."""
    _, _, _, height, width = z.shape
    step_h = int(vae.tile_latent_min_height * (1 - vae.tile_overlap_factor))
    step_w = int(vae.tile_latent_min_width * (1 - vae.tile_overlap_factor))
    return [(i, j) for i in range(0, height, step_h) for j in range(0, width, step_w)]


def parallel_tiled_decode(vae, z: torch.Tensor, group) -> torch.Tensor | None:
    """``vae.decode(z)`` with its tiles decoded across ``group``'s ranks (a CPU group); the video
    on group rank 0, ``None`` on the others.

    Rank r decodes tiles r, r + N, ... (diffusers' cuts); the tiles go to group rank 0 through the
    host, which runs diffusers' ``tiled_decode`` with the decoder replaced by their replay, so the
    blending is diffusers' own, in its order, on rank 0's device. A batch above one or an untiled
    latent is decoded whole on rank 0.
    """
    import torch.distributed as dist

    rank, world = dist.get_rank(group), dist.get_world_size(group)
    _, _, _, height, width = z.shape
    tiled = vae.use_tiling and (width > vae.tile_latent_min_width
                                or height > vae.tile_latent_min_height)
    if world == 1 or z.shape[0] != 1 or not tiled:
        return vae.decode(z, return_dict=False)[0] if rank == 0 else None
    th, tw = vae.tile_latent_min_height, vae.tile_latent_min_width
    mine = {k: vae.decoder(z[:, :, :, i:i + th, j:j + tw]).cpu()
            for k, (i, j) in enumerate(_tile_origins(vae, z)) if k % world == rank}
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(mine, gathered, dst=dist.get_global_rank(group, 0), group=group)
    if rank != 0:
        return None
    tiles = {k: t for part in gathered for k, t in part.items()}
    decoder = vae.decoder
    vae.decoder = _ReplayDecoder(tiles[k].to(z.device) for k in range(len(tiles)))
    try:
        return vae.tiled_decode(z)
    finally:
        vae.decoder = decoder


__all__ = ["NeuronHunyuanVideo15VaeDecoderApplication"]
