"""Neuron application for MiniMax-H3's video VAE decoder (a 36-block ViT, about 2.4B params).

The module is ``_HostVideoDecoder`` (``difflet/models/minimax_h3/host_components.py``):
``post_quant_conv`` then ``MiniMaxH3VideoViTDecoder3d``, with the checkpoint's own keys; the
encoder half is skipped. It decodes one fixed tile per call, the official 256 px tile of 7
latent frames (``(1, 24, 7, 16, 16)``, 1792 patch tokens + 4 registers + 1 cls), which
``decode_minimax_h3_video`` (``pipeline.py``) stitches into the clip; every request shape uses
that one tile shape, so the warm-up compiles everything a request needs.

Neuron-specific, as on Trainium: the token sequence is padded to a multiple of 128 with a key
mask (``sequence_alignment=128``), and ``_NeuronH3VaeAttnProcessor`` attends through
``difflet.ops.attention`` with the mask lowered to per-query key bounds (the bounded flash
kernel), after the q/k RMS norms in fp32 as the reference processor computes them. The 36
identical blocks compile into one shared graph; the patch projection, registers, RoPE, output
norm and unpatchify stay eager.

Precision: ``dtype`` is the weights' and the blocks' activation dtype (bf16 by default, fp32
accepted); the block norms run in fp32 as in the reference, and the caller gets fp32 frames.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase
from difflet.backends.neuron.core.checkpoint import CheckpointSlice

#: The official decode tile: 7 latent frames x 256 px (16 latents at /16).
TILE_LATENT_SHAPE = (1, 24, 7, 16, 16)
SEQUENCE_ALIGNMENT = 128


class _NeuronH3VaeAttnProcessor:
    """The reference ``MiniMaxH3VideoAttnProcessor`` with the attention through
    ``difflet.ops.attention`` and the padding mask as key bounds."""

    def __init__(self) -> None:
        from difflet.models.minimax_h3.modeling_vae import apply_h3_video_rotary
        from difflet.ops import attention

        self._attention = attention
        self._rotary = apply_h3_video_rotary

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        dtype = hidden_states.dtype
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query.float()).to(dtype)
        key = attn.norm_k(key.float()).to(dtype)
        query = self._rotary(query, rotary_emb)
        key = self._rotary(key, rotary_emb)
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        batch, heads, length, head_dim = q.shape
        kwargs = {}
        if attention_mask is not None:
            count = attention_mask.sum(dim=-1, keepdim=True).to(torch.int32)
            bound_max = (count[:, None, None, :].expand(batch, heads, length, 1)
                         .reshape(batch * heads, length, 1).contiguous())
            kwargs = {"bound_min": torch.zeros_like(bound_max), "bound_max": bound_max}
        out = self._attention(
            q.reshape(batch * heads, length, head_dim),
            k.reshape(batch * heads, length, head_dim),
            v.reshape(batch * heads, length, head_dim),
            scale=1.0 / math.sqrt(head_dim), causal=False, tp_q=True, tp_k=True, tp_out=False,
            **kwargs,
        )
        out = out.reshape(batch, heads, length, head_dim).transpose(1, 2).flatten(2, 3).to(dtype)
        return attn.to_out[0](out.contiguous())


class _NeuronH3VaeTPSwiGLU(torch.nn.Module):
    """The decoder's SwiGLU (``ff.net.0.proj`` fused ``[hidden; gate]`` rows, both with bias)
    split into column-parallel ``up_proj``/``gate_proj`` and a row-parallel ``down_proj``."""

    def __init__(self, dim: int, inner: int) -> None:
        from difflet.ops import ColumnParallelLinear, RowParallelLinear

        super().__init__()
        self.up_proj = ColumnParallelLinear(dim, inner, bias=True, gather_output=False)
        self.gate_proj = ColumnParallelLinear(dim, inner, bias=True, gather_output=False)
        self.down_proj = RowParallelLinear(inner, dim, bias=True, input_is_parallel=True,
                                           reduce_dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.up_proj(hidden_states) * torch.nn.functional.silu(self.gate_proj(hidden_states))
        )


def _shard_decoder_block(block, tp: int) -> None:
    """Head-shard one ViT block over the TP group: q/k/v column-parallel (the q/k RMS norms
    are per head and weightless), the output projection row-parallel, the SwiGLU split."""
    from difflet.ops import ColumnParallelLinear, RowParallelLinear

    attn = block.attn
    if int(attn.heads) % tp:
        raise ValueError(f"the H3 video VAE has {attn.heads} heads, not divisible by tp={tp}")
    attn.heads = int(attn.heads) // tp
    for name in ("to_q", "to_k", "to_v"):
        linear = getattr(attn, name)
        setattr(attn, name, ColumnParallelLinear(linear.in_features, linear.out_features,
                                                 bias=linear.bias is not None,
                                                 gather_output=False))
    out = attn.to_out[0]
    attn.to_out[0] = RowParallelLinear(out.in_features, out.out_features,
                                       bias=out.bias is not None, input_is_parallel=True,
                                       reduce_dtype=torch.float32)
    up = block.ff.net[0].proj
    block.ff = _NeuronH3VaeTPSwiGLU(up.in_features, up.out_features // 2)


class _NeuronH3VideoDecoder(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], *, tp_shard: bool = False) -> None:
        super().__init__()
        from difflet.models.minimax_h3.host_components import _HostVideoDecoder

        inner = _HostVideoDecoder(cfg, sequence_alignment=SEQUENCE_ALIGNMENT)
        self.post_quant_conv = inner.post_quant_conv
        self.decoder = inner.decoder
        from difflet.backends.neuron.minimax_h3.transformer import _give_fp32_host_storage

        processor = _NeuronH3VaeAttnProcessor()
        tp = _tp_size() if tp_shard else 1
        for block in self.decoder.transformer_blocks:
            if tp > 1:
                _shard_decoder_block(block, tp)
            block.attn.set_processor(processor)
            # The blocks call these norms on fp32 activations (the reference's precision), so
            # their weights stay fp32 whatever the application dtype.
            _give_fp32_host_storage(block.norm1)
            _give_fp32_host_storage(block.norm2)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.post_quant_conv.weight.dtype)
        return self.decoder(self.post_quant_conv(latents)).float()


def _tp_size() -> int:
    from difflet.models.minimax_h3.modeling_minimax_h3_tp import _safe_tp_size

    return _safe_tp_size()


class NeuronMiniMaxH3VideoVaeApplication(TorchNeuronApplicationBase):
    """The H3 video decoder, one tile per call. ``model_path`` is ``<model>/vae``.

    Under the DiT's mesh (a process holds one), by default every rank loads the whole decoder
    and decodes its own tiles (``decode_video_on_ranks``). ``tp_shard=True`` head-shards the
    36 ViT blocks over the TP group instead (a quarter of the HBM at tp4): the ranks of a TP
    group decode the same tile together and the groups split the tiles."""

    block_attrs = ("decoder.transformer_blocks",)
    skipped_checkpoint_prefixes = ("encoder.", "quant_conv.")

    def __init__(self, *, model_path, parallel=None, dtype=torch.bfloat16,
                 exec_mode: str | None = None, device="neuron", tp_shard: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        self.tp_shard = bool(tp_shard) and int(getattr(self.parallel, "tp_degree", 1) or 1) > 1
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"the H3 video VAE runs in bf16 or fp32, got {self.dtype}")
        # Under the DiT's mesh (TP, or TP x Ulysses CP) every rank holds the whole decoder and
        # decodes its own tiles; there is nothing to shard, so only DP is refused.
        if int(getattr(self.parallel, "dp_degree", 1) or 1) > 1:
            raise NotImplementedError("the H3 video VAE has no data parallelism")
        with open(Path(str(self.model_path)) / "config.json", encoding="utf-8") as handle:
            self.vae_config = json.load(handle)

    def build_module(self) -> torch.nn.Module:
        return _NeuronH3VideoDecoder(self.vae_config, tp_shard=self.tp_shard)

    def checkpoint_key(self, name: str) -> str:
        return name.replace(".ff.down_proj.", ".ff.net.2.")

    def checkpoint_slices(self) -> dict[str, CheckpointSlice]:
        """With ``tp_shard``: ``up_proj`` = rows ``[0, inner)`` and ``gate_proj`` = rows
        ``[inner, 2 inner)`` of every fused ``ff.net.0.proj`` weight and bias."""
        if not self.tp_shard:
            return {}
        dim = int(self.vae_config["decoder_num_attention_heads"]) * int(
            self.vae_config["decoder_attention_head_dim"])
        inner = dim * int(self.vae_config["decoder_ffn_mult"])
        slices = {}
        for i in range(int(self.vae_config["decoder_num_layers"])):
            prefix = f"decoder.transformer_blocks.{i}.ff"
            for kind in ("weight", "bias"):
                source = f"{prefix}.net.0.proj.{kind}"
                slices[f"{prefix}.up_proj.{kind}"] = CheckpointSlice(source, 0, 0, inner)
                slices[f"{prefix}.gate_proj.{kind}"] = CheckpointSlice(source, 0, inner, inner)
        return slices

    def tile_groups(self, rank: int, world: int) -> tuple[int, int, bool]:
        """``(group, groups, writer)`` for ``decode_video_on_ranks``: each TP group decodes a
        tile together when sharded (its first rank saves it), every rank alone otherwise."""
        if not self.tp_shard:
            return rank, world, True
        tp = int(self.parallel.tp_degree)
        return rank // tp, world // tp, rank % tp == 0

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator().manual_seed(0)
        return (torch.randn(TILE_LATENT_SHAPE, generator=gen),)


class DeviceTileDecoder:
    """``decode_minimax_h3_video``'s decoder callable on the device: a host tile in, host
    frames out, in the decoder's dtype (the decode takes them up to fp32).

    The graph returns the decoder's output cast up to fp32, so a bf16 decoder's frames go back
    to bf16 exactly and cross to the host (and between ranks) at half the size."""

    def __init__(self, app: NeuronMiniMaxH3VideoVaeApplication) -> None:
        self.app = app
        self.config = app.vae_config

    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        out = self.app(tile.float().contiguous().to(self.app.device))
        return out.to(self.app.dtype).cpu()




def video_tiles(normalized_latents: torch.Tensor, *, latents_mean, latents_std,
                clip_length: int = 17, token_drop: int = 3,
                temporal_compression_ratio: int = 4, spatial_compression_ratio: int = 16,
                tile_sample_size: int = 256, tile_min_overlap: int = 64) -> list[torch.Tensor]:
    """The latent tiles ``decode_minimax_h3_video`` (same arguments, without the decoder)
    hands its decoder, in its call order, without decoding or stitching anything.

    Its chunking and tiling arithmetic, repeated here so the decode it shares with the
    Trainium video_vae stage stays as it is; ``test_video_tiles_are_the_decodes_tiles`` pins
    the two equal.
    """
    return _clip_tiles(normalized_latents, latents_mean=latents_mean, latents_std=latents_std,
                       clip_length=clip_length, token_drop=token_drop,
                       temporal_compression_ratio=temporal_compression_ratio,
                       spatial_compression_ratio=spatial_compression_ratio,
                       tile_sample_size=tile_sample_size, tile_min_overlap=tile_min_overlap)[0]


def _clip_tiles(normalized_latents, *, latents_mean, latents_std, clip_length,
                token_drop, temporal_compression_ratio, spatial_compression_ratio,
                tile_sample_size, tile_min_overlap) -> tuple[list[torch.Tensor], int]:
    """``video_tiles`` and the tiles per temporal clip (the decode stitches clip by clip)."""
    from difflet.models.minimax_h3.pipeline import _split_tiles

    mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latents = normalized_latents.float() * std + mean
    chunk = math.ceil(clip_length / temporal_compression_ratio)
    overlap = (-token_drop) % chunk
    s = spatial_compression_ratio
    ys, y_lengths, _ = _split_tiles(latents.shape[-2] * s, tile_size=tile_sample_size,
                                    min_overlap=tile_min_overlap, alignment=s)
    xs, x_lengths, _ = _split_tiles(latents.shape[-1] * s, tile_size=tile_sample_size,
                                    min_overlap=tile_min_overlap, alignment=s)
    num_tokens = latents.shape[2] + token_drop
    pad = (-num_tokens) % chunk
    num_chunks = (num_tokens + pad) // chunk - int(token_drop > 0)
    if pad:
        latents = torch.cat([latents, latents[:, :, -1:].repeat(1, 1, pad, 1, 1)], dim=2)
    tiles = []
    for index in range(num_chunks):
        clip = latents[:, :, index * chunk : index * chunk + chunk + overlap]
        for y, height in zip(ys, y_lengths):
            for x, width in zip(xs, x_lengths):
                tiles.append(clip[..., y // s : y // s + height // s, x // s : x // s + width // s])
    return tiles, len(ys) * len(xs)


#: How long rank 0's stitching waits for a tile before it gives up (the ranks decode a
#: 768x1344x124 video in ~5 s; a rank that dies leaves the collective to fail first).
TILE_WAIT_SECONDS = 600.0


def _clip_file(scratch: Path, rank: int, clip: int) -> Path:
    return scratch / f"vae_tiles_rank{rank}_clip{clip}.pt"


def decode_video_on_ranks(decoder, latents: torch.Tensor, cfg: dict, *, rank: int, world: int,
                          scratch_dir, device, writer: bool = True,
                          threads: int | None = None,
                          wait_seconds: float = TILE_WAIT_SECONDS) -> torch.Tensor | None:
    """``decode_minimax_h3_video`` with its tiles spread over the ranks, each holding a decoder.

    The decode calls its decoder once per tile in a fixed order (``video_tiles``,
    the same on every rank: the latents are), so each rank decodes the tiles
    ``i % world == rank``, and rank 0 runs the decode with a decoder that hands it the decoded
    tiles in order, which does the official blending and stitching.

    The stitching is pipelined with the device: rank 0 runs it on a thread while its own
    device keeps decoding, taking rank 0's tiles as they come off the device and every other
    rank's from ``scratch_dir``, where each writes its tiles one temporal clip at a time (a
    background save, renamed into place when whole), so only the last clip's stitching is
    left after the device. A collective after the writes makes every rank wait for the others.
    Returns the video on rank 0, None elsewhere.

    ``rank``/``world`` are tile groups: with a TP-sharded decoder every rank of a group runs
    the group's tiles (the TP collectives need them all) and only its ``writer`` saves them.

    ``threads``: rank 0's intra-op threads for the stitching (a launcher's ranks run with one;
    single-threaded, it takes ~8 s of a 768x1344x124 decode on trn3).
    """
    import os
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import torch.distributed as dist

    from difflet.models.minimax_h3.host_components import intra_op_threads
    from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video

    kwargs = dict(latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"],
                  clip_length=int(cfg["clip_length"]), token_drop=int(cfg["token_drop"]))
    tiles, per_clip = _clip_tiles(latents, temporal_compression_ratio=4,
                                  spatial_compression_ratio=16, tile_sample_size=256,
                                  tile_min_overlap=64, **kwargs)
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    stitcher = rank == 0 and writer
    mine = [i for i in range(len(tiles)) if i % world == rank]

    own: dict[int, torch.Tensor] = {}
    ready = threading.Condition()
    failed = threading.Event()

    def own_tile(index: int) -> torch.Tensor:
        deadline = time.monotonic() + wait_seconds
        with ready:
            while index not in own:
                if failed.is_set():
                    raise RuntimeError("the device decode of rank 0's video tiles failed")
                if not ready.wait(timeout=max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"video tile {index} was not decoded in {wait_seconds:.0f}s")
            return own.pop(index)

    loaded: dict[int, torch.Tensor] = {}

    def other_tile(index: int) -> torch.Tensor:
        if index not in loaded:
            path = _clip_file(scratch, index % world, index // per_clip)
            deadline = time.monotonic() + wait_seconds
            while not path.exists():
                if failed.is_set():
                    raise RuntimeError("the device decode of rank 0's video tiles failed")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{path.name} was not written in {wait_seconds:.0f}s")
                time.sleep(0.005)
            loaded.update(torch.load(path))
            path.unlink()
        return loaded.pop(index)

    def stitch() -> torch.Tensor:
        order = iter(range(len(tiles)))

        def replay(_tile: torch.Tensor) -> torch.Tensor:
            index = next(order)
            return own_tile(index) if index % world == rank else other_tile(index)

        with intra_op_threads(threads):
            return decode_minimax_h3_video(replay, latents, **kwargs)

    def save(batch: dict[int, torch.Tensor], clip: int) -> None:
        final = _clip_file(scratch, rank, clip)
        partial = final.with_suffix(".partial")
        torch.save(batch, partial)
        os.replace(partial, final)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-vae-host") as host:
        stitching = host.submit(stitch) if stitcher else None
        saves = []
        try:
            batch: dict[int, torch.Tensor] = {}
            for position, index in enumerate(mine):
                out = decoder(tiles[index])
                if stitcher:
                    with ready:
                        own[index] = out
                        ready.notify_all()
                elif writer:
                    batch[index] = out
                    last = position + 1 == len(mine) or mine[position + 1] // per_clip != index // per_clip
                    if last:
                        saves.append(host.submit(save, batch, index // per_clip))
                        batch = {}
        except BaseException:
            failed.set()
            with ready:
                ready.notify_all()
            raise
        for job in saves:
            job.result()
        if dist.is_initialized() and dist.get_world_size() > 1:
            done = torch.ones(1, device=device)
            dist.all_reduce(done)
            if device.type == "neuron":
                torch.neuron.synchronize()
        return stitching.result() if stitching is not None else None


__all__ = [
    "DeviceTileDecoder",
    "decode_video_on_ranks",
    "video_tiles",
    "NeuronMiniMaxH3VideoVaeApplication",
    "TILE_LATENT_SHAPE",
]
