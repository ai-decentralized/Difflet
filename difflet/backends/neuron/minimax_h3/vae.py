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


class _NeuronH3VideoDecoder(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        from difflet.models.minimax_h3.host_components import _HostVideoDecoder

        inner = _HostVideoDecoder(cfg, sequence_alignment=SEQUENCE_ALIGNMENT)
        self.post_quant_conv = inner.post_quant_conv
        self.decoder = inner.decoder
        from difflet.backends.neuron.minimax_h3.transformer import _give_fp32_host_storage

        processor = _NeuronH3VaeAttnProcessor()
        for block in self.decoder.transformer_blocks:
            block.attn.set_processor(processor)
            # The blocks call these norms on fp32 activations (the reference's precision), so
            # their weights stay fp32 whatever the application dtype.
            _give_fp32_host_storage(block.norm1)
            _give_fp32_host_storage(block.norm2)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.post_quant_conv.weight.dtype)
        return self.decoder(self.post_quant_conv(latents)).float()


class NeuronMiniMaxH3VideoVaeApplication(TorchNeuronApplicationBase):
    """The H3 video decoder on one rank, one tile per call. ``model_path`` is ``<model>/vae``."""

    block_attrs = ("decoder.transformer_blocks",)
    skipped_checkpoint_prefixes = ("encoder.", "quant_conv.")

    def __init__(self, *, model_path, parallel=None, dtype=torch.bfloat16,
                 exec_mode: str | None = None, device="neuron", **kwargs: Any) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        if int(self.parallel.world_size) != 1:
            raise NotImplementedError("the H3 video VAE runs on one rank")
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"the H3 video VAE runs in bf16 or fp32, got {self.dtype}")
        with open(Path(str(self.model_path)) / "config.json", encoding="utf-8") as handle:
            self.vae_config = json.load(handle)

    def build_module(self) -> torch.nn.Module:
        return _NeuronH3VideoDecoder(self.vae_config)

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator().manual_seed(0)
        return (torch.randn(TILE_LATENT_SHAPE, generator=gen),)


class DeviceTileDecoder:
    """``decode_minimax_h3_video``'s decoder callable on the device: a host tile in, host fp32
    frames out."""

    def __init__(self, app: NeuronMiniMaxH3VideoVaeApplication) -> None:
        self.app = app
        self.config = app.vae_config

    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        out = self.app(tile.float().contiguous().to(self.app.device))
        return out.cpu()


__all__ = [
    "DeviceTileDecoder",
    "NeuronMiniMaxH3VideoVaeApplication",
    "TILE_LATENT_SHAPE",
]
