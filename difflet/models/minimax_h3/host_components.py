"""MiniMax-H3 components that run on the host CPU, for backends without a device port of them.

Backend-neutral (torch, transformers, safetensors only), used by the neuron backend's staged CLI
(``difflet/cli/orchestrators/minimax_h3.py``):

* ``HostQwen3VLTextEncoder``: the conditioner, Qwen3-VL's language model run to
  ``hidden_states[50]`` (the output of decoder layer 49), as the Trainium text stage captures it
  through NxDI. Only the first 50 decoder layers are built and read (the last 14, the final norm,
  ``lm_head`` and the vision tower are never loaded). Tokenization is the Trainium stage's: the
  raw prompt, no special tokens, right-padded to the 1024-token bucket; the padding rows of the
  output are zeroed.
  With ``vision=True`` the vision tower is loaded too and ``encode_presentation`` encodes an
  FL2VA presentation (``"<Picture i>: "`` + an image block per keyframe, then the prompt) through
  the full ``Qwen3VLModel`` (M-RoPE, DeepStack), still read at ``hidden_states[50]``.
* ``load_host_video_encoder``: the video VAE's encoder + ``quant_conv`` in fp32, spatially tiled
  like the reference; FL2VA keyframes only.
* ``load_host_video_decoder`` / ``load_host_audio_decoder``: the two decoder-only VAEs in fp32,
  with the reference attention (no sequence padding) and the audio decoder's weight norm folded
  into plain convolution weights. Their callables plug into ``decode_minimax_h3_video`` /
  ``decode_minimax_h3_audio`` (``pipeline.py``) unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from difflet.common.orchestrators import minimax_h3 as h3_common

_LM_PREFIX = "model.language_model."


@contextlib.contextmanager
def intra_op_threads(threads: int | None):
    if not threads:
        yield
        return
    before = torch.get_num_threads()
    torch.set_num_threads(int(threads))
    try:
        yield
    finally:
        torch.set_num_threads(before)


def _safetensors_index(directory: Path) -> dict[str, str]:
    """``{key: file}`` for a sharded (index) or single-file safetensors checkpoint."""
    index = sorted(directory.glob("*.safetensors.index.json"))
    if index:
        with open(index[0]) as handle:
            return {k: str(directory / v) for k, v in json.load(handle)["weight_map"].items()}
    from safetensors import safe_open

    mapping = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            mapping.update({key: str(path) for key in handle.keys()})
    return mapping


def _read_tensors(weight_map: dict[str, str], keys) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    out = {}
    for path, file_keys in by_file.items():
        with safe_open(path, framework="pt") as handle:
            for key in file_keys:
                out[key] = handle.get_tensor(key)
    return out


def _pos_embed_interpolate_fp32_weights(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Qwen3-VL's bilinear (align_corners) resampling of the learned vision position table,
    with fp32 weights and an fp32 tap sum, cast to the table dtype at the end.

    transformers 4.x (``fast_pos_embed_interpolate``) casts the weights to the bf16 table dtype
    and sums in bf16; 5.x (``get_vision_interpolation_indices_and_weights``, what the official
    H3 pipeline runs) keeps them in fp32. The ~1-ulp difference is amplified by the vision tower's
    and the language model's massive activations until it changes which vision rows become
    attention sinks (measured: 4.x flips one of the top-4 sink rows of a 768x1344 keyframe pair
    against both the fp32 and the bf16 5.x reference; this matches them).
    """
    side = int(self.num_grid_per_side)
    merge = int(self.config.spatial_merge_size)
    table = self.pos_embed.weight
    out = []
    for t, h, w in grid_thw.tolist():
        h_idx = torch.linspace(0, side - 1, h)
        w_idx = torch.linspace(0, side - 1, w)
        h_floor, w_floor = h_idx.int(), w_idx.int()
        h_ceil = (h_floor + 1).clip(max=side - 1)
        w_ceil = (w_floor + 1).clip(max=side - 1)
        dh, dw = h_idx - h_floor, w_idx - w_floor
        base_f, base_c = h_floor * side, h_ceil * side
        indices = torch.stack([
            (base_f[:, None] + w_floor[None]).flatten(),
            (base_f[:, None] + w_ceil[None]).flatten(),
            (base_c[:, None] + w_floor[None]).flatten(),
            (base_c[:, None] + w_ceil[None]).flatten(),
        ], dim=1).to(table.device)
        weights = torch.stack([
            ((1 - dh)[:, None] * (1 - dw)[None]).flatten(),
            ((1 - dh)[:, None] * dw[None]).flatten(),
            (dh[:, None] * (1 - dw)[None]).flatten(),
            (dh[:, None] * dw[None]).flatten(),
        ], dim=1).to(table.device, torch.float32)
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1).to(table.dtype)
        pos = pos.repeat(t, 1).view(t, h // merge, merge, w // merge, merge, -1)
        out.append(pos.permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
    return torch.cat(out)


# ------------------------------------------------------------------------------ text encoder


class HostQwen3VLTextEncoder:
    """Qwen3-VL's language model to ``hidden_states[layer]`` on the host CPU.

    ``model_dir`` is the H3 snapshot root (``text_encoder/`` and ``tokenizer/`` under it).
    ``encode(prompt)`` returns the Trainium text stage's record: ``encoder_hidden_states``
    ``(1, seq_len, 5120)`` bf16 with the padding rows zeroed, ``attention_mask`` ``(1, seq_len)``
    bool, ``num_text_tokens`` and ``text_encoder_layer``.
    """

    def __init__(self, model_dir, *, layer: int = h3_common.TEXT_ENCODER_LAYER,
                 seq_len: int = h3_common.TEXT_SEQ_LEN, dtype: torch.dtype = torch.bfloat16,
                 threads: int | None = None, vision: bool = False,
                 vision_dtype: torch.dtype = torch.float32) -> None:
        self.model_dir = Path(model_dir)
        self.vision = bool(vision)
        self.vision_dtype = vision_dtype
        self.full_model = None
        self.image_processor = None
        self.layer = int(layer)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self.threads = threads
        self.model = None
        self.tokenizer = None

    def load(self) -> "HostQwen3VLTextEncoder":
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        encoder_dir = self.model_dir / "text_encoder"
        text_config = AutoConfig.from_pretrained(encoder_dir).text_config
        if int(text_config.num_hidden_layers) < self.layer:
            raise ValueError(
                f"MiniMax-H3 needs Qwen3-VL hidden_states[{self.layer}], but the text encoder "
                f"has {text_config.num_hidden_layers} layers"
            )
        if getattr(text_config, "pad_token_id", None) is None:
            text_config.pad_token_id = 0
        text_config.num_hidden_layers = self.layer  # hidden_states[L] is layer L-1's output
        text_config._attn_implementation = "sdpa"
        with init_empty_weights(include_buffers=False):
            model = Qwen3VLTextModel(text_config)
        weight_map = _safetensors_index(encoder_dir)
        wanted = {}
        for key in weight_map:
            if not key.startswith(_LM_PREFIX):
                continue
            name = key[len(_LM_PREFIX):]
            if name.startswith("layers.") and int(name.split(".")[1]) >= self.layer:
                continue
            wanted[key] = name
        tensors = _read_tensors(weight_map, wanted)
        state = {wanted[k]: v.to(self.dtype) for k, v in tensors.items()}
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"Qwen3-VL text load: missing {missing[:5]}, unexpected "
                               f"{unexpected[:5]}")
        self.model = model.eval().requires_grad_(False)
        if self.vision:
            self._load_vision(encoder_dir, weight_map, text_config)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        return self

    def _load_vision(self, encoder_dir: Path, weight_map: dict[str, str], text_config) -> None:
        """The vision tower, wrapped with the already-loaded language model in a
        ``Qwen3VLModel`` (its forward builds the M-RoPE index and injects DeepStack features)."""
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        config = AutoConfig.from_pretrained(encoder_dir)
        config.text_config = text_config
        config._attn_implementation = "sdpa"
        config.vision_config._attn_implementation = "sdpa"
        with init_empty_weights(include_buffers=False):
            full = Qwen3VLModel(config)
        prefix = "model.visual."
        tensors = _read_tensors(weight_map, [k for k in weight_map if k.startswith(prefix)])
        visual_state = {k[len(prefix):]: v.to(self.vision_dtype) for k, v in tensors.items()}
        # Buffers (the vision rotary inv_freq) were built for real under
        # include_buffers=False; only the parameters come from the checkpoint.
        full.visual.load_state_dict(visual_state, strict=True, assign=True)
        full.language_model = self.model
        if hasattr(full.visual, "fast_pos_embed_interpolate"):
            full.visual.fast_pos_embed_interpolate = types.MethodType(
                _pos_embed_interpolate_fp32_weights, full.visual)
        self.full_model = full.eval().requires_grad_(False)
        processor_dir = self.model_dir / "processor"
        if not processor_dir.is_dir():
            processor_dir = encoder_dir
        self.image_processor = AutoImageProcessor.from_pretrained(processor_dir)

    def tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            prompt, add_special_tokens=False, max_length=self.seq_len, padding="max_length",
            truncation=True, return_tensors="pt", return_attention_mask=True,
        )
        return tokens.input_ids, tokens.attention_mask

    @torch.no_grad()
    def encode(self, prompt: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("HostQwen3VLTextEncoder.load() first")
        input_ids, attention_mask = self.tokenize(prompt)
        token_count = int(attention_mask.sum())
        if token_count < 1:
            raise ValueError("MiniMax-H3 prompt must produce at least one token")
        captured: list[torch.Tensor] = []
        handle = self.model.layers[self.layer - 1].register_forward_hook(
            lambda module, args, output: captured.append(
                output[0] if isinstance(output, tuple) else output)
        )
        try:
            with intra_op_threads(self.threads):
                self.model(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=torch.arange(self.seq_len).unsqueeze(0), use_cache=False)
        finally:
            handle.remove()
        hidden = captured[-1].to(torch.bfloat16).clone()
        hidden[:, token_count:] = 0  # padding rows are never part of the packed document
        return {
            "encoder_hidden_states": hidden,
            "attention_mask": attention_mask.to(torch.bool),
            "num_text_tokens": token_count,
            "text_encoder_layer": self.layer,
        }


    @torch.no_grad()
    def encode_presentation(self, prompt: str, keyframes) -> dict[str, Any]:
        """Encode an FL2VA presentation; the record of ``encode`` plus ``text_token_tags``
        (``(num_text_tokens,)``, the vision blocks' rows tagged video)."""
        from difflet.models.minimax_h3.keyframes import build_fl2va_presentation

        if self.full_model is None:
            raise RuntimeError("HostQwen3VLTextEncoder(vision=True).load() first")
        token_ids, token_tags, vision_inputs = build_fl2va_presentation(
            self.tokenizer, self.image_processor, prompt, keyframes)
        count = len(token_ids)
        if count > self.seq_len:
            raise ValueError(
                f"the FL2VA presentation has {count} tokens (keyframe vision blocks included), "
                f"more than the {self.seq_len}-token text bucket")
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        vision_kwargs = {
            name: (value.to(self.vision_dtype) if name.startswith("pixel_") else value)
            for name, value in vision_inputs.items()
        }
        captured: list[torch.Tensor] = []
        handle = self.model.layers[self.layer - 1].register_forward_hook(
            lambda module, args, output: captured.append(
                output[0] if isinstance(output, tuple) else output)
        )
        try:
            with intra_op_threads(self.threads):
                self.full_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                use_cache=False, **vision_kwargs)
        finally:
            handle.remove()
        hidden = torch.zeros(1, self.seq_len, captured[-1].shape[-1], dtype=torch.bfloat16)
        hidden[:, :count] = captured[-1].to(torch.bfloat16)
        attention_mask = torch.zeros(1, self.seq_len, dtype=torch.bool)
        attention_mask[:, :count] = True
        return {
            "encoder_hidden_states": hidden,
            "attention_mask": attention_mask,
            "num_text_tokens": count,
            "text_encoder_layer": self.layer,
            "text_token_tags": token_tags,
        }


# ------------------------------------------------------------------------------ video encoder


class HostVideoEncoder:
    """The video VAE's ``quant_conv(encoder(x))`` in fp32, spatially tiled as the reference
    (256-pixel tiles, >= 64-pixel overlaps, 16-aligned, linear blends). Single frames only:
    ``__call__(pixels)`` takes ``(1, 3, 1, H, W)`` uint8 and returns the normalized conditioning
    latents of ``keyframes.encode_keyframe_condition``."""

    def __init__(self, module: nn.Module, config: dict[str, Any], threads: int | None = None,
                 *, tile: int = 256, min_overlap: int = 64) -> None:
        self.module = module
        self.config = config
        self.threads = threads
        self.tile = int(tile)
        self.min_overlap = int(min_overlap)
        spatial = 1
        for value in config["spatial_downsample_factors"]:
            spatial *= int(value)
        self.spatial_compression = spatial

    def moments(self, x: torch.Tensor) -> torch.Tensor:
        from difflet.models.minimax_h3.pipeline import _split_tiles, _stitch_tiles

        if x.shape[2] != 1:
            raise ValueError("HostVideoEncoder encodes single keyframes only")
        height, width = x.shape[-2], x.shape[-1]
        ys, y_lengths, y_overlaps = _split_tiles(
            height, tile_size=self.tile, min_overlap=self.min_overlap,
            alignment=self.spatial_compression)
        xs, x_lengths, x_overlaps = _split_tiles(
            width, tile_size=self.tile, min_overlap=self.min_overlap,
            alignment=self.spatial_compression)
        rows = []
        with torch.no_grad(), intra_op_threads(self.threads):
            for y, y_len in zip(ys, y_lengths):
                rows.append([
                    self.module(x[..., y:y + y_len, x0:x0 + x_len].float())
                    for x0, x_len in zip(xs, x_lengths)
                ])
        ratio = self.spatial_compression
        return _stitch_tiles(rows, [o // ratio for o in y_overlaps],
                             [o // ratio for o in x_overlaps])

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        from difflet.models.minimax_h3.keyframes import encode_keyframe_condition

        return encode_keyframe_condition(
            self.moments, pixels, latents_mean=self.config["latents_mean"],
            latents_std=self.config["latents_std"])


class _HostVideoEncoderModule(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        from difflet.models.minimax_h3.modeling_vae import MiniMaxH3VideoEncoder3d

        super().__init__()
        latent = int(cfg["latent_channels"])
        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=int(cfg["in_channels"]), out_channels=2 * latent,
            block_out_channels=tuple(int(v) for v in cfg["block_out_channels"]),
            layers_per_block=int(cfg["layers_per_block"]),
            spatial_downsample_factors=tuple(int(v) for v in cfg["spatial_downsample_factors"]),
            temporal_downsample_factors=tuple(int(v) for v in cfg["temporal_downsample_factors"]),
            norm_num_groups=int(cfg["norm_num_groups"]), norm_eps=float(cfg["norm_eps"]),
            spatial_padding_mode=str(cfg["spatial_padding_mode"]),
        )
        self.quant_conv = nn.Conv3d(2 * latent, 2 * latent, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_conv(self.encoder(x))


def load_host_video_encoder(model_dir, *, threads: int | None = None) -> HostVideoEncoder:
    directory = Path(model_dir) / "vae"
    with open(directory / "config.json") as handle:
        cfg = json.load(handle)
    module = _HostVideoEncoderModule(cfg)
    _load_decoder(directory, module, ("encoder.", "quant_conv."), fold_weight_norm=False)
    return HostVideoEncoder(module.eval().requires_grad_(False), cfg, threads)


# ------------------------------------------------------------------------------ decoders


@dataclass
class HostDecoder:
    """A decoder callable plus the config values the decode helpers take."""

    module: nn.Module
    config: dict[str, Any]
    threads: int | None = None

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), intra_op_threads(self.threads):
            return self.module(latents.float())


class _HostVideoDecoder(nn.Module):
    """``post_quant_conv`` then the ViT decoder, keys as the checkpoint's. ``sequence_alignment``
    pads the token sequence (with a key mask) for kernels that need aligned lengths; 1 is the
    reference."""

    def __init__(self, cfg: dict[str, Any], *, sequence_alignment: int = 1) -> None:
        from difflet.models.minimax_h3.modeling_vae import MiniMaxH3VideoViTDecoder3d

        super().__init__()
        spatial = 1
        for value in cfg["spatial_downsample_factors"]:
            spatial *= int(value)
        temporal = 1
        for value in cfg["temporal_downsample_factors"]:
            temporal *= int(value)
        latent = int(cfg["latent_channels"])
        self.post_quant_conv = nn.Conv3d(latent, latent, kernel_size=1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=latent, out_channels=int(cfg["out_channels"]), patch_size=spatial,
            patch_size_t=temporal, num_layers=int(cfg["decoder_num_layers"]),
            num_attention_heads=int(cfg["decoder_num_attention_heads"]),
            attention_head_dim=int(cfg["decoder_attention_head_dim"]),
            num_register_tokens=int(cfg["decoder_num_register_tokens"]),
            ffn_mult=int(cfg["decoder_ffn_mult"]), rope_theta=float(cfg["decoder_rope_theta"]),
            rope_dim_ratio=float(cfg["decoder_rope_dim_ratio"]),
            norm_eps=float(cfg["decoder_norm_eps"]),
            sequence_alignment=int(sequence_alignment),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latents))


class _HostAudioDecoder(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        from difflet.models.minimax_h3.modeling_vae import MiniMaxH3AudioBigVGANDecoder

        super().__init__()
        self.dec_in_proj = nn.Conv1d(int(cfg["latent_channels"]), int(cfg["latent_dim"]), 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=int(cfg["latent_dim"]),
            upsample_initial_channel=int(cfg["decoder_dim"]),
            upsample_rates=tuple(int(v) for v in cfg["decoder_rates"]),
            upsample_kernel_sizes=tuple(int(v) for v in cfg["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(int(v) for v in cfg["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(int(v) for v in d)
                                          for d in cfg["resblock_dilation_sizes"]),
            use_weight_norm=False,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_in_proj(latents))


def _load_decoder(directory: Path, module: nn.Module, prefixes: tuple[str, ...],
                  fold_weight_norm: bool) -> None:
    weight_map = _safetensors_index(directory)
    state = _read_tensors(weight_map, [k for k in weight_map if k.startswith(prefixes)])
    if fold_weight_norm:
        # Legacy weight_norm (weight_g, weight_v) -> the identical plain weight.
        for key in [k for k in state if k.endswith(".weight_v")]:
            prefix = key[: -len(".weight_v")]
            state[f"{prefix}.weight"] = torch._weight_norm(
                state.pop(key), state.pop(f"{prefix}.weight_g"), 0)
    module.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)


def load_host_video_decoder(model_dir, *, threads: int | None = None) -> HostDecoder:
    directory = Path(model_dir) / "vae"
    with open(directory / "config.json") as handle:
        cfg = json.load(handle)
    module = _HostVideoDecoder(cfg)
    _load_decoder(directory, module, ("post_quant_conv.", "decoder."), fold_weight_norm=False)
    return HostDecoder(module.eval().requires_grad_(False), cfg, threads)


def load_host_audio_decoder(model_dir, *, threads: int | None = None) -> HostDecoder:
    directory = Path(model_dir) / "audio_vae"
    with open(directory / "config.json") as handle:
        cfg = json.load(handle)
    module = _HostAudioDecoder(cfg)
    _load_decoder(directory, module, ("dec_in_proj.", "decoder."), fold_weight_norm=True)
    return HostDecoder(module.eval().requires_grad_(False), cfg, threads)


def default_threads() -> int:
    """Intra-op threads for the host components: half the CPUs, at most 96 (the host is
    shared with the device ranks and, on the trn hosts, with other jobs)."""
    return max(1, min(96, (os.cpu_count() or 2) // 2))


__all__ = [
    "HostDecoder",
    "HostQwen3VLTextEncoder",
    "HostVideoEncoder",
    "default_threads",
    "load_host_audio_decoder",
    "load_host_video_encoder",
    "load_host_video_decoder",
]
