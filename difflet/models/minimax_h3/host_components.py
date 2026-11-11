"""MiniMax-H3 components that run on the host CPU, for backends without a device port of them.

Backend-neutral (torch, transformers, safetensors only), used by the neuron backend's staged CLI
(``difflet/cli/orchestrators/minimax_h3.py``):

* ``HostQwen3VLTextEncoder``: the conditioner, Qwen3-VL's language model run to
  ``hidden_states[50]`` (the output of decoder layer 49), as the Trainium text stage captures it
  through NxDI. Only the first 50 decoder layers are built and read (the last 14, the final norm,
  ``lm_head`` and the vision tower are never loaded). Tokenization is the Trainium stage's: the
  raw prompt, no special tokens, right-padded to the 1024-token bucket; the padding rows of the
  output are zeroed.
* ``load_host_video_decoder`` / ``load_host_audio_decoder``: the two decoder-only VAEs in fp32,
  with the reference attention (no sequence padding) and the audio decoder's weight norm folded
  into plain convolution weights. Their callables plug into ``decode_minimax_h3_video`` /
  ``decode_minimax_h3_audio`` (``pipeline.py``) unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
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
                 threads: int | None = None) -> None:
        self.model_dir = Path(model_dir)
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
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        return self

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
    "default_threads",
    "load_host_audio_decoder",
    "load_host_video_decoder",
]
