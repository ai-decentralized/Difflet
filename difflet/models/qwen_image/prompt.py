"""Prompt encoding helpers for Qwen-Image, shared by the CUDA application and the
Trainium input-caching script.

Qwen-Image's DiT takes a fixed-length text sequence (``text_seq_len``, 1024 by
default); diffusers' ``QwenImagePipeline.encode_prompt`` returns a variable
length. ``pad_or_truncate_prompt_embeds`` makes it fixed-length with a boolean
validity mask, exactly as ``scripts/qwen_image_cache_dit_inputs.py`` did for the
Trainium graph. ``encode_prompt_fixed_length`` is the whole step for an in-process
encoder on a device (CUDA plan D8).
"""

from __future__ import annotations

from typing import Any

import torch


def pad_or_truncate_prompt_embeds(
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor | None,
    *,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-length prompt embeddings and a bool mask (True = real token)."""

    batch_size, seq_len, hidden_size = prompt_embeds.shape
    if prompt_embeds_mask is None:
        prompt_embeds_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=prompt_embeds.device
        )
    else:
        prompt_embeds_mask = prompt_embeds_mask.to(device=prompt_embeds.device, dtype=torch.bool)

    if seq_len > max_sequence_length:
        return (
            prompt_embeds[:, :max_sequence_length, :].contiguous(),
            prompt_embeds_mask[:, :max_sequence_length].contiguous(),
        )
    if seq_len == max_sequence_length:
        return prompt_embeds.contiguous(), prompt_embeds_mask.contiguous()

    pad_len = max_sequence_length - seq_len
    embed_pad = torch.zeros(
        batch_size, pad_len, hidden_size, dtype=prompt_embeds.dtype, device=prompt_embeds.device
    )
    mask_pad = torch.zeros(batch_size, pad_len, dtype=torch.bool, device=prompt_embeds.device)
    return (
        torch.cat([prompt_embeds, embed_pad], dim=1).contiguous(),
        torch.cat([prompt_embeds_mask, mask_pad], dim=1).contiguous(),
    )


def load_text_pipeline(
    *,
    model_path: str,
    dtype: torch.dtype,
    device: torch.device | str,
    local_files_only: bool = True,
) -> Any:
    """diffusers' ``QwenImagePipeline`` with only the text side loaded, on ``device``.

    Mirrors ``scripts/qwen_image_cache_dit_inputs.py:_load_text_pipeline``; the VAE and
    transformer are left out so the pipeline can be used purely for ``encode_prompt``.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

    kwargs = {"local_files_only": local_files_only}
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype, **kwargs
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path, subfolder="tokenizer", **kwargs)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", **kwargs
    )
    pipe = QwenImagePipeline(
        scheduler=scheduler,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_prompt_fixed_length(
    text_pipeline: Any,
    prompt: str | list[str],
    *,
    device: torch.device | str,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``encode_prompt`` then pad/truncate to ``max_sequence_length``."""
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = text_pipeline.encode_prompt(
            prompt, device=torch.device(device), max_sequence_length=max_sequence_length
        )
    return pad_or_truncate_prompt_embeds(
        prompt_embeds, prompt_embeds_mask, max_sequence_length=max_sequence_length
    )


__all__ = ["encode_prompt_fixed_length", "load_text_pipeline", "pad_or_truncate_prompt_embeds"]
