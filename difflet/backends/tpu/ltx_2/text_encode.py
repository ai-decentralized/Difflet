"""LTX-2 prompt encoding over the prompt's real tokens only.

diffusers' ``LTX2Pipeline._get_gemma_prompt_embeds`` left-pads every prompt to
``max_sequence_length`` (1024) and runs Gemma-3 12B over all of it; the
connectors then run ``text_proj_in`` -- a (3840*49 -> 3840) linear, 0.72 B
weights -- over the same 1024 rows. A typical prompt is a few dozen tokens, and
nothing computed at a padding position reaches the DiT:

* ``per_layer_masked_mean_norm`` takes its mean / min / max over the valid
  positions only and zero-fills the padding rows;
* ``text_proj_in`` has no bias, so a zero row stays zero;
* ``LTX2ConnectorTransformer1d`` then moves the valid rows to the front and
  overwrites every padding row with its learnable registers.

So both are computed here over the last ``n`` columns (``n`` = the longest
prompt of the batch, left padding) and the result is zero-padded back to
``max_sequence_length``. Gemma is given the ``position_ids`` the padded call
would have used (``arange(L)``, transformers' default, which ignores the
mask) so every real token keeps its rotary angle; its sliding window (1024)
covers the whole prompt either way. The connector blocks still run on the
full 1024 rows (registers attend too).

Only LTX-2.0's connectors (``per_modality_projections=False``) are handled;
anything else falls back to the stock call.
"""

from __future__ import annotations

import torch


def tokenize_like_diffusers(tokenizer, prompts, max_sequence_length: int):
    """``_get_gemma_prompt_embeds``' tokenization: left padding to max_length."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    inputs = tokenizer(
        [p.strip() for p in prompts],
        padding="max_length",
        max_length=int(max_sequence_length),
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    return inputs.input_ids, inputs.attention_mask


def real_token_window(attention_mask: torch.Tensor) -> int:
    """Index of the first column any prompt of the batch uses (left padding)."""
    length = int(attention_mask.shape[1])
    n = int(attention_mask.sum(dim=1).max())
    start = length - n
    if start > 0 and bool(attention_mask[:, :start].any()):
        raise ValueError("expected left-padded prompts (a valid token precedes the window)")
    return start


@torch.no_grad()
def gemma_prompt_embeds(text_encoder, input_ids, attention_mask, *, dtype=None):
    """Packed (B, L, hidden * (layers + 1)) Gemma states, computed on the real tokens.

    Padding rows are zero; the stock call leaves whatever Gemma produced there,
    which nothing downstream reads.
    """
    batch, length = int(input_ids.shape[0]), int(input_ids.shape[1])
    start = real_token_window(attention_mask)
    position_ids = torch.arange(start, length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
    out = text_encoder(
        input_ids=input_ids[:, start:],
        attention_mask=attention_mask[:, start:],
        position_ids=position_ids,
        output_hidden_states=True,
    )
    packed = torch.stack(out.hidden_states, dim=-1).flatten(2, 3)
    if dtype is not None:
        packed = packed.to(dtype)
    full = packed.new_zeros(batch, length, packed.shape[-1])
    full[:, start:] = packed
    return full


@torch.no_grad()
def connectors_forward(connectors, text_encoder_hidden_states, attention_mask,
                       padding_side: str = "left", scale_factor: int = 8):
    """``LTX2TextConnectors.forward`` (diffusers 0.38) with ``text_proj_in`` on the real rows.

    Same return value: ``(video_text_embedding, audio_text_embedding, binary_mask)``.
    """
    from diffusers.pipelines.ltx2.connectors import per_layer_masked_mean_norm

    cfg = connectors.config
    if cfg.per_modality_projections or padding_side != "left":
        return connectors(text_encoder_hidden_states, attention_mask,
                          padding_side=padding_side, scale_factor=scale_factor)
    hidden = text_encoder_hidden_states
    if hidden.ndim == 3:
        hidden = hidden.unflatten(2, (cfg.caption_channels, -1))
    batch, length = int(hidden.shape[0]), int(hidden.shape[1])
    start = real_token_window(attention_mask)
    window_mask = attention_mask[:, start:]
    norm = per_layer_masked_mean_norm(
        text_hidden_states=hidden[:, start:],
        sequence_lengths=window_mask.sum(dim=-1),
        device=hidden.device,
        padding_side="left",
        scale_factor=scale_factor,
    )
    proj = connectors.text_proj_in(norm)
    text_emb_proj = proj.new_zeros(batch, length, proj.shape[-1])
    text_emb_proj[:, start:] = proj

    # From here on, the stock forward unchanged.
    text_dtype = text_emb_proj.dtype
    mask = (attention_mask.to(torch.int64) - 1).to(text_dtype)
    mask = mask.reshape(mask.shape[0], 1, -1, mask.shape[-1])
    add_attn_mask = mask * torch.finfo(text_dtype).max
    video, video_attn_mask = connectors.video_connector(text_emb_proj, add_attn_mask)
    binary = (video_attn_mask < 1e-6).to(torch.int64)
    binary = binary.reshape(video.shape[0], video.shape[1], 1)
    video = video * binary
    audio, _ = connectors.audio_connector(text_emb_proj, add_attn_mask)
    return video, audio, binary.squeeze(-1)


__all__ = [
    "connectors_forward",
    "gemma_prompt_embeds",
    "real_token_window",
    "tokenize_like_diffusers",
]
