# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LTX one-stage host arithmetic with explicit BF16 sampler materialization.

The schedule and materialization order follow the LTX one-stage implementation.
Only the joint text-to-audiovisual, single-CFG-scale workload is exposed here.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch


def official_sigmas(config, steps: int, *, device) -> torch.Tensor:
    """Build the max-anchor LTX schedule in torch FP32, including terminal zero."""
    base_anchor = config.get("base_image_seq_len", 1024)
    max_anchor = config.get("max_image_seq_len", 4096)
    base_shift = config.get("base_shift", 0.95)
    max_shift = config.get("max_shift", 2.05)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")
    slope = (max_shift - base_shift) / (max_anchor - base_anchor)
    shift = max_anchor * slope + (base_shift - slope * base_anchor)
    exp_shift = math.exp(shift)
    sigmas = torch.where(sigmas != 0, exp_shift / (exp_shift + (1 / sigmas - 1)), 0)
    terminal = config.get("shift_terminal")
    if terminal is not None:
        non_zero = sigmas != 0
        one_minus = 1.0 - sigmas[non_zero]
        scale = one_minus[-1] / (1.0 - terminal)
        sigmas[non_zero] = 1.0 - one_minus / scale
    return sigmas.to(device=device, dtype=torch.float32)


def resolve_sigmas(config, steps: int, sigmas, *, device) -> torch.Tensor:
    """Validate final sigma boundaries; custom values are never shifted again."""
    if sigmas is None:
        values = official_sigmas(config, steps, device=device)
        if not torch.isfinite(values).all() or not (values[:-1] > values[1:]).all():
            raise ValueError("LTX-2 schedule must be finite and strictly decreasing")
        return values
    if not isinstance(sigmas, (list, tuple)) or any(isinstance(x, bool) for x in sigmas):
        raise ValueError("LTX-2 sigmas must be a sequence of finite decreasing numbers")
    try:
        values = torch.tensor(sigmas, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError):
        raise ValueError("LTX-2 sigmas must be a sequence of finite decreasing numbers") from None
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("LTX-2 sigmas must contain at least two finite boundary values")
    if values[-1] != 0:
        values = torch.cat([values, values.new_zeros(1)])
    if not (values[:-1] > 0).all() or not (values[:-1] > values[1:]).all():
        raise ValueError("LTX-2 sigmas must decrease strictly to terminal zero")
    if values.numel() != steps + 1:
        raise ValueError("LTX-2 sigmas must match num_inference_steps")
    return values.to(device=device)


def guided_velocity(sample, positive, negative, sigma, scale: float) -> torch.Tensor:
    """Guide x0 with model-dtype rounding at every LTX materialization boundary."""
    sigma = sigma.float()
    positive_x0 = (sample.float() - positive.float() * sigma).to(sample.dtype).contiguous()
    if negative is None:
        guided_x0 = positive_x0
    else:
        negative_x0 = (sample.float() - negative.float() * sigma).to(sample.dtype).contiguous()
        positive_x0 = positive_x0.float()
        guided_x0 = (positive_x0 + (scale - 1) * (positive_x0 - negative_x0.float())).to(
            sample.dtype
        )
    # Materializing the Python scalar is part of LTX's division arithmetic.
    return ((sample.float() - guided_x0.float()) / sigma.item()).to(sample.dtype)


def euler_step(sample, velocity, sigma, sigma_next) -> torch.Tensor:
    return (sample.float() + velocity.float() * (sigma_next.float() - sigma.float())).to(
        sample.dtype
    )


def _gemma(pipe, texts, *, max_sequence_length, device):
    tokenizer = pipe.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("LTX-2 prompt tokenization requires bos_token_id")
    input_ids = []
    for text in texts:
        ids = tokenizer(
            text.strip(),
            padding=False,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors=None,
        ).input_ids
        if not ids or ids[0] != bos:
            ids = [bos, *ids][:max_sequence_length]
        input_ids.append(ids)
    tokens = tokenizer.pad(
        {"input_ids": input_ids},
        padding="max_length",
        max_length=max_sequence_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    mask = tokens.attention_mask.to(device)
    hidden = pipe.text_encoder(
        input_ids=tokens.input_ids.to(device), attention_mask=mask, output_hidden_states=True
    ).hidden_states
    return torch.stack(hidden, dim=-1).flatten(2, 3).to(pipe.text_encoder.dtype), mask


def _conditioning(
    pipe,
    prompt,
    negative_prompt,
    prompt_embeds,
    negative_prompt_embeds,
    prompt_attention_mask,
    negative_prompt_attention_mask,
    *,
    do_cfg,
    max_sequence_length,
    device,
):
    positive = [prompt] if isinstance(prompt, str) else prompt
    negative = [negative_prompt or ""] if not isinstance(negative_prompt, list) else negative_prompt
    kwargs = dict(max_sequence_length=max_sequence_length, device=device)
    if do_cfg and prompt_embeds is None and negative_prompt_embeds is None:
        embeddings, masks = _gemma(pipe, positive + negative, **kwargs)
        prompt_embeds, negative_prompt_embeds = embeddings[:1], embeddings[1:]
        prompt_attention_mask, negative_prompt_attention_mask = masks[:1], masks[1:]
    else:
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = _gemma(pipe, positive, **kwargs)
        if do_cfg and negative_prompt_embeds is None:
            negative_prompt_embeds, negative_prompt_attention_mask = _gemma(
                pipe, negative, **kwargs
            )
    dtype = pipe.transformer.dtype
    video, audio, _ = pipe.connectors(
        prompt_embeds.to(device=device, dtype=dtype),
        prompt_attention_mask.to(device),
        padding_side="left",
    )
    if do_cfg:
        negative_video, negative_audio, _ = pipe.connectors(
            negative_prompt_embeds.to(device=device, dtype=dtype),
            negative_prompt_attention_mask.to(device),
            padding_side="left",
        )
        video, audio = torch.cat([video, negative_video]), torch.cat([audio, negative_audio])
    return video, audio


def _noised_state(value, generator):
    from diffusers.utils.torch_utils import randn_tensor

    # Even zero noise scale consumes the request RNG at this public input boundary.
    noise = randn_tensor(value.shape, generator=generator, device=value.device, dtype=value.dtype)
    return torch.lerp(value.float(), noise.float(), 0.0).to(value.dtype)


def _initial_states(pipe, latents, audio_latents, *, video_shape, audio_shape, generator, device):
    from diffusers.utils.torch_utils import randn_tensor

    dtype = pipe.transformer.dtype
    if latents is None:
        video = randn_tensor(video_shape, generator=generator, device=device, dtype=dtype)
    else:
        video = latents.to(device)
        if video.ndim == 5:
            video = pipe._normalize_latents(
                video, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
            )
            video = pipe._pack_latents(
                video, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
            )
        video = _noised_state(video, generator).to(dtype)
    video = video.transpose(1, 2).contiguous().transpose(1, 2)
    if audio_latents is None:
        audio = randn_tensor(audio_shape, generator=generator, device=device, dtype=dtype)
    else:
        audio = audio_latents.to(device)
        if audio.ndim == 4:
            audio = pipe._pack_audio_latents(audio)
        audio = pipe._normalize_audio_latents(
            audio, pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
        )
        audio = _noised_state(audio, generator).to(dtype)
    return video, audio


@contextmanager
def _vocoder_determinism(device):
    previous = torch.backends.cudnn.deterministic
    try:
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = True
        yield
    finally:
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = previous


@torch.no_grad()
def generate(
    pipe,
    *,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    frame_rate,
    num_inference_steps,
    guidance_scale,
    generator,
    latents,
    audio_latents,
    prompt_embeds,
    negative_prompt_embeds,
    prompt_attention_mask,
    negative_prompt_attention_mask,
    output_type,
    max_sequence_length,
    device,
    sigmas,
):
    """Run explicit LTX BF16 host numerics around the resident Difflet transformer."""
    device = torch.device(device)
    boundaries = resolve_sigmas(pipe.scheduler.config, num_inference_steps, sigmas, device=device)
    do_cfg = not math.isclose(guidance_scale, 1.0)
    video_context, audio_context = _conditioning(
        pipe,
        prompt,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        do_cfg=do_cfg,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    h, w = height // pipe.vae_spatial_compression_ratio, width // pipe.vae_spatial_compression_ratio
    patch, patch_t = pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
    tokens = (frames // patch_t) * (h // patch) * (w // patch)
    channels = pipe.transformer.config.in_channels * patch_t * patch * patch
    audio_frames = round(
        num_frames
        / frame_rate
        * pipe.audio_sampling_rate
        / pipe.audio_hop_length
        / float(pipe.audio_vae_temporal_compression_ratio)
    )
    mel_bins = pipe.audio_vae.config.mel_bins // pipe.audio_vae_mel_compression_ratio
    video, audio = _initial_states(
        pipe,
        latents,
        audio_latents,
        video_shape=(1, tokens, channels),
        audio_shape=(1, audio_frames, pipe.audio_vae.config.latent_channels * mel_bins),
        generator=generator,
        device=device,
    )
    batch = 2 if do_cfg else 1
    video_coords = pipe.transformer.rope.prepare_video_coords(
        batch, frames, h, w, device, fps=frame_rate
    )
    audio_coords = pipe.transformer.audio_rope.prepare_audio_coords(batch, audio_frames, device)
    for index in range(num_inference_steps):
        timestep = (
            boundaries[index] * pipe.scheduler.config.get("num_train_timesteps", 1000)
        ).expand(batch)
        with pipe.transformer.cache_context("guided"):
            video_velocity, audio_velocity = pipe.transformer(
                hidden_states=video.repeat(batch, 1, 1) if do_cfg else video,
                audio_hidden_states=audio.repeat(batch, 1, 1) if do_cfg else audio,
                encoder_hidden_states=video_context,
                audio_encoder_hidden_states=audio_context,
                timestep=timestep.reshape(-1, 1).expand(-1, tokens),
                audio_timestep=timestep.reshape(-1, 1).expand(-1, audio_frames),
                sigma=timestep,
                audio_sigma=timestep,
                encoder_attention_mask=None,
                audio_encoder_attention_mask=None,
                num_frames=frames,
                height=h,
                width=w,
                fps=frame_rate,
                audio_num_frames=audio_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                isolate_modalities=False,
                use_cross_timestep=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                attention_kwargs=None,
                return_dict=False,
            )
        video_parts, audio_parts = video_velocity.chunk(batch), audio_velocity.chunk(batch)
        video_velocity = guided_velocity(
            video,
            video_parts[0],
            video_parts[1] if do_cfg else None,
            boundaries[index],
            guidance_scale,
        )
        audio_velocity = guided_velocity(
            audio,
            audio_parts[0],
            audio_parts[1] if do_cfg else None,
            boundaries[index],
            guidance_scale,
        )
        video = euler_step(video, video_velocity, boundaries[index], boundaries[index + 1])
        audio = euler_step(audio, audio_velocity, boundaries[index], boundaries[index + 1])
    video = pipe._unpack_latents(video, frames, h, w, patch, patch_t)
    video = pipe._denormalize_latents(
        video, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
    )
    audio = pipe._denormalize_audio_latents(
        audio, pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
    )
    audio = pipe._unpack_audio_latents(audio, audio_frames, num_mel_bins=mel_bins)
    if output_type == "latent":
        return video, audio
    timestep_decode = None
    if pipe.vae.config.timestep_conditioning:
        # The public LTX-2 profile exposes decode_timestep=decode_noise_scale=0 only.
        from diffusers.utils.torch_utils import randn_tensor

        noise = randn_tensor(video.shape, generator=generator, device=device, dtype=video.dtype)
        timestep_decode = torch.zeros(1, device=device, dtype=video.dtype)
        video = video + timestep_decode[:, None, None, None, None] * noise
    frames_out = pipe.vae.decode(video.to(pipe.vae.dtype), timestep_decode, return_dict=False)[0]
    frames_out = pipe.video_processor.postprocess_video(frames_out, output_type=output_type)
    mel = pipe.audio_vae.decode(audio.to(pipe.audio_vae.dtype), return_dict=False)[0]
    with _vocoder_determinism(device):
        waveform = pipe.vocoder(mel)
    return frames_out, waveform
