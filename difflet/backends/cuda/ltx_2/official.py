"""Host-side sampler for the opt-in ``ltx_official_bf16`` LTX-2 profile.

The stock diffusers pipeline keeps its sampler state in float32. This driver instead reproduces
the native LTX-2 one-stage loop: the video and audio latents stay in BF16, every guidance and
Euler update is computed in float32 and rounded back at the native points, and the sigma
schedule follows the native float32 shift. Text encoding, connectors, latent packing and
decoding reuse the loaded diffusers components.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

__all__ = ["euler_step", "generate", "guided_velocity", "official_sigmas", "resolve_sigmas"]


def official_sigmas(config: Mapping[str, Any], steps: int, *, device) -> torch.Tensor:
    """Return the ``steps + 1`` shifted float32 sigma boundaries, ending at 0."""
    base_len = config.get("base_image_seq_len", 1024)
    max_len = config.get("max_image_seq_len", 4096)
    base_shift = config.get("base_shift", 0.95)
    max_shift = config.get("max_shift", 2.05)
    terminal = config.get("shift_terminal", None)

    t = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
    # The shift is the max-length end of the resolution line, whatever the actual size.
    slope = (max_shift - base_shift) / (max_len - base_len)
    mu = max_len * slope + (base_shift - slope * base_len)
    exp_mu = math.exp(mu)

    active = t != 0
    shifted = exp_mu / (exp_mu + (1 / t[active] - 1))
    if terminal is not None:
        remaining = 1 - shifted
        scale = remaining[-1] / (1 - terminal)
        shifted = 1 - remaining / scale
    sigmas = torch.zeros_like(t)
    sigmas[active] = shifted
    return sigmas.to(device)


def resolve_sigmas(
    config: Mapping[str, Any], steps: int, sigmas: Sequence[float] | None, *, device
) -> torch.Tensor:
    """Return the request's float32 sigma boundaries on ``device``.

    Without ``sigmas`` this is the official schedule. Given ``sigmas`` are final boundaries,
    never shifted again; a missing terminal 0 is appended, and the result must hold exactly one
    interval per inference step.
    """
    if sigmas is None:
        return official_sigmas(config, steps, device=device)
    if not isinstance(sigmas, (list, tuple)) or any(isinstance(v, bool) for v in sigmas):
        raise ValueError("LTX-2 sigmas must be a list or tuple of numbers")
    try:
        values = torch.tensor(sigmas, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"LTX-2 sigmas must be a flat sequence of numbers: {error}") from error
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("LTX-2 sigmas must be a flat sequence of at least two values")
    if not torch.isfinite(values).all():
        raise ValueError("LTX-2 sigmas must be finite")
    if values[-1] != 0:
        values = torch.cat((values, values.new_zeros(1)))
    if not (values[:-1] > 0).all():
        raise ValueError("LTX-2 sigmas must be positive before the terminal 0")
    if not (values[1:] < values[:-1]).all():
        raise ValueError("LTX-2 sigmas must be strictly decreasing")
    if values.numel() != steps + 1:
        raise ValueError(
            f"LTX-2 sigmas give {values.numel() - 1} intervals but num_inference_steps={steps}"
        )
    return values.to(device)


def guided_velocity(
    sample: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor | None,
    sigma: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return the classifier-free-guided velocity, guiding in the denoised (x0) space."""
    dtype = sample.dtype
    s = sigma.float()
    state = sample.float()
    denoised = (state - positive.float() * s).to(dtype).contiguous()
    if negative is not None:
        uncond = (state - negative.float() * s).to(dtype).contiguous()
        cond = denoised.float()
        denoised = (cond + (scale - 1) * (cond - uncond.float())).to(dtype)
    # A host-scalar divisor, not a tensor one: the two take different kernel paths.
    return ((state - denoised.float()) / s.item()).to(dtype)


def euler_step(
    sample: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor
) -> torch.Tensor:
    """Advance ``sample`` from ``sigma`` to ``sigma_next`` in float32, kept in its own dtype."""
    step = sigma_next.float() - sigma.float()
    return (sample.float() + velocity.float() * step).to(sample.dtype)


def _encode_texts(pipe, texts: list[str], max_len: int, device: torch.device):
    tokenizer, encoder = pipe.tokenizer, pipe.text_encoder
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("LTX-2 prompt encoding requires a tokenizer with a BOS token")
    rows = []
    for text in texts:
        ids = list(
            tokenizer(
                text.strip(),
                padding=False,
                truncation=True,
                max_length=max_len,
                return_tensors=None,
            ).input_ids
        )
        if not ids or ids[0] != bos:
            ids = [bos, *ids][:max_len]
        rows.append(ids)
    batch = tokenizer.pad(
        {"input_ids": rows},
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = batch.input_ids.to(device)
    mask = batch.attention_mask.to(device)
    output = encoder(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
    # Every hidden state, embeddings included, with the state index fastest per feature.
    states = torch.stack(tuple(output.hidden_states), dim=-1)
    return states.flatten(-2).to(encoder.dtype), mask


def _conditioning(
    pipe,
    *,
    prompt,
    negative_prompt,
    prompt_embeds,
    prompt_attention_mask,
    negative_prompt_embeds,
    negative_prompt_attention_mask,
    do_cfg: bool,
    max_len: int,
    device: torch.device,
):
    positive_texts = [prompt] if isinstance(prompt, str) else prompt
    negative_texts = (
        negative_prompt if isinstance(negative_prompt, list) else [negative_prompt or ""]
    )
    if do_cfg and prompt_embeds is None and negative_prompt_embeds is None:
        # The native encode runs both prompts as one Gemma batch, and batch size can change
        # which kernels the encoder uses.
        embeds, mask = _encode_texts(pipe, [*positive_texts, *negative_texts], max_len, device)
        prompt_embeds, prompt_attention_mask = embeds[:1], mask[:1]
        negative_prompt_embeds, negative_prompt_attention_mask = embeds[1:], mask[1:]
    else:
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = _encode_texts(
                pipe, positive_texts, max_len, device
            )
        if do_cfg and negative_prompt_embeds is None:
            negative_prompt_embeds, negative_prompt_attention_mask = _encode_texts(
                pipe, negative_texts, max_len, device
            )

    dtype = pipe.transformer.dtype

    def connect(embeds, mask):
        video, audio, _ = pipe.connectors(
            embeds.to(device=device, dtype=dtype), mask.to(device), padding_side="left"
        )
        return video, audio

    # The connectors run once per prompt, never on a merged CFG batch.
    video, audio = connect(prompt_embeds, prompt_attention_mask)
    if do_cfg:
        negative_video, negative_audio = connect(
            negative_prompt_embeds, negative_prompt_attention_mask
        )
        video = torch.cat((video, negative_video))
        audio = torch.cat((audio, negative_audio))
    return video, audio


def _noised(value: torch.Tensor, generator) -> torch.Tensor:
    noise = randn_tensor(value.shape, generator=generator, device=value.device, dtype=value.dtype)
    # Zero-weight noise still consumes the request RNG, keeping later draws aligned.
    return torch.lerp(value.float(), noise.float(), 0.0).to(value.dtype)


def _initial_video(pipe, latents, *, tokens, channels, generator, device, dtype):
    if latents is None:
        video = randn_tensor((1, tokens, channels), generator=generator, device=device, dtype=dtype)
    else:
        video = latents.to(device)
        if video.ndim == 5:
            vae = pipe.vae
            video = pipe._normalize_latents(
                video, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
            )
            video = pipe._pack_latents(
                video, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
            )
        video = _noised(video, generator).to(dtype)
    # Channels-major storage, as the native sampler holds it: same values, different strides.
    return video.transpose(1, 2).contiguous().transpose(1, 2)


def _initial_audio(pipe, latents, *, frames, features, generator, device, dtype):
    if latents is None:
        return randn_tensor((1, frames, features), generator=generator, device=device, dtype=dtype)
    audio = latents.to(device)
    if audio.ndim == 4:
        audio = pipe._pack_audio_latents(audio)
    audio = pipe._normalize_audio_latents(
        audio, pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
    )
    return _noised(audio, generator).to(dtype)


def _vocode(vocoder, mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        return vocoder(mel)
    previous = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True
    try:
        return vocoder(mel)
    finally:
        torch.backends.cudnn.deterministic = previous


def _split_guidance(output: torch.Tensor, batch: int, do_cfg: bool):
    parts = output.chunk(batch)
    return parts[0], parts[1] if do_cfg else None


@torch.no_grad()
def generate(
    pipe,
    *,
    prompt,
    negative_prompt,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    guidance_scale: float,
    generator,
    latents: torch.Tensor | None,
    audio_latents: torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
    negative_prompt_embeds: torch.Tensor | None,
    prompt_attention_mask: torch.Tensor | None,
    negative_prompt_attention_mask: torch.Tensor | None,
    output_type: str,
    max_sequence_length: int,
    device,
    sigmas: Sequence[float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one joint video/audio request through the BF16 profile.

    Returns the denormalized ``(video, audio)`` latents for ``output_type="latent"``, otherwise
    the post-processed frames and the vocoder waveform.
    """
    device = torch.device(device)
    # Validated before any conditioning work or RNG draw.
    bounds = resolve_sigmas(pipe.scheduler.config, num_inference_steps, sigmas, device=device)
    do_cfg = not math.isclose(guidance_scale, 1.0)
    video_context, audio_context = _conditioning(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        do_cfg=do_cfg,
        max_len=max_sequence_length,
        device=device,
    )

    latent_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio
    patch = pipe.transformer_spatial_patch_size
    patch_t = pipe.transformer_temporal_patch_size
    tokens = (latent_frames // patch_t) * (latent_height // patch) * (latent_width // patch)
    channels = pipe.transformer.config.in_channels * patch_t * patch * patch
    audio_frames = round(
        num_frames
        / frame_rate
        * pipe.audio_sampling_rate
        / pipe.audio_hop_length
        / float(pipe.audio_vae_temporal_compression_ratio)
    )
    mel_bins = pipe.audio_vae.config.mel_bins // pipe.audio_vae_mel_compression_ratio
    dtype = pipe.transformer.dtype
    video = _initial_video(
        pipe,
        latents,
        tokens=tokens,
        channels=channels,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    audio = _initial_audio(
        pipe,
        audio_latents,
        frames=audio_frames,
        features=pipe.audio_vae.config.latent_channels * mel_bins,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    batch = 2 if do_cfg else 1
    video_coords = pipe.transformer.rope.prepare_video_coords(
        batch, latent_frames, latent_height, latent_width, device, fps=frame_rate
    )
    audio_coords = pipe.transformer.audio_rope.prepare_audio_coords(batch, audio_frames, device)
    train_timesteps = pipe.scheduler.config.get("num_train_timesteps", 1000)

    for i in range(num_inference_steps):
        t = (bounds[i] * train_timesteps).expand(batch)
        with pipe.transformer.cache_context("guided"):
            video_out, audio_out = pipe.transformer(
                hidden_states=video.repeat(batch, 1, 1) if do_cfg else video,
                audio_hidden_states=audio.repeat(batch, 1, 1) if do_cfg else audio,
                encoder_hidden_states=video_context,
                audio_encoder_hidden_states=audio_context,
                timestep=t.reshape(-1, 1).expand(-1, tokens),
                audio_timestep=t.reshape(-1, 1).expand(-1, audio_frames),
                sigma=t,
                audio_sigma=t,
                encoder_attention_mask=None,
                audio_encoder_attention_mask=None,
                num_frames=latent_frames,
                height=latent_height,
                width=latent_width,
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
        video_positive, video_negative = _split_guidance(video_out, batch, do_cfg)
        audio_positive, audio_negative = _split_guidance(audio_out, batch, do_cfg)
        video_velocity = guided_velocity(
            video, video_positive, video_negative, bounds[i], guidance_scale
        )
        audio_velocity = guided_velocity(
            audio, audio_positive, audio_negative, bounds[i], guidance_scale
        )
        video = euler_step(video, video_velocity, bounds[i], bounds[i + 1])
        audio = euler_step(audio, audio_velocity, bounds[i], bounds[i + 1])

    vae, audio_vae = pipe.vae, pipe.audio_vae
    video = pipe._unpack_latents(video, latent_frames, latent_height, latent_width, patch, patch_t)
    video = pipe._denormalize_latents(
        video, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
    )
    audio = pipe._denormalize_audio_latents(audio, audio_vae.latents_mean, audio_vae.latents_std)
    audio = pipe._unpack_audio_latents(audio, audio_frames, num_mel_bins=mel_bins)
    if output_type == "latent":
        return video, audio

    if vae.config.timestep_conditioning:
        # The decode timestep and noise scale are zero; the draw only advances the RNG.
        noise = randn_tensor(video.shape, generator=generator, device=device, dtype=video.dtype)
        decode_timestep = torch.zeros(1, device=device, dtype=video.dtype)
        video = video + decode_timestep[:, None, None, None, None] * noise
    else:
        decode_timestep = None
    frames = vae.decode(video.to(vae.dtype), decode_timestep, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(frames, output_type=output_type)
    mel = audio_vae.decode(audio.to(audio_vae.dtype), return_dict=False)[0]
    return frames, _vocode(pipe.vocoder, mel, device)
