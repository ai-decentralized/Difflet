"""LTX-2 hybrid pipeline orchestration.

The first M4c runtime path is intentionally narrow: cached connector/text
embeddings plus packed video/audio latents enter one dual-stream DiT call. The
host-side encoder, connector, VAE, audio VAE, and vocoder closures remain
separate follow-ups.
"""

from __future__ import annotations

import copy
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from difflet.models.ltx_2.application import LTX2DiTInputBundle
from difflet.models.ltx_2.application import LTX_2_DEFAULT_NUM_FRAMES
from difflet.models.ltx_2.application import LTX_2_DEFAULT_TEXT_SEQ_LEN


@dataclass
class LTX2PipelineOutput:
    frames: torch.Tensor
    audio: torch.Tensor
    latents: torch.Tensor
    audio_latents: torch.Tensor
    trajectory: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class LTX2Orchestrator:
    """Host-side scheduler loop for cached LTX-2 dual-stream inputs."""

    def __init__(
        self,
        *,
        model_path: str,
        transformer: Any = None,
        vae: Any = None,
        audio_vae: Any = None,
        vocoder: Any = None,
        video_processor: Any = None,
        host_pipeline: Any = None,
        dtype: torch.dtype = torch.bfloat16,
        height: int = 512,
        width: int = 768,
        num_frames: int = LTX_2_DEFAULT_NUM_FRAMES,
        text_seq_len: int = LTX_2_DEFAULT_TEXT_SEQ_LEN,
        audio_text_seq_len: int | None = None,
        audio_num_frames: int | None = None,
        frame_rate: float = 24.0,
        scheduler: Any = None,
        teacache_calibration_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.video_processor = video_processor
        self.host_pipeline = host_pipeline
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.text_seq_len = int(text_seq_len)
        self.audio_text_seq_len = int(audio_text_seq_len or text_seq_len)
        self.frame_rate = float(frame_rate)
        self.vae_temporal_compression_ratio = int(
            getattr(host_pipeline, "vae_temporal_compression_ratio", 8)
        )
        self.vae_spatial_compression_ratio = int(
            getattr(host_pipeline, "vae_spatial_compression_ratio", 32)
        )
        self.audio_sampling_rate = int(getattr(host_pipeline, "audio_sampling_rate", 16000))
        self.audio_hop_length = int(getattr(host_pipeline, "audio_hop_length", 160))
        self.audio_vae_temporal_compression_ratio = int(
            getattr(host_pipeline, "audio_vae_temporal_compression_ratio", 4)
        )
        self.audio_vae_mel_compression_ratio = int(
            getattr(host_pipeline, "audio_vae_mel_compression_ratio", 4)
        )
        audio_vae_config = getattr(audio_vae, "config", None)
        if audio_vae_config is None and host_pipeline is not None:
            audio_vae_config = getattr(getattr(host_pipeline, "audio_vae", None), "config", None)
        self.num_mel_bins = int(getattr(audio_vae_config, "mel_bins", 64))
        self.audio_num_frames = int(audio_num_frames) if audio_num_frames is not None else None
        self.scheduler = (
            scheduler
            if scheduler is not None
            else _load_scheduler(model_path, warn_if_missing=transformer is not None)
        )
        # TeaCache (cclog 87): adaptive step-skipping. Built lazily on first denoise.
        self.teacache_calibration_path = teacache_calibration_path
        self._teacache_controller = None

    def _maybe_init_teacache(self) -> bool:
        """Build the TeaCache controller once. Returns whether TeaCache is enabled."""
        if not self.teacache_calibration_path:
            return False
        if self._teacache_controller is not None:
            return True
        from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController

        calibration = TeaCacheCalibration.from_json(self.teacache_calibration_path)
        self._teacache_controller = TeaCacheController(calibration)
        return True

    def has_runtime_components(self) -> bool:
        return (
            self.transformer is not None
            or self.vae is not None
            or self.audio_vae is not None
            or self.vocoder is not None
            or self.host_pipeline is not None
        )

    @property
    def latent_num_frames(self) -> int:
        return (self.num_frames - 1) // self.vae_temporal_compression_ratio + 1

    @property
    def latent_height(self) -> int:
        return self.height // self.vae_spatial_compression_ratio

    @property
    def latent_width(self) -> int:
        return self.width // self.vae_spatial_compression_ratio

    @property
    def video_seq_len(self) -> int:
        return self.latent_num_frames * self.latent_height * self.latent_width

    @property
    def inferred_audio_num_frames(self) -> int:
        if self.audio_num_frames is not None:
            return self.audio_num_frames
        duration_s = self.num_frames / self.frame_rate
        latents_per_second = (
            self.audio_sampling_rate
            / self.audio_hop_length
            / float(self.audio_vae_temporal_compression_ratio)
        )
        return round(duration_s * latents_per_second)

    @property
    def latent_mel_bins(self) -> int:
        return self.num_mel_bins // self.audio_vae_mel_compression_ratio

    @property
    def audio_seq_len(self) -> int:
        return self.inferred_audio_num_frames

    def prepare_latents(
        self,
        *,
        batch_size: int,
        channels: int = 128,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 3:
                expected = (int(batch_size), self.video_seq_len, int(channels))
                if tuple(latents.shape) != expected:
                    raise ValueError(
                        f"Expected packed LTX-2 video latents shape {expected}, "
                        f"got {tuple(latents.shape)}"
                    )
                return latents
            if latents.ndim == 5:
                return pack_ltx_2_video_latents(latents)
            raise ValueError("LTX-2 video latents must be packed 3D or unpacked 5D tensors.")

        unpacked = torch.randn(
            (
                int(batch_size),
                int(channels),
                self.latent_num_frames,
                self.latent_height,
                self.latent_width,
            ),
            generator=generator,
            device=device,
            dtype=dtype or self.dtype,
        )
        return pack_ltx_2_video_latents(unpacked)

    def prepare_audio_latents(
        self,
        *,
        batch_size: int,
        channels: int = 8,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_dim = int(channels) * self.latent_mel_bins
        if latents is not None:
            if latents.ndim == 3:
                expected = (int(batch_size), self.audio_seq_len, feature_dim)
                if tuple(latents.shape) != expected:
                    raise ValueError(
                        f"Expected packed LTX-2 audio latents shape {expected}, "
                        f"got {tuple(latents.shape)}"
                    )
                return latents
            if latents.ndim == 4:
                return pack_ltx_2_audio_latents(latents)
            raise ValueError("LTX-2 audio latents must be packed 3D or unpacked 4D tensors.")

        unpacked = torch.randn(
            (int(batch_size), int(channels), self.audio_seq_len, self.latent_mel_bins),
            generator=generator,
            device=device,
            dtype=dtype or self.dtype,
        )
        return pack_ltx_2_audio_latents(unpacked)

    @torch.no_grad()
    def __call__(
        self,
        *,
        bundle: LTX2DiTInputBundle | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        audio_encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        batch_size: int = 1,
        video_channels: int = 128,
        audio_channels: int = 8,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        audio_guidance_scale: float | None = None,
        stg_scale: float = 0.0,
        audio_stg_scale: float | None = None,
        modality_scale: float = 1.0,
        audio_modality_scale: float | None = None,
        guidance_rescale: float = 0.0,
        audio_guidance_rescale: float | None = None,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        use_cross_timestep: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        output_type: str = "latent",
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        return_trajectory: bool = False,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
    ) -> LTX2PipelineOutput | tuple[torch.Tensor, torch.Tensor]:
        if output_type not in {"latent", "pt"}:
            raise ValueError("LTX-2 output_type currently supports only 'latent' or 'pt'.")
        audio_guidance_scale = guidance_scale if audio_guidance_scale is None else audio_guidance_scale
        audio_guidance_rescale = (
            guidance_rescale if audio_guidance_rescale is None else audio_guidance_rescale
        )
        audio_stg_scale = stg_scale if audio_stg_scale is None else audio_stg_scale
        audio_modality_scale = modality_scale if audio_modality_scale is None else audio_modality_scale
        if (float(stg_scale) > 0.0 or float(audio_stg_scale) > 0.0) and not (
            spatio_temporal_guidance_blocks
        ):
            raise ValueError(
                "LTX-2 STG requires spatio_temporal_guidance_blocks when stg_scale is nonzero."
            )
        if bundle is None:
            if encoder_hidden_states is None:
                (
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                ) = self.prepare_conditioning(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    guidance_scale=max(float(guidance_scale), float(audio_guidance_scale)),
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length or self.text_seq_len,
                )
                batch_size = _latent_batch_size_from_conditioning(
                    int(encoder_hidden_states.shape[0]),
                    guidance_scale=guidance_scale,
                    audio_guidance_scale=audio_guidance_scale,
                )
            packed_latents = self.prepare_latents(
                batch_size=batch_size,
                channels=video_channels,
                dtype=self.dtype,
                generator=generator,
                latents=latents,
            )
            packed_audio_latents = self.prepare_audio_latents(
                batch_size=batch_size,
                channels=audio_channels,
                dtype=self.dtype,
                generator=generator,
                latents=audio_latents,
            )
            if video_coords is None:
                video_coords = make_ltx_2_video_coords(
                    batch_size=packed_latents.shape[0],
                    num_frames=self.latent_num_frames,
                    height=self.latent_height,
                    width=self.latent_width,
                    device=packed_latents.device,
                    fps=self.frame_rate,
                )
            if audio_coords is None:
                audio_coords = make_ltx_2_audio_coords(
                    batch_size=packed_audio_latents.shape[0],
                    audio_num_frames=self.audio_seq_len,
                    device=packed_audio_latents.device,
                    sampling_rate=self.audio_sampling_rate,
                    hop_length=self.audio_hop_length,
                )
            bundle = _bundle_from_tensors(
                latents=packed_latents,
                audio_latents=packed_audio_latents,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
                video_coords=video_coords,
                audio_coords=audio_coords,
                dtype=self.dtype,
                text_seq_len=self.text_seq_len,
                audio_text_seq_len=self.audio_text_seq_len,
            )

        packed_latents = bundle.hidden_states
        packed_audio_latents = bundle.audio_hidden_states
        trajectory: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if return_trajectory else None
        )
        if trajectory is not None:
            trajectory.append((packed_latents.detach().cpu(), packed_audio_latents.detach().cpu()))

        if self.transformer is not None:
            if timesteps is None:
                timesteps = self._timesteps(num_inference_steps, device=packed_latents.device)
            else:
                timesteps = torch.as_tensor(timesteps, device=packed_latents.device)
                if timesteps.ndim == 0:
                    timesteps = timesteps[None]
            packed_latents, packed_audio_latents = self._denoise(
                bundle=bundle,
                timesteps=timesteps,
                trajectory=trajectory,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                stg_scale=stg_scale,
                audio_stg_scale=audio_stg_scale,
                modality_scale=modality_scale,
                audio_modality_scale=audio_modality_scale,
                guidance_rescale=guidance_rescale,
                audio_guidance_rescale=audio_guidance_rescale,
                spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                use_cross_timestep=use_cross_timestep,
                attention_kwargs=attention_kwargs,
            )

        frames = packed_latents
        audio = packed_audio_latents
        if output_type == "pt":
            frames, audio = self._decode_latents(
                packed_latents,
                packed_audio_latents,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                generator=generator,
            )

        if return_dict:
            return LTX2PipelineOutput(
                frames=frames,
                audio=audio,
                latents=packed_latents,
                audio_latents=packed_audio_latents,
                trajectory=trajectory,
            )
        return frames, audio

    def prepare_conditioning(
        self,
        *,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = LTX_2_DEFAULT_TEXT_SEQ_LEN,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.host_pipeline is None:
            raise ValueError("Missing LTX-2 DiT input: encoder_hidden_states")
        if prompt is None and prompt_embeds is None:
            raise ValueError("LTX-2 host prompt path requires prompt or prompt_embeds.")

        pipe = self.host_pipeline
        device = getattr(pipe, "_execution_device", None) or torch.device("cpu")
        encode_prompt = getattr(pipe, "encode_prompt")
        do_cfg = float(guidance_scale) > 1.0
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.dtype,
        )
        if do_cfg:
            if negative_prompt_embeds is None or negative_prompt_attention_mask is None:
                raise ValueError("LTX-2 CFG prompt encoding did not produce negative embeddings.")
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask],
                dim=0,
            )

        tokenizer_padding_side = "left"
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None:
            tokenizer_padding_side = getattr(tokenizer, "padding_side", "left")
        connectors = getattr(pipe, "connectors")
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = (
            connectors(
                prompt_embeds,
                prompt_attention_mask,
                padding_side=tokenizer_padding_side,
            )
        )
        return (
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            connector_attention_mask.to(dtype=torch.bool),
            connector_attention_mask.to(dtype=torch.bool),
        )

    def _timesteps(self, num_inference_steps: int, *, device: torch.device) -> torch.Tensor:
        num_inference_steps = max(int(num_inference_steps), 1)
        if self.scheduler is None:
            return torch.linspace(
                1.0,
                1.0 / float(num_inference_steps),
                steps=num_inference_steps,
                device=device,
                dtype=torch.float32,
            )
        if num_inference_steps < 2:
            raise ValueError(
                "LTX-2 diffusers scheduler requires num_inference_steps >= 2. "
                "The upstream FlowMatchEulerDiscreteScheduler produces non-finite "
                "timesteps for a one-step LTX-2 sigma schedule."
            )
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        timesteps, _ = _retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            "cpu",
            sigmas=sigmas,
            mu=ltx_2_scheduler_mu(getattr(self.scheduler, "config", {})),
        )
        return timesteps.to(device=device)

    def _denoise(
        self,
        *,
        bundle: LTX2DiTInputBundle,
        timesteps: torch.Tensor,
        trajectory: list[tuple[torch.Tensor, torch.Tensor]] | None,
        guidance_scale: float,
        audio_guidance_scale: float,
        stg_scale: float,
        audio_stg_scale: float,
        modality_scale: float,
        audio_modality_scale: float,
        guidance_rescale: float,
        audio_guidance_rescale: float,
        spatio_temporal_guidance_blocks: list[int] | None,
        use_cross_timestep: bool,
        attention_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = bundle.hidden_states
        audio_latents = bundle.audio_hidden_states
        audio_scheduler = copy.deepcopy(self.scheduler) if self.scheduler is not None else None
        # TeaCache: the controller tracks only the video stream; the audio residual is
        # maintained manually here. cclog 87 fix: cache the raw VELOCITY model output (what
        # the scheduler consumes), NOT the x0 prediction — caching x0 and converting back
        # via (latents-x0)/sigma amplifies the reused-x0 error at small sigma (late steps),
        # which destroyed cosine (0.27). Velocity caching matches Wan (cosine 0.985).
        ctrl = self._teacache_controller if self._maybe_init_teacache() else None
        prev_audio_vel: torch.Tensor | None = None
        cached_audio_vel_res: torch.Tensor | None = None
        for step_index, timestep in enumerate(timesteps):
            model_dtype = _component_dtype(self.transformer, self.dtype)
            do_cfg = float(guidance_scale) > 1.0 or float(audio_guidance_scale) > 1.0
            do_stg = float(stg_scale) > 0.0 or float(audio_stg_scale) > 0.0
            do_modality = float(modality_scale) > 1.0 or float(audio_modality_scale) > 1.0
            model_latents = torch.cat([latents] * 2) if do_cfg else latents
            model_audio_latents = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
            timestep_batch = _batch_timestep(
                timestep,
                model_latents.shape[0],
                latents.device,
                model_dtype,
            )
            model_bundle = LTX2DiTInputBundle(
                hidden_states=model_latents.to(dtype=model_dtype),
                audio_hidden_states=model_audio_latents.to(dtype=model_dtype),
                encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=model_dtype),
                audio_encoder_hidden_states=bundle.audio_encoder_hidden_states.to(
                    dtype=model_dtype
                ),
                timestep=timestep_batch,
                sigma=timestep_batch,
                encoder_attention_mask=bundle.encoder_attention_mask,
                audio_encoder_attention_mask=bundle.audio_encoder_attention_mask,
                video_coords=_cfg_repeat_tensor(bundle.video_coords, latents.shape[0], do_cfg),
                audio_coords=_cfg_repeat_tensor(
                    bundle.audio_coords,
                    audio_latents.shape[0],
                    do_cfg,
                ),
            )
            if do_cfg:
                _validate_cfg_batch(
                    model_bundle.encoder_hidden_states,
                    latents.shape[0],
                    "encoder_hidden_states",
                )
                _validate_cfg_batch(
                    model_bundle.audio_encoder_hidden_states,
                    audio_latents.shape[0],
                    "audio_encoder_hidden_states",
                )
                _validate_cfg_batch(
                    model_bundle.encoder_attention_mask,
                    latents.shape[0],
                    "encoder_attention_mask",
                )
                _validate_cfg_batch(
                    model_bundle.audio_encoder_attention_mask,
                    audio_latents.shape[0],
                    "audio_encoder_attention_mask",
                )
            # video_sigma / audio_sigma are needed for the scheduler step whether the
            # full DiT runs or the step is skipped, so compute them up front.
            video_sigma = _scheduler_sigma(
                self.scheduler,
                step_index,
                device=latents.device,
                dtype=latents.dtype,
            )
            audio_sigma = _scheduler_sigma(
                audio_scheduler,
                step_index,
                device=audio_latents.device,
                dtype=audio_latents.dtype,
            )

            # TeaCache skip decision. The block-0 modulated-input signal is timestep-only
            # (identical across the cond/uncond CFG halves), so the probe uses the single
            # un-doubled latent + the un-doubled per-batch timestep.
            mod_input = None
            skip = False
            if ctrl is not None:
                timestep_batch_single = _batch_timestep(
                    timestep, latents.shape[0], latents.device, model_dtype
                )
                mod_input = self.transformer.teacache_mod_input(
                    latents.to(dtype=model_dtype), timestep_batch_single
                )
                diff_norm = None
                if ctrl.prev_mod_input is not None:
                    prev = ctrl.prev_mod_input
                    cur = mod_input.detach().float().cpu()
                    denom = prev.abs().mean().clamp_min(1e-8)
                    diff_norm = float((cur - prev).abs().mean() / denom)
                skip = ctrl.should_skip(step_index, mod_input, diff_norm=diff_norm)

            if skip:
                # Reuse cached VELOCITY (raw model output) — no x0/sigma round-trip.
                noise_pred_video = ctrl.skip_noise_pred(mod_input=mod_input)
                noise_pred_audio = prev_audio_vel + cached_audio_vel_res
            else:
                noise_pred_video_x0, noise_pred_audio_x0, video_cond_x0, audio_cond_x0 = (
                    self._full_dit_step(
                        model_bundle=model_bundle,
                        latents=latents,
                        audio_latents=audio_latents,
                        video_sigma=video_sigma,
                        audio_sigma=audio_sigma,
                        do_cfg=do_cfg,
                        do_stg=do_stg,
                        do_modality=do_modality,
                        guidance_scale=guidance_scale,
                        audio_guidance_scale=audio_guidance_scale,
                        stg_scale=stg_scale,
                        audio_stg_scale=audio_stg_scale,
                        modality_scale=modality_scale,
                        audio_modality_scale=audio_modality_scale,
                        spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                        use_cross_timestep=use_cross_timestep,
                        attention_kwargs=attention_kwargs,
                    )
                )
                if float(guidance_rescale) > 0.0:
                    noise_pred_video_x0 = rescale_ltx_2_noise_cfg(
                        noise_pred_video_x0,
                        video_cond_x0,
                        guidance_rescale=float(guidance_rescale),
                    )
                if float(audio_guidance_rescale) > 0.0:
                    noise_pred_audio_x0 = rescale_ltx_2_noise_cfg(
                        noise_pred_audio_x0,
                        audio_cond_x0,
                        guidance_rescale=float(audio_guidance_rescale),
                    )
                # Convert to velocity (what the scheduler consumes) on the FULL step, then
                # cache THIS velocity — so skips reuse a velocity residual, not an x0 one.
                noise_pred_video = _convert_x0_to_velocity(latents, noise_pred_video_x0, video_sigma)
                noise_pred_audio = _convert_x0_to_velocity(audio_latents, noise_pred_audio_x0, audio_sigma)
                if ctrl is not None:
                    ctrl.record_full_step(noise_pred_video, mod_input=mod_input)
                    if prev_audio_vel is not None:
                        cached_audio_vel_res = noise_pred_audio - prev_audio_vel
            if ctrl is not None:
                prev_audio_vel = noise_pred_audio

            latents = self._scheduler_step(
                self.scheduler,
                noise_pred_video,
                timestep,
                latents,
                len(timesteps),
            )
            audio_latents = self._scheduler_step(
                audio_scheduler,
                noise_pred_audio,
                timestep,
                audio_latents,
                len(timesteps),
            )
            if trajectory is not None:
                trajectory.append((latents.detach().cpu(), audio_latents.detach().cpu()))
        if ctrl is not None:
            print(f"[ltx2-teacache] {ctrl.stats()}", flush=True)
        return latents, audio_latents

    def _full_dit_step(
        self,
        *,
        model_bundle: LTX2DiTInputBundle,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        do_cfg: bool,
        do_stg: bool,
        do_modality: bool,
        guidance_scale: float,
        audio_guidance_scale: float,
        stg_scale: float,
        audio_stg_scale: float,
        modality_scale: float,
        audio_modality_scale: float,
        spatio_temporal_guidance_blocks: list[int] | None,
        use_cross_timestep: bool,
        attention_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full DiT evaluation + CFG/STG/modality guidance for one step.

        Returns ``(noise_pred_video_x0, noise_pred_audio_x0, video_cond_x0,
        audio_cond_x0)``. The ``*_cond_x0`` baselines are returned so the caller
        can apply ``guidance_rescale`` (which references the cond prediction).
        """
        noise_pred_video, noise_pred_audio = _first_tensor_pair(self.transformer(model_bundle))
        if do_cfg:
            video_uncond, video_cond = noise_pred_video.float().chunk(2, dim=0)
            audio_uncond, audio_cond = noise_pred_audio.float().chunk(2, dim=0)
            video_uncond_x0 = _convert_velocity_to_x0(latents, video_uncond, video_sigma)
            video_cond_x0 = _convert_velocity_to_x0(latents, video_cond, video_sigma)
            audio_uncond_x0 = _convert_velocity_to_x0(audio_latents, audio_uncond, audio_sigma)
            audio_cond_x0 = _convert_velocity_to_x0(audio_latents, audio_cond, audio_sigma)
            noise_pred_video_x0 = video_cond_x0 + (float(guidance_scale) - 1.0) * (
                video_cond_x0 - video_uncond_x0
            )
            noise_pred_audio_x0 = audio_cond_x0 + (
                float(audio_guidance_scale) - 1.0
            ) * (
                audio_cond_x0 - audio_uncond_x0
            )
            positive_bundle = _positive_ltx_2_bundle(model_bundle, latents.shape[0])
        else:
            video_cond_x0 = _convert_velocity_to_x0(latents, noise_pred_video.float(), video_sigma)
            audio_cond_x0 = _convert_velocity_to_x0(
                audio_latents,
                noise_pred_audio.float(),
                audio_sigma,
            )
            noise_pred_video_x0 = video_cond_x0
            noise_pred_audio_x0 = audio_cond_x0
            positive_bundle = model_bundle

        if do_stg:
            stg_video, stg_audio = _first_tensor_pair(
                self._call_transformer_with_ltx_2_kwargs(
                    positive_bundle,
                    isolate_modalities=False,
                    spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                    perturbation_mask=None,
                    use_cross_timestep=use_cross_timestep,
                    attention_kwargs=attention_kwargs,
                )
            )
            stg_video_x0 = _convert_velocity_to_x0(latents, stg_video.float(), video_sigma)
            stg_audio_x0 = _convert_velocity_to_x0(
                audio_latents,
                stg_audio.float(),
                audio_sigma,
            )
            noise_pred_video_x0 = noise_pred_video_x0 + float(stg_scale) * (
                video_cond_x0 - stg_video_x0
            )
            noise_pred_audio_x0 = noise_pred_audio_x0 + float(audio_stg_scale) * (
                audio_cond_x0 - stg_audio_x0
            )

        if do_modality:
            modality_video, modality_audio = _first_tensor_pair(
                self._call_transformer_with_ltx_2_kwargs(
                    positive_bundle,
                    isolate_modalities=True,
                    spatio_temporal_guidance_blocks=None,
                    perturbation_mask=None,
                    use_cross_timestep=use_cross_timestep,
                    attention_kwargs=attention_kwargs,
                )
            )
            modality_video_x0 = _convert_velocity_to_x0(
                latents,
                modality_video.float(),
                video_sigma,
            )
            modality_audio_x0 = _convert_velocity_to_x0(
                audio_latents,
                modality_audio.float(),
                audio_sigma,
            )
            noise_pred_video_x0 = noise_pred_video_x0 + (float(modality_scale) - 1.0) * (
                video_cond_x0 - modality_video_x0
            )
            noise_pred_audio_x0 = noise_pred_audio_x0 + (
                float(audio_modality_scale) - 1.0
            ) * (
                audio_cond_x0 - modality_audio_x0
            )

        return noise_pred_video_x0, noise_pred_audio_x0, video_cond_x0, audio_cond_x0

    def _scheduler_step(
        self,
        scheduler: Any,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        if scheduler is None:
            return latents - noise_pred.to(dtype=latents.dtype) / float(max(num_inference_steps, 1))
        return _first_tensor(
            scheduler.step(
                noise_pred.to(dtype=latents.dtype),
                timestep,
                latents,
                return_dict=False,
            )
        )

    def _call_transformer_with_ltx_2_kwargs(
        self,
        bundle: LTX2DiTInputBundle,
        **kwargs: Any,
    ) -> Any:
        if not bool(getattr(self.transformer, "supports_ltx_2_extra_kwargs", False)):
            raise NotImplementedError(
                "LTX-2 STG/modality guidance requires a transformer that supports "
                "extra LTX-2 guidance kwargs."
            )
        return self.transformer(bundle, **kwargs)

    def _decode_latents(
        self,
        packed_latents: torch.Tensor,
        packed_audio_latents: torch.Tensor,
        *,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.vae is None or self.audio_vae is None or self.vocoder is None:
            raise ValueError(
                "LTX-2 output_type='pt' requires active VAE, audio VAE, and vocoder components."
            )
        video_latents = unpack_ltx_2_video_latents(
            packed_latents,
            num_frames=self.latent_num_frames,
            height=self.latent_height,
            width=self.latent_width,
        )
        video_latents = video_latents.to(dtype=_component_dtype(self.vae, video_latents.dtype))

        timestep = None
        vae_config = getattr(self.vae, "config", None)
        if bool(getattr(vae_config, "timestep_conditioning", False)):
            timestep = _decode_timestep_tensor(
                decode_timestep,
                batch_size=video_latents.shape[0],
                device=video_latents.device,
                dtype=video_latents.dtype,
            )
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            noise_scale = _decode_timestep_tensor(
                decode_noise_scale,
                batch_size=video_latents.shape[0],
                device=video_latents.device,
                dtype=video_latents.dtype,
            ).view(-1, 1, 1, 1, 1)
            noise = torch.randn(
                video_latents.shape,
                generator=generator,
                device=video_latents.device,
                dtype=video_latents.dtype,
            )
            video_latents = (1.0 - noise_scale) * video_latents + noise_scale * noise

        video_latents = _denormalize_ltx_2_video_latents(video_latents, self.vae)

        packed_audio_latents = _denormalize_ltx_2_audio_latents(
            packed_audio_latents,
            self.audio_vae,
        )
        audio_latents = unpack_ltx_2_audio_latents(
            packed_audio_latents,
            latent_length=self.audio_seq_len,
            num_mel_bins=self.latent_mel_bins,
        )
        audio_latents = audio_latents.to(
            dtype=_component_dtype(self.audio_vae, audio_latents.dtype)
        )
        video = _first_tensor(self.vae.decode(video_latents, timestep, return_dict=False))
        if self.video_processor is not None:
            video = self.video_processor.postprocess_video(video, output_type="pt")
        mel = _first_tensor(self.audio_vae.decode(audio_latents, return_dict=False))
        audio = _first_tensor(self.vocoder(mel))
        return video, audio


def pack_ltx_2_video_latents(
    latents: torch.Tensor,
    *,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError("LTX-2 unpacked video latents must have shape (B, C, F, H, W).")
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_size_t != 0 or height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("LTX-2 video latent dimensions must be divisible by patch sizes.")
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.flatten(4, 7).flatten(1, 3)


def unpack_ltx_2_video_latents(
    latents: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError("LTX-2 packed video latents must have shape (B, S, D).")
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size,
        int(num_frames),
        int(height),
        int(width),
        -1,
        int(patch_size_t),
        int(patch_size),
        int(patch_size),
    )
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return latents.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def pack_ltx_2_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError("LTX-2 unpacked audio latents must have shape (B, C, L, M).")
    return latents.transpose(1, 2).flatten(2, 3)


def unpack_ltx_2_audio_latents(
    latents: torch.Tensor,
    *,
    latent_length: int,
    num_mel_bins: int,
) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError("LTX-2 packed audio latents must have shape (B, S, D).")
    del latent_length
    return latents.unflatten(2, (-1, int(num_mel_bins))).transpose(1, 2)


def make_ltx_2_video_coords(
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device | str | None,
    patch_size: int = 1,
    patch_size_t: int = 1,
    scale_factors: tuple[int, int, int] = (8, 32, 32),
    causal_offset: int = 1,
    fps: float = 24.0,
) -> torch.Tensor:
    frames = torch.arange(0, num_frames, patch_size_t, dtype=torch.float32, device=device)
    rows = torch.arange(0, height, patch_size, dtype=torch.float32, device=device)
    cols = torch.arange(0, width, patch_size, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(frames, rows, cols, indexing="ij"), dim=0)
    patch_size_tensor = torch.tensor(
        (patch_size_t, patch_size, patch_size),
        dtype=grid.dtype,
        device=grid.device,
    )
    latent_coords = torch.stack(
        [grid, grid + patch_size_tensor.view(3, 1, 1, 1)],
        dim=-1,
    )
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    scale_tensor = torch.tensor(
        scale_factors,
        dtype=latent_coords.dtype,
        device=latent_coords.device,
    )
    pixel_coords = latent_coords * scale_tensor.view(1, 3, 1, 1)
    pixel_coords[:, 0, ...] = (
        pixel_coords[:, 0, ...] + int(causal_offset) - int(scale_factors[0])
    ).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return pixel_coords


def make_ltx_2_audio_coords(
    *,
    batch_size: int,
    audio_num_frames: int,
    device: torch.device | str | None,
    patch_size_t: int = 1,
    scale_factor: int = 4,
    causal_offset: int = 1,
    sampling_rate: int = 16000,
    hop_length: int = 160,
    shift: int = 0,
) -> torch.Tensor:
    coords = torch.arange(
        shift,
        audio_num_frames + shift,
        patch_size_t,
        dtype=torch.float32,
        device=device,
    )
    start_mel = (coords * scale_factor + int(causal_offset) - int(scale_factor)).clamp(min=0)
    end_mel = (
        (coords + int(patch_size_t)) * scale_factor
        + int(causal_offset)
        - int(scale_factor)
    ).clamp(min=0)
    seconds_per_mel = float(hop_length) / float(sampling_rate)
    coords = torch.stack([start_mel * seconds_per_mel, end_mel * seconds_per_mel], dim=-1)
    return coords.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)


def _bundle_from_tensors(
    *,
    latents: torch.Tensor,
    audio_latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    audio_encoder_hidden_states: torch.Tensor | None,
    encoder_attention_mask: torch.Tensor | None,
    audio_encoder_attention_mask: torch.Tensor | None,
    video_coords: torch.Tensor | None,
    audio_coords: torch.Tensor | None,
    dtype: torch.dtype,
    text_seq_len: int,
    audio_text_seq_len: int,
) -> LTX2DiTInputBundle:
    if encoder_hidden_states is None:
        raise ValueError("Missing LTX-2 DiT input: encoder_hidden_states")
    if audio_encoder_hidden_states is None:
        audio_encoder_hidden_states = encoder_hidden_states
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(
            (encoder_hidden_states.shape[0], text_seq_len),
            dtype=torch.bool,
            device=encoder_hidden_states.device,
        )
    if audio_encoder_attention_mask is None:
        audio_encoder_attention_mask = torch.ones(
            (audio_encoder_hidden_states.shape[0], audio_text_seq_len),
            dtype=torch.bool,
            device=audio_encoder_hidden_states.device,
        )
    if video_coords is None:
        video_coords = make_ltx_2_video_coords(
            batch_size=latents.shape[0],
            num_frames=latents.shape[1],
            height=1,
            width=1,
            device=latents.device,
        )
    if audio_coords is None:
        audio_coords = make_ltx_2_audio_coords(
            batch_size=audio_latents.shape[0],
            audio_num_frames=audio_latents.shape[1],
            device=audio_latents.device,
        )
    return LTX2DiTInputBundle(
        hidden_states=latents,
        audio_hidden_states=audio_latents,
        encoder_hidden_states=encoder_hidden_states,
        audio_encoder_hidden_states=audio_encoder_hidden_states,
        timestep=torch.zeros([latents.shape[0]], dtype=dtype, device=latents.device),
        sigma=torch.zeros([latents.shape[0]], dtype=dtype, device=latents.device),
        encoder_attention_mask=encoder_attention_mask,
        audio_encoder_attention_mask=audio_encoder_attention_mask,
        video_coords=video_coords,
        audio_coords=audio_coords,
    )


def _load_scheduler(model_path: str, *, warn_if_missing: bool = True):
    scheduler_path = os.path.join(model_path, "scheduler")
    if not os.path.exists(os.path.join(scheduler_path, "scheduler_config.json")):
        if warn_if_missing:
            warnings.warn(_missing_scheduler_message(model_path), RuntimeWarning, stacklevel=2)
        return None
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except ImportError:
        if warn_if_missing:
            warnings.warn(
                "LTX-2 scheduler config was found, but diffusers is not installed.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    return FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)


def ltx_2_scheduler_mu(scheduler_config: Any) -> float:
    """Match diffusers LTX2Pipeline's scheduler shift defaults."""

    def get(name: str, default: float) -> float:
        if hasattr(scheduler_config, "get"):
            return float(scheduler_config.get(name, default))
        return float(getattr(scheduler_config, name, default))

    image_seq_len = get("max_image_seq_len", 4096.0)
    base_seq_len = get("base_image_seq_len", 1024.0)
    max_seq_len = get("max_image_seq_len", 4096.0)
    base_shift = get("base_shift", 0.95)
    max_shift = get("max_shift", 2.05)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + base_shift - m * base_seq_len


def _retrieve_timesteps(scheduler: Any, num_inference_steps: int, device: str, **kwargs: Any):
    accepted = set(inspect.signature(scheduler.set_timesteps).parameters)
    set_kwargs: dict[str, Any] = {}
    if "mu" in accepted and kwargs.get("mu") is not None:
        set_kwargs["mu"] = kwargs["mu"]
    if "timesteps" in kwargs and kwargs["timesteps"] is not None:
        scheduler.set_timesteps(timesteps=kwargs["timesteps"], device=device, **set_kwargs)
    elif "sigmas" in kwargs and kwargs["sigmas"] is not None:
        scheduler.set_timesteps(sigmas=kwargs["sigmas"], device=device, **set_kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **set_kwargs)
    return scheduler.timesteps, num_inference_steps


def _missing_scheduler_message(model_path: str) -> str:
    return (
        f"LTX-2 scheduler_config.json was not found under {model_path!r}; "
        "using the fixed-shape fallback denoise step for tests/smoke only."
    )


def disable_ltx_2_xla_lazy_import() -> None:
    try:
        import diffusers.utils.import_utils as import_utils
    except ImportError:
        return
    import_utils._torch_xla_available = False


def _component_dtype(component: Any, default: torch.dtype) -> torch.dtype:
    dtype = getattr(component, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    config = getattr(component, "config", None)
    neuron_config = getattr(config, "neuron_config", None)
    dtype = getattr(neuron_config, "torch_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else default


def _denormalize_ltx_2_video_latents(latents: torch.Tensor, vae: Any) -> torch.Tensor:
    mean = getattr(vae, "latents_mean", None)
    std = getattr(vae, "latents_std", None)
    if mean is None or std is None:
        return latents
    mean = torch.as_tensor(mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    config = getattr(vae, "config", None)
    scaling_factor = float(getattr(config, "scaling_factor", 1.0))
    return latents * std / scaling_factor + mean


def _denormalize_ltx_2_audio_latents(latents: torch.Tensor, audio_vae: Any) -> torch.Tensor:
    mean = getattr(audio_vae, "latents_mean", None)
    std = getattr(audio_vae, "latents_std", None)
    if mean is None or std is None:
        return latents
    mean = torch.as_tensor(mean, device=latents.device, dtype=latents.dtype)
    std = torch.as_tensor(std, device=latents.device, dtype=latents.dtype)
    return latents * std + mean


def _decode_timestep_tensor(
    decode_timestep: float | list[float],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(decode_timestep, list):
        decode_timestep = [decode_timestep] * batch_size
    if len(decode_timestep) != batch_size:
        raise ValueError("LTX-2 decode_timestep list length must match batch size.")
    return torch.tensor(decode_timestep, device=device, dtype=dtype)


def rescale_ltx_2_noise_cfg(
    noise_cfg: torch.Tensor,
    noise_pred_text: torch.Tensor,
    *,
    guidance_rescale: float,
) -> torch.Tensor:
    dims = list(range(1, noise_cfg.ndim))
    std_text = noise_pred_text.std(dim=dims, keepdim=True)
    std_cfg = noise_cfg.std(dim=dims, keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg


def _scheduler_sigma(
    scheduler: Any,
    step_index: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if scheduler is None or not hasattr(scheduler, "sigmas"):
        return torch.tensor(1.0, device=device, dtype=dtype)
    sigmas = torch.as_tensor(getattr(scheduler, "sigmas"), device=device, dtype=dtype)
    index = min(int(step_index), sigmas.numel() - 1)
    return sigmas[index]


def _convert_velocity_to_x0(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return sample - velocity.to(dtype=sample.dtype) * sigma


def _convert_x0_to_velocity(
    sample: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return (sample - x0.to(dtype=sample.dtype)) / sigma


def _latent_batch_size_from_conditioning(
    conditioning_batch_size: int,
    *,
    guidance_scale: float,
    audio_guidance_scale: float,
) -> int:
    if float(guidance_scale) <= 1.0 and float(audio_guidance_scale) <= 1.0:
        return int(conditioning_batch_size)
    if int(conditioning_batch_size) % 2 != 0:
        raise ValueError(
            "LTX-2 CFG conditioning batch must contain negative and positive halves."
        )
    return int(conditioning_batch_size) // 2


def _cfg_repeat_tensor(tensor: torch.Tensor, latent_batch_size: int, do_cfg: bool) -> torch.Tensor:
    if not do_cfg:
        return tensor
    if tensor.shape[0] == latent_batch_size * 2:
        return tensor
    if tensor.shape[0] != latent_batch_size:
        raise ValueError(
            f"LTX-2 CFG tensor batch {tensor.shape[0]} does not match latent batch "
            f"{latent_batch_size} or doubled batch {latent_batch_size * 2}."
        )
    return tensor.repeat((2,) + (1,) * (tensor.ndim - 1))


def _validate_cfg_batch(tensor: torch.Tensor, latent_batch_size: int, name: str) -> None:
    expected = int(latent_batch_size) * 2
    if tensor.shape[0] != expected:
        raise ValueError(
            f"LTX-2 CFG {name} batch must be {expected}, got {tensor.shape[0]}."
        )


def _positive_ltx_2_bundle(bundle: LTX2DiTInputBundle, latent_batch_size: int) -> LTX2DiTInputBundle:
    return LTX2DiTInputBundle(
        hidden_states=_positive_tensor(bundle.hidden_states, latent_batch_size),
        audio_hidden_states=_positive_tensor(bundle.audio_hidden_states, latent_batch_size),
        encoder_hidden_states=_positive_tensor(bundle.encoder_hidden_states, latent_batch_size),
        audio_encoder_hidden_states=_positive_tensor(
            bundle.audio_encoder_hidden_states,
            latent_batch_size,
        ),
        timestep=_positive_tensor(bundle.timestep, latent_batch_size),
        sigma=_positive_tensor(bundle.sigma, latent_batch_size),
        encoder_attention_mask=_positive_tensor(bundle.encoder_attention_mask, latent_batch_size),
        audio_encoder_attention_mask=_positive_tensor(
            bundle.audio_encoder_attention_mask,
            latent_batch_size,
        ),
        video_coords=_positive_tensor(bundle.video_coords, latent_batch_size),
        audio_coords=_positive_tensor(bundle.audio_coords, latent_batch_size),
    )


def _positive_tensor(tensor: torch.Tensor, latent_batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == latent_batch_size * 2:
        return tensor.chunk(2, dim=0)[1]
    return tensor


def _batch_timestep(
    timestep: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    timestep = torch.as_tensor(timestep, device=device, dtype=dtype)
    if timestep.ndim == 0:
        timestep = timestep[None]
    if timestep.numel() == 1:
        timestep = timestep.expand(batch_size)
    return timestep.to(dtype=dtype)


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("sample", "audio_sample", "frames", "audio", "latents"):
            item = value.get(key)
            if torch.is_tensor(item):
                return item
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Could not extract tensor from {type(value)!r}")


def _first_tensor_pair(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, dict):
        sample = value.get("sample")
        audio_sample = value.get("audio_sample")
        if torch.is_tensor(sample) and torch.is_tensor(audio_sample):
            return sample, audio_sample
    sample = getattr(value, "sample", None)
    audio_sample = getattr(value, "audio_sample", None)
    if torch.is_tensor(sample) and torch.is_tensor(audio_sample):
        return sample, audio_sample
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        first, second = value[0], value[1]
        if torch.is_tensor(first) and torch.is_tensor(second):
            return first, second
    raise TypeError(f"Could not extract tensor pair from {type(value)!r}")
