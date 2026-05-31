"""Qwen-Image hybrid pipeline orchestration.

The first M4a runtime path keeps Qwen2.5-VL prompt encoding, scheduler setup,
and VAE decode on the host side. The Trainium boundary is the packed-latent
DiT call represented by ``QwenImageDiTInputBundle``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nova.models.qwen_image.application import QwenImageDiTInputBundle


@dataclass
class QwenImagePipelineOutput:
    images: torch.Tensor
    latents: torch.Tensor
    trajectory: list[torch.Tensor] | None = None


class QwenImageOrchestrator:
    """Host-side scheduler loop for cached Qwen-Image embeddings and packed latents."""

    def __init__(
        self,
        *,
        model_path: str,
        transformer: Any = None,
        vae: Any = None,
        dtype: torch.dtype = torch.bfloat16,
        height: int = 1024,
        width: int = 1024,
        text_seq_len: int = 1024,
        scheduler: Any = None,
        teacache_speedup: float | None = None,
        teacache_calibration_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.transformer = transformer
        self.vae = vae
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)
        self.text_seq_len = int(text_seq_len)
        self.vae_scale_factor = 8
        self.scheduler = (
            scheduler
            if scheduler is not None
            else _load_scheduler(model_path, warn_if_missing=transformer is not None)
        )
        self.teacache_speedup = teacache_speedup
        self.teacache_controller = None
        if teacache_speedup is not None:
            from nova.pipeline.teacache import (
                TeaCacheController,
                load_teacache_calibration_or_raise,
            )

            shape_label = _teacache_shape_label(height=self.height, width=self.width)
            calibration = load_teacache_calibration_or_raise(
                teacache_calibration_path,
                model="qwen_image",
                shape_label=shape_label,
            )
            if (
                calibration.target_speedup is not None
                and float(teacache_speedup) > float(calibration.target_speedup) + 1e-6
            ):
                raise ValueError(
                    "TeaCache calibration target speedup is lower than requested: "
                    f"requested {teacache_speedup}, calibration has "
                    f"{calibration.target_speedup}."
                )
            self.teacache_controller = TeaCacheController(calibration)

    def has_runtime_components(self) -> bool:
        return self.transformer is not None or self.vae is not None

    @property
    def latent_height(self) -> int:
        return 2 * (self.height // (self.vae_scale_factor * 2))

    @property
    def latent_width(self) -> int:
        return 2 * (self.width // (self.vae_scale_factor * 2))

    @property
    def packed_seq_len(self) -> int:
        return (self.latent_height // 2) * (self.latent_width // 2)

    def prepare_latents(
        self,
        *,
        batch_size: int,
        channels: int = 16,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 3:
                expected = (int(batch_size), self.packed_seq_len, int(channels) * 4)
                if tuple(latents.shape) != expected:
                    raise ValueError(f"Expected packed latents shape {expected}, got {tuple(latents.shape)}")
                return latents
            if latents.ndim == 5:
                return pack_qwen_image_latents(latents)
            raise ValueError("Qwen-Image latents must be packed 3D or unpacked 5D tensors.")

        latent_shape = (
            int(batch_size),
            1,
            int(channels),
            self.latent_height,
            self.latent_width,
        )
        unpacked = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=dtype or self.dtype,
        )
        return pack_qwen_image_latents(unpacked)

    @torch.no_grad()
    def __call__(
        self,
        *,
        bundle: QwenImageDiTInputBundle | None = None,
        latents: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        batch_size: int = 1,
        channels: int = 16,
        num_inference_steps: int = 4,
        guidance_scale: float | None = None,
        output_type: str = "latent",
        generator: torch.Generator | None = None,
        return_dict: bool = True,
        return_trajectory: bool = False,
    ) -> QwenImagePipelineOutput | tuple[torch.Tensor]:
        if output_type not in {"latent", "pt"}:
            raise ValueError("Qwen-Image output_type currently supports only 'latent' or 'pt'.")
        if bundle is None:
            packed_latents = self.prepare_latents(
                batch_size=batch_size,
                channels=channels,
                dtype=self.dtype,
                generator=generator,
                latents=latents,
            )
            bundle = _bundle_from_tensors(
                latents=packed_latents,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                guidance=guidance,
                guidance_scale=guidance_scale,
                dtype=self.dtype,
            )

        packed_latents = bundle.hidden_states
        trajectory: list[torch.Tensor] | None = [] if return_trajectory else None
        if trajectory is not None:
            trajectory.append(packed_latents.detach().cpu())

        if self.transformer is not None:
            if timesteps is None:
                timesteps = self._timesteps(num_inference_steps, device=packed_latents.device)
            else:
                timesteps = torch.as_tensor(timesteps, device=packed_latents.device)
                if timesteps.ndim == 0:
                    timesteps = timesteps[None]
            packed_latents = self._denoise(bundle=bundle, timesteps=timesteps, trajectory=trajectory)

        images = packed_latents
        if output_type == "pt":
            images = self._decode_latents(packed_latents)

        if return_dict:
            return QwenImagePipelineOutput(
                images=images,
                latents=packed_latents,
                trajectory=trajectory,
            )
        return (images,)

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
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        timesteps, _ = _retrieve_timesteps(self.scheduler, num_inference_steps, "cpu", sigmas=sigmas)
        return timesteps.to(device=device)

    def _denoise(
        self,
        *,
        bundle: QwenImageDiTInputBundle,
        timesteps: torch.Tensor,
        trajectory: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        latents = bundle.hidden_states
        # fused-A (cclog 81): prev_mod is a persistent on-device Parameter; the
        # probe returns only the scalar delta — no host prev_mod copy. Mirrors
        # the HunyuanVideo fused branch (cclog 80).
        fused_probe = (
            self.teacache_controller is not None
            and getattr(self.transformer, "teacache_probe_fused", False)
            and hasattr(self.transformer, "teacache_delta")
        )
        for step_index, timestep in enumerate(timesteps):
            model_dtype = _component_dtype(self.transformer, self.dtype)
            timestep_batch = _batch_timestep(timestep, latents.shape[0], latents.device, model_dtype)
            model_bundle = QwenImageDiTInputBundle(
                hidden_states=latents.to(dtype=model_dtype),
                # The diffusers QwenImage pipeline feeds timestep/1000 to the DiT
                # (sigma in [0,1]); the scheduler keeps the raw timestep. Passing
                # the raw 0-1000 timestep makes the AdaLN modulation swing wildly
                # step-to-step (and inflates the TeaCache rel-L1 signal ~30x).
                timestep=(timestep_batch / 1000.0).to(dtype=model_dtype),
                encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=model_dtype),
                encoder_hidden_states_mask=bundle.encoder_hidden_states_mask,
                guidance=bundle.guidance.to(dtype=model_dtype),
            )
            delta_scalar: float | None = None
            if fused_probe and self.teacache_controller.needs_signal():
                # prev_mod persists on device; probe returns only delta. The
                # garbage step-0 delta (zero prev_mod) is absorbed by warmup.
                # Skipped entirely in fixed-cadence mode (probe-free, cclog 84).
                delta_t = self.transformer.teacache_delta(model_bundle)
                delta_scalar = float(delta_t.detach().cpu().item())
            if (
                self.teacache_controller is not None
                and self.teacache_controller.should_skip(
                    step_index,
                    None,
                    diff_norm=delta_scalar,
                )
            ):
                noise_pred = self.teacache_controller.skip_noise_pred(None)
            else:
                noise_pred = _first_tensor(self.transformer(model_bundle))
                if self.teacache_controller is not None:
                    self.teacache_controller.record_full_step(noise_pred, None)
            latents = self._scheduler_step(noise_pred, timestep, latents, len(timesteps))
            if trajectory is not None:
                trajectory.append(latents.detach().cpu())
        return latents

    def _scheduler_step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        if self.scheduler is None:
            return latents - noise_pred.to(dtype=latents.dtype) / float(max(num_inference_steps, 1))
        return _first_tensor(
            self.scheduler.step(
                noise_pred.to(dtype=latents.dtype),
                timestep,
                latents,
                return_dict=False,
            )
        )

    def _decode_latents(self, packed_latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("Qwen-Image output_type='pt' requires an active VAE decoder.")
        latents = unpack_qwen_image_latents(
            packed_latents,
            height=self.height,
            width=self.width,
            vae_scale_factor=self.vae_scale_factor,
        )
        dtype = _component_dtype(self.vae, self.dtype)
        config = _component_config(self.vae)
        latents_mean = getattr(config, "latents_mean", None)
        latents_std = getattr(config, "latents_std", None)
        if latents_mean is not None and latents_std is not None:
            mean = torch.tensor(latents_mean, dtype=dtype, device=latents.device).view(1, -1, 1, 1, 1)
            std = torch.tensor(latents_std, dtype=dtype, device=latents.device).view(1, -1, 1, 1, 1)
            latents = latents.to(dtype=dtype) * std + mean
        decode = getattr(self.vae, "decode", None)
        if decode is None:
            return _first_tensor(self.vae(latents.to(dtype=dtype)))
        return _first_tensor(decode(latents.to(dtype=dtype), return_dict=False))


def pack_qwen_image_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError("Qwen-Image unpacked latents must have shape (B, 1, C, H, W).")
    batch_size, frames, channels, height, width = latents.shape
    if frames != 1:
        raise ValueError("Qwen-Image image latents must have exactly one latent frame.")
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Qwen-Image latent height/width must be divisible by 2.")
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def unpack_qwen_image_latents(
    latents: torch.Tensor,
    *,
    height: int,
    width: int,
    vae_scale_factor: int,
) -> torch.Tensor:
    batch_size, _num_patches, channels = latents.shape
    height = 2 * (int(height) // (int(vae_scale_factor) * 2))
    width = 2 * (int(width) // (int(vae_scale_factor) * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, 1, height, width)


def _bundle_from_tensors(
    *,
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    encoder_hidden_states_mask: torch.Tensor | None,
    guidance: torch.Tensor | None,
    guidance_scale: float | None,
    dtype: torch.dtype,
) -> QwenImageDiTInputBundle:
    if encoder_hidden_states is None:
        raise ValueError("Missing Qwen-Image DiT input: encoder_hidden_states")
    if encoder_hidden_states_mask is None:
        encoder_hidden_states_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.bool,
            device=encoder_hidden_states.device,
        )
    if guidance is None:
        guidance = torch.zeros(latents.shape[0], dtype=dtype, device=latents.device)
        if guidance_scale is not None:
            guidance.fill_(float(guidance_scale))
    return QwenImageDiTInputBundle(
        hidden_states=latents,
        timestep=torch.zeros([latents.shape[0]], dtype=dtype, device=latents.device),
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        guidance=guidance,
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
                "Qwen-Image scheduler config was found, but diffusers is not installed.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    return FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)


def _retrieve_timesteps(scheduler: Any, num_inference_steps: int, device: str, **kwargs: Any):
    if "sigmas" in kwargs:
        scheduler.set_timesteps(sigmas=kwargs["sigmas"], device=device)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler.timesteps, num_inference_steps


def _missing_scheduler_message(model_path: str) -> str:
    return (
        f"Qwen-Image scheduler_config.json was not found under {model_path!r}; "
        "using the fixed-shape fallback denoise step for tests/smoke only."
    )


def _teacache_shape_label(*, height: int, width: int) -> str:
    return f"{int(height)}x{int(width)}"


def _component_dtype(component: Any, default: torch.dtype) -> torch.dtype:
    dtype = getattr(component, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    config = _component_config(component)
    neuron_config = getattr(config, "neuron_config", None)
    dtype = getattr(neuron_config, "torch_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else default


def _component_config(component: Any) -> Any:
    return getattr(component, "config", None)


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
        for key in ("sample", "images", "frames", "latents"):
            item = value.get(key)
            if torch.is_tensor(item):
                return item
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Could not extract tensor from {type(value)!r}")
