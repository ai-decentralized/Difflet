"""HunyuanVideo hybrid pipeline orchestration.

M3 v0 keeps Llama3, CLIP, scheduler setup, and VAE decode on the host side.
The Trainium boundary is the DiT backbone call represented by
``HunyuanVideoDiTInputBundle``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nova.models.hunyuan_video.application import HunyuanVideoDiTInputBundle


@dataclass
class HunyuanVideoPipelineOutput:
    frames: torch.Tensor
    latents: torch.Tensor
    trajectory: list[torch.Tensor] | None = None


class HunyuanVideoOrchestrator:
    """Host-side scheduler loop for the HunyuanVideo M3 hybrid path."""

    def __init__(
        self,
        *,
        model_path: str,
        transformer: Any = None,
        vae: Any = None,
        dtype: torch.dtype = torch.bfloat16,
        height: int = 320,
        width: int = 512,
        num_frames: int = 61,
        scheduler: Any = None,
    ) -> None:
        self.model_path = model_path
        self.transformer = transformer
        self.vae = vae
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.scheduler = scheduler if scheduler is not None else _load_scheduler(model_path)

    def has_runtime_components(self) -> bool:
        return self.transformer is not None or self.vae is not None

    @torch.no_grad()
    def __call__(
        self,
        *,
        bundle: HunyuanVideoDiTInputBundle | None = None,
        latents: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 6.0,
        output_type: str = "latent",
        return_dict: bool = True,
        return_trajectory: bool = False,
    ) -> HunyuanVideoPipelineOutput | tuple[torch.Tensor]:
        if output_type not in {"latent", "pt"}:
            raise ValueError("HunyuanVideo output_type currently supports only 'latent' or 'pt'.")
        if bundle is None:
            bundle = _bundle_from_tensors(
                latents=latents,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                pooled_projections=pooled_projections,
                guidance=guidance,
                guidance_scale=guidance_scale,
                dtype=self.dtype,
            )

        latents = bundle.hidden_states
        trajectory: list[torch.Tensor] | None = [] if return_trajectory else None
        if trajectory is not None:
            trajectory.append(latents.detach().cpu())

        if self.transformer is not None:
            if timesteps is None:
                timesteps = self._timesteps(num_inference_steps, device=latents.device)
            else:
                timesteps = self._prepare_explicit_timesteps(timesteps, device=latents.device)
            latents = self._denoise(bundle=bundle, timesteps=timesteps, trajectory=trajectory)

        frames = latents
        if output_type == "pt":
            frames = self._decode_latents(latents)

        if return_dict:
            return HunyuanVideoPipelineOutput(frames=frames, latents=latents, trajectory=trajectory)
        return (frames,)

    def _timesteps(self, num_inference_steps: int, *, device: torch.device) -> torch.Tensor:
        num_inference_steps = max(int(num_inference_steps), 1)
        if self.scheduler is None:
            return torch.linspace(1000.0, 0.0, steps=num_inference_steps, device=device)
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, _ = _retrieve_timesteps(self.scheduler, num_inference_steps, "cpu", sigmas=sigmas)
        return timesteps.to(device=device)

    def _prepare_explicit_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = torch.as_tensor(timesteps, device=device)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if self.scheduler is None:
            return timesteps

        scheduler_timesteps = self._timesteps(int(timesteps.numel()), device=device)
        if scheduler_timesteps.shape != timesteps.shape or not torch.allclose(
            scheduler_timesteps.to(dtype=timesteps.dtype).float(),
            timesteps.float(),
            atol=1e-3,
            rtol=1e-4,
        ):
            raise ValueError(
                "Explicit HunyuanVideo timesteps do not match the scheduler timesteps "
                "derived from the M3 linear sigma schedule."
            )
        return scheduler_timesteps

    def _denoise(
        self,
        *,
        bundle: HunyuanVideoDiTInputBundle,
        timesteps: torch.Tensor,
        trajectory: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        latents = bundle.hidden_states
        for timestep in timesteps:
            model_dtype = _component_dtype(self.transformer, self.dtype)
            timestep_batch = _batch_timestep(
                timestep,
                batch_size=latents.shape[0],
                device=latents.device,
                dtype=model_dtype,
            )
            model_bundle = HunyuanVideoDiTInputBundle(
                hidden_states=latents.to(dtype=model_dtype),
                timestep=timestep_batch,
                encoder_hidden_states=bundle.encoder_hidden_states.to(dtype=model_dtype),
                encoder_attention_mask=bundle.encoder_attention_mask,
                pooled_projections=bundle.pooled_projections.to(dtype=model_dtype),
                guidance=bundle.guidance.to(dtype=model_dtype),
            )
            noise_pred = _first_tensor(self.transformer(model_bundle))
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
            step = noise_pred.to(dtype=latents.dtype) / float(max(num_inference_steps, 1))
            return latents - step
        return _first_tensor(
            self.scheduler.step(
                noise_pred.to(dtype=latents.dtype),
                timestep,
                latents,
                return_dict=False,
            )
        )

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            self.vae = _load_vae(self.model_path, self.dtype)
        dtype = _component_dtype(self.vae, self.dtype)
        config = _component_config(self.vae)
        scaling_factor = float(getattr(config, "scaling_factor", 1.0))
        latents = latents.to(dtype=dtype) / scaling_factor
        decode = getattr(self.vae, "decode", None)
        if decode is None:
            return _first_tensor(self.vae(latents))
        return _first_tensor(decode(latents, return_dict=False))


def _bundle_from_tensors(
    *,
    latents: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor | None,
    encoder_attention_mask: torch.Tensor | None,
    pooled_projections: torch.Tensor | None,
    guidance: torch.Tensor | None,
    guidance_scale: float,
    dtype: torch.dtype,
) -> HunyuanVideoDiTInputBundle:
    missing = [
        name
        for name, value in (
            ("latents", latents),
            ("encoder_hidden_states", encoder_hidden_states),
            ("encoder_attention_mask", encoder_attention_mask),
            ("pooled_projections", pooled_projections),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing HunyuanVideo DiT inputs: {', '.join(missing)}")
    assert latents is not None
    if guidance is None:
        guidance = torch.full(
            [latents.shape[0]],
            float(guidance_scale) * 1000.0,
            dtype=dtype,
            device=latents.device,
        )
    return HunyuanVideoDiTInputBundle(
        hidden_states=latents,
        timestep=torch.zeros([latents.shape[0]], dtype=dtype, device=latents.device),
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        pooled_projections=pooled_projections,
        guidance=guidance,
    )


def _load_scheduler(model_path: str):
    scheduler_path = os.path.join(model_path, "scheduler")
    if not os.path.exists(os.path.join(scheduler_path, "scheduler_config.json")):
        return None
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except ImportError:
        return None
    return FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)


def _load_vae(model_path: str, dtype: torch.dtype):
    vae_path = os.path.join(model_path, "vae")
    if not os.path.exists(os.path.join(vae_path, "config.json")):
        raise ValueError(
            "HunyuanVideo output_type='pt' requires a VAE decoder or model_path/vae."
        )
    try:
        from diffusers import AutoencoderKLHunyuanVideo
    except ImportError as exc:
        raise RuntimeError("HunyuanVideo HF VAE decode requires diffusers.") from exc
    vae = AutoencoderKLHunyuanVideo.from_pretrained(vae_path, torch_dtype=dtype).eval()
    enable_tiling = getattr(vae, "enable_tiling", None)
    if enable_tiling is not None:
        enable_tiling()
    return vae


def _retrieve_timesteps(
    scheduler,
    num_inference_steps: int,
    device: str | torch.device | None,
    *,
    sigmas,
):
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps, num_inference_steps


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        if "sample" in value:
            return value["sample"]
        return _first_tensor(value[next(iter(value))])
    if isinstance(value, (tuple, list)):
        return _first_tensor(value[0])
    if hasattr(value, "sample"):
        return value.sample
    raise TypeError(f"Expected tensor-like HunyuanVideo component output, got {type(value)!r}")


def _component_dtype(component: Any, fallback: torch.dtype) -> torch.dtype:
    dtype = getattr(component, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    config = _component_config(component)
    neuron_config = getattr(config, "neuron_config", None)
    dtype = getattr(neuron_config, "torch_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else fallback


def _component_config(component: Any) -> Any:
    return getattr(component, "config", None)


def _batch_timestep(
    timestep: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor(timestep, device=device)
    timestep = timestep.to(device=device, dtype=dtype)
    if timestep.ndim == 0:
        return timestep.expand(batch_size)
    return timestep
