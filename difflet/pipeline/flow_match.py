"""Flow-match schedules exactly as diffusers' FLUX and Qwen-Image pipelines build them."""

from __future__ import annotations

from typing import Any

import numpy as np


def _config_get(config: Any):
    return config.get if hasattr(config, "get") else lambda key, default: getattr(config, key, default)


def sigmas_and_mu(scheduler_config: Any, num_inference_steps: int, seq_len: int) -> tuple[np.ndarray, float]:
    """``(sigmas, mu)`` as ``FluxPipeline`` / ``QwenImagePipeline.__call__`` derive them: linear
    sigmas, and the dynamic shift for an image of ``seq_len`` packed latent tokens."""
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    steps = int(num_inference_steps)
    get = _config_get(scheduler_config)
    mu = calculate_shift(
        int(seq_len),
        get("base_image_seq_len", 256),
        get("max_image_seq_len", 4096),
        get("base_shift", 0.5),
        get("max_shift", 1.15),
    )
    return np.linspace(1.0, 1.0 / steps, steps), float(mu)


def set_timesteps(scheduler: Any, num_inference_steps: int, seq_len: int, *, device: Any = "cpu"):
    """Give ``scheduler`` the pipeline's schedule for ``seq_len`` tokens; return its timesteps.

    ``mu`` is passed only to a scheduler that shifts dynamically (``use_dynamic_shifting``), the
    way diffusers' ``retrieve_timesteps`` forwards it."""
    sigmas, mu = sigmas_and_mu(getattr(scheduler, "config", {}), num_inference_steps, seq_len)
    if _config_get(getattr(scheduler, "config", {}))("use_dynamic_shifting", False):
        scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    else:
        scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps
