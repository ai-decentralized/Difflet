"""HunyuanVideo 1.5 text-to-video on the neuron backend: the denoising loop around our DiT.

The loop is diffusers' HunyuanVideo15Pipeline.__call__ for text-to-video, step for step, with its
own helpers (``retrieve_timesteps`` on the snapshot's scheduler, ``randn_tensor`` for the noise):
sigmas ``linspace(1, 0, steps + 1)[:-1]``; latents ``[B, C, (F-1)/4+1, H/16, W/16]`` in the DiT's
dtype; conditioning latents and mask channel zero; image embeddings zero; each step the DiT on
``cat([latents, 0, 0])`` with ``timestep = t``, classifier-free guidance as the snapshot's guider
config defines it (``ClassifierFreeGuidance`` with ``use_original_formulation=False``):

    noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

then ``scheduler.step``. Only how the two passes run differs: one call on a batch of 2 under CFG
parallelism (each cfg group computes one branch), two calls otherwise. Guider settings this loop
does not implement (guidance rescale, the original formulation, a partial start/stop) are refused.

TeaCache: the shared controller acts on the guided prediction, so a skipped step skips both guidance
passes and extrapolates the last full prediction by its last change. Probe-free modes (fixed cadence,
online delta) decide on the host; the adaptive mode, given a calibration, decides from the probe's
signal (block 0's timestep-only modulated input, as the Trainium probe computes it), measured every
step before the decision. Every rank runs the loop on the same inputs and (all-gathered) predictions,
so every rank makes the same decisions.

Text embeddings are precomputed (the host encoder stage); the result is latents for the VAE stage.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from difflet.backends.neuron.hunyuan_video15.transformer import (
    VAE_SPATIAL,
    VAE_TEMPORAL,
    NeuronHunyuanVideo15TransformerApplication,
)


def load_guidance(model_path: str, guidance_scale: float | None = None) -> float | None:
    """The snapshot's classifier-free guidance scale (None: disabled), refusing variants the loop
    does not implement. ``guidance_scale`` overrides the configured scale."""
    with open(os.path.join(model_path, "guider", "guider_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("_class_name") != "ClassifierFreeGuidance":
        raise NotImplementedError(f"guider {cfg.get('_class_name')!r} is not supported")
    if not cfg.get("enabled", True):
        return None
    unsupported = {k: cfg.get(k) for k, default in (("guidance_rescale", 0.0), ("use_original_formulation", False),
                                                    ("start", 0.0), ("stop", 1.0)) if cfg.get(k, default) != default}
    if unsupported:
        raise NotImplementedError(f"guider settings not implemented: {unsupported}")
    scale = float(cfg["guidance_scale"] if guidance_scale is None else guidance_scale)
    return scale if scale > 1.0 else None  # CFG at scale <= 1 is a single conditional pass


class TorchNeuronHunyuanVideo15Pipeline:
    """``load()``, then ``__call__(prompt_embeds, ..., negative_prompt_embeds, ...) -> latents``."""

    def __init__(self, *, model_path, parallel=None, height: int = 480, width: int = 848, num_frames: int = 121,
                 dtype=torch.bfloat16, exec_mode: str | None = None, device="neuron",
                 guidance_scale: float | None = None, text_seq_len: int = 1000, text_2_seq_len: int = 256,
                 image_tokens: int = 729, teacache_cadence: int | None = None,
                 teacache_online_delta_alpha: float | None = None, teacache_speedup: float | None = None,
                 teacache_calibration_path: str | None = None) -> None:
        self.model_path = str(model_path)
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        self.dtype = dtype
        self.guidance_scale = load_guidance(self.model_path, guidance_scale)
        self.dit = NeuronHunyuanVideo15TransformerApplication(
            model_path=os.path.join(self.model_path, "transformer"), height=height, width=width,
            num_frames=num_frames, text_seq_len=text_seq_len, text_2_seq_len=text_2_seq_len,
            image_tokens=image_tokens, parallel=parallel, dtype=dtype, exec_mode=exec_mode, device=device)
        if self.dit.cfg > 1 and self.guidance_scale is None:
            raise ValueError("CFG parallelism needs classifier-free guidance (the guider is disabled)")
        self.scheduler = None
        self.teacache = self.teacache_probe = None
        adaptive = teacache_speedup is not None or teacache_calibration_path is not None
        if adaptive and (teacache_cadence or teacache_online_delta_alpha):
            raise ValueError("adaptive TeaCache (speedup/calibration) and the probe-free modes are exclusive")
        if adaptive:
            from difflet.backends.neuron.core.teacache import NeuronTeaCacheProbe
            from difflet.pipeline.teacache import TeaCacheController, load_teacache_calibration_or_raise

            calibration = load_teacache_calibration_or_raise(
                teacache_calibration_path, model="hunyuan_video_15", shape_label=f"{height}x{width}x{num_frames}")
            if (teacache_speedup is not None and calibration.target_speedup is not None
                    and float(teacache_speedup) > float(calibration.target_speedup) + 1e-6):
                raise ValueError(f"the calibration's target speedup is {calibration.target_speedup}, "
                                 f"lower than the requested {teacache_speedup}")
            self.teacache = TeaCacheController(calibration)
            self.teacache_probe = NeuronTeaCacheProbe(self.dit.teacache_mod_input)
        elif teacache_cadence or teacache_online_delta_alpha:
            from difflet.pipeline.teacache import build_probe_free_controller

            self.teacache = build_probe_free_controller(
                model="hunyuan_video_15", shape_label=f"{height}x{width}x{num_frames}",
                cadence=teacache_cadence, online_delta_alpha=teacache_online_delta_alpha)

    def load(self, skip_warmup: bool = False) -> None:
        import diffusers

        with open(os.path.join(self.model_path, "model_index.json"), encoding="utf-8") as f:
            _, scheduler_cls = json.load(f)["scheduler"]
        self.scheduler = getattr(diffusers, scheduler_cls).from_pretrained(os.path.join(self.model_path, "scheduler"))
        self.dit.load(skip_warmup=skip_warmup)

    def _dit(self, latent_input, timestep, embeds, masks, embeds_2, masks_2, image_embeds):
        return self.dit(hidden_states=latent_input, timestep=timestep, encoder_hidden_states=embeds,
                        encoder_attention_mask=masks, encoder_hidden_states_2=embeds_2,
                        encoder_attention_mask_2=masks_2, image_embeds=image_embeds)[0].to("cpu")

    @torch.no_grad()
    def __call__(self, *, prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2,
                 negative_prompt_embeds=None, negative_prompt_embeds_mask=None, negative_prompt_embeds_2=None,
                 negative_prompt_embeds_mask_2=None, num_inference_steps: int = 50, generator=None,
                 latents=None, callback=None) -> torch.Tensor:
        import inspect

        import diffusers
        from diffusers.utils.torch_utils import randn_tensor

        # the pipeline module's own helper (whatever its file is called)
        retrieve_timesteps = inspect.getmodule(diffusers.HunyuanVideo15Pipeline).retrieve_timesteps

        dt = self.dtype
        cfg = self.guidance_scale is not None
        if cfg and negative_prompt_embeds is None:
            raise ValueError("classifier-free guidance needs the negative prompt's embeddings")
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, "cpu",
                                                            sigmas=sigmas)
        shape = (1, int(self.dit.model_config["out_channels"]), (self.num_frames - 1) // VAE_TEMPORAL + 1,
                 self.height // VAE_SPATIAL, self.width // VAE_SPATIAL)
        latents = (randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=dt)
                   if latents is None else latents.to("cpu", dt))
        cond = torch.zeros(shape, dtype=dt)
        mask = torch.zeros(shape[:1] + (1,) + shape[2:], dtype=dt)
        image_embeds = torch.zeros(1, self.dit.image_tokens, int(self.dit.model_config["image_embed_dim"]), dtype=dt)
        pos = [t.to(dt) for t in (prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2)]
        neg = [t.to(dt) for t in (negative_prompt_embeds, negative_prompt_embeds_mask, negative_prompt_embeds_2,
                                  negative_prompt_embeds_mask_2)] if cfg else None
        ctrl, probe = self.teacache, self.teacache_probe
        if ctrl is not None:
            from difflet.pipeline.teacache import sync_probe_free_num_steps

            ctrl.reset()
            sync_probe_free_num_steps(ctrl, len(timesteps))
            if probe is not None:
                probe.reset()
        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents, cond, mask], dim=1)
            timestep = t.expand(latent_input.shape[0]).to(latent_input.dtype)
            diff = None
            if probe is not None and ctrl.needs_signal():  # every step: the probe keeps m_{t-1}
                diff = float(probe.teacache_delta(latent_input, timestep))
                ctrl.note_probe()
            skipped = ctrl is not None and ctrl.should_skip(i, None, diff_norm=diff)
            if skipped:
                noise_pred = ctrl.skip_noise_pred()
            elif not cfg:
                noise_pred = self._dit(latent_input, timestep, *pos, image_embeds)
            elif self.dit.cfg > 1:  # both branches in one call, one per cfg group: [uncond, cond]
                stacked = [torch.cat([n, p]) for n, p in zip(neg, pos)]
                out = self._dit(torch.cat([latent_input, latent_input]), torch.cat([timestep, timestep]),
                                *stacked, image_embeds)
                pred_uncond, pred_cond = out[0:1], out[1:2]
                noise_pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
            else:
                pred_cond = self._dit(latent_input, timestep, *pos, image_embeds)
                pred_uncond = self._dit(latent_input, timestep, *neg, image_embeds)
                noise_pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
            if ctrl is not None and not skipped:
                ctrl.record_full_step(noise_pred)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if callback is not None:
                callback(i, t, latents)
        if ctrl is not None:
            print(f"[teacache] stats: {ctrl.stats()}", flush=True)
        return latents


__all__ = ["TorchNeuronHunyuanVideo15Pipeline", "load_guidance"]
