"""HunyuanVideo 1.5 text-to-video denoising loop for the AOT backend.

Written against diffusers' ``HunyuanVideo15Pipeline.__call__`` (t2v path): sigmas
``linspace(1, 0, steps + 1)[:-1]``, noise ``[1, C, (F-1)//4+1, H//16, W//16]`` in the working
dtype, conditioning latents / mask channel / image embeddings all zero, one DiT call on
``cat([latents, 0, 0])`` per guidance branch, classifier-free guidance

    noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

then ``scheduler.step``. A DiT with ``cfg == 2`` runs both branches in one batch-2 call
(``[uncond, cond]`` for every per-sample input); otherwise two batch-1 calls, cond first.

TeaCache acts on the guided prediction: a skipped step skips both guidance passes.
"""

from __future__ import annotations

import json
import os
import typing

import numpy as np
import torch

from difflet.pipeline.teacache import (
    TeaCacheController,
    build_probe_free_controller,
    load_teacache_calibration_or_raise,
    sync_probe_free_num_steps,
)

VAE_SPATIAL, VAE_TEMPORAL = 16, 4

_MODEL = "hunyuan_video_15"


def load_hv15_guidance(model_path: str, guidance_scale: float | None = None) -> float | None:
    """The snapshot's classifier-free guidance scale (None: disabled), refusing unsupported variants."""
    with open(os.path.join(model_path, "guider", "guider_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("_class_name") != "ClassifierFreeGuidance":
        raise NotImplementedError(f"guider {cfg.get('_class_name')!r} is not supported")
    if not cfg.get("enabled", True):
        return None
    unsupported = {
        k: cfg.get(k)
        for k, default in (
            ("guidance_rescale", 0.0),
            ("use_original_formulation", False),
            ("start", 0.0),
            ("stop", 1.0),
        )
        if cfg.get(k, default) != default
    }
    if unsupported:
        raise NotImplementedError(f"guider settings not implemented: {unsupported}")
    scale = float(cfg["guidance_scale"] if guidance_scale is None else guidance_scale)
    return scale if scale > 1.0 else None


def cfg_on(scale: float | None) -> bool:
    return scale is not None and float(scale) > 1.0


class TeaCacheProbe(typing.Protocol):
    def reset(self) -> None: ...

    def teacache_delta(self, latent_input: torch.Tensor, timestep: torch.Tensor) -> float: ...


class HunyuanVideo15DiT(typing.Protocol):
    cfg: int
    model_config: dict
    image_tokens: int

    def load(self, skip_warmup: bool = False) -> None: ...

    def begin_request(self) -> None: ...

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        encoder_attention_mask_2: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor: ...

    def make_teacache_probe(self) -> TeaCacheProbe: ...


class HunyuanVideo15Denoiser:
    def __init__(
        self,
        dit: HunyuanVideo15DiT,
        *,
        model_path: str,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype = torch.bfloat16,
        guidance_scale: float | None = None,
        teacache_cadence: int | None = None,
        teacache_online_delta_alpha: float | None = None,
        teacache_speedup: float | None = None,
        teacache_calibration_path: str | None = None,
    ) -> None:
        self.dit = dit
        self.model_path = str(model_path)
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        self.dtype = dtype
        self.guidance_scale = load_hv15_guidance(self.model_path, guidance_scale)
        if dit.cfg > 1 and self.guidance_scale is None:
            raise ValueError("CFG parallelism needs classifier-free guidance (the guider is disabled)")
        self.scheduler = None
        self.teacache: TeaCacheController | None = None
        self.teacache_probe: TeaCacheProbe | None = None
        shape_label = f"{self.height}x{self.width}x{self.num_frames}"
        adaptive = teacache_speedup is not None or teacache_calibration_path is not None
        if adaptive and (teacache_cadence or teacache_online_delta_alpha):
            raise ValueError("adaptive TeaCache (speedup/calibration) and the probe-free modes are exclusive")
        if adaptive:
            calibration = load_teacache_calibration_or_raise(
                teacache_calibration_path, model=_MODEL, shape_label=shape_label
            )
            if (
                teacache_speedup is not None
                and calibration.target_speedup is not None
                and float(teacache_speedup) > float(calibration.target_speedup) + 1e-6
            ):
                raise ValueError(
                    f"the calibration's target speedup is {calibration.target_speedup}, "
                    f"lower than the requested {teacache_speedup}"
                )
            self.teacache = TeaCacheController(calibration)
            self.teacache_probe = dit.make_teacache_probe()
        elif teacache_cadence or teacache_online_delta_alpha:
            self.teacache = build_probe_free_controller(
                model=_MODEL,
                shape_label=shape_label,
                cadence=teacache_cadence,
                online_delta_alpha=teacache_online_delta_alpha,
            )

    def load(self, skip_warmup: bool = False) -> None:
        import diffusers

        with open(os.path.join(self.model_path, "model_index.json"), encoding="utf-8") as f:
            _, scheduler_cls = json.load(f)["scheduler"]
        self.scheduler = getattr(diffusers, scheduler_cls).from_pretrained(
            os.path.join(self.model_path, "scheduler")
        )
        self.dit.load(skip_warmup=skip_warmup)

    def _dit(self, latent_input, timestep, text, image_embeds) -> torch.Tensor:
        embeds, masks, embeds_2, masks_2 = text
        return self.dit(
            hidden_states=latent_input,
            timestep=timestep,
            encoder_hidden_states=embeds,
            encoder_attention_mask=masks,
            encoder_hidden_states_2=embeds_2,
            encoder_attention_mask_2=masks_2,
            image_embeds=image_embeds,
        ).to("cpu")

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt_embeds,
        prompt_embeds_mask,
        prompt_embeds_2,
        prompt_embeds_mask_2,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_embeds_2=None,
        negative_prompt_embeds_mask_2=None,
        num_inference_steps: int = 50,
        generator=None,
        latents=None,
        callback=None,
    ) -> torch.Tensor:
        import inspect

        import diffusers
        from diffusers.utils.torch_utils import randn_tensor

        retrieve_timesteps = inspect.getmodule(diffusers.HunyuanVideo15Pipeline).retrieve_timesteps

        dt = self.dtype
        scale = self.guidance_scale
        cfg = cfg_on(scale)
        if cfg and negative_prompt_embeds is None:
            raise ValueError("classifier-free guidance needs the negative prompt's embeddings")
        if self.dit.cfg > 1 and not cfg:
            raise ValueError("CFG parallelism needs classifier-free guidance (guidance_scale > 1)")

        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, "cpu", sigmas=sigmas
        )
        shape = (
            1,
            int(self.dit.model_config["out_channels"]),
            (self.num_frames - 1) // VAE_TEMPORAL + 1,
            self.height // VAE_SPATIAL,
            self.width // VAE_SPATIAL,
        )
        latents = (
            randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=dt)
            if latents is None
            else latents.to("cpu", dt)
        )
        cond = torch.zeros(shape, dtype=dt)
        mask = torch.zeros(shape[:1] + (1,) + shape[2:], dtype=dt)
        image_dim = int(self.dit.model_config["image_embed_dim"])
        image_embeds = torch.zeros(1, self.dit.image_tokens, image_dim, dtype=dt)
        pos = [
            t.to(dt)
            for t in (prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2)
        ]
        neg = (
            [
                t.to(dt)
                for t in (
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    negative_prompt_embeds_2,
                    negative_prompt_embeds_mask_2,
                )
            ]
            if cfg
            else None
        )

        self.dit.begin_request()
        ctrl, probe = self.teacache, self.teacache_probe
        if ctrl is not None:
            ctrl.reset()
            sync_probe_free_num_steps(ctrl, len(timesteps))
            if probe is not None:
                probe.reset()

        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents, cond, mask], dim=1)
            timestep = t.expand(latent_input.shape[0]).to(latent_input.dtype)
            diff = None
            if probe is not None and ctrl.needs_signal():
                diff = float(probe.teacache_delta(latent_input, timestep))
                ctrl.note_probe()
            skipped = ctrl is not None and ctrl.should_skip(i, None, diff_norm=diff)
            if skipped:
                noise_pred = ctrl.skip_noise_pred()
            elif not cfg:
                noise_pred = self._dit(latent_input, timestep, pos, image_embeds)
            elif self.dit.cfg > 1:
                stacked = [torch.cat([n, p]) for n, p in zip(neg, pos)]
                out = self._dit(
                    torch.cat([latent_input, latent_input]),
                    torch.cat([timestep, timestep]),
                    stacked,
                    torch.cat([image_embeds, image_embeds]),
                )
                pred_uncond, pred_cond = out[0:1], out[1:2]
                noise_pred = pred_uncond + scale * (pred_cond - pred_uncond)
            else:
                pred_cond = self._dit(latent_input, timestep, pos, image_embeds)
                pred_uncond = self._dit(latent_input, timestep, neg, image_embeds)
                noise_pred = pred_uncond + scale * (pred_cond - pred_uncond)
            if ctrl is not None and not skipped:
                ctrl.record_full_step(noise_pred)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if callback is not None:
                callback(i, t, latents)
        if ctrl is not None:
            print(f"[teacache] stats: {ctrl.stats()}", flush=True)
        return latents
