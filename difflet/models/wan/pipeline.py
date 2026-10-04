"""Wan end-to-end pipeline orchestration.

This module keeps pipeline control flow independent from the Trainium
component wrappers.  The wrappers expose fixed-shape callables; this layer
prepares prompt embeddings, latents, scheduler timesteps, classifier-free
guidance, and VAE decode normalization around them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class WanPipelineOutput:
    """Minimal Wan pipeline output compatible with diffusers-style callers."""

    frames: torch.Tensor
    latents: torch.Tensor
    prompt_embeds: torch.Tensor | None = None


class WanOrchestrator:
    """CPU-side orchestration for compiled Wan sub-components."""

    def __init__(
        self,
        *,
        model_path: str,
        text_encoder: Any = None,
        transformer: Any = None,
        transformer_2: Any = None,
        vae_decoder: Any = None,
        dtype: torch.dtype = torch.bfloat16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 9,
        boundary_ratio: float | None = None,
        tokenizer_path: str | None = None,
        max_text_length: int = 512,
        teacache_calibration_path: str | None = None,
        teacache_probes: dict[int, Any] | None = None,
        teacache_cadence: int | None = None,
        teacache_online_delta_alpha: float | None = None,
    ) -> None:
        self.model_path = model_path
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.vae_decoder = vae_decoder
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.boundary_ratio = (
            _read_boundary_ratio(model_path) if boundary_ratio is None else boundary_ratio
        )
        self.scheduler = _load_scheduler(model_path)
        self.tokenizer_path = tokenizer_path or os.path.join(model_path, "tokenizer")
        self.max_text_length = int(max_text_length)
        self._tokenizer = None
        # Adaptive TeaCache uses per-stage device probes when supplied by the
        # Trainium application; host-only callers retain a CPU shadow fallback.
        self.teacache_calibration_path = teacache_calibration_path
        self._teacache_controller = None
        self._teacache_shadows: dict[int, Any] = {}
        self._teacache_probes = teacache_probes or {}
        self._teacache_last_model_id: int | None = None
        # Probe-free TeaCache modes (fixed cadence / online-delta): host-side
        # skip decisions only — no CPU shadow, no calibration file, no NEFF
        # change. num_steps is synced to the request in _denoise.
        if teacache_cadence is not None or teacache_online_delta_alpha is not None:
            if teacache_calibration_path:
                raise ValueError(
                    "teacache_cadence/teacache_online_delta_alpha are mutually "
                    "exclusive with teacache_calibration_path."
                )
            from difflet.pipeline.teacache import build_probe_free_controller

            self._teacache_controller = build_probe_free_controller(
                model="wan",
                shape_label=f"{self.height}x{self.width}x{self.num_frames}",
                cadence=teacache_cadence,
                online_delta_alpha=teacache_online_delta_alpha,
            )

    def _maybe_init_teacache(self) -> bool:
        """Build the controller; host-only runtimes retain their CPU signal fallback."""
        if self._teacache_controller is not None:
            return True
        if not self.teacache_calibration_path:
            return False
        from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController

        calibration = TeaCacheCalibration.from_json(self.teacache_calibration_path)
        self._teacache_controller = TeaCacheController(calibration)
        # Fixed-cadence mode (cclog 84/89): the skip decision is purely index-based, so the
        # block-0 CPU shadow is never consulted — don't build it (saves load + per-step cost).
        if not self._teacache_controller.needs_signal():
            return True
        stages = {"transformer": self.transformer, "transformer_2": self.transformer_2}
        for subfolder, model in stages.items():
            if model is None:
                continue
            if id(model) in self._teacache_probes:
                continue
            from difflet.backends.trainium.wan.teacache_cpu_shadow import WanTeacacheCPUShadow

            self._teacache_shadows[id(model)] = WanTeacacheCPUShadow(
                os.path.join(self.model_path, subfolder), dtype=self.dtype
            )
        return True

    def has_runtime_components(self) -> bool:
        return any(
            component is not None
            for component in (
                self.text_encoder,
                self.transformer,
                self.transformer_2,
                self.vae_decoder,
            )
        )

    def prepare_latents(
        self,
        *,
        batch_size: int,
        channels: int = 16,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        height = self.height if height is None else int(height)
        width = self.width if width is None else int(width)
        num_frames = self.num_frames if num_frames is None else int(num_frames)
        latent_frames = (num_frames - 1) // 4 + 1
        shape = (int(batch_size), int(channels), latent_frames, height // 8, width // 8)
        if latents is not None:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Expected latents shape {shape}, got {tuple(latents.shape)}")
            return latents
        return torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=dtype or self.dtype,
        )

    def encode_prompt(
        self,
        *,
        prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if prompt_embeds is not None:
            return prompt_embeds
        if input_ids is not None:
            if self.text_encoder is None:
                raise ValueError("input_ids were provided but no Wan text_encoder is active.")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.int32)
            embeds = _first_tensor(self.text_encoder(input_ids, attention_mask))
            return _zero_padding_embeds(embeds, attention_mask)
        if prompt is not None:
            if self.text_encoder is None:
                raise ValueError("prompt was provided but no Wan text_encoder is active.")
            tokenizer = self._get_tokenizer()
            prompts = prompt if isinstance(prompt, list) else [prompt]
            tokenized = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(torch.int64)
            attention_mask = tokenized["attention_mask"].to(torch.int32)
            embeds = _first_tensor(self.text_encoder(input_ids, attention_mask))
            return _zero_padding_embeds(embeds, attention_mask)
        return None

    def _negative_prompt_embeds(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Unconditional embeddings for CFG.

        diffusers encodes negative_prompt="" through the text encoder (a real
        EOS-token embedding, zero-padded); a zeros_like tensor is not a valid
        embedding and degrades the CFG direction. Falls back to zeros when no
        text encoder / tokenizer is available (e.g. precomputed-embeds runs).
        """
        if self.text_encoder is not None:
            try:
                neg = self.encode_prompt(prompt="")
                if neg is not None:
                    return neg.to(dtype=prompt_embeds.dtype)
            except Exception as exc:  # tokenizer assets missing, etc.
                print(f"[wan] empty-prompt negative encode failed ({exc}); "
                      "falling back to zeros", flush=True)
        return torch.zeros_like(prompt_embeds)

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Wan prompt tokenization requires `transformers`; install it or pass input_ids."
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        except (OSError, EnvironmentError) as exc:
            raise FileNotFoundError(
                f"Could not load Wan tokenizer from {self.tokenizer_path!r}. "
                "Most local snapshots ship only tokenizer_config.json; download the "
                "tokenizer assets (e.g. spiece.model) before running prompt-based smoke."
            ) from exc
        return self._tokenizer

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        *,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        negative_input_ids: torch.Tensor | None = None,
        negative_attention_mask: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        batch_size: int = 1,
        channels: int = 16,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 1.0,
        guidance_scale_2: float | None = None,
        output_type: str = "latent",
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> WanPipelineOutput | tuple[torch.Tensor]:
        if output_type not in {"latent", "pt"}:
            raise ValueError("Wan output_type currently supports only 'latent' or 'pt'.")

        # Hold loop latents in fp32 (diffusers convention): UniPC's order-2
        # corrector computes small differences of near-equal terms and collapses
        # in bf16. The model input is cast to the component dtype per step.
        latents = self.prepare_latents(
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            generator=generator,
            latents=latents,
        ).to(torch.float32)
        prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if negative_prompt_embeds is None and (
            negative_input_ids is not None or negative_prompt is not None
        ):
            negative_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=None,
                input_ids=negative_input_ids,
                attention_mask=negative_attention_mask,
            )

        if self.transformer is not None:
            if prompt_embeds is None:
                raise ValueError(
                    "Wan denoising requires prompt_embeds or input_ids when transformer is active."
                )
            latents = self._denoise(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                guidance_scale_2=guidance_scale_2,
            )

        frames = latents
        if output_type == "pt":
            if self.vae_decoder is None:
                raise ValueError("Wan output_type='pt' requires an active VAE decoder.")
            frames = self._decode_latents(latents)

        if return_dict:
            return WanPipelineOutput(frames=frames, latents=latents, prompt_embeds=prompt_embeds)
        return (frames,)

    def _denoise(
        self,
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_scale_2: float | None,
    ) -> torch.Tensor:
        num_inference_steps = max(int(num_inference_steps), 1)
        scheduler = self.scheduler
        if scheduler is not None:
            scheduler.set_timesteps(num_inference_steps, device=latents.device)
            timesteps = list(scheduler.timesteps)
            train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        else:
            timesteps = list(
                torch.linspace(
                    999,
                    0,
                    steps=num_inference_steps,
                    device=latents.device,
                    dtype=torch.float32,
                )
            )
            train_timesteps = 1000

        boundary_timestep = None
        if self.boundary_ratio is not None:
            boundary_timestep = float(self.boundary_ratio) * float(train_timesteps)

        # CFG parallel runs uncond+cond as a single batched DiT call (one branch
        # per data-parallel rank), so the serial TeaCache residual path doesn't
        # apply; keep them mutually exclusive (mirrors Flux).
        cfg_parallel = bool(
            getattr(_component_config(self.transformer), "cfg_parallel_enabled", False)
        )
        teacache_on = False if cfg_parallel else self._maybe_init_teacache()
        ctrl = self._teacache_controller if teacache_on else None
        if ctrl is not None:
            from difflet.pipeline.teacache import sync_probe_free_num_steps

            # Probe-free controllers are built before the request's step count
            # is known; sync it so the cooldown window protects the real tail.
            sync_probe_free_num_steps(ctrl, len(timesteps))
            ctrl.reset()
            self._teacache_last_model_id = None

        from difflet.pipeline.step_timing import timed_steps

        for step_index, timestep in timed_steps("wan", timesteps):
            current_model = self._select_transformer(timestep, boundary_timestep)
            scale = self._select_guidance_scale(timestep, boundary_timestep, guidance_scale, guidance_scale_2)
            model_dtype = _component_dtype(current_model, self.dtype)

            if cfg_parallel:
                # Stack [uncond, cond] into batch=2; the transformer scatters one
                # branch to each DP rank and gathers the result back to batch=2.
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = self._negative_prompt_embeds(prompt_embeds)
                batched_latents = torch.cat([latents, latents], dim=0).to(dtype=model_dtype)
                batched_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                ).to(dtype=model_dtype)
                batched_timestep = _batch_timestep(timestep, 2, latents.device, model_dtype)
                out = _first_tensor(
                    current_model(batched_latents, batched_timestep, batched_embeds)
                )
                uncond, cond = out[0:1], out[1:2]
                noise_pred = uncond + scale * (cond - uncond)
                latents = self._scheduler_step(noise_pred, timestep, latents, num_inference_steps)
                continue

            timestep_batch = _batch_timestep(timestep, latents.shape[0], latents.device, model_dtype)

            # TeaCache: block-0's modulated input depends on latents and timestep,
            # shared by cond/uncond, so one probe per step drives the skip decision; the
            # cached post-CFG residual is reused when skipping the full DiT evaluation.
            mod_input = None
            should_skip = False
            if ctrl is not None:
                if self._teacache_last_model_id != id(current_model):
                    ctrl.reset()  # stage switch (high->low noise) invalidates the residual
                    self._teacache_last_model_id = id(current_model)
                diff_norm = None
                # Fixed-cadence mode skips the block-0 CPU shadow entirely (index-based decision).
                if ctrl.needs_signal():
                    probe = self._teacache_probes.get(id(current_model))
                    if probe is not None:
                        # Probe every step to preserve the calibrated step-to-step
                        # signal, including skip runs. A fresh request/stage first
                        # overwrites device state while ctrl has no residual yet.
                        delta = probe.teacache_delta(latents.to(dtype=model_dtype), timestep_batch)
                        diff_norm = float(delta.detach().cpu().item())
                        ctrl.note_probe()
                    else:
                        shadow = self._teacache_shadows[id(current_model)]
                        mod_input = shadow.teacache_mod_input(
                            latents.to(dtype=model_dtype), timestep_batch, prompt_embeds.to(dtype=model_dtype)
                        )
                        if ctrl.prev_mod_input is not None:
                            prev = ctrl.prev_mod_input
                            cur = mod_input.detach().float().cpu()
                            denom = prev.abs().mean().clamp_min(1e-8)
                            diff_norm = float((cur - prev).abs().mean() / denom)
                should_skip = ctrl.should_skip(step_index, mod_input, diff_norm=diff_norm)

            if should_skip:
                noise_pred = ctrl.skip_noise_pred(mod_input=mod_input)
            else:
                noise_pred = _first_tensor(
                    current_model(
                        latents.to(dtype=model_dtype),
                        timestep_batch,
                        prompt_embeds.to(dtype=model_dtype),
                    )
                )
                if scale > 1.0:
                    if negative_prompt_embeds is None:
                        negative_prompt_embeds = self._negative_prompt_embeds(prompt_embeds)
                    uncond = _first_tensor(
                        current_model(
                            latents.to(dtype=model_dtype),
                            timestep_batch,
                            negative_prompt_embeds.to(dtype=model_dtype),
                        )
                    )
                    noise_pred = uncond + scale * (noise_pred - uncond)
                if ctrl is not None:
                    ctrl.record_full_step(noise_pred, mod_input=mod_input)

            latents = self._scheduler_step(noise_pred, timestep, latents, num_inference_steps)

        if ctrl is not None:
            self._teacache_last_stats = ctrl.stats()
            print(f"[teacache] stats: {self._teacache_last_stats}", flush=True)
        return latents

    def _select_transformer(self, timestep: torch.Tensor, boundary_timestep: float | None):
        if (
            self.transformer_2 is not None
            and boundary_timestep is not None
            and float(timestep.detach().float().item()) < boundary_timestep
        ):
            return self.transformer_2
        return self.transformer

    @staticmethod
    def _select_guidance_scale(
        timestep: torch.Tensor,
        boundary_timestep: float | None,
        guidance_scale: float,
        guidance_scale_2: float | None,
    ) -> float:
        if (
            guidance_scale_2 is not None
            and boundary_timestep is not None
            and float(timestep.detach().float().item()) < boundary_timestep
        ):
            return float(guidance_scale_2)
        return float(guidance_scale)

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
        dtype = _component_dtype(self.vae_decoder, self.dtype)
        config = _component_config(self.vae_decoder)
        latents = latents.to(dtype=dtype)
        mean = getattr(config, "latents_mean", None)
        std = getattr(config, "latents_std", None)
        if mean is not None and std is not None and len(mean) > 0 and len(std) > 0:
            mean_tensor = torch.tensor(mean, dtype=dtype, device=latents.device).view(1, -1, 1, 1, 1)
            std_tensor = torch.tensor(std, dtype=dtype, device=latents.device).view(1, -1, 1, 1, 1)
            latents = latents * std_tensor + mean_tensor
        return _first_tensor(self.vae_decoder(latents))


def has_wan_components(app: Any) -> bool:
    pipeline = getattr(app, "pipeline", None)
    return bool(pipeline is not None and pipeline.has_runtime_components())


def _zero_padding_embeds(embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Zero embedding rows at padding positions.

    Reference Wan convention (diffusers ``_get_t5_prompt_embeds``): the DiT
    cross-attends unmasked over the full text sequence and was trained with
    zero-padded embeddings — raw UMT5 pad-token outputs at the padding
    positions poison cross-attention and collapse sampling to noise.
    """
    mask = attention_mask.to(dtype=embeds.dtype, device=embeds.device)
    return embeds * mask.unsqueeze(-1)


def _load_scheduler(model_path: str):
    scheduler_path = os.path.join(model_path, "scheduler")
    if not os.path.exists(os.path.join(scheduler_path, "scheduler_config.json")):
        return None
    try:
        from diffusers import UniPCMultistepScheduler
    except ImportError:
        return None
    return UniPCMultistepScheduler.from_pretrained(scheduler_path)


def _read_boundary_ratio(model_path: str) -> float | None:
    index_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get("boundary_ratio")
    return None if value is None else float(value)


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return _first_tensor(value[0])
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    if hasattr(value, "sample"):
        return value.sample
    raise TypeError(f"Expected tensor-like Wan component output, got {type(value)!r}")


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
