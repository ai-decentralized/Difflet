"""The HunyuanVideo 1.5 Trainium AOT application behind the AOT denoising loop.

``AotHunyuanVideo15DiT`` adapts ``NeuronHunyuanVideoApplication`` (model_version 1.5) to the
``HunyuanVideo15DiT`` protocol of ``pipeline15.HunyuanVideo15Denoiser``; ``build_aot_denoiser``
wires the two together. The probe for adaptive TeaCache is the app's fused
``teacache_probe`` component (compiled and loaded with the DiT).
"""

from __future__ import annotations

import json
import os

import torch

from difflet.models.hunyuan_video.application import (
    HunyuanVideo15DiTInputBundle,
    NeuronHunyuanVideoApplication,
)
from difflet.models.hunyuan_video.pipeline15 import HunyuanVideo15Denoiser


class _AotTeaCacheProbe:
    """The app's fused probe component as a ``pipeline15.TeaCacheProbe``.

    Cheap to build before ``load``: the component is resolved at call time. The previous
    step's modulated input (``prev_mod``) lives on device inside the probe NEFF.
    """

    def __init__(self, app: NeuronHunyuanVideoApplication) -> None:
        self.app = app

    def reset(self) -> None:
        self.app.teacache_probe.reset_state()

    def teacache_delta(self, latent_input: torch.Tensor, timestep: torch.Tensor) -> float:
        # Block-0 modulation depends only on the latent input and t, which the cond and
        # uncond rows share: row 0 is the whole signal. The probe is compiled at batch 1.
        dt = self.app.dtype
        rel_l1 = self.app.teacache_probe.teacache_delta(
            latent_input[:1].to("cpu", dt),
            timestep[:1].to("cpu", dt),
            torch.zeros(1, dtype=dt),
        )
        return float(rel_l1)


class AotHunyuanVideo15DiT:
    """``HunyuanVideo15DiT`` over the compiled AOT application."""

    def __init__(self, app: NeuronHunyuanVideoApplication, compiled_model_path: str, *, cfg: int):
        self.app = app
        self.compiled_model_path = str(compiled_model_path)
        self.cfg = int(cfg)
        with open(os.path.join(app.transformer_path, "config.json"), encoding="utf-8") as f:
            self.model_config: dict = json.load(f)
        self.image_tokens = int(app.image_seq_len)

    def load(self, skip_warmup: bool = False) -> None:
        self.app.load(self.compiled_model_path, skip_warmup=skip_warmup)

    def begin_request(self) -> None:
        self.app.reset_refiner_cache()

    def make_teacache_probe(self) -> _AotTeaCacheProbe:
        if not self.app.teacache_probe_fused or self.app.teacache_probe is None:
            raise RuntimeError("adaptive TeaCache needs the teacache_probe component compiled")
        return _AotTeaCacheProbe(self.app)

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
    ) -> torch.Tensor:
        if bool(torch.any(image_embeds != 0)):
            raise ValueError(
                "HunyuanVideo 1.5 AOT runs text-to-video only: the compiled DiT bakes the "
                "t2v image stream, so non-zero image_embeds (image-to-video) are refused."
            )
        dt = self.app.dtype
        batch = int(hidden_states.shape[0])
        bundle = HunyuanVideo15DiTInputBundle(
            hidden_states=hidden_states.to(dt).contiguous(),
            timestep=timestep.to(dt).contiguous(),
            encoder_hidden_states=encoder_hidden_states.to(dt).contiguous(),
            # encode_prompt hands masks over in the working dtype; the DiT takes int64.
            encoder_attention_mask=encoder_attention_mask.round().to(torch.int64).contiguous(),
            timestep_r=torch.zeros(batch, dtype=dt),
            encoder_hidden_states_2=encoder_hidden_states_2.to(dt).contiguous(),
            encoder_attention_mask_2=encoder_attention_mask_2.round().to(torch.int64).contiguous(),
            image_embeds=image_embeds.to(dt).contiguous(),
        )
        out = self.app.forward_dit(bundle)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.to("cpu")


def build_aot_application(
    *,
    model_path: str,
    parallel,
    height: int,
    width: int,
    num_frames: int,
    teacache_fused: bool,
) -> NeuronHunyuanVideoApplication:
    """The HV1.5 AOT application (DiT, plus the fused probe when ``teacache_fused``)."""
    from difflet.models.hunyuan_video.entry import create_hunyuan_video15_application

    return create_hunyuan_video15_application(
        model_path=str(model_path),
        parallel=parallel,
        dtype=torch.bfloat16,
        shape={"height": int(height), "width": int(width), "num_frames": int(num_frames)},
        teacache_fused=bool(teacache_fused),
    )


def build_aot_denoiser(
    *,
    model_path: str,
    compiled_model_path: str,
    parallel,
    height: int,
    width: int,
    num_frames: int,
    guidance_scale: float | None = None,
    teacache_cadence: int | None = None,
    teacache_online_delta_alpha: float | None = None,
    teacache_speedup: float | None = None,
    teacache_calibration_path: str | None = None,
) -> HunyuanVideo15Denoiser:
    app = build_aot_application(
        model_path=model_path,
        parallel=parallel,
        height=height,
        width=width,
        num_frames=num_frames,
        teacache_fused=bool(teacache_speedup or teacache_calibration_path),
    )
    dit = AotHunyuanVideo15DiT(
        app,
        compiled_model_path,
        cfg=2 if bool(getattr(parallel, "cfg_parallel_enabled", False)) else 1,
    )
    return HunyuanVideo15Denoiser(
        dit,
        model_path=str(model_path),
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        dtype=app.dtype,
        guidance_scale=guidance_scale,
        teacache_cadence=teacache_cadence,
        teacache_online_delta_alpha=teacache_online_delta_alpha,
        teacache_speedup=teacache_speedup,
        teacache_calibration_path=teacache_calibration_path,
    )


__all__ = ["AotHunyuanVideo15DiT", "build_aot_application", "build_aot_denoiser"]
