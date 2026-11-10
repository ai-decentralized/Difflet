"""Stock LTX-2 audiovisual pipeline driven by the Difflet GPU transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from difflet.backends.cuda.ltx_2.dit_config import Ltx2DiTConfig
from difflet.backends.gpu.application import GpuApplication
from difflet.models.ltx_2.contract import LTX2DiTInputBundle
from difflet.models.ltx_2.defaults import LTX_2_DEFAULT_TEXT_SEQ_LEN
from difflet.models.ltx_2.spec import SPEC

_PROFILES = ("diffusers", "ltx_official_bf16")


@dataclass(frozen=True)
class Ltx2AudioVideoOutput:
    """Decoded joint output, or both denormalized latent streams for ``latent``.

    Decoded frames follow diffusers: BFCHW in [0, 1]. Audio uses the vocoder's
    output sample rate. Latent output has no decoded frames, audio or sample rate.
    """

    frames: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    sample_rate: int | None = None
    video_latents: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None


class CudaLtx2Application(GpuApplication):
    spec = SPEC
    input_bundle_type = LTX2DiTInputBundle

    def _configure(self, kwargs: dict[str, Any]) -> None:
        # The one place the profile, dtype, text length and frame rate are checked.
        self.generation_profile = kwargs.get("generation_profile", "diffusers")
        if self.generation_profile not in _PROFILES:
            raise ValueError(f"LTX-2 generation_profile must be one of {_PROFILES}")
        official = self.generation_profile == "ltx_official_bf16"
        cpu_diagnostic = self.dtype is torch.float32 and self.device.type == "cpu" and not official
        if self.dtype is not torch.bfloat16 and not cpu_diagnostic:
            raise ValueError(
                "LTX-2 runs in bfloat16; float32 is only for CPU diagnostics of the default profile"
            )
        if official and self.device.type == "cuda":
            from .official_compatibility import prepare_official_runtime

            prepare_official_runtime()
        self.load_text_encoders = bool(kwargs.get("load_text_encoders", True))
        self.load_vae = bool(kwargs.get("load_vae", True))
        self.local_files_only = bool(kwargs.get("local_files_only", True))
        self.text_seq_len = kwargs.get("text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN)
        if not 1 <= self.text_seq_len <= LTX_2_DEFAULT_TEXT_SEQ_LEN:
            raise ValueError(f"LTX-2 text_seq_len must be in [1, {LTX_2_DEFAULT_TEXT_SEQ_LEN}]")
        self.frame_rate = kwargs.get("frame_rate", SPEC.default_fps)
        if self.frame_rate != SPEC.default_fps:
            raise ValueError(f"LTX-2 GPU inference requires frame_rate={SPEC.default_fps}")
        self.pipe = None
        self.adapter = None

    def _build_transformers(self) -> None:
        from difflet.backends.cuda.ltx_2.transformer import CudaLtx2TransformerApplication

        self.config = Ltx2DiTConfig.from_pretrained(
            self.transformer_path,
            **self.shape,
            frame_rate=self.frame_rate,
            text_seq_len=self.text_seq_len,
            torch_dtype=self.dtype,
        )
        self.transformer = CudaLtx2TransformerApplication(
            model_path=self.transformer_path,
            config=self.config,
            device=self.device,
            torch_compile=self.torch_compile,
            generation_profile=self.generation_profile,
        )

    def load_companions(self) -> None:
        from diffusers import LTX2Pipeline

        from difflet.backends.cuda.ltx_2.transformer import Ltx2DiTAdapter

        # The adapter requires the loaded component. Passing it to from_pretrained
        # would trigger diffusers' upstream-class check and a duplicate DiT load.
        self.adapter = Ltx2DiTAdapter(self._require_transformer("load companions"))
        skipped = {"transformer": None}
        if not self.load_text_encoders:
            skipped["text_encoder"] = None
        if not self.load_vae:
            skipped.update(vae=None, audio_vae=None, vocoder=None)
        base = LTX2Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            **skipped,
        )
        for name in ("text_encoder", "connectors", "vae", "audio_vae", "vocoder"):
            module = getattr(base, name, None)
            if module is not None:
                module.to(self.device)
        if self.generation_profile == "ltx_official_bf16":
            from .official_compatibility import configure_official_connectors

            configure_official_connectors(base.connectors)
        self.pipe = LTX2Pipeline(**{**base.components, "transformer": self.adapter})

    def dit_input_contract(self) -> dict[str, dict[str, Any]]:
        component = self._require_transformer("input contract")
        return {
            name: {"shape": tuple(tensor.shape), "dtype": tensor.dtype}
            for name, tensor in zip(
                LTX2DiTInputBundle.__dataclass_fields__, component.get_example_inputs(), strict=True
            )
        }

    def _validate_latents(self, latents, audio_latents) -> None:
        cfg = self._require_transformer("generation").dit_config
        video_shapes = {
            (1, cfg.video_seq_len, cfg.in_channels),
            (1, cfg.in_channels, cfg.latent_num_frames, cfg.latent_height, cfg.latent_width),
        }
        # The audio VAE's latent channel count and mel compression define packing.
        audio_vae = self.pipe.audio_vae
        channels = audio_vae.config.latent_channels
        bins = audio_vae.config.mel_bins // audio_vae.mel_compression_ratio
        audio_shapes = {
            (1, cfg.audio_seq_len, channels * bins),
            (1, channels, cfg.audio_seq_len, bins),
        }
        for name, value, shapes in (
            ("latents", latents, video_shapes),
            ("audio_latents", audio_latents, audio_shapes),
        ):
            if value is not None and tuple(value.shape) not in shapes:
                raise ValueError(f"LTX-2 {name} must match the resident shape: {sorted(shapes)}")

    @torch.no_grad()
    def generate_audio_video(
        self,
        prompt: str | list[str] | None = None,
        *,
        negative_prompt: str | list[str] | None = None,
        num_inference_steps: int = SPEC.default_steps,
        guidance_scale: float = SPEC.default_guidance_scale,
        seed: int | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        output_type: str = "pt",
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        sigmas: list[float] | None = None,
    ) -> Ltx2AudioVideoOutput:
        """Generate both streams using the selected host numerical profile.

        Initial video/audio latents may be packed or unpacked, with stock
        diffusers normalization semantics. ``seed`` resets a request-local RNG.
        Prompt embeddings are the upstream pre-connector Gemma embeddings.
        The default profile preserves stock diffusers behavior. The explicit
        ``ltx_official_bf16`` profile uses BF16 sampler state and accepts final
        ``sigmas`` boundaries, including optional terminal zero, with one interval
        per ``num_inference_steps``. Both profiles expose joint CFG only.
        """
        # Only what diffusers would not refuse itself: the resident shape and batch, this
        # application's own options, and the components a reduced load leaves out.
        official = self.generation_profile == "ltx_official_bf16"
        if sigmas is not None and not official:
            raise ValueError("LTX-2 final sigmas require generation_profile='ltx_official_bf16'")
        if official and prompt is None and prompt_embeds is None:
            raise ValueError("LTX-2 generation requires a prompt or prompt embeddings")
        if num_inference_steps < 1:
            raise ValueError("LTX-2 num_inference_steps must be at least 1")
        if output_type not in {"pt", "latent"}:
            raise ValueError("LTX-2 joint output_type must be 'pt' or 'latent'")
        for name, value in (("height", height), ("width", width), ("num_frames", num_frames)):
            if value is not None and value != self.shape[name]:
                raise ValueError(f"LTX-2 {name} must match resident shape {self.shape[name]}")
        if frame_rate is not None and frame_rate != self.frame_rate:
            raise ValueError(f"LTX-2 frame_rate must match resident frame_rate={self.frame_rate}")
        for name, value in (("prompt", prompt), ("negative_prompt", negative_prompt)):
            if isinstance(value, list) and len(value) != 1:
                raise ValueError(f"LTX-2 generates one {name} per request (batch 1)")
        if self.pipe is None:
            raise RuntimeError("application is not loaded; call load() first")
        # Both normalization buffers are used even by upstream latent output.
        if self.pipe.vae is None or self.pipe.audio_vae is None:
            raise RuntimeError(
                "LTX-2 generation requires load_vae=True (both latent normalization buffers)"
            )
        if self.pipe.connectors is None:
            raise RuntimeError("LTX-2 generation requires text connectors")
        if output_type != "latent" and self.pipe.vocoder is None:
            raise RuntimeError("LTX-2 decoded generation requires the vocoder")
        do_cfg = not math.isclose(guidance_scale, 1.0) if official else guidance_scale > 1
        if self.pipe.text_encoder is None and (
            prompt_embeds is None or do_cfg and negative_prompt_embeds is None
        ):
            raise RuntimeError(
                "LTX-2 prompt generation requires load_text_encoders=True or complete prompt embeddings"
            )
        for name, embedding, mask in (
            ("prompt", prompt_embeds, prompt_attention_mask),
            ("negative_prompt", negative_prompt_embeds, negative_prompt_attention_mask),
        ):
            # The official profile bypasses diffusers' check_inputs, which refuses this.
            if official and embedding is not None and mask is None:
                raise ValueError(f"LTX-2 {name} embeddings require their attention mask")
        self._validate_latents(latents, audio_latents)
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        if official:
            from difflet.backends.cuda.ltx_2.official import generate

            video, audio = generate(
                self.pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **self.shape,
                frame_rate=self.frame_rate,
                num_inference_steps=num_inference_steps,
                guidance_scale=float(guidance_scale),
                generator=generator,
                latents=latents,
                audio_latents=audio_latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=self.text_seq_len,
                output_type=output_type,
                device=self.device,
                sigmas=sigmas,
            )
            if output_type == "latent":
                return Ltx2AudioVideoOutput(video_latents=video, audio_latents=audio)
            return Ltx2AudioVideoOutput(
                frames=video,
                audio=audio,
                sample_rate=int(self.pipe.vocoder.config.output_sampling_rate),
            )
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **self.shape,
            frame_rate=self.frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=float(guidance_scale),
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=self.text_seq_len,
            output_type=output_type,
        )
        if output_type == "latent":
            return Ltx2AudioVideoOutput(video_latents=out.frames, audio_latents=out.audio)
        return Ltx2AudioVideoOutput(
            frames=out.frames,
            audio=out.audio,
            sample_rate=int(self.pipe.vocoder.config.output_sampling_rate),
        )

    def generate(self, prompt=None, *, output_type="pt", **kwargs) -> torch.Tensor:
        """Return silent video as CPU fp32 BCTHW in [-1, 1] for CLI and serving."""
        if output_type != "pt":
            raise ValueError(
                "LTX-2 generic generate supports output_type='pt'; use generate_audio_video for latents"
            )
        out = self.generate_audio_video(prompt, output_type="pt", **kwargs)
        return (
            out.frames.detach()
            .to(device="cpu", dtype=torch.float32)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            * 2
            - 1
        )


__all__ = ["CudaLtx2Application", "Ltx2AudioVideoOutput"]
