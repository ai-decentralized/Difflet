"""MiniMax-H3 on one CUDA GPU: text-to-video and first-frame image-to-video, with audio.

Every component stays on the device, about 122 GB of weights, so H3 needs a GPU of the
B200's size. ``load`` brings up the transformer (with compiled blocks under
``torch_compile``) and its companions: the Qwen3-VL text encoder with its vision tower, the
visual VAE's encoder (for the keyframe) and decoder, and the audio VAE's decoder. A request
encodes the prompt (and the keyframe), denoises (``denoise.denoise``) and decodes both
streams; the media are returned on the CPU.

The modules are CUDA's own (``modeling``, ``vae``, ``audio_vae``; the text encoder is
Transformers' Qwen3-VL). The keyframe's canvas and its presentation to the text encoder, the
packed layout and the schedules are the shared FL2VA helpers in ``difflet.models.minimax_h3``,
as on the neuron backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from difflet.backends.cuda.minimax_h3.audio_vae import decode_audio, load_audio_decoder
from difflet.backends.cuda.minimax_h3.denoise import denoise
from difflet.backends.cuda.minimax_h3.spec import SPEC
from difflet.backends.cuda.minimax_h3.vae import (
    decode_video,
    load_video_decoder,
    load_video_encoder,
)
from difflet.backends.gpu.application import GpuApplication
from difflet.models.minimax_h3.contracts import build_t2va_layout
from difflet.models.minimax_h3.keyframes import build_fl2va_presentation, prepare_keyframes
from difflet.models.minimax_h3.pipeline import load_minimax_h3_schedulers

# The visual VAE decodes 256-pixel tiles.
_MIN_SIDE = 256
_MAX_PROMPT_TOKENS = SPEC.prompt_budgets[0].max_tokens


@dataclass(frozen=True)
class MiniMaxH3Output:
    #: ``(1, 3, F, H, W)`` in [0, 1].
    video: torch.Tensor
    #: ``(1, 2, samples)`` in [-1, 1] at ``sample_rate``.
    audio: torch.Tensor
    sample_rate: int


class CudaMiniMaxH3Application(GpuApplication):
    spec = SPEC

    def _configure(self, kwargs) -> None:
        if self.dtype != torch.bfloat16:
            raise ValueError("MiniMax-H3 on CUDA runs in bf16")
        if min(self.shape["height"], self.shape["width"]) < _MIN_SIDE:
            raise ValueError(f"MiniMax-H3 on CUDA needs height and width of at least {_MIN_SIDE}")
        # The reference layout checks the shape before any checkpoint I/O.
        self.num_latent_frames = build_t2va_layout(
            num_text_tokens=1, **self.shape
        ).num_latent_frames

    def _build_transformers(self) -> None:
        from difflet.backends.cuda.minimax_h3.transformer import CudaMiniMaxH3TransformerApplication

        self.transformer = CudaMiniMaxH3TransformerApplication(
            model_path=self.transformer_path, device=self.device, torch_compile=self.torch_compile
        )

    def load_companions(self) -> None:
        from transformers import AutoImageProcessor, AutoTokenizer

        from difflet.backends.cuda.minimax_h3.text_encoder import load_text_encoder

        root = Path(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        # Only the image processor: the full Qwen3-VL processor also builds a video processor
        # that needs torchvision.
        self.image_processor = AutoImageProcessor.from_pretrained(
            root / "processor", local_files_only=True
        )
        self.text_encoder = load_text_encoder(
            root / "text_encoder", dtype=self.dtype, device=self.device
        )
        self.video_encoder = load_video_encoder(root / "vae", device=self.device)
        self.video_decoder = load_video_decoder(root / "vae", device=self.device)
        self.audio_decoder = load_audio_decoder(root / "audio_vae", device=self.device)
        self.video_scheduler, self.audio_scheduler = load_minimax_h3_schedulers(root)

    # ------------------------------------------------------------ encoders

    def encode_prompt(self, prompt: str, keyframes=()) -> tuple[torch.Tensor, torch.Tensor]:
        """The hidden states of the shared FL2VA presentation, ``(1, tokens, 5120)``, and each
        token's row tag. T2VA (no keyframes) is the prompt through the language model; with a
        keyframe, ``"<Picture 1>: "`` and its image block precede the prompt and go through the
        vision tower too, the image block video-tagged."""
        num_prompt_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if not 1 <= num_prompt_tokens <= _MAX_PROMPT_TOKENS:
            raise ValueError(
                f"MiniMax-H3 prompts take 1-{_MAX_PROMPT_TOKENS} tokens, got {num_prompt_tokens}"
            )
        token_ids, token_tags, vision = build_fl2va_presentation(
            self.tokenizer, self.image_processor, prompt, keyframes
        )
        ids = torch.tensor([token_ids], device=self.device)
        if not vision:
            hidden = self.text_encoder.language_model(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
            ).last_hidden_state
            return hidden, token_tags
        pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        hidden = self.text_encoder(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=vision["pixel_values"].to(self.device),
            image_grid_thw=vision["image_grid_thw"].to(self.device),
            mm_token_type_ids=(ids == pad).to(torch.int32),
            use_cache=False,
        ).last_hidden_state
        return hidden, token_tags

    def encode_keyframe(self, image, *, seed: int) -> torch.Tensor:
        """The keyframe's fixed, noise-augmented anchor rows ``(1, rows_per_frame, 96)``."""
        from difflet.backends.cuda.minimax_h3.keyframe import encode_keyframe, keyframe_anchor_rows

        latent = encode_keyframe(self.video_encoder, image)
        return keyframe_anchor_rows(
            latent, seed=seed, num_target_latent_frames=self.num_latent_frames
        )

    # ------------------------------------------------------------- request

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        num_inference_steps: int = SPEC.default_steps,
        seed: int = 0,
        num_frames: int | None = None,
        first_image=None,
        last_image=None,
        output_type: str = "pt",
    ) -> MiniMaxH3Output:
        """One clip with its audio. ``num_inference_steps`` counts denoising steps (the
        reference's convention); ``first_image`` (a PIL image) conditions the first frame."""
        if self.transformer is None or not self.transformer.is_loaded:
            raise RuntimeError("MiniMax-H3 is not loaded; call load() first")
        if output_type != "pt":
            raise ValueError("MiniMax-H3 returns tensors (output_type='pt')")
        if num_frames not in (None, self.shape["num_frames"]):
            raise ValueError(f"MiniMax-H3 was loaded for {self.shape['num_frames']} frames")
        if last_image is not None:
            raise ValueError("MiniMax-H3 on CUDA conditions on a first frame only")
        keyframes, keyframe_rows = [], None
        if first_image is not None:
            keyframes, _, _, _ = prepare_keyframes(
                first_image, None, height=self.shape["height"], width=self.shape["width"]
            )
        text, text_tags = self.encode_prompt(prompt, keyframes)
        if keyframes:
            keyframe_rows = self.encode_keyframe(keyframes[0], seed=seed)
        latents = denoise(
            self.transformer,
            encoder_hidden_states=text,
            **self.shape,
            num_inference_steps=num_inference_steps,
            seed=seed,
            video_scheduler=self.video_scheduler,
            audio_scheduler=self.audio_scheduler,
            device=self.device,
            text_token_tags=text_tags,
            keyframe_rows=keyframe_rows,
        )
        video = decode_video(self.video_decoder, latents.video_latents).cpu()
        audio = decode_audio(self.audio_decoder, latents.audio_latents).cpu()
        sample_rate = int(self.audio_decoder.config.sampling_rate)
        return MiniMaxH3Output(video=video, audio=audio, sample_rate=sample_rate)


__all__ = ["CudaMiniMaxH3Application", "MiniMaxH3Output"]
