"""Trainium wrapper for the LTX-2 dual-stream transformer."""

from __future__ import annotations

import os
from typing import List

import torch
from torch import nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.models.ltx_2.application import LTX_2_DEFAULT_TEXT_SEQ_LEN


def _difflet_apply_split_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Equivalent split RoPE with explicit per-head reshape for XLA tracing."""

    cos, sin = freqs
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        batch_size, num_heads, seq_len, _ = cos.shape
        head_dim = x.shape[-1] // num_heads
        x = x.reshape(batch_size, seq_len, num_heads, head_dim).swapaxes(1, 2)
        needs_reshape = True

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"Expected x.shape[-1] to be even for split rotary, got {last}.")
    rotary_dim = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, rotary_dim).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]

    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]

    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)

    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(batch_size, seq_len, -1)
    return out.to(dtype=x_dtype)


class LTX2TransformerInferenceConfig(InferenceConfig):
    """Inference config for ``LTX2VideoTransformer3DModel``."""

    def add_derived_config(self):
        super().add_derived_config()
        if getattr(self, "out_channels", None) is None:
            self.out_channels = self.in_channels
        if getattr(self, "audio_out_channels", None) is None:
            self.audio_out_channels = self.audio_in_channels
        if isinstance(getattr(self, "vae_scale_factors", None), list):
            self.vae_scale_factors = tuple(self.vae_scale_factors)
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = LTX_2_DEFAULT_TEXT_SEQ_LEN
        if not hasattr(self, "audio_text_seq_len"):
            self.audio_text_seq_len = self.text_seq_len
        if not hasattr(self, "audio_num_frames") or self.audio_num_frames is None:
            self.audio_num_frames = self._infer_audio_num_frames()
        self.video_text_dim = (
            int(self.caption_channels)
            if bool(getattr(self, "use_prompt_embeddings", True))
            else int(self.cross_attention_dim)
        )
        self.audio_text_dim = (
            int(self.caption_channels)
            if bool(getattr(self, "use_prompt_embeddings", True))
            else int(self.audio_cross_attention_dim)
        )

    def get_required_attributes(self) -> List[str]:
        return [
            "in_channels",
            "out_channels",
            "patch_size",
            "patch_size_t",
            "num_attention_heads",
            "attention_head_dim",
            "cross_attention_dim",
            "vae_scale_factors",
            "audio_in_channels",
            "audio_out_channels",
            "audio_patch_size",
            "audio_patch_size_t",
            "audio_num_attention_heads",
            "audio_attention_head_dim",
            "audio_cross_attention_dim",
            "audio_scale_factor",
            "audio_sampling_rate",
            "audio_hop_length",
            "num_layers",
            "caption_channels",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def latent_num_frames(self) -> int:
        temporal_scale = int(self.vae_scale_factors[0])
        return (int(self.num_frames) - 1) // temporal_scale + 1

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.vae_scale_factors[1])

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.vae_scale_factors[2])

    @property
    def video_seq_len(self) -> int:
        return (
            (self.latent_num_frames // int(self.patch_size_t))
            * (self.latent_height // int(self.patch_size))
            * (self.latent_width // int(self.patch_size))
        )

    @property
    def audio_seq_len(self) -> int:
        # LTX-2 audio latents are packed as [B, audio_frames, channels * mel_bins].
        # The mel axis is part of the feature dimension, not the sequence axis.
        return int(self.audio_num_frames) // int(self.audio_patch_size_t)

    def _infer_audio_num_frames(self) -> int:
        frame_rate = float(getattr(self, "frame_rate", 24.0))
        duration_s = int(self.num_frames) / frame_rate
        audio_latents_per_second = (
            int(self.audio_sampling_rate)
            / int(self.audio_hop_length)
            / float(getattr(self, "audio_vae_temporal_compression_ratio", 4))
        )
        return round(duration_s * audio_latents_per_second)

    def validate_config(self):
        super().validate_config()
        if int(self.patch_size) != 1 or int(self.patch_size_t) != 1:
            raise NotImplementedError(
                "LTX-2 M4c currently supports video patch_size=patch_size_t=1."
            )
        if int(self.audio_patch_size_t) != 1:
            raise NotImplementedError("LTX-2 M4c currently supports audio_patch_size_t=1.")
        if int(self.height) % int(self.vae_scale_factors[1]) != 0:
            raise ValueError("LTX-2 compile height must be divisible by the VAE spatial scale.")
        if int(self.width) % int(self.vae_scale_factors[2]) != 0:
            raise ValueError("LTX-2 compile width must be divisible by the VAE spatial scale.")
        if self.latent_num_frames % int(self.patch_size_t) != 0:
            raise ValueError("LTX-2 latent frame count must be divisible by patch_size_t.")
        if self.latent_height % int(self.patch_size) != 0:
            raise ValueError("LTX-2 latent height must be divisible by patch_size.")
        if self.latent_width % int(self.patch_size) != 0:
            raise ValueError("LTX-2 latent width must be divisible by patch_size.")
        if int(self.audio_num_frames) % int(self.audio_patch_size_t) != 0:
            raise ValueError(
                "LTX-2 audio latent frame count must be divisible by audio_patch_size_t."
            )


class _LTX2TransformerTraceModule(nn.Module):
    """Fix non-tensor LTX-2 transformer args at trace time."""

    def __init__(self, config: LTX2TransformerInferenceConfig):
        super().__init__()
        import diffusers.models.transformers.transformer_ltx2 as ltx2_transformer

        ltx2_transformer.apply_split_rotary_emb = _difflet_apply_split_rotary_emb
        LTX2VideoTransformer3DModel = ltx2_transformer.LTX2VideoTransformer3DModel

        self.config = config
        self.transformer = LTX2VideoTransformer3DModel(
            in_channels=int(config.in_channels),
            out_channels=int(config.out_channels),
            patch_size=int(config.patch_size),
            patch_size_t=int(config.patch_size_t),
            num_attention_heads=int(config.num_attention_heads),
            attention_head_dim=int(config.attention_head_dim),
            cross_attention_dim=int(config.cross_attention_dim),
            vae_scale_factors=tuple(config.vae_scale_factors),
            pos_embed_max_pos=int(getattr(config, "pos_embed_max_pos", 20)),
            base_height=int(getattr(config, "base_height", 2048)),
            base_width=int(getattr(config, "base_width", 2048)),
            gated_attn=bool(getattr(config, "gated_attn", False)),
            cross_attn_mod=bool(getattr(config, "cross_attn_mod", False)),
            audio_in_channels=int(config.audio_in_channels),
            audio_out_channels=int(config.audio_out_channels),
            audio_patch_size=int(config.audio_patch_size),
            audio_patch_size_t=int(config.audio_patch_size_t),
            audio_num_attention_heads=int(config.audio_num_attention_heads),
            audio_attention_head_dim=int(config.audio_attention_head_dim),
            audio_cross_attention_dim=int(config.audio_cross_attention_dim),
            audio_scale_factor=int(config.audio_scale_factor),
            audio_pos_embed_max_pos=int(getattr(config, "audio_pos_embed_max_pos", 20)),
            audio_sampling_rate=int(config.audio_sampling_rate),
            audio_hop_length=int(config.audio_hop_length),
            audio_gated_attn=bool(getattr(config, "audio_gated_attn", False)),
            audio_cross_attn_mod=bool(getattr(config, "audio_cross_attn_mod", False)),
            num_layers=int(config.num_layers),
            activation_fn=str(getattr(config, "activation_fn", "gelu-approximate")),
            qk_norm=str(getattr(config, "qk_norm", "rms_norm_across_heads")),
            norm_elementwise_affine=bool(getattr(config, "norm_elementwise_affine", False)),
            norm_eps=float(getattr(config, "norm_eps", 1e-6)),
            caption_channels=int(config.caption_channels),
            attention_bias=bool(getattr(config, "attention_bias", True)),
            attention_out_bias=bool(getattr(config, "attention_out_bias", True)),
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
            rope_double_precision=bool(getattr(config, "rope_double_precision", True)),
            causal_offset=int(getattr(config, "causal_offset", 1)),
            timestep_scale_multiplier=int(getattr(config, "timestep_scale_multiplier", 1000)),
            cross_attn_timestep_scale_multiplier=int(
                getattr(config, "cross_attn_timestep_scale_multiplier", 1000)
            ),
            rope_type=str(getattr(config, "rope_type", "interleaved")),
            use_prompt_embeddings=bool(getattr(config, "use_prompt_embeddings", True)),
            perturbed_attn=bool(getattr(config, "perturbed_attn", False)),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        sigma: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        audio_encoder_attention_mask: torch.Tensor,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            audio_timestep=timestep,
            sigma=sigma,
            audio_sigma=sigma,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=int(self.config.latent_num_frames),
            height=int(self.config.latent_height),
            width=int(self.config.latent_width),
            fps=float(getattr(self.config, "frame_rate", 24.0)),
            audio_num_frames=int(self.config.audio_num_frames),
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=bool(getattr(self.config, "use_cross_timestep", False)),
            return_dict=False,
        )


class ModelWrapperLTX2Transformer(ModelWrapper):
    """ModelBuilder wrapper for LTX-2 transformer compile inputs."""

    def __init__(
        self,
        config: InferenceConfig,
        model_cls,
        tag: str = "",
        compiler_args: str | None = None,
        priority_model_idx: int | None = None,
        model_init_kwargs=None,
    ) -> None:
        super().__init__(
            config,
            model_cls,
            tag,
            compiler_args,
            priority_model_idx,
            model_init_kwargs or {},
        )
        self.bucket_config = None

    def input_generator(self) -> list[tuple[torch.Tensor, ...]]:
        batch_size = int(getattr(self.config.neuron_config, "batch_size", 1))
        dtype = self.config.neuron_config.torch_dtype
        text_seq_len = int(getattr(self.config, "text_seq_len", LTX_2_DEFAULT_TEXT_SEQ_LEN))
        audio_text_seq_len = int(getattr(self.config, "audio_text_seq_len", text_seq_len))
        video_coords = _make_video_coords(
            batch_size=batch_size,
            num_frames=int(self.config.latent_num_frames),
            height=int(self.config.latent_height),
            width=int(self.config.latent_width),
            patch_size=int(self.config.patch_size),
            patch_size_t=int(self.config.patch_size_t),
            scale_factors=tuple(self.config.vae_scale_factors),
            causal_offset=int(getattr(self.config, "causal_offset", 1)),
            fps=float(getattr(self.config, "frame_rate", 24.0)),
        )
        audio_coords = _make_audio_coords(
            batch_size=batch_size,
            audio_num_frames=int(self.config.audio_num_frames),
            patch_size_t=int(self.config.audio_patch_size_t),
            scale_factor=int(self.config.audio_scale_factor),
            causal_offset=int(getattr(self.config, "causal_offset", 1)),
            sampling_rate=int(self.config.audio_sampling_rate),
            hop_length=int(self.config.audio_hop_length),
        )
        return [
            (
                torch.randn(
                    [batch_size, self.config.video_seq_len, self.config.in_channels],
                    dtype=dtype,
                ),
                torch.randn(
                    [batch_size, self.config.audio_seq_len, self.config.audio_in_channels],
                    dtype=dtype,
                ),
                torch.randn([batch_size, text_seq_len, self.config.video_text_dim], dtype=dtype),
                torch.randn(
                    [batch_size, audio_text_seq_len, self.config.audio_text_dim],
                    dtype=dtype,
                ),
                torch.ones([batch_size], dtype=dtype),
                torch.ones([batch_size], dtype=dtype),
                torch.ones([batch_size, text_seq_len], dtype=torch.bool),
                torch.ones([batch_size, audio_text_seq_len], dtype=torch.bool),
                video_coords,
                audio_coords,
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        sigma,
        encoder_attention_mask,
        audio_encoder_attention_mask,
        video_coords,
        audio_coords,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            audio_encoder_hidden_states,
            timestep,
            sigma,
            encoder_attention_mask,
            audio_encoder_attention_mask,
            video_coords,
            audio_coords,
        )


class NeuronLTX2TransformerApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``LTX2VideoTransformer3DModel``."""

    _model_cls = _LTX2TransformerTraceModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag="LTX2VideoTransformer3DModel",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype
        self._cpu_transformer = None

    def _load_cpu_transformer(self):
        """Host CPU copy of the LTX-2 transformer (for the TeaCache signal, cclog 87).

        Single-mode counterpart to the segmented runtime's host model — the TeaCache
        block-0 signal only needs a host transformer copy, independent of the device
        execution mode.
        """
        if self._cpu_transformer is None:
            from difflet.backends.trainium.ltx_2.segmented import _patch_ltx2_rope

            _patch_ltx2_rope()
            from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

            model = LTX2VideoTransformer3DModel.from_pretrained(self.model_path, torch_dtype=self.dtype)
            self._cpu_transformer = model.to(dtype=self.dtype).eval()
        return self._cpu_transformer

    @torch.no_grad()
    def teacache_mod_input(self, hidden_states, timestep):
        """TeaCache signal: block-0 modulated video self-attn input (cclog 87).

        ``norm1(proj_in(latent)) * (1 + scale_msa) + shift_msa`` from the host CPU
        transformer; modulation is timestep-only (identical for cond/uncond, so the
        caller passes the un-doubled latent + per-batch timestep).
        """
        model = self._load_cpu_transformer()
        hidden_states = hidden_states.to(dtype=self.dtype)
        batch_size = hidden_states.shape[0]
        hidden_states = model.proj_in(hidden_states)
        temb, _ = model.time_embed(
            timestep.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        block0 = model.transformer_blocks[0]
        video_ada_params = block0.get_mod_params(block0.scale_shift_table, temb, batch_size)
        shift_msa, scale_msa = video_ada_params[0], video_ada_params[1]
        return block0.norm1(hidden_states) * (1 + scale_msa) + shift_msa

    @classmethod
    def get_config_cls(cls):
        return LTX2TransformerInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperLTX2Transformer

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        compiler_args = (
            "--model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap' "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return compiler_args

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        del config
        return {
            key if key.startswith("transformer.") else f"transformer.{key}": value
            for key, value in state_dict.items()
        }

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


def _make_video_coords(
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    patch_size_t: int,
    scale_factors: tuple[int, int, int],
    causal_offset: int,
    fps: float,
) -> torch.Tensor:
    frames = torch.arange(0, num_frames, patch_size_t, dtype=torch.float32)
    rows = torch.arange(0, height, patch_size, dtype=torch.float32)
    cols = torch.arange(0, width, patch_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(frames, rows, cols, indexing="ij"), dim=0)
    patch_size_tensor = torch.tensor(
        (patch_size_t, patch_size, patch_size),
        dtype=grid.dtype,
    )
    latent_coords = torch.stack(
        [grid, grid + patch_size_tensor.view(3, 1, 1, 1)],
        dim=-1,
    )
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    scale_tensor = torch.tensor(scale_factors, dtype=latent_coords.dtype)
    pixel_coords = latent_coords * scale_tensor.view(1, 3, 1, 1)
    pixel_coords[:, 0, ...] = (
        pixel_coords[:, 0, ...] + int(causal_offset) - int(scale_factors[0])
    ).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return pixel_coords


def _make_audio_coords(
    *,
    batch_size: int,
    audio_num_frames: int,
    patch_size_t: int,
    scale_factor: int,
    causal_offset: int,
    sampling_rate: int,
    hop_length: int,
) -> torch.Tensor:
    coords = torch.arange(0, audio_num_frames, patch_size_t, dtype=torch.float32)
    start_mel = (coords * scale_factor + int(causal_offset) - int(scale_factor)).clamp(min=0)
    end_mel = (
        (coords + int(patch_size_t)) * scale_factor
        + int(causal_offset)
        - int(scale_factor)
    ).clamp(min=0)
    seconds_per_mel = float(hop_length) / float(sampling_rate)
    coords = torch.stack([start_mel * seconds_per_mel, end_mel * seconds_per_mel], dim=-1)
    return coords.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
