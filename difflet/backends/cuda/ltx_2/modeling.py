"""Single-device LTX-2 wrapper; shared accelerator modeling is unchanged."""

from __future__ import annotations

import torch
from torch import nn


class CudaLtx2TransformerModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

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
            norm_eps=float(getattr(config, "norm_eps", 1e-06)),
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
        video_out, audio_out = self.transformer(
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
        return (video_out, audio_out)
