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
            pos_embed_max_pos=int(config.pos_embed_max_pos),
            base_height=int(config.base_height),
            base_width=int(config.base_width),
            gated_attn=bool(config.gated_attn),
            cross_attn_mod=bool(config.cross_attn_mod),
            audio_in_channels=int(config.audio_in_channels),
            audio_out_channels=int(config.audio_out_channels),
            audio_patch_size=int(config.audio_patch_size),
            audio_patch_size_t=int(config.audio_patch_size_t),
            audio_num_attention_heads=int(config.audio_num_attention_heads),
            audio_attention_head_dim=int(config.audio_attention_head_dim),
            audio_cross_attention_dim=int(config.audio_cross_attention_dim),
            audio_scale_factor=int(config.audio_scale_factor),
            audio_pos_embed_max_pos=int(config.audio_pos_embed_max_pos),
            audio_sampling_rate=int(config.audio_sampling_rate),
            audio_hop_length=int(config.audio_hop_length),
            audio_gated_attn=bool(config.audio_gated_attn),
            audio_cross_attn_mod=bool(config.audio_cross_attn_mod),
            num_layers=int(config.num_layers),
            activation_fn=str(config.activation_fn),
            qk_norm=str(config.qk_norm),
            norm_elementwise_affine=bool(config.norm_elementwise_affine),
            norm_eps=float(config.norm_eps),
            caption_channels=int(config.caption_channels),
            attention_bias=bool(config.attention_bias),
            attention_out_bias=bool(config.attention_out_bias),
            rope_theta=float(config.rope_theta),
            rope_double_precision=bool(config.rope_double_precision),
            causal_offset=int(config.causal_offset),
            timestep_scale_multiplier=int(config.timestep_scale_multiplier),
            cross_attn_timestep_scale_multiplier=int(config.cross_attn_timestep_scale_multiplier),
            rope_type=str(config.rope_type),
            use_prompt_embeddings=bool(config.use_prompt_embeddings),
            perturbed_attn=bool(config.perturbed_attn),
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
            fps=float(self.config.frame_rate),
            audio_num_frames=int(self.config.audio_num_frames),
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep=bool(self.config.use_cross_timestep),
            return_dict=False,
        )
        return (video_out, audio_out)
