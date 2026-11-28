"""LTX-2 wrapper for the CUDA backend; shared accelerator modeling is unchanged.

Under CFG-parallel each of two processes holds the whole transformer. A call carrying
guidance's ``[negative, positive]`` batch of 2 runs this rank's branch at batch 1 and
all-gathers both outputs over the CFG group, so every rank returns the full batch.
"""

from __future__ import annotations

import torch
from torch import nn

from difflet.backends.cuda.ops_impl import parallel_mesh
from difflet.backends.cuda.ops_impl.collectives import (
    gather_from_tensor_model_parallel_region_with_dim,
    scatter_to_process_group_spmd,
)

#: The transformer's keyword arguments that carry guidance's batch of 2, in both profiles'
#: calls: the latent streams, the prompt embeddings and masks, timesteps, sigmas and
#: coordinates. The others are shapes and flags, or inputs only a batch-1 call carries.
_CFG_BATCHED = (
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "audio_encoder_hidden_states",
    "timestep",
    "audio_timestep",
    "sigma",
    "audio_sigma",
    "encoder_attention_mask",
    "audio_encoder_attention_mask",
    "video_coords",
    "audio_coords",
)


class CudaLtx2TransformerModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

        self.config = config
        self.cfg_group = None
        if config.cfg_parallel_enabled:
            # The component initialised the cfg=2 mesh before building this module.
            self.cfg_group = parallel_mesh.get_cfg_group()
            self.cfg_rank = parallel_mesh.get_axis_rank("cfg")
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
        return self.transformer_call(
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

    def transformer_call(self, **kwargs):
        """The inner transformer's keyword call, one CFG branch per rank under CFG-parallel.

        A batch of 1 (a request without guidance) has no pair to split: every rank runs it
        whole, so the result is one GPU's.
        """
        if self.cfg_group is None or kwargs["hidden_states"].shape[0] == 1:
            return self.transformer(**kwargs)
        for name in _CFG_BATCHED:
            if kwargs.get(name) is not None:
                kwargs[name] = scatter_to_process_group_spmd(
                    kwargs[name], 0, self.cfg_rank, process_group=self.cfg_group
                )
        video, audio = self.transformer(**kwargs)
        return tuple(
            gather_from_tensor_model_parallel_region_with_dim(out, 0, process_group=self.cfg_group)
            for out in (video, audio)
        )
