"""Trainium wrapper for the Qwen-Image DiT transformer."""

from __future__ import annotations

import math
import os
from typing import List

import torch
from torch import nn
from diffusers.models.attention_dispatch import dispatch_attention_fn

from nova.backends.trainium.core.application_base import NeuronApplicationBase
from nova.backends.trainium.core.config import InferenceConfig
from nova.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from nova.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    attention,
    get_tensor_model_parallel_size,
)


class QwenImageTransformerInferenceConfig(InferenceConfig):
    """Inference config for the Qwen-Image transformer component."""

    def add_derived_config(self):
        super().add_derived_config()
        if getattr(self, "out_channels", None) is None:
            self.out_channels = self.in_channels // 4
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = 1024
        if not hasattr(self, "vae_scale_factor"):
            self.vae_scale_factor = 8

    def get_required_attributes(self) -> List[str]:
        return [
            "patch_size",
            "in_channels",
            "out_channels",
            "num_layers",
            "attention_head_dim",
            "num_attention_heads",
            "joint_attention_dim",
            "guidance_embeds",
            "axes_dims_rope",
            "height",
            "width",
        ]

    @property
    def latent_height(self) -> int:
        return 2 * (int(self.height) // (int(self.vae_scale_factor) * 2))

    @property
    def latent_width(self) -> int:
        return 2 * (int(self.width) // (int(self.vae_scale_factor) * 2))

    @property
    def packed_height(self) -> int:
        return self.latent_height // int(self.patch_size)

    @property
    def packed_width(self) -> int:
        return self.latent_width // int(self.patch_size)

    @property
    def image_seq_len(self) -> int:
        return self.packed_height * self.packed_width

    def validate_config(self):
        super().validate_config()
        if isinstance(self.axes_dims_rope, list):
            self.axes_dims_rope = tuple(self.axes_dims_rope)
        if sum(int(dim) for dim in self.axes_dims_rope) != int(self.attention_head_dim):
            raise ValueError("Qwen-Image axes_dims_rope must sum to attention_head_dim.")
        if any(int(dim) % 2 != 0 for dim in self.axes_dims_rope):
            raise ValueError("Qwen-Image axes_dims_rope entries must be even.")
        if int(self.patch_size) != 2:
            raise NotImplementedError("Qwen-Image M4a currently supports patch_size=2.")
        if int(self.height) % (int(self.vae_scale_factor) * 2) != 0:
            raise ValueError("Qwen-Image compile height must be divisible by 16.")
        if int(self.width) % (int(self.vae_scale_factor) * 2) != 0:
            raise ValueError("Qwen-Image compile width must be divisible by 16.")
        if self.latent_height % int(self.patch_size) != 0:
            raise ValueError("Qwen-Image latent height must be divisible by patch_size.")
        if self.latent_width % int(self.patch_size) != 0:
            raise ValueError("Qwen-Image latent width must be divisible by patch_size.")
        if getattr(self, "use_additional_t_cond", False):
            raise NotImplementedError("Qwen-Image M4a does not support additional_t_cond yet.")


def _qwen_complex_rope_to_real(freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs.real.repeat_interleave(2, dim=-1)
    sin = freqs.imag.repeat_interleave(2, dim=-1)
    return cos, sin


def _qwen_rope_as_real(freqs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
    if isinstance(freqs, tuple):
        return freqs
    return _qwen_complex_rope_to_real(freqs)


def _apply_qwen_rope_real(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = freqs
    cos = cos[None, :, None, :].to(x.device)
    sin = sin[None, :, None, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class _StaticQwenImageRealRope(nn.Module):
    """Static real-valued Qwen RoPE for one compiled image/text shape."""

    def __init__(
        self,
        *,
        img_rope: tuple[torch.Tensor, torch.Tensor],
        txt_rope: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        super().__init__()
        self.register_buffer("img_cos", img_rope[0], persistent=False)
        self.register_buffer("img_sin", img_rope[1], persistent=False)
        self.register_buffer("txt_cos", txt_rope[0], persistent=False)
        self.register_buffer("txt_sin", txt_rope[1], persistent=False)

    def forward(self, *args, **kwargs):
        del args, kwargs
        return (self.img_cos, self.img_sin), (self.txt_cos, self.txt_sin)


class _ZeroQwenImageAttention(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        return torch.zeros_like(hidden_states), torch.zeros_like(encoder_hidden_states)


class _ZeroLikeModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _column_parallel_like(linear: nn.Linear, *, gather_output: bool) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        gather_output=gather_output,
    )


def _row_parallel_like(linear: nn.Linear, *, input_is_parallel: bool) -> RowParallelLinear:
    return RowParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        input_is_parallel=input_is_parallel,
    )


def _safe_tensor_parallel_size() -> int:
    try:
        return int(get_tensor_model_parallel_size())
    except AssertionError:
        return 1


def _replace_qwen_linears_for_tp(transformer: nn.Module) -> None:
    """Shard Qwen's largest transformer linears across tensor-parallel ranks."""

    tp_degree = _safe_tensor_parallel_size()
    if tp_degree <= 1:
        return

    replicate_modulation = _env_flag("NOVA_QWEN_TP_REPLICATE_MOD")
    replicate_attention = _env_flag("NOVA_QWEN_TP_REPLICATE_ATTN")
    replicate_mlp = _env_flag("NOVA_QWEN_TP_REPLICATE_MLP")
    replicate_io = _env_flag("NOVA_QWEN_TP_REPLICATE_IO")
    replicate_time = _env_flag("NOVA_QWEN_TP_REPLICATE_TIME")
    replicate_out = _env_flag("NOVA_QWEN_TP_REPLICATE_OUT")

    if not replicate_time:
        transformer.time_text_embed.timestep_embedder.linear_1 = _column_parallel_like(
            transformer.time_text_embed.timestep_embedder.linear_1,
            gather_output=True,
        )
        transformer.time_text_embed.timestep_embedder.linear_2 = _column_parallel_like(
            transformer.time_text_embed.timestep_embedder.linear_2,
            gather_output=True,
        )
    if not replicate_io:
        transformer.img_in = _column_parallel_like(transformer.img_in, gather_output=True)
        transformer.txt_in = _column_parallel_like(transformer.txt_in, gather_output=True)
    if not replicate_out:
        transformer.norm_out.linear = _column_parallel_like(
            transformer.norm_out.linear,
            gather_output=True,
        )

    for block in transformer.transformer_blocks:
        if not replicate_modulation:
            block.img_mod[1] = _column_parallel_like(block.img_mod[1], gather_output=True)
            block.txt_mod[1] = _column_parallel_like(block.txt_mod[1], gather_output=True)

        attn = block.attn
        if not replicate_attention:
            if int(attn.heads) % int(tp_degree) != 0:
                raise ValueError(f"Qwen-Image attention heads {attn.heads} must divide tp={tp_degree}.")
            attn.heads = int(attn.heads) // int(tp_degree)
            attn.to_q = _column_parallel_like(attn.to_q, gather_output=False)
            attn.to_k = _column_parallel_like(attn.to_k, gather_output=False)
            attn.to_v = _column_parallel_like(attn.to_v, gather_output=False)
            attn.add_q_proj = _column_parallel_like(attn.add_q_proj, gather_output=False)
            attn.add_k_proj = _column_parallel_like(attn.add_k_proj, gather_output=False)
            attn.add_v_proj = _column_parallel_like(attn.add_v_proj, gather_output=False)
            attn.to_out[0] = _row_parallel_like(attn.to_out[0], input_is_parallel=True)
            attn.to_add_out = _row_parallel_like(attn.to_add_out, input_is_parallel=True)

        if not replicate_mlp:
            block.img_mlp.net[0].proj = _column_parallel_like(
                block.img_mlp.net[0].proj,
                gather_output=False,
            )
            block.img_mlp.net[2] = _row_parallel_like(
                block.img_mlp.net[2],
                input_is_parallel=True,
            )
            block.txt_mlp.net[0].proj = _column_parallel_like(
                block.txt_mlp.net[0].proj,
                gather_output=False,
            )
            block.txt_mlp.net[2] = _row_parallel_like(
                block.txt_mlp.net[2],
                input_is_parallel=True,
            )


def _apply_qwen_block_diagnostics(transformer: nn.Module) -> None:
    zero_attention = _env_flag("NOVA_QWEN_ZERO_BLOCK_ATTN")
    zero_mlp = _env_flag("NOVA_QWEN_ZERO_BLOCK_MLP")
    if not zero_attention and not zero_mlp:
        return

    for block in transformer.transformer_blocks:
        if zero_attention:
            block.attn = _ZeroQwenImageAttention()
        if zero_mlp:
            block.img_mlp = _ZeroLikeModule()
            block.txt_mlp = _ZeroLikeModule()


class _QwenImageTrainiumAttnProcessor:
    """Qwen double-stream attention with real-valued RoPE for Neuron tracing."""

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del encoder_hidden_states_mask
        if encoder_hidden_states is None:
            raise ValueError("Qwen-Image Trainium attention requires encoder_hidden_states.")

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_rope = _qwen_rope_as_real(img_freqs)
            txt_rope = _qwen_rope_as_real(txt_freqs)
            img_query = _apply_qwen_rope_real(img_query, img_rope)
            img_key = _apply_qwen_rope_real(img_key, img_rope)
            txt_query = _apply_qwen_rope_real(txt_query, txt_rope)
            txt_key = _apply_qwen_rope_real(txt_key, txt_rope)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        head_dim = joint_query.shape[-1]
        if attention_mask is None:
            # Unmasked Qwen-Image joint attention → flash attention_cte
            # (device-only, ~4x over compiled SDPA — cclog 90/m10). q/k/v are
            # (B, S, heads, dim); the kernel wants (B*heads, S, dim) with the
            # heads folded into the batch axis. Attention is per-rank: q/k/v
            # already carry this rank's head shard from the ColumnParallel
            # projections (_parallel_config is None — no context parallelism).
            q = joint_query.transpose(1, 2)
            k = joint_key.transpose(1, 2)
            v = joint_value.transpose(1, 2)
            b, h, s_q, d = q.shape
            s_k = k.shape[2]
            attn_out = attention(
                q.reshape(b * h, s_q, d),
                k.reshape(b * h, s_k, d),
                v.reshape(b * h, s_k, d),
                scale=1.0 / math.sqrt(head_dim),
                causal=False,
                attention_mask=None,
                tp_q=True,
                tp_k=True,
                tp_out=False,
            )
            joint_hidden_states = attn_out.reshape(b, h, s_q, d).transpose(1, 2)
        else:
            joint_hidden_states = dispatch_attention_fn(
                joint_query,
                joint_key,
                joint_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


class _QwenImageTransformerTraceModule(nn.Module):
    """Fix non-tensor Qwen-Image pipeline args at trace time."""

    def __init__(self, config: QwenImageTransformerInferenceConfig):
        super().__init__()
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformer2DModel,
        )

        self.config = config
        self.guidance_embeds = bool(getattr(config, "guidance_embeds", False))
        self.img_shapes = [[(1, int(config.packed_height), int(config.packed_width))]]
        self.transformer = QwenImageTransformer2DModel(
            patch_size=int(config.patch_size),
            in_channels=int(config.in_channels),
            out_channels=int(config.out_channels),
            num_layers=int(config.num_layers),
            attention_head_dim=int(config.attention_head_dim),
            num_attention_heads=int(config.num_attention_heads),
            joint_attention_dim=int(config.joint_attention_dim),
            guidance_embeds=self.guidance_embeds,
            axes_dims_rope=tuple(config.axes_dims_rope),
            zero_cond_t=bool(getattr(config, "zero_cond_t", False)),
            use_layer3d_rope=bool(getattr(config, "use_layer3d_rope", False)),
        )
        _replace_qwen_linears_for_tp(self.transformer)
        _apply_qwen_block_diagnostics(self.transformer)
        img_freqs, txt_freqs = self.transformer.pos_embed(
            self.img_shapes,
            max_txt_seq_len=int(config.text_seq_len),
            device=torch.device("cpu"),
        )
        self.transformer.pos_embed = _StaticQwenImageRealRope(
            img_rope=_qwen_complex_rope_to_real(img_freqs),
            txt_rope=_qwen_complex_rope_to_real(txt_freqs),
        )
        for block in self.transformer.transformer_blocks:
            block.attn.processor = _QwenImageTrainiumAttnProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        # The diffusers mask path computes active text length with arange/where.
        # That currently fails during XLA tracing for the fixed-shape M4a
        # boundary, so the first Trainium closure uses full-length cached text
        # embeddings and keeps mask support as a separate follow-up.
        del encoder_hidden_states_mask
        guidance_arg = guidance if self.guidance_embeds else None
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            guidance=guidance_arg,
            img_shapes=self.img_shapes * int(hidden_states.shape[0]),
            return_dict=False,
        )[0]

    def teacache_mod_input(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        """Block-0 modulated image input for TeaCache (cclog 81 — fused-A port).

        Mirrors QwenImageTransformer2DModel.forward up to the first block's
        image modulation: img_in (patchify) -> time_text_embed -> block0.img_mod
        / img_norm1 / _modulate. Calls the diffusers model's own submodules so
        it stays faithful to the real forward. Assumes guidance_embeds=False and
        zero_cond_t=False (the M4a Trainium config); guidance is forwarded only
        when guidance_embeds is set.
        """
        t = self.transformer
        hs = t.img_in(hidden_states)
        timestep = timestep.to(hs.dtype)
        guidance_arg = (guidance.to(hs.dtype) * 1000) if self.guidance_embeds else None
        if guidance_arg is None:
            temb = t.time_text_embed(timestep, hs, None)
        else:
            temb = t.time_text_embed(timestep, guidance_arg, hs, None)
        block0 = t.transformer_blocks[0]
        img_mod1 = block0.img_mod(temb).chunk(2, dim=-1)[0]
        img_normed = block0.img_norm1(hs)
        modulated, _gate = block0._modulate(img_normed, img_mod1, None)
        return modulated


class ModelWrapperQwenImageTransformer(ModelWrapper):
    """ModelBuilder wrapper for Qwen-Image transformer compile inputs."""

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
        text_seq_len = int(getattr(self.config, "text_seq_len", 1024))

        return [
            (
                torch.randn(
                    [batch_size, self.config.image_seq_len, self.config.in_channels],
                    dtype=dtype,
                ),
                torch.ones([batch_size], dtype=dtype),
                torch.randn(
                    [batch_size, text_seq_len, self.config.joint_attention_dim],
                    dtype=dtype,
                ),
                torch.ones([batch_size, text_seq_len], dtype=torch.bool),
                torch.ones([batch_size], dtype=dtype),
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
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        guidance,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            guidance,
        )


class NeuronQwenImageTransformerApplication(NeuronApplicationBase):
    """Compile/load wrapper for ``QwenImageTransformer2DModel``."""

    _model_cls = _QwenImageTransformerTraceModule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag="QwenImageTransformer2DModel",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return QwenImageTransformerInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperQwenImageTransformer

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
