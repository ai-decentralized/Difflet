"""Trainium application wrapper for the HunyuanVideo 1.5 DiT backbone."""

from __future__ import annotations

import math
import os
import types
from typing import List

import torch
import torch.nn as nn

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper
from difflet.backends.trainium.hunyuan_video.parallel15 import (
    JointBoundedProcessor,
    JointGatherKvProcessor,
    JointUlyssesProcessor,
    install_tensor_parallel,
)
from difflet.ops import (
    SPMDRank,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
    get_cfg_group,
    get_cfg_rank_spmd,
    get_cp_group,
    get_cp_rank_spmd,
    get_world_group,
    init_parallel_mesh,
    scatter_to_process_group_spmd,
)


class _HunyuanVideo15NoMaskAttnProcessor:
    """Experimental HV1.5 attention processor for all-valid token compiles."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, *, query_chunk_size: int | None = None) -> None:
        self.query_chunk_size = query_chunk_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        if self.query_chunk_size is None or query.shape[1] <= self.query_chunk_size:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            chunks = []
            for query_chunk in torch.split(query, self.query_chunk_size, dim=1):
                chunks.append(
                    dispatch_attention_fn(
                        query_chunk,
                        key,
                        value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        backend=self._attention_backend,
                        parallel_config=self._parallel_config,
                    )
                )
            hidden_states = torch.cat(chunks, dim=1)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class _HunyuanVideo15NkiNoMaskAttnProcessor(_HunyuanVideo15NoMaskAttnProcessor):
    """Experimental no-mask processor using Difflet's NKI attention kernel."""

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        from difflet.ops import attention

        batch_size, query_len, heads, head_dim = query.shape
        key_len = key.shape[1]
        value_len = value.shape[1]
        query = query.permute(0, 2, 1, 3).reshape(batch_size * heads, query_len, head_dim)
        key = key.permute(0, 2, 1, 3).reshape(batch_size * heads, key_len, head_dim)
        value = value.permute(0, 2, 1, 3).reshape(batch_size * heads, value_len, head_dim)
        hidden_states = attention(
            query,
            key,
            value,
            scale=1 / math.sqrt(head_dim),
            causal=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
        return hidden_states.reshape(batch_size, heads, query_len, head_dim).permute(0, 2, 1, 3)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        hidden_states = self._attention(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class _HunyuanVideo15ZeroAttnProcessor:
    """Experimental processor that removes block attention for capacity probes."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attn, attention_mask, image_rotary_emb
        context = None if encoder_hidden_states is None else torch.zeros_like(encoder_hidden_states)
        return torch.zeros_like(hidden_states), context


class _ZeroLikeModule(nn.Module):
    """Experimental replacement for FFN capacity probes."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


class HunyuanVideo15BackboneInferenceConfig(InferenceConfig):
    """Inference config for the HunyuanVideo 1.5 transformer component."""

    def add_derived_config(self):
        super().add_derived_config()
        if getattr(self, "out_channels", None) is None:
            self.out_channels = 32
        if not hasattr(self, "text_seq_len"):
            self.text_seq_len = 1000
        if not hasattr(self, "text_seq_len_2"):
            self.text_seq_len_2 = 256
        if not hasattr(self, "image_seq_len"):
            self.image_seq_len = 729
        if not hasattr(self, "spatial_compression_ratio"):
            self.spatial_compression_ratio = 16
        if not hasattr(self, "temporal_compression_ratio"):
            self.temporal_compression_ratio = 4
        for name, default in (
            ("cfg_parallel_enabled", False),
            ("context_parallel_enabled", False),
            ("cp_mode", "ulysses"),
            ("sp_enabled", False),
        ):
            if not hasattr(self, name):
                setattr(self, name, default)

    def get_required_attributes(self) -> List[str]:
        return [
            "in_channels",
            "out_channels",
            "num_attention_heads",
            "attention_head_dim",
            "num_layers",
            "num_refiner_layers",
            "mlp_ratio",
            "patch_size",
            "patch_size_t",
            "qk_norm",
            "text_embed_dim",
            "text_embed_2_dim",
            "image_embed_dim",
            "rope_theta",
            "rope_axes_dim",
            "height",
            "width",
            "num_frames",
        ]

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def latent_height(self) -> int:
        return int(self.height) // int(self.spatial_compression_ratio)

    @property
    def latent_width(self) -> int:
        return int(self.width) // int(self.spatial_compression_ratio)

    @property
    def latent_frames(self) -> int:
        ratio = int(self.temporal_compression_ratio)
        return (int(self.num_frames) - 1) // ratio + 1

    @property
    def num_image_tokens(self) -> int:
        """Full image-stream length N after patchify (before any sequence sharding)."""
        p, p_t = int(self.patch_size), int(self.patch_size_t)
        return (
            (self.latent_frames // p_t) * (self.latent_height // p) * (self.latent_width // p)
        )

    @property
    def cp_degree(self) -> int:
        """CP degree as the mesh infers it: world / (tp * cfg) when CP is enabled."""
        if not self.context_parallel_enabled:
            return 1
        cfg = 2 if self.cfg_parallel_enabled else 1
        return int(self.neuron_config.world_size) // (int(self.neuron_config.tp_degree) * cfg)

    def validate_config(self):
        super().validate_config()
        if isinstance(self.rope_axes_dim, list):
            self.rope_axes_dim = tuple(self.rope_axes_dim)
        if self.qk_norm != "rms_norm":
            raise NotImplementedError(
                "HunyuanVideo 1.5 currently supports only qk_norm='rms_norm'."
            )
        spatial_ratio = int(self.spatial_compression_ratio)
        if self.height % spatial_ratio != 0 or self.width % spatial_ratio != 0:
            raise ValueError(
                "HunyuanVideo 1.5 compile height/width must be divisible by "
                f"{spatial_ratio}."
            )
        if self.latent_height % self.patch_size != 0 or self.latent_width % self.patch_size != 0:
            raise ValueError("HunyuanVideo 1.5 latent height/width must be divisible by patch_size.")
        if self.latent_frames % self.patch_size_t != 0:
            raise ValueError("HunyuanVideo 1.5 latent frame count must be divisible by patch_size_t.")
        if self.cfg_parallel_enabled and self.context_parallel_enabled:
            from difflet.pipeline.parallel_config import cfg_cp_composable

            if not cfg_cp_composable():
                raise NotImplementedError(
                    "HunyuanVideo 1.5 CFG-parallel and context-parallel do not compose on "
                    "this backend (both consume the data-parallel lanes, see "
                    "cfg_cp_composable); enable only one of them."
                )
        heads, tp = int(self.num_attention_heads), int(self.neuron_config.tp_degree)
        if heads % tp != 0:
            options = [d for d in range(1, heads + 1) if heads % d == 0]
            raise ValueError(
                f"HunyuanVideo 1.5 num_attention_heads={heads} must be divisible by "
                f"tp_degree={tp}; use a tensor-parallel degree in {options}."
            )
        batch = int(self.neuron_config.batch_size)
        if self.cfg_parallel_enabled and batch != 2:
            raise ValueError(
                "HunyuanVideo 1.5 CFG-parallel stacks [uncond, cond] into one batch-2 "
                f"compile and gives each cfg rank one branch; got batch_size={batch}, "
                "use batch_size=2."
            )
        if self.sp_enabled:
            self._validate_sequence_parallel(heads, tp)
        if self.cp_degree > 1:
            self._validate_context_parallel(heads, tp, self.cp_degree)

    def _divisible_frames(self, degree: int) -> list[int]:
        """num_frames nearest below and above whose image-token count divides ``degree``."""
        p, p_t = int(self.patch_size), int(self.patch_size_t)
        hw = (self.latent_height // p) * (self.latent_width // p)
        step = p_t * (degree // math.gcd(hw, degree))
        lower = (self.latent_frames // step) * step
        ratio = int(self.temporal_compression_ratio)
        return [(t - 1) * ratio + 1 for t in (lower, lower + step) if t >= 1]

    def _validate_context_parallel(self, heads: int, tp: int, cp: int) -> None:
        if self.cp_mode not in ("ulysses", "gather_kv"):
            raise NotImplementedError(
                f"HunyuanVideo 1.5 AOT context parallelism supports cp_mode 'ulysses' and "
                f"'gather_kv', got {self.cp_mode!r}; use --cp-mode ulysses."
            )
        if self.cp_mode == "ulysses" and (heads // tp) % cp:
            cps = [d for d in range(2, heads // tp + 1) if (heads // tp) % d == 0]
            raise ValueError(
                f"HunyuanVideo 1.5 Ulysses splits the num_attention_heads={heads} // "
                f"tp_degree={tp} = {heads // tp} heads per TP rank over cp_degree={cp}, "
                f"which does not divide them; use cp_degree in {cps or [1]}, or "
                "cp_mode='gather_kv'."
            )
        tokens = self.num_image_tokens
        if tokens % cp:
            raise ValueError(
                f"HunyuanVideo 1.5 context parallelism splits the N={tokens} image tokens "
                f"of {self.height}x{self.width}x{self.num_frames} over cp_degree={cp}, but "
                f"N % {cp} = {tokens % cp}; use num_frames in {self._divisible_frames(cp)} "
                "at this resolution."
            )

    def _validate_sequence_parallel(self, heads: int, tp: int) -> None:
        if tp <= 1:
            options = [d for d in range(2, heads + 1) if heads % d == 0]
            raise ValueError(
                "HunyuanVideo 1.5 sequence parallelism shards the image stream over the "
                f"tensor-parallel group; got tp_degree={tp}, use a tp_degree in {options}."
            )
        if self.context_parallel_enabled:
            raise NotImplementedError(
                "HunyuanVideo 1.5 sequence parallelism does not compose with "
                "context parallelism (both shard the image sequence); enable only one."
            )
        tokens = self.num_image_tokens
        if tokens % tp:
            frames = self._divisible_frames(tp)
            tps = [d for d in range(2, heads + 1) if heads % d == 0 and tokens % d == 0]
            raise ValueError(
                f"HunyuanVideo 1.5 sequence parallelism splits the N={tokens} image tokens "
                f"of {self.height}x{self.width}x{self.num_frames} over tp_degree={tp}, but "
                f"N % {tp} = {tokens % tp}; use num_frames in {frames} at this resolution"
                + (f", or tp_degree in {tps}." if tps else ".")
            )


class ModelWrapperHunyuanVideo15Backbone(ModelWrapper):
    """ModelBuilder wrapper for HunyuanVideo 1.5 DiT compile inputs."""

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
        text_seq_len = int(getattr(self.config, "text_seq_len", 1000))
        text_seq_len_2 = int(getattr(self.config, "text_seq_len_2", 256))
        image_seq_len = int(getattr(self.config, "image_seq_len", 729))
        # cclog 86 (host-refiner): the refiner runs on host; the NEFF receives the
        # ALREADY-REFINED mllm embeds (inner_dim), not the raw text-encoder output.
        mllm_dim = self.config.inner_dim
        text_len = image_seq_len + text_seq_len_2 + text_seq_len

        return [
            (
                torch.randn(
                    [
                        batch_size,
                        self.config.in_channels,
                        self.config.latent_frames,
                        self.config.latent_height,
                        self.config.latent_width,
                    ],
                    dtype=dtype,
                ),
                torch.ones([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len, mllm_dim], dtype=dtype),
                torch.ones([batch_size, text_seq_len], dtype=torch.int64),
                torch.ones([batch_size], dtype=dtype),
                torch.randn([batch_size, text_seq_len_2, self.config.text_embed_2_dim], dtype=dtype),
                torch.ones([batch_size, text_seq_len_2], dtype=torch.int64),
                torch.zeros([batch_size, image_seq_len, self.config.image_embed_dim], dtype=dtype),
                # text_order: host-computed valid-first permutation of [image, byt5, mllm].
                torch.arange(text_len, dtype=torch.int64).expand(batch_size, text_len).contiguous(),
            )
        ]

    def get_model_instance(self):
        def _create_model():
            model = _HunyuanVideo15TraceModule(self.config)
            model = model.to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep_r,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds,
        text_order,
    ):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep_r,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            image_embeds,
            text_order,
        )


class NeuronHunyuanVideo15BackboneApplication(NeuronApplicationBase):
    """Compile/load wrapper for diffusers ``HunyuanVideo15Transformer3DModel``."""

    _model_cls = object

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_wrapper = self.get_model_wrapper_cls()
        self.model = self.model_wrapper(
            config=self.config,
            model_cls=self._model_cls,
            tag="HunyuanVideo15Transformer3DModel",
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)
        self.dtype = self.config.neuron_config.torch_dtype

    @classmethod
    def get_config_cls(cls):
        return HunyuanVideo15BackboneInferenceConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperHunyuanVideo15Backbone

    def forward(self, *model_inputs, **kwargs):
        return self.models[0](*model_inputs, **kwargs)

    def get_compiler_args(self) -> str:
        override = os.environ.get("DIFFLET_HUNYUAN15_COMPILER_ARGS")
        if override:
            os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
            return override
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
        state_dict = {
            key if key.startswith("transformer.") else f"transformer.{key}": value
            for key, value in state_dict.items()
        }
        # The trace module's root global_rank SPMDRank must hold arange(world_size)
        # whenever a sharding mode is on (CFG, CP or Megatron-SP), so each rank reads its
        # own rank; otherwise every rank acts as rank 0. Unprefixed: it lives on the root.
        if (
            getattr(config, "cfg_parallel_enabled", False)
            or getattr(config, "context_parallel_enabled", False)
            or getattr(config, "sp_enabled", False)
        ):
            world_size = config.neuron_config.world_size
            state_dict["global_rank.rank"] = torch.arange(0, world_size, dtype=torch.int32)
        # Megatron-SP scatters by the TP-local rank (root tp_rank_util, sharded over the
        # TP group as in Wan): under SP+CFG the global ranks reach 2*tp-1.
        if getattr(config, "sp_enabled", False):
            tp_degree = config.neuron_config.tp_degree
            state_dict["tp_rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass


def _model_kwargs_from_config(config: InferenceConfig) -> dict[str, object]:
    names = (
        "in_channels",
        "out_channels",
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "num_refiner_layers",
        "mlp_ratio",
        "patch_size",
        "patch_size_t",
        "qk_norm",
        "text_embed_dim",
        "text_embed_2_dim",
        "image_embed_dim",
        "rope_theta",
        "rope_axes_dim",
        "target_size",
        "task_type",
        "use_meanflow",
    )
    kwargs = {name: getattr(config, name) for name in names if hasattr(config, name)}
    if isinstance(kwargs.get("rope_axes_dim"), list):
        kwargs["rope_axes_dim"] = tuple(kwargs["rope_axes_dim"])
    return kwargs


def _hv15_static_reorder_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask,
    timestep_r=None,
    encoder_hidden_states_2=None,
    encoder_attention_mask_2=None,
    image_embeds=None,
    text_order=None,
    attention_kwargs=None,
    return_dict=True,
):
    """Static-reorder fork of HunyuanVideo15Transformer3DModel.forward (cclog 86).

    Bound to the trace module's transformer INSTANCE (never the diffusers class).
    Differences from diffusers:

    * ``encoder_hidden_states`` is the host-refined mllm stream (inner_dim): the token
      refiner runs on host, so ``context_embedder`` is skipped here.
    * The data-dependent valid-first reorder (``text[text_mask]`` boolean indexing, a
      dynamic shape that cannot lower to a static NEFF) is replaced by a gather with the
      host-computed ``text_order`` permutation (``parallel15.hv15_text_order``). After it
      the valid text tokens are the contiguous prefix, so the blocks get ``text_valid``
      (int32 ``[B]``, counted in fp32) in the ``attention_mask`` slot and the processor
      attends keys ``[0, N + text_valid)``. The encoder stream has no rope, so the order
      only has to be valid-first; padding embeds are zeroed.
    * Under context parallelism the image stream and its rope are scattered over the CP
      group once before the blocks and the image stream is gathered once after; the text
      stream stays replicated. Megatron-SP does the same over the TP group.
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    image_rotary_emb = self.rope(hidden_states)
    temb = self.time_embed(timestep, timestep_r=timestep_r)
    hidden_states = self.x_embedder(hidden_states)

    encoder_hidden_states = encoder_hidden_states + self.cond_type_embed(
        torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
    )
    encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)
    encoder_hidden_states_2 = encoder_hidden_states_2 + self.cond_type_embed(
        torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
    )
    encoder_hidden_states_3 = self.image_embedder(image_embeds)
    is_t2v = torch.all(image_embeds == 0)
    if is_t2v:
        encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
        encoder_attention_mask_3 = torch.zeros(
            (batch_size, encoder_hidden_states_3.shape[1]),
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
    else:
        encoder_attention_mask_3 = torch.ones(
            (batch_size, encoder_hidden_states_3.shape[1]),
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
    encoder_hidden_states_3 = encoder_hidden_states_3 + self.cond_type_embed(
        2 * torch.ones_like(encoder_hidden_states_3[:, :, 0], dtype=torch.long)
    )

    # [image_3, byt5, mllm] in place, padding embeds zeroed, then the static valid-first
    # gather (replaces diffusers' boolean-index reorder).
    m1 = encoder_attention_mask.bool()
    m2 = encoder_attention_mask_2.bool()
    m3 = encoder_attention_mask_3.bool()
    ehs1 = encoder_hidden_states * m1.unsqueeze(-1).to(encoder_hidden_states.dtype)
    ehs2 = encoder_hidden_states_2 * m2.unsqueeze(-1).to(encoder_hidden_states_2.dtype)
    ehs3 = encoder_hidden_states_3 * m3.unsqueeze(-1).to(encoder_hidden_states_3.dtype)
    encoder_hidden_states = torch.cat([ehs3, ehs2, ehs1], dim=1)
    encoder_attention_mask = torch.cat([m3, m2, m1], dim=1)
    encoder_hidden_states = torch.gather(
        encoder_hidden_states,
        1,
        text_order[..., None].expand(-1, -1, encoder_hidden_states.shape[-1]),
    )
    encoder_attention_mask = torch.gather(encoder_attention_mask, 1, text_order)
    # fp32 count, then int32: Trainium LNC2 miscompiles long integer reductions.
    text_valid = encoder_attention_mask.float().sum(1).to(torch.int32)

    # Megatron-SP: only the image stream is sequence-sharded over the TP group, once
    # here for all blocks (the processor/FFN gather and reduce-scatter internally). The
    # scatter takes the materialized SPMD rank buffer: nxd's
    # scatter_to_sequence_parallel_region bakes group.rank() into a trace constant.
    sp_rank_util = getattr(self, "_sp_rank_util", None)
    if sp_rank_util is not None:
        hidden_states = scatter_to_process_group_spmd(
            hidden_states, partition_dim=1, rank=sp_rank_util.get_rank(), process_group=None
        )

    # Context parallel: the image stream and its rope (cos, sin) are scattered along N
    # over the CP group by this rank's CP rank (from the root SPMD rank buffer), once for
    # all blocks; the processors take the full N from their build-time num_image_tokens.
    cp_rank_source = getattr(self, "_cp_rank_source", None)
    if cp_rank_source is not None:
        cp_group = get_cp_group()
        cp_rank = get_cp_rank_spmd(cp_rank_source.get_rank())
        hidden_states = scatter_to_process_group_spmd(
            hidden_states, partition_dim=1, rank=cp_rank, process_group=cp_group
        )
        image_rotary_emb = tuple(
            scatter_to_process_group_spmd(t, partition_dim=0, rank=cp_rank, process_group=cp_group)
            for t in image_rotary_emb
        )

    for block in self.transformer_blocks:
        hidden_states, encoder_hidden_states = block(
            hidden_states,
            encoder_hidden_states,
            temb,
            text_valid,
            image_rotary_emb,
        )

    if sp_rank_util is not None:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)
    if cp_rank_source is not None:
        hidden_states = gather_from_tensor_model_parallel_region_with_dim(
            hidden_states, gather_dim=1, process_group=cp_group
        )

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (hidden_states,)
    return Transformer2DModelOutput(sample=hidden_states)


class _HunyuanVideo15TraceModule(nn.Module):
    """Fixed-tuple wrapper around diffusers HunyuanVideo15Transformer3DModel.

    The transformer runs the static-reorder forward (bound to this instance), the
    key-bounded ``JointBoundedProcessor`` (or its context-parallel ulysses / gather_kv
    variant) in every dual-stream block, and tensor-parallel linears in those blocks.
    The token refiner runs on host (application.forward_dit).
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        from diffusers.models.transformers.transformer_hunyuan_video15 import (
            HunyuanVideo15Transformer3DModel,
        )

        self.use_meanflow = bool(getattr(config, "use_meanflow", False))
        sp = bool(getattr(config, "sp_enabled", False))
        if (
            getattr(config, "cfg_parallel_enabled", False)
            or getattr(config, "context_parallel_enabled", False)
            or sp
        ):
            # Sharding modes need the cfg/cp/tp mesh and each rank's global rank (the
            # root ``global_rank.rank`` is filled by convert_hf_to_neuron_state_dict).
            init_parallel_mesh(config)
            self.global_rank = SPMDRank(world_size=get_world_group().size())
        self.cfg_parallel = bool(getattr(config, "cfg_parallel_enabled", False))
        if self.cfg_parallel:
            self.cfg_group = get_cfg_group()

        self.transformer = HunyuanVideo15Transformer3DModel(**_model_kwargs_from_config(config))
        self.transformer.forward = types.MethodType(_hv15_static_reorder_forward, self.transformer)
        if sp:
            # One TP-local rank buffer at the root (``tp_rank_util.rank``), referenced by
            # the transformer without registering it there (no second state_dict key).
            self.tp_rank_util = SPMDRank(world_size=int(config.neuron_config.tp_degree))
            object.__setattr__(self.transformer, "_sp_rank_util", self.tp_rank_util)
        if getattr(config, "context_parallel_enabled", False):
            # The forward's one CP scatter reads the root SPMD rank buffer, referenced
            # without registering it (the state_dict keeps one ``global_rank.rank``).
            object.__setattr__(self.transformer, "_cp_rank_source", self.global_rank)
            cp_processor = {
                "ulysses": JointUlyssesProcessor,
                "gather_kv": JointGatherKvProcessor,
            }[config.cp_mode]
            processor = cp_processor(num_image_tokens=config.num_image_tokens)
        else:
            processor = JointBoundedProcessor(sp=sp)
        for block in self.transformer.transformer_blocks:
            block.attn.set_processor(processor)
        install_tensor_parallel(self.transformer, sp=sp)

        # Diagnostic switches (capacity / attention probes).
        if os.environ.get("DIFFLET_HUNYUAN15_NKI_ATTENTION") == "1":
            processor = _HunyuanVideo15NkiNoMaskAttnProcessor()
            for block in self.transformer.transformer_blocks:
                block.attn.set_processor(processor)
        elif os.environ.get("DIFFLET_HUNYUAN15_NO_ATTENTION_MASK") == "1":
            raw_chunk_size = os.environ.get("DIFFLET_HUNYUAN15_ATTENTION_QUERY_CHUNK_SIZE")
            chunk_size = int(raw_chunk_size) if raw_chunk_size else None
            processor = _HunyuanVideo15NoMaskAttnProcessor(query_chunk_size=chunk_size)
            for block in self.transformer.transformer_blocks:
                block.attn.set_processor(processor)
        if os.environ.get("DIFFLET_HUNYUAN15_ZERO_BLOCK_ATTN") == "1":
            processor = _HunyuanVideo15ZeroAttnProcessor()
            for block in self.transformer.transformer_blocks:
                block.attn.set_processor(processor)
        if os.environ.get("DIFFLET_HUNYUAN15_ZERO_BLOCK_FF") == "1":
            for block in self.transformer.transformer_blocks:
                block.ff = _ZeroLikeModule()
                block.ff_context = _ZeroLikeModule()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states_2: torch.Tensor,
        encoder_attention_mask_2: torch.Tensor,
        image_embeds: torch.Tensor,
        text_order: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg_parallel:
            # The caller stacks [uncond, cond] into batch 2; each cfg rank denoises one
            # branch. Every input (incl. text_order and both masks) is batch-major.
            cfg_rank = get_cfg_rank_spmd(self.global_rank.get_rank())
            (
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep_r,
                encoder_hidden_states_2,
                encoder_attention_mask_2,
                image_embeds,
                text_order,
            ) = (
                scatter_to_process_group_spmd(
                    x, partition_dim=0, rank=cfg_rank, process_group=self.cfg_group
                )
                for x in (
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_r,
                    encoder_hidden_states_2,
                    encoder_attention_mask_2,
                    image_embeds,
                    text_order,
                )
            )
        out = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep_r=timestep_r if self.use_meanflow else None,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            image_embeds=image_embeds,
            text_order=text_order,
            return_dict=False,
        )[0]
        if self.cfg_parallel:
            out = gather_from_tensor_model_parallel_region_with_dim(
                out, gather_dim=0, process_group=self.cfg_group
            )
        return out
