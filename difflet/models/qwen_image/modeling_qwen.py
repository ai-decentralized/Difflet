"""Qwen-Image MMDiT fork with Megatron-style sequence parallelism (SP).

Pure SP: whenever ``sp_enabled`` and tp > 1, *both* streams — the image
residual stream and the text residual stream — are sharded along their own
sequence dimension between the transformer blocks. LayerNorm / modulation /
gate / residual run on the shards (that is the activation-memory saving);
each TP region (joint attention, per-stream MLP) is entered with an
all-gather ``g`` and left with a reduce-scatter ``ḡ`` that replaces the
row-parallel all-reduce — the exact operator placement of
``difflet/models/wan/modeling_wan.py``.

The block is a faithful fork of diffusers 0.38's ``QwenImageTransformerBlock``
(``diffusers/models/transformers/transformer_qwenimage.py``): submodule
names and layout are identical so ``from_pretrained`` weight loading and the
TP monkey-patches in ``difflet.backends.trainium.qwen_image.transformer``
apply unchanged. The model class subclasses diffusers'
``QwenImageTransformer2DModel`` and only rebuilds the block list and overrides
``forward`` to add the entry scatter / exit gather.

Non-SP behaviour (``sp_enabled=False``) is bit-identical to the parent
classes — every insertion below is behind ``if self.sp_enabled:``.
"""

from __future__ import annotations

from math import prod
from typing import Any

import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
    compute_text_seq_len_from_mask,
)

from difflet.ops import (
    SPMDRank,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_process_group_spmd,
)


def _safe_tp_size() -> int:
    """TP group size, 1 when no group is initialized (CPU reference / tp=1)."""
    try:
        from neuronx_distributed.parallel_layers.utils import (
            get_tensor_model_parallel_size,
        )

        return int(get_tensor_model_parallel_size())
    except Exception:
        return 1


def _sp_unbias(x: torch.Tensor, row_linear: nn.Module) -> torch.Tensor:
    """Correct the bias double-count of a ``reduce_output=False`` row-parallel
    linear under Megatron-SP (same fix as ``modeling_wan._sp_unbias``).

    nxd's ``RowParallelLinear`` adds the *full* bias to each rank's
    un-reduced partial; the ``ḡ`` reduce-scatter then sums across the TP
    group, so the bias lands ``tp×``. Subtract the ``(tp-1)×`` overcount so
    it is applied exactly once. No-op at tp == 1.
    """
    bias = getattr(row_linear, "bias", None)
    if bias is None:
        return x
    tp = _safe_tp_size()
    if tp <= 1:
        return x
    return x - (tp - 1) * bias.to(x.dtype)


def _shard_sequence(
    x: torch.Tensor, rank_util: SPMDRank | None, *, what: str
) -> torch.Tensor:
    """Megatron-SP forward-entry sequence scatter across the TP group.

    Must use the materialized SPMD rank buffer (``SPMDRank``) rather than a
    Python-side ``chunk(tp)[rank]``: under nxd SPMD tracing the Python rank
    resolves to a single constant and every rank would keep the same chunk
    (the failure mode the qwen registry note documented; wan's
    ``_sp_seq_scatter`` uses the same primitive, verified on device).
    Identity when no util is wired (CPU reference / tp == 1).
    """
    tp = _safe_tp_size()
    if tp <= 1 or rank_util is None:
        return x
    if x.shape[1] % tp != 0:
        raise ValueError(
            f"Qwen-Image SP requires {what} sequence length ({x.shape[1]}) "
            f"to divide tp ({tp}); the ḡ/g collectives need even shards"
        )
    return scatter_to_process_group_spmd(
        x, partition_dim=1, rank=rank_util.get_rank(), process_group=None
    )


class QwenImageSPTransformerBlock(nn.Module):
    """Fork of diffusers' ``QwenImageTransformerBlock`` with pure dual-stream SP.

    With ``sp_enabled`` the two residual streams arrive sequence-sharded as
    ``[B, S/tp, D]`` (image and text independently) and leave the same way.
    ``attn.to_out[0]`` / ``attn.to_add_out`` / ``img_mlp.net[2]`` /
    ``txt_mlp.net[2]`` must have been built with ``reduce_output=False`` —
    the ``_replace_qwen_linears_for_tp(sp_enabled=True)`` patch does that.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
        sp_enabled: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.sp_enabled = sp_enabled

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.zero_cond_t = zero_cond_t

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor (verbatim from diffusers)."""
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            index_expanded = index.unsqueeze(-1)
            shift_0_exp = shift_0.unsqueeze(1)
            shift_1_exp = shift_1.unsqueeze(1)
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        modulate_index: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: img stream, [B, S_img/tp, D] when sp_enabled
        # encoder_hidden_states: txt stream, [B, S_txt/tp, D] when sp_enabled
        img_mod_params = self.img_mod(temb)  # [B, 6*dim], no sequence dim

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # ---- segment 1: norm1 + modulation on the sequence shards ----
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # ---- enter the joint-attention TP region: g back to full sequences ----
        if self.sp_enabled:
            img_modulated = gather_from_sequence_parallel_region(img_modulated, dim=1)
            txt_modulated = gather_from_sequence_parallel_region(txt_modulated, dim=1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # to_out[0] / to_add_out run with reduce_output=False under SP, so
        # these are un-reduced partial sums over the full sequences.
        img_attn_partial, txt_attn_partial = attn_output

        # ---- leave the TP region: ḡ instead of all-reduce, back on shards ----
        if self.sp_enabled:
            img_attn_output = reduce_scatter_to_sequence_parallel_region(img_attn_partial, dim=1)
            img_attn_output = _sp_unbias(img_attn_output, self.attn.to_out[0])
            txt_attn_output = reduce_scatter_to_sequence_parallel_region(txt_attn_partial, dim=1)
            txt_attn_output = _sp_unbias(txt_attn_output, self.attn.to_add_out)
        else:
            img_attn_output, txt_attn_output = img_attn_partial, txt_attn_partial

        # gate + residual on the shards (sequence-independent elementwise)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # ---- segment 2: per-stream norm2 + MLP, same g/ḡ pair per stream ----
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        if self.sp_enabled:
            img_modulated2 = gather_from_sequence_parallel_region(img_modulated2, dim=1)
        img_mlp_partial = self.img_mlp(img_modulated2)
        if self.sp_enabled:
            img_mlp_output = reduce_scatter_to_sequence_parallel_region(img_mlp_partial, dim=1)
            img_mlp_output = _sp_unbias(img_mlp_output, self.img_mlp.net[2])
        else:
            img_mlp_output = img_mlp_partial
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        if self.sp_enabled:
            txt_modulated2 = gather_from_sequence_parallel_region(txt_modulated2, dim=1)
        txt_mlp_partial = self.txt_mlp(txt_modulated2)
        if self.sp_enabled:
            txt_mlp_output = reduce_scatter_to_sequence_parallel_region(txt_mlp_partial, dim=1)
            txt_mlp_output = _sp_unbias(txt_mlp_output, self.txt_mlp.net[2])
        else:
            txt_mlp_output = txt_mlp_partial
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageSPTransformer2DModel(QwenImageTransformer2DModel):
    """``QwenImageTransformer2DModel`` with SP-aware blocks and entry/exit
    scatter/gather. Constructed only when sp is requested; every other path
    keeps using the diffusers class untouched."""

    def __init__(self, *, sp_enabled: bool = False, tp_rank_util: SPMDRank | None = None, **kwargs):
        num_layers = int(kwargs["num_layers"])
        num_attention_heads = int(kwargs["num_attention_heads"])
        attention_head_dim = int(kwargs["attention_head_dim"])
        zero_cond_t = bool(kwargs.get("zero_cond_t", False))
        super().__init__(**kwargs)
        self.sp_enabled = bool(sp_enabled) and _safe_tp_size() > 1
        # Materialized per-rank buffer for the entry scatter (see
        # ``_shard_sequence``); the trace module wires it after construction.
        self.tp_rank_util = tp_rank_util

        dim = attention_head_dim * num_attention_heads
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageSPTransformerBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                    sp_enabled=self.sp_enabled,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        # Faithful copy of the diffusers forward with two SP insertions:
        # (1) shard both streams right before the block loop (after the joint
        #     mask is built from the full image length), (2) gather the image
        #     stream after the loop for norm_out/proj_out. The text stream is
        #     discarded after the loop, so it stays sharded throughout.
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            joint_attention_mask = joint_attention_mask[:, None, None, :]
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        # --- SP insertion 1: shard both residual streams along their own
        # sequence dims via the materialized rank buffer. (modulate_index is
        # per-token and rank-dependent — zero_cond_t + SP is unsupported.)
        if self.sp_enabled:
            if modulate_index is not None:
                raise NotImplementedError(
                    "Qwen-Image SP does not support zero_cond_t (rank-dependent "
                    "modulate_index); it is off in the released configs"
                )
            hidden_states = _shard_sequence(hidden_states, self.tp_rank_util, what="image")
            encoder_hidden_states = _shard_sequence(
                encoder_hidden_states, self.tp_rank_util, what="text"
            )

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,  # mask travels via attention_kwargs
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]

        # --- SP insertion 2: reassemble the image stream for the head. ---
        if self.sp_enabled:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=output)
