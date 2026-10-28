"""Backend-neutral modeling for the Qwen-Image DiT transformer.

Extracted verbatim from ``difflet/backends/trainium/qwen_image/transformer.py``
so a second hardware backend can reuse it. Nothing here imports a backend:
only ``torch``, pure-torch diffusers pieces, and ``difflet.ops`` — the same
shape Wan and HunyuanVideo already use. The Trainium file keeps the pieces
that genuinely are Trainium (``InferenceConfig``, ``ModelWrapper``,
``NeuronApplicationBase``) and re-exports everything below, so every existing
importer keeps working unchanged.

This code patches an upstream diffusers ``QwenImageTransformer2DModel``: it
swaps the large linears for tensor-parallel ones, replaces the complex-valued
RoPE with a static real-valued module fixed to one compiled shape, and
installs a double-stream attention processor. ``config`` is duck-typed — it
needs the Qwen-Image geometry attributes, not any particular config class.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn

from difflet.ops.collectives import safe_tensor_parallel_size
from difflet.ops import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
    attention,
    gather_from_tensor_model_parallel_region_with_dim,
    get_cp_group,
    get_cp_rank_spmd,
    get_tensor_model_parallel_size,
    get_world_group,
    init_parallel_mesh,
    joint_ring_attention,
    joint_ulysses_attention,
    scatter_to_process_group_spmd,
)


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
    """Static real-valued Qwen RoPE for the compiled image/text shape(s).

    Holds one precomputed (cos, sin) image table per compiled packed shape,
    selected at trace time from the ``img_shapes`` argument (a Python constant
    per bucket, so each bucket NEFF bakes exactly its own table). Text RoPE is
    shape-independent.

    Under context parallelism the image RoPE buffers are scattered along the
    sequence axis (dim 0 of ``(S_img, head_dim)``) so each CP rank applies the
    rotary embedding for its own image-token shard. Text RoPE stays full.
    """

    def __init__(
        self,
        *,
        img_ropes: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
        txt_rope: tuple[torch.Tensor, torch.Tensor],
        context_parallel_enabled: bool = False,
        global_rank: SPMDRank | None = None,
        cp_group=None,
    ) -> None:
        super().__init__()
        self._img_keys = list(img_ropes)
        for (ph, pw), (cos, sin) in img_ropes.items():
            self.register_buffer(f"img_cos_{ph}x{pw}", cos, persistent=False)
            self.register_buffer(f"img_sin_{ph}x{pw}", sin, persistent=False)
        self.register_buffer("txt_cos", txt_rope[0], persistent=False)
        self.register_buffer("txt_sin", txt_rope[1], persistent=False)
        self.context_parallel_enabled = context_parallel_enabled
        # Hold the shared SPMDRank without registering it as a submodule here
        # (it is owned by the trace module so the state dict has a single
        # ``global_rank.rank`` key).
        self._global_rank_ref = (global_rank,) if global_rank is not None else ()
        self.cp_group = cp_group

    def forward(self, img_shapes=None, *args, max_txt_seq_len: int | None = None, **kwargs):
        del args, kwargs
        if img_shapes is not None:
            _batch, ph, pw = img_shapes[0][0]
            key = (int(ph), int(pw))
        else:
            key = self._img_keys[0]
        img_cos = getattr(self, f"img_cos_{key[0]}x{key[1]}")
        img_sin = getattr(self, f"img_sin_{key[0]}x{key[1]}")
        if self.context_parallel_enabled:
            cp_rank = get_cp_rank_spmd(self._global_rank_ref[0].get_rank())
            img_cos = scatter_to_process_group_spmd(
                img_cos, partition_dim=0, rank=cp_rank, process_group=self.cp_group
            )
            img_sin = scatter_to_process_group_spmd(
                img_sin, partition_dim=0, rank=cp_rank, process_group=self.cp_group
            )
        txt_cos, txt_sin = self.txt_cos, self.txt_sin
        if max_txt_seq_len is not None and int(max_txt_seq_len) < int(txt_cos.shape[0]):
            # diffusers passes the current text length; a trimmed prompt is shorter
            # than the precomputed table. Trainium/TPU pass the full length: no-op there.
            # ``.clone()``: a view keeps the 1024-row table as ``_base`` and torch.compile's
            # duck sizing then ties the prompt length to it and recompiles per prompt.
            txt_cos = txt_cos[:max_txt_seq_len].clone()
            txt_sin = txt_sin[:max_txt_seq_len].clone()
        return (img_cos, img_sin), (txt_cos, txt_sin)


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


def _row_parallel_like(
    linear: nn.Linear,
    *,
    input_is_parallel: bool,
    reduce_dtype: torch.dtype,
    reduce_output: bool = True,
) -> RowParallelLinear:
    # reduce_dtype has no default on purpose: NxD's own default is fp32.
    return RowParallelLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        input_is_parallel=input_is_parallel,
        reduce_output=reduce_output,
        reduce_dtype=reduce_dtype,
    )


def _safe_tensor_parallel_size() -> int:
    """tp_degree if a TP group is initialized, else 1 where the backend means a tp=1 build
    (``difflet.ops.collectives.safe_tensor_parallel_size``: NxD asserts, the TPU mesh raises
    ``ParallelMeshNotInitializedError``); the neuron backend's uninitialised multi-rank mesh
    raises instead of silently building unsharded."""
    return safe_tensor_parallel_size(get_tensor_model_parallel_size)


def _replace_qwen_linears_for_tp(
    transformer: nn.Module,
    sp_enabled: bool = False,
    *,
    reduce_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Shard Qwen's largest transformer linears across tensor-parallel ranks.

    Under Megatron-SP (``sp_enabled``) the four row-parallel outputs keep
    their partial sums (``reduce_output=False``); the forked block forward's
    ``ḡ`` reduce-scatter performs the cross-rank sum while re-sharding the
    sequence, and the fork's ``_sp_unbias`` corrects the bias double-count.
    ``reduce_dtype`` is the precision of the row-parallel all-reduce.
    """

    tp_degree = _safe_tensor_parallel_size()
    if tp_degree <= 1:
        return

    replicate_modulation = _env_flag("DIFFLET_QWEN_TP_REPLICATE_MOD")
    replicate_attention = _env_flag("DIFFLET_QWEN_TP_REPLICATE_ATTN")
    replicate_mlp = _env_flag("DIFFLET_QWEN_TP_REPLICATE_MLP")
    replicate_io = _env_flag("DIFFLET_QWEN_TP_REPLICATE_IO")
    replicate_time = _env_flag("DIFFLET_QWEN_TP_REPLICATE_TIME")
    replicate_out = _env_flag("DIFFLET_QWEN_TP_REPLICATE_OUT")

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
            attn.to_out[0] = _row_parallel_like(
                attn.to_out[0],
                input_is_parallel=True,
                reduce_dtype=reduce_dtype,
                reduce_output=not sp_enabled,
            )
            attn.to_add_out = _row_parallel_like(
                attn.to_add_out,
                input_is_parallel=True,
                reduce_dtype=reduce_dtype,
                reduce_output=not sp_enabled,
            )

        if not replicate_mlp:
            block.img_mlp.net[0].proj = _column_parallel_like(
                block.img_mlp.net[0].proj,
                gather_output=False,
            )
            block.img_mlp.net[2] = _row_parallel_like(
                block.img_mlp.net[2],
                input_is_parallel=True,
                reduce_dtype=reduce_dtype,
                reduce_output=not sp_enabled,
            )
            block.txt_mlp.net[0].proj = _column_parallel_like(
                block.txt_mlp.net[0].proj,
                gather_output=False,
            )
            block.txt_mlp.net[2] = _row_parallel_like(
                block.txt_mlp.net[2],
                input_is_parallel=True,
                reduce_dtype=reduce_dtype,
                reduce_output=not sp_enabled,
            )


def _apply_qwen_block_diagnostics(transformer: nn.Module) -> None:
    zero_attention = _env_flag("DIFFLET_QWEN_ZERO_BLOCK_ATTN")
    zero_mlp = _env_flag("DIFFLET_QWEN_ZERO_BLOCK_MLP")
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

    def __init__(
        self,
        *,
        context_parallel_enabled: bool = False,
        cp_group=None,
        cp_mode: str = "gather_kv",
    ) -> None:
        self.context_parallel_enabled = context_parallel_enabled
        self.cp_group = cp_group
        self.cp_mode = cp_mode

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        text_valid_len: torch.Tensor | None = None,
        image_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``text_valid_len`` (int ``[B]``, optional; with ``image_seq_len``, the full image token
        count before any CP split): the text stream is right-padded and only its first
        ``text_valid_len[b]`` keys are attended, diffusers' masked semantics
        (``_masked_joint_attention``). None keeps the unmasked joint attention the Trainium and
        TPU closures trace."""
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

        # Context parallel: choose ring-attention or gather-KV based on cp_mode.
        # Under ring: image K,V are sequence-sharded and rotated via the ring
        # collective; text K,V are replicated and folded in as a local partial.
        # Under gather_kv (default): each rank all-gathers the full image K,V
        # before running joint attention. Text K,V are always replicated.
        head_dim = txt_query.shape[-1]
        if text_valid_len is not None:
            if attention_mask is not None:
                raise ValueError("pass text_valid_len or attention_mask, not both")
            joint_hidden_states = self._masked_joint_attention(
                img_query, img_key, img_value, txt_query, txt_key, txt_value,
                text_valid_len=text_valid_len, image_seq_len=image_seq_len,
            )
        elif self.context_parallel_enabled and self.cp_mode == "ring":
            if attention_mask is not None:
                raise NotImplementedError("ring cp_mode does not support attention_mask")
            # Joint ring: image K,V sharded → rotated; text K,V replicated →
            # local partial. q_joint is [txt ‖ img] to match the gather_kv
            # joint concat order so the downstream split at seq_txt is unchanged.
            # img/txt tensors are [B, S, H, d]; the op wants [B, H, S, d].
            scale = 1.0 / math.sqrt(head_dim)
            q_joint = torch.cat([txt_query, img_query], dim=1).transpose(1, 2)  # [B, H, Sq, d]
            o_joint = joint_ring_attention(
                q_joint,
                img_key.transpose(1, 2), img_value.transpose(1, 2),  # sharded image K,V
                txt_key.transpose(1, 2), txt_value.transpose(1, 2),  # replicated text K,V
                scale=scale, causal=False,
            )  # [B, H, Sq, d]
            joint_hidden_states = o_joint.transpose(1, 2)  # [B, Sq, H, d]
        elif self.context_parallel_enabled and self.cp_mode == "ulysses":
            if attention_mask is not None:
                raise NotImplementedError("ulysses cp_mode does not support attention_mask")
            # Joint ulysses: image K,V sharded → all-to-all'd into a head shard at full
            # sequence length; text K,V replicated → reduced to the same head shard. The
            # op hands each stream back in the sharding it arrived with, so they are
            # re-joined here as [txt ‖ img] to match the gather_kv concat order and keep
            # the downstream split at seq_txt unchanged.
            scale = 1.0 / math.sqrt(head_dim)
            img_out, txt_out = joint_ulysses_attention(
                img_query.transpose(1, 2), txt_query.transpose(1, 2),
                img_key.transpose(1, 2), img_value.transpose(1, 2),      # sharded image K,V
                txt_key.transpose(1, 2), txt_value.transpose(1, 2),      # replicated text K,V
                scale=scale, causal=False,
            )  # [B, H, S_img/cp, d], [B, H, S_txt, d]
            joint_hidden_states = torch.cat([txt_out, img_out], dim=2).transpose(1, 2)
        else:
            # gather_kv path: all-gather image K,V before joint attention.
            if self.context_parallel_enabled:
                stacked_kv = torch.stack([img_key, img_value], dim=0)  # [2, B, S/cp, H, d]
                stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv, gather_dim=2, process_group=self.cp_group
                )  # [2, B, S, H, d]
                img_key, img_value = torch.unbind(stacked_kv, dim=0)

            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_key = torch.cat([txt_key, img_key], dim=1)
            joint_value = torch.cat([txt_value, img_value], dim=1)

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
                # Masked joint attention (a padded prompt on an eager backend): the
                # same op in (B, H, S, d) layout, so the (B, 1, 1, S_k) key mask
                # broadcasts per batch row instead of against folded heads.
                q = joint_query.transpose(1, 2)
                k = joint_key.transpose(1, 2)
                v = joint_value.transpose(1, 2)
                attn_out = attention(
                    q,
                    k,
                    v,
                    scale=1.0 / math.sqrt(head_dim),
                    causal=False,
                    attention_mask=attention_mask,
                )
                joint_hidden_states = attn_out.transpose(1, 2)

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(txt_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output

    def _masked_joint_attention(
        self, img_query, img_key, img_value, txt_query, txt_key, txt_value, *,
        text_valid_len: torch.Tensor, image_seq_len: int | None,
    ) -> torch.Tensor:
        """Joint attention with the text padding masked out of the keys, as diffusers masks it
        (``[text_mask | ones(image)]`` on the key axis, every query). In ``[B, S, H, d]``, out
        ``[B, S_txt + S_img_local, H, d]`` in the ``[txt ‖ img]`` order the caller splits.

        No dense mask: the keys are ordered ``[image ‖ text]``, so the valid ones are the
        contiguous prefix ``[0, image_seq_len + text_valid_len)``, which ``attention``'s
        ``bound_min``/``bound_max`` take (the key-bounded flash kernel on the neuron device).
        Ring CP masks the replicated text partial by count (``joint_ring_attention``); Ulysses
        CP bounds its ``[image_full ‖ text]`` keys the same way (``joint_ulysses_attention``).
        """
        scale = 1.0 / math.sqrt(txt_query.shape[-1])
        valid = text_valid_len.to(torch.int32)

        def bhsd(t):
            return t.transpose(1, 2)

        if self.context_parallel_enabled and self.cp_mode == "ring":
            q_joint = torch.cat([txt_query, img_query], dim=1)
            out = joint_ring_attention(
                bhsd(q_joint), bhsd(img_key), bhsd(img_value), bhsd(txt_key), bhsd(txt_value),
                scale=scale, causal=False, text_valid_len=valid,
            )
            return bhsd(out)
        if image_seq_len is None:
            raise ValueError("the masked joint attention needs image_seq_len (full image tokens)")
        if self.context_parallel_enabled and self.cp_mode == "ulysses":
            img_out, txt_out = joint_ulysses_attention(
                bhsd(img_query), bhsd(txt_query), bhsd(img_key), bhsd(img_value),
                bhsd(txt_key), bhsd(txt_value), scale=scale, causal=False,
                key_valid_len=valid + int(image_seq_len),
            )
            return bhsd(torch.cat([txt_out, img_out], dim=2))
        if self.context_parallel_enabled:
            stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                torch.stack([img_key, img_value], dim=0), gather_dim=2,
                process_group=self.cp_group,
            )
            img_key, img_value = torch.unbind(stacked_kv, dim=0)
        if int(img_key.shape[1]) != int(image_seq_len):
            raise ValueError(
                f"image keys {int(img_key.shape[1])} != image_seq_len {int(image_seq_len)}"
            )
        q = bhsd(torch.cat([img_query, txt_query], dim=1))
        k = bhsd(torch.cat([img_key, txt_key], dim=1))
        v = bhsd(torch.cat([img_value, txt_value], dim=1))
        b, h, s_q, d = q.shape
        s_k = k.shape[2]
        bound_max = (valid + int(image_seq_len)).view(b, 1, 1).expand(b, h, s_q)
        bound_max = bound_max.reshape(b * h, s_q, 1).contiguous()
        out = attention(
            q.reshape(b * h, s_q, d), k.reshape(b * h, s_k, d), v.reshape(b * h, s_k, d),
            scale=scale, causal=False, bound_min=torch.zeros_like(bound_max),
            bound_max=bound_max, tp_q=True, tp_k=True, tp_out=False,
        ).reshape(b, h, s_q, d)
        n_img = img_query.shape[1]
        return bhsd(torch.cat([out[:, :, n_img:], out[:, :, :n_img]], dim=2))


def _trim_text_padding(
    encoder_hidden_states: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Slice the text stream to the last valid token across the batch.

    Exact for the image output: a masked key contributes nothing to any query, text
    positions past the last valid one are keys to nobody and their own outputs are
    discarded, so dropping those columns changes the image tokens only through
    reduction order. Rows shorter than the longest keep their mask; when every row is
    full after the slice the mask is dropped (no mask → flash SDPA). An all-False mask
    (no valid token anywhere) is left as it is, as diffusers does.
    """
    mask = mask.to(torch.bool)
    if not bool(mask.any()):
        return encoder_hidden_states, mask
    positions = torch.arange(mask.shape[1], device=mask.device)
    length = int((positions * mask).amax().item()) + 1  # one host sync per forward
    encoder_hidden_states = encoder_hidden_states[:, :length]
    mask = mask[:, :length]
    if bool(mask.all()):
        return encoder_hidden_states, None
    return encoder_hidden_states, mask


class _QwenImageTransformerTraceModule(nn.Module):
    """Fix non-tensor Qwen-Image pipeline args at trace time."""

    def __init__(self, config, *, apply_text_mask: bool = False, trim_text_padding: bool = False):
        super().__init__()
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformer2DModel,
        )

        self.config = config
        # Whether the prompt padding mask reaches the joint attention. False keeps the
        # Trainium/TPU behaviour (mask dropped; see forward). The CUDA backend sets True so
        # padded prompts match upstream diffusers.
        self.apply_text_mask = bool(apply_text_mask)
        # Whether the padding is sliced off before the DiT instead of masked:
        # the same output as masking (see _trim_text_padding), but the joint sequence shrinks
        # to the real prompt and, when no row is ragged, the mask disappears so SDPA can take
        # the flash kernel. Eager backends only; needs the mask to know what is padding.
        if trim_text_padding and not apply_text_mask:
            raise ValueError("trim_text_padding needs apply_text_mask")
        self.trim_text_padding = bool(trim_text_padding)
        self.guidance_embeds = bool(getattr(config, "guidance_embeds", False))
        self.context_parallel_enabled = bool(getattr(config, "context_parallel_enabled", False))
        self.cp_mode = str(getattr(config, "cp_mode", "gather_kv"))
        self.sp_enabled = bool(getattr(config, "sp_enabled", False))
        if self.context_parallel_enabled:
            # CP scatters RoPE/KV over the cp axis subgroup; the mesh must be
            # initialized before the rope module / attention processors below
            # capture the group reference.
            init_parallel_mesh(config)
            self.cp_group = get_cp_group()
            self.global_rank = SPMDRank(world_size=get_world_group().size())
        else:
            self.cp_group = None
            self.global_rank = None
        # SP reuses the TP group: the materialized SPMDRank buffer is the
        # per-rank id the fork's entry scatter reads (loaded as
        # ``sp_rank_util.rank = arange(tp)``; see convert_hf_to_neuron_state_dict).
        # The Trainium config nests tp under ``neuron_config``; the neuron
        # backend's NeuronQwenImageConfig carries it directly (its SPMDRank is
        # stateless: one process per rank, the rank is a Python int).
        neuron_config = getattr(config, "neuron_config", None)
        sp_tp_degree = int(getattr(neuron_config, "tp_degree", None) or config.tp_degree)
        self.sp_rank_util = SPMDRank(world_size=sp_tp_degree) if self.sp_enabled else None
        # One packed shape per compiled bucket, keyed by image_seq_len. The
        # forward selects by hidden_states.shape[1], which is a concrete value
        # during each bucket's trace, so every bucket NEFF bakes its own RoPE
        # constants and img_shapes.
        from difflet.backends.trainium.core.bucketing import resolve_compile_shapes

        self._packed_by_seq: dict[int, tuple[int, int]] = {}
        vsf2 = int(config.vae_scale_factor) * 2
        patch = int(config.patch_size)
        for height, width, _frames in resolve_compile_shapes(config):
            ph = (2 * (int(height) // vsf2)) // patch
            pw = (2 * (int(width) // vsf2)) // patch
            self._packed_by_seq[ph * pw] = (ph, pw)
        self.img_shapes = [[(1, int(config.packed_height), int(config.packed_width))]]
        if self.sp_enabled:
            # SP fork: same submodule layout as diffusers' class, so weight
            # loading and the TP patches below apply unchanged.
            from difflet.models.qwen_image.modeling_qwen import QwenImageSPTransformer2DModel

            transformer_cls = QwenImageSPTransformer2DModel
            sp_ctor_kwargs = {"sp_enabled": True, "tp_rank_util": self.sp_rank_util}
        else:
            transformer_cls = QwenImageTransformer2DModel
            sp_ctor_kwargs = {}
        self.transformer = transformer_cls(
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
            **sp_ctor_kwargs,
        )
        _replace_qwen_linears_for_tp(
            self.transformer,
            sp_enabled=self.sp_enabled,
            # Non-NxD backends (CUDA, TPU) build this from a config without neuron_config.
            reduce_dtype=getattr(getattr(config, "neuron_config", None), "rpl_reduce_dtype", None)
            or torch.bfloat16,
        )
        _apply_qwen_block_diagnostics(self.transformer)
        img_ropes: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        txt_rope = None
        for ph, pw in self._packed_by_seq.values():
            img_freqs, txt_freqs = self.transformer.pos_embed(
                [[(1, ph, pw)]],
                max_txt_seq_len=int(config.text_seq_len),
                device=torch.device("cpu"),
            )
            img_ropes[(ph, pw)] = _qwen_complex_rope_to_real(img_freqs)
            if txt_rope is None:
                txt_rope = _qwen_complex_rope_to_real(txt_freqs)
        self.transformer.pos_embed = _StaticQwenImageRealRope(
            img_ropes=img_ropes,
            txt_rope=txt_rope,
            context_parallel_enabled=self.context_parallel_enabled,
            global_rank=self.global_rank,
            cp_group=self.cp_group,
        )
        for block in self.transformer.transformer_blocks:
            block.attn.processor = _QwenImageTrainiumAttnProcessor(
                context_parallel_enabled=self.context_parallel_enabled,
                cp_group=self.cp_group,
                cp_mode=self.cp_mode,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        guidance: torch.Tensor,
        *,
        text_valid_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The diffusers mask path computes active text length with arange/where.
        # That currently fails during XLA tracing for the fixed-shape M4a
        # boundary, so the Trainium/TPU closures (apply_text_mask=False) use
        # full-length cached text embeddings and drop the mask. Eager backends
        # (CUDA) pass it through: diffusers then builds the joint key mask
        # [text_mask | ones(image)] and the processor applies it, so a prompt
        # padded to text_seq_len gives the same output as diffusers' unpadded run.
        # ``text_valid_len`` (int [B], the count of a right-padded prompt's
        # valid rows) masks the padding instead, by key count in the attention
        # processor (``_masked_joint_attention``); the neuron backend passes it
        # with apply_text_mask=False.
        if self.apply_text_mask:
            if self.context_parallel_enabled:
                raise NotImplementedError(
                    "the text mask with context parallelism is not supported on CUDA yet"
                )
            mask_arg = encoder_hidden_states_mask
            if self.trim_text_padding and mask_arg is not None:
                encoder_hidden_states, mask_arg = _trim_text_padding(
                    encoder_hidden_states, mask_arg
                )
        else:
            mask_arg = None
        del encoder_hidden_states_mask
        guidance_arg = guidance if self.guidance_embeds else None

        # Per-bucket packed shape: hidden_states.shape[1] is concrete during
        # each bucket's trace, so this lookup resolves statically per NEFF.
        seq_len = int(hidden_states.shape[1])
        ph, pw = self._packed_by_seq.get(
            seq_len, (int(self.img_shapes[0][0][1]), int(self.img_shapes[0][0][2]))
        )
        img_shapes = [[(1, ph, pw)]]
        attention_kwargs = None
        if text_valid_len is not None:
            attention_kwargs = {"text_valid_len": text_valid_len, "image_seq_len": seq_len}

        # Context parallel: scatter the image tokens along the sequence axis
        # across the cp axis subgroup before the diffusers forward. The
        # static RoPE scatters the matching image rope, the attention processor
        # gathers image K/V, and we gather the packed output back below. Text
        # tokens stay replicated, so encoder_hidden_states is left full.
        if self.context_parallel_enabled:
            cp_rank = get_cp_rank_spmd(self.global_rank.get_rank())
            hidden_states = scatter_to_process_group_spmd(
                hidden_states,
                partition_dim=1,
                rank=cp_rank,
                process_group=self.cp_group,
            )

        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=mask_arg,
            guidance=guidance_arg,
            img_shapes=img_shapes * int(hidden_states.shape[0]),
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        if self.context_parallel_enabled:
            output = gather_from_tensor_model_parallel_region_with_dim(
                output, gather_dim=1, process_group=self.cp_group
            )
        return output

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


__all__ = [
    "_QwenImageTrainiumAttnProcessor",
    "_QwenImageTransformerTraceModule",
    "_StaticQwenImageRealRope",
    "_ZeroLikeModule",
    "_ZeroQwenImageAttention",
    "_apply_qwen_block_diagnostics",
    "_apply_qwen_rope_real",
    "_column_parallel_like",
    "_env_flag",
    "_qwen_complex_rope_to_real",
    "_qwen_rope_as_real",
    "_replace_qwen_linears_for_tp",
    "_row_parallel_like",
    "_safe_tensor_parallel_size",
    "_trim_text_padding",
]
