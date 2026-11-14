# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
# Copyright 2026 Difflet contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The HunyuanVideo 1.5 text-to-video DiT for the CUDA backend.

Diffusers' building blocks keep the checkpoint layout, normalization and cast boundaries;
Difflet's projections and attention replace theirs. Eager single-sample blocks run the
joint attention with packed QKV, Difflet's fused Q/K RMSNorm + RoPE op (``qk_norm_rope``)
and cuDNN SDPA; compiled blocks attend each sample's valid token prefix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_hunyuan_video15 import (
    HunyuanVideo15ByT5TextProjection,
    HunyuanVideo15ImageProjection,
    HunyuanVideo15PatchEmbed,
    HunyuanVideo15RotaryPosEmbed,
    HunyuanVideo15TimeEmbedding,
    HunyuanVideo15TokenRefiner,
    HunyuanVideo15TransformerBlock,
)

from difflet.backends.cuda.hunyuan_video_15.qk_norm_rope import (
    HAS_TRITON,
    fused_joint_qk_norm_rope,
    pack_rope_table,
)
from difflet.ops import ColumnParallelLinear, attention


def _bind_projections(module):
    """Keep original state keys and fused linear bias arithmetic at TP=1."""
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            replacement = ColumnParallelLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                gather_output=True,
            )
            replacement.weight = child.weight
            replacement.bias = child.bias
            setattr(module, name, replacement)
        else:
            _bind_projections(child)


def _prepare_compact_attention_bias(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The token refiner's additive key bias, ``[B, 1, 1, S]``: zero for valid keys and
    -inf for padding, one value per key rather than an S-by-S mask."""
    bias = torch.zeros_like(valid, dtype=dtype).masked_fill(~valid, float("-inf"))
    return bias[:, None, None, :]


@torch.library.custom_op("difflet_hv15::prefix_attention", mutates_args=())
def prefix_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attend each sample over its valid joint-token prefix; padded query rows are zero.

    Equals valid-query AND valid-key masking when valid tokens lead the sequence, as the
    DiT's packed [video | glyph | text | padding] layout guarantees. The prefix is
    attended unmasked, so SDPA can pick its fastest kernel (cuDNN on recent GPUs, against a
    slower kernel for the padded, biased call). Opaque to the compiler, so blocks keep one
    static-shape graph for every prompt length. ``lengths`` is a CPU int64 vector.
    """
    output = torch.zeros_like(query)
    for index, length in enumerate(lengths.tolist()):
        window = (slice(index, index + 1), slice(None), slice(0, length))
        output[window] = attention(query[window], key[window], value[window], scale=scale)
    return output


@prefix_attention.register_fake
def _(query, key, value, lengths, scale):
    return torch.empty_like(query)


class _AttentionProcessor:
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if encoder_hidden_states is not None:
            cq = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            ck = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            cv = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
            cq = attn.norm_added_q(cq)
            ck = attn.norm_added_k(ck)
            query, key, value = (
                torch.cat(pair, dim=1) for pair in ((query, cq), (key, ck), (value, cv))
            )
        result = self._attend(query, key, value, attn.scale, attention_mask)
        result = result.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is None:
            return attn.to_out[1](attn.to_out[0](result))
        result, context = result.split(
            (hidden_states.shape[1], encoder_hidden_states.shape[1]), dim=1
        )
        return attn.to_out[1](attn.to_out[0](result)), attn.to_add_out(context)

    @staticmethod
    def _attend(query, key, value, scale, attention_mask):
        """[B, S, H, D] in and out; masked key bias, then zeroed invalid query rows."""
        result = attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=scale,
            attention_mask=attention_mask,
        ).transpose(1, 2)
        # Stock uses valid-query AND valid-key masking. Zero invalid query results
        # before projections, reproducing it without ever building an S-by-S mask.
        if attention_mask is not None:
            keep = attention_mask == 0
            result = result.masked_fill(~keep[:, 0, 0, :, None, None], 0)
        return result


class _PrefixAttentionProcessor(_AttentionProcessor):
    """Joint DiT attention over each sample's valid prefix; ``attention_mask`` is lengths."""

    @staticmethod
    def _attend(query, key, value, scale, lengths):
        return prefix_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), lengths, scale
        ).transpose(1, 2)


def _cudnn_attention_available(device: torch.device) -> bool:
    """cuDNN SDPA needs NVIDIA cuDNN 9 and sm_80+."""
    return bool(
        torch.version.hip is None
        and torch.backends.cudnn.is_available()
        and (torch.backends.cudnn.version() or 0) >= 90000
        and torch.cuda.get_device_capability(device) >= (8, 0)
    )


def joint_attention_supported(weight: torch.Tensor, head_dim: int) -> bool:
    """Whether a block whose projection weight is ``weight`` can take the joint path: CUDA
    with Triton for the fused op (BF16, head_dim 128) and cuDNN SDPA. Decided once, at load;
    otherwise the blocks keep the prefix path."""
    return (
        weight.is_cuda
        and HAS_TRITON
        and weight.dtype == torch.bfloat16
        and head_dim == 128
        and _cudnn_attention_available(weight.device)
    )


class _JointAttentionProcessor(_PrefixAttentionProcessor):
    """Joint DiT attention for one sample whose context is trimmed to its valid tokens.

    The same arithmetic as vLLM-Omni's HunyuanVideo 1.5 attention: packed QKV projections,
    a fused Q/K RMSNorm + RoPE op and cuDNN SDPA.

    ``image_rotary_emb`` is ``(cos, sin, table)`` with the packed table from
    ``pack_rope_table`` and ``attention_mask`` is ``None`` (every context token kept is
    valid). Any other call takes the prefix path. Sequence: one QKV GEMM per stream,
    the fused joint RMSNorm/RoPE/cat op, cuDNN-only SDPA on ``[B, H, S, D]`` views,
    flatten, cast, split, output projections.
    """

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        if attention_mask is not None:  # prefix lengths: the prefix path
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )
        heads, head_dim = attn.heads, attn.norm_q.weight.shape[0]
        inner = heads * head_dim
        qkv = F.linear(hidden_states, attn._packed_qkv_weight, attn._packed_qkv_bias)
        query, key, value = (t.unflatten(-1, (heads, -1)) for t in qkv.split([inner] * 3, dim=-1))
        added = F.linear(
            encoder_hidden_states, attn._packed_added_qkv_weight, attn._packed_added_qkv_bias
        )
        cq, ck, cv = (t.unflatten(-1, (heads, -1)) for t in added.split([inner] * 3, dim=-1))
        query, key, value = fused_joint_qk_norm_rope(
            query,
            key,
            value,
            cq,
            ck,
            cv,
            attn.norm_q.weight,
            attn.norm_k.weight,
            attn.norm_added_q.weight,
            attn.norm_added_k.weight,
            image_rotary_emb[2],
            attn.norm_q.eps,
        )
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            result = F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / (head_dim**0.5),
                enable_gqa=False,
            ).permute(0, 2, 1, 3)
        result = result.flatten(2, 3).to(query.dtype)
        context_len = encoder_hidden_states.shape[1]
        result, context = result.split_with_sizes(
            [result.shape[1] - context_len, context_len], dim=1
        )
        return attn.to_out[0](result), attn.to_add_out(context)


class _CompactRefiner(nn.Module):
    def __init__(self, refiner):
        super().__init__()
        self.refiner_blocks = refiner.refiner_blocks
        for block in self.refiner_blocks:
            block.attn.set_processor(_AttentionProcessor())

    def forward(self, hidden_states, temb, attention_mask=None):
        mask = (
            None
            if attention_mask is None
            else _prepare_compact_attention_bias(attention_mask.bool(), hidden_states.dtype)
        )
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, mask)
        return hidden_states


class HunyuanVideo15Transformer3DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        inner = config.num_attention_heads * config.attention_head_dim
        self.x_embedder = HunyuanVideo15PatchEmbed((1, 1, 1), config.in_channels, inner)
        self.image_embedder = HunyuanVideo15ImageProjection(config.image_embed_dim, inner)
        self.context_embedder = HunyuanVideo15TokenRefiner(
            config.text_embed_dim,
            config.num_attention_heads,
            config.attention_head_dim,
            num_layers=config.num_refiner_layers,
        )
        self.context_embedder.token_refiner = _CompactRefiner(self.context_embedder.token_refiner)
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(
            config.text_embed_2_dim, 2048, inner
        )
        self.time_embed = HunyuanVideo15TimeEmbedding(inner)
        self.cond_type_embed = nn.Embedding(3, inner)
        self.rope = HunyuanVideo15RotaryPosEmbed(1, 1, config.rope_axes_dim, config.rope_theta)
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideo15TransformerBlock(
                    config.num_attention_heads,
                    config.attention_head_dim,
                    config.mlp_ratio,
                    qk_norm=config.qk_norm,
                )
                for _ in range(config.num_layers)
            ]
        )
        for block in self.transformer_blocks:
            block.attn.set_processor(_JointAttentionProcessor())
        self.norm_out = AdaLayerNormContinuous(inner, inner, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner, config.out_channels)
        _bind_projections(self)

    @torch.no_grad()
    def pack_attention_projections(self):
        """One Q/K/V weight per stream, so each stream's projection is one GEMM.

        Run once the weights are on their device: each block's ``to_q/to_k/to_v`` (and
        ``add_*_proj``) weights and biases are concatenated into one contiguous tensor
        and rebound as views of it, so the checkpoint keys and memory are unchanged and
        the joint processor runs one GEMM per stream.
        """
        for block in self.transformer_blocks:
            attn = block.attn
            for names, prefix in (
                (("to_q", "to_k", "to_v"), "_packed_qkv"),
                (("add_q_proj", "add_k_proj", "add_v_proj"), "_packed_added_qkv"),
            ):
                modules = [getattr(attn, name) for name in names]
                weight = torch.cat([m.weight for m in modules]).contiguous()
                bias = torch.cat([m.bias for m in modules]).contiguous()
                rows = [m.weight.shape[0] for m in modules]
                for module, w, b in zip(modules, weight.split(rows), bias.split(rows)):
                    module.weight = nn.Parameter(w, requires_grad=False)
                    module.bias = nn.Parameter(b, requires_grad=False)
                attn.register_buffer(prefix + "_weight", weight, persistent=False)
                attn.register_buffer(prefix + "_bias", bias, persistent=False)

    def _joint_attention_ready(self, hidden_states):
        """Projections packed at load (eager, on a supported device) and one sample."""
        return (
            hasattr(self.transformer_blocks[0].attn, "_packed_qkv_weight")
            and hidden_states.shape[0] == 1
        )

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        image_embeds,
        *,
        return_dict=True,
    ):
        batch, _, frames, height, width = hidden_states.shape
        rope = self.rope(hidden_states)
        temb = self.time_embed(timestep)
        hidden_states = self.x_embedder(hidden_states)
        text = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)
        text = text + self.cond_type_embed(torch.zeros_like(text[:, :, 0], dtype=torch.long))
        glyph = self.context_embedder_2(encoder_hidden_states_2)
        glyph = glyph + self.cond_type_embed(torch.ones_like(glyph[:, :, 0], dtype=torch.long))
        image = self.image_embedder(image_embeds) * 0.0
        image = image + self.cond_type_embed(torch.full_like(image[:, :, 0], 2, dtype=torch.long))
        text_mask, glyph_mask = encoder_attention_mask.bool(), encoder_attention_mask_2.bool()
        image_mask = torch.zeros(image.shape[:2], device=image.device, dtype=torch.bool)
        contexts, masks = [], []
        for txt, tm, gl, gm, im, imask in zip(
            text, text_mask, glyph, glyph_mask, image, image_mask
        ):
            contexts.append(
                torch.cat(
                    (
                        im[imask],
                        gl[gm],
                        txt[tm],
                        im[~imask],
                        torch.zeros_like(gl[~gm]),
                        torch.zeros_like(txt[~tm]),
                    )
                )
            )
            masks.append(torch.cat((imask[imask], gm[gm], tm[tm], imask[~imask], gm[~gm], tm[~tm])))
        context = torch.stack(contexts)
        valid = torch.cat(
            (
                torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool),
                torch.stack(masks),
            ),
            dim=1,
        )
        # Valid tokens lead every sample by construction, so per-sample prefix lengths (one
        # host sync per DiT call) replace an S-sized key bias.
        lengths = valid.sum(dim=1).cpu()
        if self._joint_attention_ready(hidden_states):
            # Keep only the sample's valid context tokens (then all valid, so no mask) and
            # pack one RoPE table per forward.
            context = context[:, : int(lengths[0]) - hidden_states.shape[1]]
            table = pack_rope_table(rope[0], rope[1], context.shape[1], hidden_states.dtype)
            joint = (rope[0], rope[1], table)
            for block in self.transformer_blocks:
                hidden_states, context = block(hidden_states, context, temb, None, joint)
        else:
            for block in self.transformer_blocks:
                hidden_states, context = block(hidden_states, context, temb, lengths, rope)
        output = self.proj_out(self.norm_out(hidden_states, temb))
        output = output.reshape(batch, frames, height, width, -1).permute(0, 4, 1, 2, 3)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)
