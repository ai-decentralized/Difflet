# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted attention and BF16 rotary operations for the opt-in LTX-2 CUDA profile.
"""Opt-in native LTX-2 connector arithmetic and startup BLAS policy.

Attention follows Diffusers' LTX-2 processor; BF16 rotary arithmetic follows
vLLM-Omni's LTX-2 implementation. Both upstream implementations are Apache-2.0.
Only the opt-in profile configures this startup policy and replaces its processors.
"""

from __future__ import annotations

import os

import torch
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AudioVideoAttnProcessor,
    dispatch_attention_fn,
)

_WORKSPACE_CONFIG = ":4096:8"


def prepare_official_runtime() -> None:
    """Select the qualified 32 MiB BLAS workspace before any CUDA handle exists."""
    preferred = torch.backends.cuda.preferred_blas_library()
    if preferred.name not in {"Cublas", "Default"}:
        raise ValueError("ltx_official_bf16 requires the cuBLAS library preference")
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is not None and configured != _WORKSPACE_CONFIG:
        raise ValueError("ltx_official_bf16 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if configured is None:
        # Changing this after handle allocation can advertise a larger workspace
        # than the cached allocation in PyTorch 2.11. Require a fresh startup.
        if torch.cuda.is_initialized():
            raise ValueError(
                "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization "
                "for ltx_official_bf16"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _WORKSPACE_CONFIG


def apply_official_rotary(x, freqs, *, rope_type, head_dim):
    """Retain native BF16 coefficient, multiply and addcmul materialization."""
    cos, sin = (value.to(x.dtype) for value in freqs)
    if rope_type == "interleaved":
        real, imag = x.unflatten(2, (x.shape[2] // 2, 2)).unbind(-1)
        rotated = torch.stack((-imag, real), dim=-1).flatten(2)
        return x * cos + rotated * sin
    if rope_type != "split":
        raise ValueError(f"Unsupported LTX-2 rotary layout: {rope_type}")
    reshape = x.ndim != 4 and cos.ndim == 4
    if reshape:
        batch = x.shape[0]
        _, heads, tokens, _ = cos.shape
        x = x.reshape(batch, tokens, heads, head_dim).swapaxes(1, 2)
    last = x.shape[-1]
    if last % 2:
        raise ValueError("LTX-2 split rotary requires an even head dimension")
    split = x.reshape(*x.shape[:-1], 2, last // 2)
    first, second = split[..., :1, :], split[..., 1:, :]
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    out = split * cos
    out[..., :1, :].addcmul_(-sin, second)
    out[..., 1:, :].addcmul_(sin, first)
    out = out.reshape(*out.shape[:-2], last)
    if reshape:
        out = out.swapaxes(1, 2).reshape(batch, tokens, heads * head_dim)
    return out


class OfficialLtx2AttentionProcessor(LTX2AudioVideoAttnProcessor):
    """Native connector precision without mutating shared upstream functions."""

    def __init__(self, *, learned_registers: bool):
        super().__init__()
        self.learned_registers = learned_registers

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
    ):
        batch, length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # Native LTX-2 drops the redundant additive mask after learned registers
        # replace padding. No device-to-host mask inspection belongs in inference.
        if self.learned_registers:
            attention_mask = None
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, length, batch)
            attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        gates = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(context))
        value = attn.to_v(context)
        if query_rotary_emb is not None:
            key_freqs = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            query = apply_official_rotary(
                query, query_rotary_emb, rope_type=attn.rope_type, head_dim=attn.head_dim
            )
            key = apply_official_rotary(
                key, key_freqs, rope_type=attn.rope_type, head_dim=attn.head_dim
            )
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        hidden_states = (
            dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            .flatten(2, 3)
            .to(query.dtype)
        )
        if gates is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            hidden_states = (hidden_states * (2.0 * torch.sigmoid(gates)).unsqueeze(-1)).flatten(
                2, 3
            )
        return attn.to_out[1](attn.to_out[0](hidden_states))


def configure_official_connectors(connectors) -> int:
    """Install only this connector instance's processors; preserve its weights."""
    pending = []
    for connector in (connectors.video_connector, connectors.audio_connector):
        learned = connector.learnable_registers is not None
        for block in connector.transformer_blocks:
            attention = block.attn1
            if isinstance(attention.processor, OfficialLtx2AttentionProcessor):
                if attention.processor.learned_registers != learned:
                    raise ValueError("LTX-2 connector register policy changed")
            elif type(attention.processor) is LTX2AudioVideoAttnProcessor:
                pending.append((attention, learned))
            else:
                raise ValueError("ltx_official_bf16 requires the stock LTX-2 connector processor")
    for attention, learned in pending:
        attention.set_processor(OfficialLtx2AttentionProcessor(learned_registers=learned))
    return len(pending)


def configure_official_transformer(transformer) -> dict[str, int]:
    """Retain native BF16 rotary and unlearned RMSNorm on this DiT instance."""
    from diffusers.models.normalization import RMSNorm

    norms = []
    processors = []
    norm_names = (
        "norm1",
        "audio_norm1",
        "norm2",
        "audio_norm2",
        "audio_to_video_norm",
        "video_to_audio_norm",
        "norm3",
        "audio_norm3",
    )
    attention_names = (
        "attn1",
        "audio_attn1",
        "attn2",
        "audio_attn2",
        "audio_to_video_attn",
        "video_to_audio_attn",
    )
    # Admit the entire instance before applying any replacement.
    for block in transformer.transformer_blocks:
        for name in norm_names:
            norm = getattr(block, name)
            if type(norm) is torch.nn.RMSNorm:
                if norm.weight is not None or len(norm.normalized_shape) != 1:
                    raise ValueError(
                        "Official LTX-2 requires unlearned one-dimensional block norms"
                    )
                continue
            if (
                type(norm) is not RMSNorm
                or norm.weight is not None
                or norm.bias is not None
                or len(norm.dim) != 1
            ):
                raise ValueError("Official LTX-2 requires stock unlearned block norms")
            replacement = torch.nn.RMSNorm(norm.dim, eps=norm.eps, elementwise_affine=False)
            replacement.train(norm.training)
            norms.append((block, name, replacement))
        for name in attention_names:
            attention = getattr(block, name)
            processor = attention.processor
            if (
                type(processor) is OfficialLtx2AttentionProcessor
                and not processor.learned_registers
            ):
                continue
            if type(processor) is not LTX2AudioVideoAttnProcessor:
                raise ValueError("Official LTX-2 requires stock attention processors")
            replacement = OfficialLtx2AttentionProcessor(learned_registers=False)
            replacement._attention_backend = processor._attention_backend
            replacement._parallel_config = processor._parallel_config
            processors.append((attention, replacement))
    for block, name, norm in norms:
        setattr(block, name, norm)
    for attention, processor in processors:
        attention.set_processor(processor)
    return {"norms": len(norms), "processors": len(processors)}


def _official_transformer_call(transformer, forward, args, kwargs):
    """Use the native opposite-modality sigma for each cross-attention gate.

    Diffusers shares the scale/shift timestep with the gate. The native profile
    retains that scale/shift input but computes each gate from the other
    modality's global sigma, including its scalar batch shape.
    """
    import inspect

    signature = transformer.forward if forward is transformer else forward
    inputs = inspect.signature(signature).bind(*args, **kwargs).arguments
    sigma = inputs.get("sigma")
    audio_sigma = inputs.get("audio_sigma")
    if not isinstance(sigma, torch.Tensor) or not isinstance(audio_sigma, torch.Tensor):
        raise ValueError("Official LTX-2 requires explicit video and audio sigmas")
    factor = (
        transformer.config.cross_attn_timestep_scale_multiplier
        / transformer.config.timestep_scale_multiplier
    )
    hooks = []
    try:
        for name, value in (
            ("av_cross_attn_video_a2v_gate", audio_sigma),
            ("av_cross_attn_audio_v2a_gate", sigma),
        ):
            timestep = value.flatten() * factor

            def replace(module, gate_args, gate_kwargs, timestep=timestep):
                if len(gate_args) != 1 or "timestep" in gate_kwargs:
                    raise ValueError("Official LTX-2 requires the stock gate call signature")
                return (timestep,), gate_kwargs

            hooks.append(
                getattr(transformer, name).register_forward_pre_hook(replace, with_kwargs=True)
            )
        return forward(*args, **kwargs)
    finally:
        for hook in reversed(hooks):
            hook.remove()


def official_transformer_forward(transformer, *args, **kwargs):
    """Invoke the native gate policy once, including for an already configured instance."""
    if getattr(getattr(transformer.forward, "__func__", None), "_ltx_official_policy", False):
        return transformer(*args, **kwargs)
    return _official_transformer_call(transformer, transformer, args, kwargs)


def configure_official_forward(transformer) -> int:
    """Install native gate inputs for every entry point on this transformer instance."""
    from functools import wraps
    from types import MethodType

    current = getattr(transformer.forward, "__func__", None)
    if getattr(current, "_ltx_official_policy", False):
        return 0
    original = type(transformer).forward
    if current is not original:
        raise ValueError("Official LTX-2 requires the stock transformer forward")

    @wraps(original)
    def forward(self, *args, **kwargs):
        # Capture only the unbound function; deep copies bind their own instance.
        return _official_transformer_call(self, original.__get__(self, type(self)), args, kwargs)

    forward._ltx_official_policy = True
    transformer.forward = MethodType(forward, transformer)
    return 1
