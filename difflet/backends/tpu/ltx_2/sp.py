"""LTX-2 on TPU with sequence parallelism over collective matmuls (``DIFFLET_TPU_SP=1``).

Under TP every LTX-2 block ends eight row-parallel linears in an all-reduce: the
video and audio self-attention, text cross-attention and a2v/v2a outputs, and
both FFNs. As for Wan, Qwen-Image and FLUX, the **video** ones become a
reduce-scatter fused into the row-parallel matmul plus an all-gather fused into
the next column-parallel one (``ops_impl.collective_matmul``), and the per-token
work between them runs on this rank's 1/tp of the video rows. The audio stream
(a few dozen to a few hundred rows) stays TP and replicated: its collectives are
small, and sharding it would not pay for the rings.

Layout. TP rank r holds video rows ``[r*Sp/tp, (r+1)*Sp/tp)`` of the sequence
padded to ``Sp``, a multiple of ``2*tp`` (the rings split each shard in two):
480x704x49 has 2310 rows, padded to 2312 at tp=4. The pad rows are dropped as
keys (video self-attention, and v2a where video is the key side), so the real
rows see exactly the unpadded attention; as queries they compute values that
are cut off after the model.

Padding happens **outside the compiled graph** (:func:`pad_video_inputs` /
:func:`unpad_video_output`, called eagerly by the TPU transformer application):
the latents and the video coordinates come in already padded, and the output
is cut back after. Padding inside the graph -- a ``cat`` of the sequence, then
the shard slice -- broke the rings on torch_tpu 0.2.0 (hangs, core halts, inf or
wrong values; v5e-4, 2026-11-10), whether the whole sequence or only the last
shard was padded, while the same rings were right at every unpadded length
(1024..6144, incl. 2312 itself), in plain JAX at 2312, and with the padding done
eagerly before the graph (benchmark/tpu_profile/cm_padded_repro.py).

    attn1     norm -AG+[to_q,to_k,to_v]-> attention (keys < S) -to_out+RS->
    attn2     norm -AG+[to_q]-> ; text -to_k,to_v (TP)-> attention -to_out+RS->
    a2v       video norm -AG+[to_q]-> ; audio -to_k,to_v (TP)-> -to_out+RS->
    v2a       audio -to_q (TP)-> ; video norm -AG+[to_k,to_v]-> (keys < S)
              -to_out (TP all-reduce, audio rows)->
    ff        -AG+net[0].proj-> gelu -net[2]+RS->
    audio_*   unchanged (TP)

The blocks receive the full (padded) video stream; the first block keeps this
rank's rows, the last one gathers them back, so the model forward (embedders,
AdaLN tables, RoPE, ``proj_out``) is diffusers' own.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.tpu.ops_impl import collective_matmul as cm

#: The collective matmuls LTX-2 uses: RS over one pair, AG over 1, 2, 3 weights.
LTX2_OPS = (("rs", 1), ("ag", 1), ("ag", 2), ("ag", 3))

#: ``DIFFLET_TPU_LTX2_SP_SHARED_CA_GATHER`` (default on; ``0`` for the two rings): a2v
#: and v2a share one all-gather of ``audio_to_video_norm`` (XLA all-gather, then each
#: modulation on the full rows and plain column-parallel matmuls) instead of two AG
#: rings over differently modulated copies of it. Exact: the modulation is per batch
#: (scale/shift (B, 1, D)), so it commutes with the row gather; block 0 vs TP is the
#: same rel L2 either way (0.00475). v5e-4, 480x704x49: window 189.1 -> 181.0 ms/step
#: (one x gather a layer fewer, and the narrow outputs' DUS assembly gone); the video
#: differs from the rings' by bf16 accumulation order only (PSNR 34.5 dB, 2026-11-13).
SHARED_CA_GATHER_ENV = "DIFFLET_TPU_LTX2_SP_SHARED_CA_GATHER"


def _shared_ca_gather() -> bool:
    import os

    raw = os.environ.get(SHARED_CA_GATHER_ENV, "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _linear(x, linear):
    """``linear`` (a column-parallel TP shard) on already-gathered rows."""
    return F.linear(x, linear.weight, linear.bias.to(x.dtype) if linear.bias is not None else None)


def padded_length(seq_len: int, tp: int, row_align: int = 1) -> int:
    """``seq_len`` rounded up to a multiple of ``2 * tp * row_align``. LTX-2 keeps
    ``row_align=1`` (2310 -> 2312 at tp=4): aligning the ring halves to the bf16
    tiles (16 -> 2432) cost +17..20 ms/step on v5e-4 -- the padded rows' compute,
    and the assembly copies were not alignment's doing (2026-11-13)."""
    return cm.ring_padded_length(seq_len, tp, row_align)


def _pad_seq(x: torch.Tensor, dim: int, length: int, value: float) -> torch.Tensor:
    pad = int(length) - int(x.shape[dim])
    if pad <= 0:
        return x
    shape = list(x.shape)
    shape[dim] = pad
    return torch.cat([x, x.new_full(shape, value)], dim=dim)


def pad_video_inputs(hidden_states: torch.Tensor, video_coords: torch.Tensor,
                     padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the packed latents (B, S, C) with zeros and the video coordinates
    (B, 3, S, 2) by repeating the last row, to ``padded`` rows. Eager only:
    see the module docstring."""
    pad = int(padded) - int(hidden_states.shape[1])
    if pad <= 0:
        return hidden_states, video_coords
    hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(
        hidden_states.shape[0], pad, hidden_states.shape[2])], dim=1)
    video_coords = torch.cat([video_coords, video_coords[:, :, -1:].expand(
        -1, -1, pad, -1)], dim=2)
    return hidden_states, video_coords


def unpad_video_output(video: torch.Tensor, seq_len: int) -> torch.Tensor:
    """The real rows of the model's video output (eager)."""
    return video[:, :int(seq_len)].contiguous()


def _to_bh(x: torch.Tensor, heads: int) -> torch.Tensor:
    b, s, inner = x.shape
    return x.reshape(b, s, heads, inner // heads).permute(0, 2, 1, 3).reshape(b * heads, s, -1)


def _attend(attn, q, k, v, *, q_rope=None, k_rope=None, text_mask=None, valid_keys=None):
    """The TP processor's attention on precomputed projections: (global) qk norm,
    RoPE, attention; returns the pre-``to_out`` context (B, Sq, inner_local)."""
    import diffusers.models.transformers.transformer_ltx2 as ltx2

    from difflet.ops import attention

    norm = getattr(attn.processor, "_global_rms_norm", None)
    if norm is not None:  # TP: the RMS spans every rank's heads
        q, k = norm(attn.norm_q, q), norm(attn.norm_k, k)
    else:  # tp == 1: the stock across-heads norm is already global
        q, k = attn.norm_q(q), attn.norm_k(k)
    if q_rope is not None:
        rotate = (ltx2.apply_interleaved_rotary_emb if attn.rope_type == "interleaved"
                  else ltx2.apply_split_rotary_emb)
        q = rotate(q, q_rope)
        k = rotate(k, k_rope if k_rope is not None else q_rope)

    heads = int(attn.heads)
    b, sq, inner = q.shape
    sk = int(k.shape[1])
    kwargs = {}
    if text_mask is not None:
        from difflet.backends.tpu.ltx_2.sharding import _key_window

        mask = attn.prepare_attention_mask(text_mask, sk, b)
        mask = mask.view(b, heads, -1, mask.shape[-1])
        kwargs = _key_window(mask, b * heads, sq, sk)
    elif valid_keys is not None and int(valid_keys) < sk:
        # The pad keys are dropped: exact, and the plain flash path TP uses.
        k, v = k[:, :int(valid_keys)], v[:, :int(valid_keys)]
    out = attention(_to_bh(q, heads), _to_bh(k, heads), _to_bh(v, heads),
                    scale=1.0 / math.sqrt(inner // heads), causal=False,
                    tp_q=True, tp_k=True, tp_out=False, **kwargs)
    return out.reshape(b, heads, sq, -1).permute(0, 2, 1, 3).reshape(b, sq, inner).to(q.dtype)


def _ag(x, linears):
    return cm.all_gather_matmul(x, linears)


def _rs(x, linear):
    return cm.matmul_reduce_scatter(x, linear)


def _out_rs(attn, ctx):
    """``to_out`` of a video-query attention: row-parallel matmul + reduce-scatter."""
    return attn.to_out[1](_rs(ctx, attn.to_out[0]))


def _ff(ff, x):
    """diffusers ``FeedForward``: [GELU(proj), Dropout, Linear, (Dropout)]."""
    act = ff.net[0]
    (h,) = _ag(x, (act.proj,))
    h = ff.net[1](F.gelu(h, approximate=getattr(act, "approximate", "none")))
    out = _rs(h, ff.net[2])
    return ff.net[3](out) if len(ff.net) > 3 else out


def _block_forward(
    self,
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    temb,
    temb_audio,
    temb_ca_scale_shift,
    temb_ca_audio_scale_shift,
    temb_ca_gate,
    temb_ca_audio_gate,
    temb_prompt=None,
    temb_prompt_audio=None,
    video_rotary_emb=None,
    audio_rotary_emb=None,
    ca_video_rotary_emb=None,
    ca_audio_rotary_emb=None,
    encoder_attention_mask=None,
    audio_encoder_attention_mask=None,
    self_attention_mask=None,
    audio_self_attention_mask=None,
    a2v_cross_attention_mask=None,
    v2a_cross_attention_mask=None,
    use_a2v_cross_attention=True,
    use_v2a_cross_attention=True,
    perturbation_mask=None,
    all_perturbed=None,
):
    """``LTX2VideoTransformerBlock.forward`` (diffusers 0.38) with the video rows sharded."""
    sp = self._difflet_sp
    if (self_attention_mask is not None or a2v_cross_attention_mask is not None
            or v2a_cross_attention_mask is not None or perturbation_mask is not None):
        raise NotImplementedError("LTX-2 SP: self/a2v/v2a masks and STG are not supported")
    if sp.first:
        if int(hidden_states.shape[1]) != sp.padded:
            raise ValueError(f"LTX-2 SP expects the video stream padded to {sp.padded} rows "
                             f"(pad_video_inputs), got {hidden_states.shape[1]}")
        n = sp.padded // sp.tp
        hidden_states = hidden_states[:, sp.rank * n:(sp.rank + 1) * n]

    batch_size = hidden_states.size(0)

    # 1.1 video self-attention
    video_ada = self.get_mod_params(self.scale_shift_table, temb, batch_size)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = video_ada[:6]
    if self.video_cross_attn_adaln:
        shift_text_q, scale_text_q, gate_text_q = video_ada[6:9]
    norm = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
    attn = self.attn1
    q, k, v = _ag(norm, (attn.to_q, attn.to_k, attn.to_v))
    ctx = _attend(attn, q, k, v, q_rope=video_rotary_emb, valid_keys=sp.seq)
    hidden_states = hidden_states + (_out_rs(attn, ctx)) * gate_msa

    # 1.2 audio self-attention (TP, unchanged)
    audio_ada = self.get_mod_params(self.audio_scale_shift_table, temb_audio, batch_size)
    (audio_shift_msa, audio_scale_msa, audio_gate_msa,
     audio_shift_mlp, audio_scale_mlp, audio_gate_mlp) = audio_ada[:6]
    if self.audio_cross_attn_adaln:
        audio_shift_text_q, audio_scale_text_q, audio_gate_text_q = audio_ada[6:9]
    norm_audio = self.audio_norm1(audio_hidden_states) * (1 + audio_scale_msa) + audio_shift_msa
    audio_hidden_states = audio_hidden_states + (self.audio_attn1(
        hidden_states=(norm_audio), encoder_hidden_states=None,
        query_rotary_emb=audio_rotary_emb, attention_mask=audio_self_attention_mask,
    )) * audio_gate_msa

    # 2. text cross-attention
    if self.cross_attn_adaln:
        shift_text_kv, scale_text_kv = self.get_mod_params(
            self.prompt_scale_shift_table, temb_prompt, batch_size)
        audio_shift_text_kv, audio_scale_text_kv = self.get_mod_params(
            self.audio_prompt_scale_shift_table, temb_prompt_audio, batch_size)
    norm = self.norm2(hidden_states)
    if self.video_cross_attn_adaln:
        norm = norm * (1 + scale_text_q) + shift_text_q
    if self.cross_attn_adaln:
        encoder_hidden_states = encoder_hidden_states * (1 + scale_text_kv) + shift_text_kv
    attn = self.attn2
    (q,) = _ag((norm), (attn.to_q,))
    ctx = _attend(attn, q, attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states),
                  text_mask=encoder_attention_mask)
    out = (_out_rs(attn, ctx))
    if self.video_cross_attn_adaln:
        out = out * gate_text_q
    hidden_states = hidden_states + out

    norm_audio = self.audio_norm2(audio_hidden_states)
    if self.audio_cross_attn_adaln:
        norm_audio = norm_audio * (1 + audio_scale_text_q) + audio_shift_text_q
    if self.cross_attn_adaln:
        audio_encoder_hidden_states = (
            audio_encoder_hidden_states * (1 + audio_scale_text_kv) + audio_shift_text_kv)
    out_audio = (self.audio_attn2(
        (norm_audio), encoder_hidden_states=audio_encoder_hidden_states,
        query_rotary_emb=None, attention_mask=audio_encoder_attention_mask,
    ))
    if self.audio_cross_attn_adaln:
        out_audio = out_audio * audio_gate_text_q
    audio_hidden_states = audio_hidden_states + out_audio

    # 3. audio-to-video and video-to-audio cross-attention
    if use_a2v_cross_attention or use_v2a_cross_attention:
        norm = self.audio_to_video_norm(hidden_states)
        norm_audio = self.video_to_audio_norm(audio_hidden_states)
        video_ca = self.get_mod_params(
            self.video_a2v_cross_attn_scale_shift_table[:4, :], temb_ca_scale_shift, batch_size)
        video_gate = self.get_mod_params(
            self.video_a2v_cross_attn_scale_shift_table[4:, :], temb_ca_gate, batch_size)
        video_a2v_scale, video_a2v_shift, video_v2a_scale, video_v2a_shift = video_ca
        a2v_gate = video_gate[0].squeeze(2)
        audio_ca = self.get_mod_params(
            self.audio_a2v_cross_attn_scale_shift_table[:4, :], temb_ca_audio_scale_shift, batch_size)
        audio_gate = self.get_mod_params(
            self.audio_a2v_cross_attn_scale_shift_table[4:, :], temb_ca_audio_gate, batch_size)
        audio_a2v_scale, audio_a2v_shift, audio_v2a_scale, audio_v2a_shift = audio_ca
        v2a_gate = audio_gate[0].squeeze(2)

        # One gather of the norm for both, when the modulation is per batch.
        full = None
        if _shared_ca_gather() and use_a2v_cross_attention and use_v2a_cross_attention and all(
                t.squeeze(2).shape[1] == 1 for t in video_ca):
            from difflet.ops import gather_from_sequence_parallel_region

            full = gather_from_sequence_parallel_region(norm, dim=1)

        if use_a2v_cross_attention:  # Q video (sharded), K/V audio (replicated)
            mod = (full if full is not None else norm) * (
                1 + video_a2v_scale.squeeze(2)) + video_a2v_shift.squeeze(2)
            mod_audio = norm_audio * (1 + audio_a2v_scale.squeeze(2)) + audio_a2v_shift.squeeze(2)
            attn = self.audio_to_video_attn
            (q,) = (_linear(mod, attn.to_q),) if full is not None else _ag((mod), (attn.to_q,))
            ctx = _attend(attn, q, attn.to_k(mod_audio), attn.to_v(mod_audio),
                          q_rope=ca_video_rotary_emb, k_rope=ca_audio_rotary_emb)
            hidden_states = hidden_states + a2v_gate * (_out_rs(attn, ctx))

        if use_v2a_cross_attention:  # Q audio (replicated), K/V video (sharded)
            mod = (full if full is not None else norm) * (
                1 + video_v2a_scale.squeeze(2)) + video_v2a_shift.squeeze(2)
            mod_audio = norm_audio * (1 + audio_v2a_scale.squeeze(2)) + audio_v2a_shift.squeeze(2)
            attn = self.video_to_audio_attn
            if full is not None:
                k, v = _linear(mod, attn.to_k), _linear(mod, attn.to_v)
            else:
                k, v = _ag((mod), (attn.to_k, attn.to_v))
            ctx = _attend(attn, attn.to_q(mod_audio), k, v,
                          q_rope=ca_audio_rotary_emb, k_rope=ca_video_rotary_emb, valid_keys=sp.seq)
            out_audio = (attn.to_out[1](attn.to_out[0](ctx)))  # TP all-reduce, audio rows
            audio_hidden_states = audio_hidden_states + v2a_gate * out_audio

    # 4. feed-forward
    norm = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    hidden_states = hidden_states + (_ff(self.ff, (norm))) * gate_mlp
    norm_audio = self.audio_norm3(audio_hidden_states) * (1 + audio_scale_mlp) + audio_shift_mlp
    audio_hidden_states = audio_hidden_states + (
        self.audio_ff((norm_audio))) * audio_gate_mlp

    if sp.last:
        from difflet.ops import gather_from_sequence_parallel_region

        hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)
    return hidden_states, audio_hidden_states


def install(module: nn.Module, *, video_seq_len: int) -> bool:
    """Swap the SP block forward into a TP-sharded ``LTX2VideoTransformer3DModel``.

    Returns False (and changes nothing) when there is nothing to shard.
    """
    from difflet.backends.tpu.ops_impl import parallel_mesh

    tp, rank = parallel_mesh.get_tp_size(), parallel_mesh.get_tp_rank()
    blocks = list(module.transformer_blocks)
    if not blocks:
        return False
    for block in blocks:
        if getattr(block, "perturbed_attn", False):
            raise NotImplementedError("LTX-2 SP does not support perturbed attention (STG)")
        if getattr(block.attn1, "to_gate_logits", None) is not None:
            raise NotImplementedError("LTX-2 SP does not support gated attention")
    seq = int(video_seq_len)
    padded = padded_length(seq, tp)
    cm.prepare(LTX2_OPS)
    for index, block in enumerate(blocks):
        block._difflet_sp = types.SimpleNamespace(
            seq=seq, padded=padded, tp=tp, rank=rank,
            first=index == 0, last=index == len(blocks) - 1,
        )
        block.forward = types.MethodType(_block_forward, block)
    module._difflet_sp_rows = (seq, padded)  # read by the application to pad / unpad
    print(f"[difflet tpu] LTX-2 sequence-parallel with collective matmuls ({len(blocks)} blocks, "
          f"video S={seq} padded to {padded}, tp={tp}; audio stays TP)", flush=True)
    return True


__all__ = [
    "LTX2_OPS", "SHARED_CA_GATHER_ENV", "install", "pad_video_inputs",
    "padded_length", "unpad_video_output"]
