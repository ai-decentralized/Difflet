"""Wan on TPU with sequence parallelism over collective matmuls (``DIFFLET_TPU_SP=1``).

``difflet/models/wan/modeling_wan.py`` already implements Megatron-SP for
Trainium (``sp_enabled``): the model scatters the patch sequence across the TP
group on entry and gathers it on exit, and inside each attention and FFN it
all-gathers before the column-parallel projections and reduce-scatters after
the row-parallel one. That file is shared with Trainium, so it is left as is.
On TPU the separate gather / reduce-scatter are slower than the all-reduce they
replace; what pays is fusing each into its matmul
(``ops_impl.collective_matmul``). :func:`install` therefore swaps the forward
of every block's ``attn1``, ``attn2`` and ``ffn`` for the fused equivalent, on
this module instance only.

What runs where, per block (B, S, H = batch, sequence, model dim):

    self-attn   x (B,S/tp,H) -AG+[to_q,to_k,to_v]-> q,k,v (B,S,heads/tp)
                attention as before (full sequence, local heads), then
                to_out[0] -matmul+RS-> (B,S/tp,H)
    cross-attn  as above with AG+[to_q] only (k, v come from the replicated text)
    FFN         x -AG+net_in-> gelu -net_out+RS-> (B,S/tp,H)

The per-token work between them (residual, adaLN, the three LayerNorms) is the
shared block forward, now on S/tp tokens. The across-heads q/k RMSNorm keeps its
(B,S,1) all-reduce: it needs every head's sum of squares.

Row padding (``DIFFLET_TPU_WAN_SP_PAD``, default on). The rings and the Pallas kernels split
each rank's rows into halves and address them as whole HBM tiles; when a half is
not a multiple of 16 rows (480x832x9: S=4680, 585-row halves) every all-gather's
input split and output assembly, and the reduce-scatter kernel's input, are
relayout copies. Padded, the sequence runs as ``Sp`` rows, a multiple of
``2 * tp * 16`` (4680 -> 4736 at tp=4). The model forward is the shared one with
its own scatter / gather turned off: the first block pads the patch sequence and
keeps this rank's rows (an ``index_select`` on a per-rank buffer -- a runtime
value, so every rank compiles the same program; a constant slice of a padded
sequence broke the rings, see collective_matmul), the last block gathers and
cuts the pad. In self-attention the pad rows are not keys (exact) and get the
identity rotation; as queries they compute values that are cut off.
480x832x9, v5e-4, 2026-11-11: step 269.2 -> 262.2 ms; the relayouts of the rings'
input splits and output assembly (138 ms of a 20-step denoise) are gone, the
cut of the pad keys and the wider final gather cost ~50 ms back.

Split-halves RoPE (``DIFFLET_TPU_WAN_SPLIT_ROPE``, default on). Wan rotates each
head's channel pairs ``(2i, 2i+1)``: a ``(..., D/2, 2)`` interleave of every q and
k, which the trace above put at ~90 ms of relayout per denoise. As for FLUX
(backends/tpu/flux/rope.py): each head's channels of self-attention's ``to_q`` /
``to_k`` rows and of the q/k RMSNorm weights are permuted once at install to
``[0, 2, ..., D-2, 1, 3, ..., D-1]``, so the pairs become ``(i, D/2 + i)`` and the
rotation is a slice and a concat. ``q . k`` per head is unchanged by a permutation
of both, V and everything after are untouched; only the RMSNorm's summation
order differs from the interleaved layout. 480x832x9 (padded), v5e-4,
2026-11-11: step 261.7 -> 249.2 ms; the interleave's reshapes / copies and the
q/k/v output relayout (~320 ms of a 20-step denoise) are gone.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from difflet.backends.tpu.ops_impl import collective_matmul as cm

#: ``DIFFLET_TPU_WAN_SP_PAD`` (default on, ``0`` turns it off): pad the sequence so
#: each ring half is a multiple of ``ROW_ALIGN`` rows (see the module docstring).
PAD_ENV = "DIFFLET_TPU_WAN_SP_PAD"

#: Rows per ring half are padded to a multiple of this (bf16 HBM tiles hold 16 rows).
ROW_ALIGN = 16


def _pad_requested() -> bool:
    import os

    return os.environ.get(PAD_ENV, "1").strip().lower() in ("1", "true", "yes", "on")


#: ``DIFFLET_TPU_WAN_SPLIT_ROPE`` (default on, ``0`` turns it off): see the module docstring.
SPLIT_ROPE_ENV = "DIFFLET_TPU_WAN_SPLIT_ROPE"


def _split_rope_requested() -> bool:
    import os

    return os.environ.get(SPLIT_ROPE_ENV, "1").strip().lower() in ("1", "true", "yes", "on")


def padded_length(seq_len: int, tp: int) -> int:
    """``seq_len`` rounded up to a multiple of ``2 * tp * ROW_ALIGN``."""
    step = 2 * int(tp) * ROW_ALIGN
    return (int(seq_len) + step - 1) // step * step


def _attention_forward(self, hidden_states, encoder_hidden_states=None, rotary_emb=None):
    from difflet.models.wan.modeling_wan import _attn_kernel, apply_rotary_emb

    if encoder_hidden_states is None:
        q, k, v = cm.all_gather_matmul(hidden_states, (self.to_q, self.to_k, self.to_v))
    else:
        (q,) = cm.all_gather_matmul(hidden_states, (self.to_q,))
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

    q = self._global_rms_norm(self.norm_q, q)
    k = self._global_rms_norm(self.norm_k, k)

    q = q.unflatten(-1, (self.local_heads, self.head_dim))
    k = k.unflatten(-1, (self.local_heads, self.head_dim))
    v = v.unflatten(-1, (self.local_heads, self.head_dim))

    if rotary_emb is not None and not self.is_cross_attention:
        cos, sin = rotary_emb
        pad = q.shape[1] - cos.shape[1]
        if pad > 0:  # padded sequence: the pad rows get the identity rotation
            cos = torch.cat([cos, cos.new_ones(cos.shape[0], pad, *cos.shape[2:])], dim=1)
            sin = torch.cat([sin, sin.new_zeros(sin.shape[0], pad, *sin.shape[2:])], dim=1)
        rotate = (_rotate_split_halves if getattr(self, "_difflet_split_rope", False)
                  else apply_rotary_emb)
        q = rotate(q, cos, sin)
        k = rotate(k, cos, sin)
    valid_keys = getattr(self, "_difflet_sp_valid_keys", None)
    if valid_keys is not None and not self.is_cross_attention:
        k, v = k[:, :valid_keys], v[:, :valid_keys]  # the pad rows are not keys

    out = _attn_kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                       head_dim=self.head_dim)
    out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
    out = cm.matmul_reduce_scatter(out, self.to_out[0])
    return self.to_out[1](out)


def _rotate_split_halves(x, cos, sin):
    """Wan's rotary on channels permuted to split halves: pair i is ``(i, D/2 + i)``.
    ``cos`` / ``sin`` hold one value per pair (:class:`_SplitHalvesRope`); the
    elementwise math is the interleaved ``apply_rotary_emb``'s."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class _SplitHalvesRope(torch.nn.Module):
    """``WanRotaryPosEmbed`` returning one (cos, sin) per channel pair, (1, S, 1, D/2).
    The stock tables repeat each pair's value (``repeat_interleave_real``); the
    interleaved rotation reads cos at even and sin at odd channels, so these do too."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states):
        cos, sin = self.inner(hidden_states)
        return cos[..., 0::2], sin[..., 1::2]


def _install_split_rope(module) -> None:
    from difflet.backends.tpu.flux.rope import pair_split_permutation

    head_dim = int(module.config.attention_head_dim)
    perm = pair_split_permutation(head_dim)

    def permuted(tensor, rows_per_group):
        # On the host, then back as a fresh tensor: permuting on the device left
        # the parameter in a layout that every forward copied back (v5e-4,
        # 2026-11-11: copy bf16[1280,5120] x 80 per step, 50 ms per denoise).
        host = tensor.detach().cpu()
        host = host.view(-1, rows_per_group, *host.shape[1:]).index_select(1, perm)
        return host.reshape(tensor.shape).contiguous().to(tensor.device)

    with torch.no_grad():
        for block in module.blocks:
            attn = block.attn1
            for linear in (attn.to_q, attn.to_k):  # this rank's heads
                linear.weight.data = permuted(linear.weight, head_dim)
                if linear.bias is not None:
                    linear.bias.data = permuted(linear.bias, head_dim)
            for norm in (attn.norm_q, attn.norm_k):  # every head (sliced per rank later)
                weight = getattr(norm, "weight", None)
                if weight is not None:
                    weight.data = permuted(weight, head_dim)
            attn._difflet_split_rope = True
    module.rope = _SplitHalvesRope(module.rope)


def _ffn_forward(self, x):
    (h,) = cm.all_gather_matmul(x, (self.net_in,))
    h = F.gelu(h, approximate="tanh")
    return cm.matmul_reduce_scatter(h, self.net_out)


def _padded_block_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
    """A block of the padded layout: the first pads the full patch sequence and
    keeps this rank's rows, the last gathers them and cuts the pad."""
    from difflet.ops import gather_from_sequence_parallel_region

    first, last, seq_len, padded = self._difflet_sp_pad
    if first:
        if temb.ndim == 4:
            raise NotImplementedError(f"{PAD_ENV}=1 does not cover per-token (Ti2V) modulation")
        pad = padded - hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(
            hidden_states.shape[0], pad, hidden_states.shape[2])], dim=1)
        hidden_states = hidden_states.index_select(1, self._difflet_sp_rows)
    hidden_states = type(self).forward(self, hidden_states, encoder_hidden_states, temb,
                                       rotary_emb)
    if last:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, dim=1)
        hidden_states = hidden_states[:, :seq_len]
    return hidden_states


def _install_padding(module, *, seq_len: int, padded: int, tp: int) -> None:
    from difflet.backends.tpu.ops_impl import parallel_mesh

    rank, rows = parallel_mesh.get_tp_rank(), padded // tp
    device = next(module.parameters()).device
    blocks = list(module.blocks)
    for i, block in enumerate(blocks):
        block._difflet_sp_pad = (i == 0, i == len(blocks) - 1, int(seq_len), int(padded))
        block.forward = types.MethodType(_padded_block_forward, block)
        block.attn1._difflet_sp_valid_keys = int(seq_len)
    blocks[0].register_buffer(
        "_difflet_sp_rows", torch.arange(rank * rows, (rank + 1) * rows, device=device),
        persistent=False)
    # The shared forward's own sequence scatter / gather would split and join the
    # unpadded sequence; the blocks above do it on the padded one instead.
    module.sp_enabled = False


def install(module: torch.nn.Module, *, seq_len: int) -> bool:
    """Swap the fused forwards into a ``WanTransformer3DModel`` built with
    ``sp_enabled``. Returns False (and changes nothing) when the shape does not
    split, so the caller keeps the shared Megatron-SP path."""
    if not getattr(module, "sp_enabled", False):
        raise ValueError("install() needs a Wan model built with sp_enabled=True")
    from difflet.backends.tpu.ops_impl import parallel_mesh

    tp = parallel_mesh.get_tp_size()
    padded = padded_length(seq_len, tp) if _pad_requested() else int(seq_len)
    if not cm.supports(padded, tp):
        print(f"[difflet tpu] {cm.SP_ENV}: sequence {seq_len} does not split into "
              f"2 x tp={tp}; using the unfused Megatron-SP path", flush=True)
        return False
    # The Pallas AG kernel: -8.3% per step at 480x832x9 (v5e-4, 2026-11-11);
    # DIFFLET_TPU_SP_KERNEL=jax keeps the ring.
    cm.set_kernel_default("pallas")
    cm.prepare()
    if padded != seq_len:
        _install_padding(module, seq_len=seq_len, padded=padded, tp=tp)
    split_rope = _split_rope_requested()
    if split_rope:
        _install_split_rope(module)
    for block in module.blocks:
        for attn in (block.attn1, block.attn2):
            if getattr(attn, "context_parallel_enabled", False):
                raise NotImplementedError("sequence parallelism excludes context parallelism")
            attn.forward = types.MethodType(_attention_forward, attn)
        block.ffn.forward = types.MethodType(_ffn_forward, block.ffn)
    print(f"[difflet tpu] Wan sequence-parallel with collective matmuls "
          f"({len(module.blocks)} blocks, S={seq_len}"
          f"{f' padded to {padded}' if padded != seq_len else ''}, tp={tp}"
          f"{', split-halves RoPE' if split_rope else ''})", flush=True)
    return True


__all__ = ["PAD_ENV", "SPLIT_ROPE_ENV", "install", "padded_length"]
