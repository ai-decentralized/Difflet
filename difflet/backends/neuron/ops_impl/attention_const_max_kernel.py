# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Difflet: forked from nkilib 2.14 (the copy bundled in aws_neuronx_venv_torch_neuronx_2_14_0:
# torch-neuronx 2.14, nki 0.7.0), experimental/attention/attention_const_max.py, sha256
# 4aa030f09121d3f3e26f664a7a6bc92a7cdb299fb73170a7e6eb3a02907d3d76 (experiment E1 for the
# key-bounded models: docs/verification/2026-11-18-attn-constmax-kernel.md). Only the functions
# that change are copied (``attention_const_max``, ``_shard_inputs``,
# ``_attention_const_max_compute``, ``attention_const_max_core``); every other helper is imported
# from the installed upstream file. Added: the optional key-prefix bound ``key_bound``. Without it
# the instruction stream is upstream's, instruction for instruction.

"""Constant-max attention with an optional key-prefix bound (Difflet fork of nkilib's
``attention_const_max``; see its docstring for the algorithm and for choosing softmax_max).

``key_bound``: int32 ``[N, Sq, 1]`` @ HBM, attention_cte's ``bound_max`` layout, read as a
key-prefix mask: head ``n`` attends keys ``[0, key_bound[n, 0])``, the same for every query (rows
``0..127`` of each head are read, so ``Sq >= 128``). Keys at index ``>= key_bound`` contribute
exactly zero to numerator and denominator.

How: MM1 is computed transposed (keys on partitions) and the exp already takes the softmax max
as a per-partition ``[128, 1]`` operand. With a bound, each head gets one such operand per K tile,
``[128, num_k_tiles]`` f32, the max plus ``BIG * (key_index >= key_bound)`` (sign as the exp path
wants it), built once per head from a GpSimd iota and two VectorE ops; K tile ``t``'s exp reads
column ``t``. A masked key's exp is exp(score - max - BIG) = 0. No control flow depends on the
bound.
"""

from typing import Optional, Tuple, Union

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.common_types import AttentionLayout
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.experimental import neurotile as nt
from nkilib.experimental.attention.attention_const_max import (
    _O_TILE_SIZE,
    _P,
    _KVBlockStream,
    _extract_dims,
    _has_vector_engine_exp,
    _make_q_loader,
    _normalize,
    _prepare_softmax_max,
    _validate_inputs,
)

#: Added to a masked key's softmax max: exp(score - max - _MASK_BIG) is exactly 0 for any score
#: under _MASK_BIG - 88 (f32 / bf16 exp underflow).
_MASK_BIG = 30000.0


@nki.jit
def attention_const_max(
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    v_hbm: nl.NkiTensor,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: Optional[float] = None,
    q_layout: AttentionLayout = AttentionLayout.HEAD_MAJOR,
    k_layout: AttentionLayout = AttentionLayout.HEAD_MAJOR,
    key_bound: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """Upstream's ``attention_const_max`` plus ``key_bound`` (module docstring)."""
    Nq, Sq, d = _extract_dims(q_hbm, q_layout)
    _, Sk, d_k = _extract_dims(k_hbm, k_layout)
    Sv, d_v = v_hbm.shape[1], v_hbm.shape[2]

    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    _validate_inputs(Nq, Sq, d, Sk, d_k, Sv, d_v, q_hbm.dtype, k_hbm.dtype, v_hbm.dtype, softmax_max)
    if key_bound != None:
        kernel_assert(tuple(key_bound.shape) == (Nq, Sq, 1), f"key_bound must be [N, Sq, 1], got {key_bound.shape}")
        kernel_assert(Sq >= _P, f"key_bound is read from query rows 0..127: Sq must be >= {_P}, got {Sq}")

    # Prepare the constant max for the exp path once, up front. A scalar passes through as a
    # (pre-signed) float; a runtime HBM tensor is loaded into SBUF. See _prepare_softmax_max.
    prepared_softmax_max = _prepare_softmax_max(softmax_max)

    o_hbm = nl.ndarray(shape=(Nq, Sq, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    # Shard the independent (N, Sq) work across LNC cores by flattening N*Sq and splitting.
    # Q is [N, d, Sq] or [N, Sq, d] and O is [N, Sq, d] — both have N and Sq as concurrent dims.
    # K/V are fully replicated (all cores need all KV tokens).
    q_local, o_local, k_local, v_local, bound_local = _shard_inputs(
        q_hbm,
        k_hbm,
        v_hbm,
        o_hbm,
        q_layout=q_layout,
        key_bound=key_bound,
    )

    _attention_const_max_compute(
        q_local,
        k_local,
        v_local,
        o_local,
        q_layout=q_layout,
        k_layout=k_layout,
        softmax_max=prepared_softmax_max,
        softmax_scale=softmax_scale,
        key_bound=bound_local,
    )

    return o_hbm


def _shard_inputs(
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    v_hbm: nl.NkiTensor,
    o_hbm: nl.NkiTensor,
    q_layout: AttentionLayout,
    key_bound: Optional[nl.NkiTensor] = None,
) -> Tuple[nl.NkiTensor, ...]:
    """Shard Q/O across heads when possible, otherwise across Q's sequence axis. The key bound
    follows K (its rows are the same along queries, so the Sq split keeps it whole)."""
    Nq, Sq, _ = _extract_dims(q_hbm, q_layout)
    num_cores = nl.num_programs()
    core_id = nl.program_id(0)
    bound_local = key_bound

    if Nq % num_cores == 0:
        heads_per_core = Nq // num_cores
        head_start = core_id * heads_per_core
        head_end = head_start + heads_per_core
        q_local = q_hbm[head_start:head_end, :, :]
        o_local = o_hbm[head_start:head_end, :, :]
        k_local = k_hbm[head_start:head_end, :, :]
        v_local = v_hbm[head_start:head_end, :, :]
        if key_bound != None:
            bound_local = key_bound[head_start:head_end, :, :]
    else:
        sq_per_core = Sq // num_cores
        sq_start = core_id * sq_per_core
        sq_end = Sq if (core_id == num_cores - 1) else (sq_start + sq_per_core)
        q_local = (
            q_hbm[:, sq_start:sq_end, :] if q_layout == AttentionLayout.SEQ_MAJOR else q_hbm[:, :, sq_start:sq_end]
        )
        o_local = o_hbm[:, sq_start:sq_end, :]
        k_local = k_hbm
        v_local = v_hbm

    return q_local, o_local, k_local, v_local, bound_local


def _key_bound_max(
    key_bound_head: nl.NkiTensor,
    key_index: nl.NkiTensor,
    softmax_max: Union[float, nl.NkiTensor],
) -> nl.NkiTensor:
    """One head's per-K-tile softmax max, ``[128, num_k_tiles]`` f32 SBUF, pre-signed like
    ``softmax_max``: the max, plus ``_MASK_BIG`` (in the exp's sign) at keys ``>= key_bound``.

    key_bound_head: [Sq, 1] int32 @ HBM (rows 0..127 read; equal along queries).
    key_index: [128, num_k_tiles] f32 SBUF, key_index[p, t] = 128 t + p.
    """
    num_k_tiles = key_index.shape[1]
    bound_i32 = nl.ndarray(shape=(_P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bound_i32, src=key_bound_head[0:_P, 0:1])
    bound = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bound, src=bound_i32)
    # vector exp subtracts a positive max_value; the activation adds a negated bias.
    big = _MASK_BIG if _has_vector_engine_exp() else -_MASK_BIG
    tile_max = nl.ndarray(shape=(_P, num_k_tiles), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tile_max, data=key_index, op0=nl.greater_equal, operand0=bound, op1=nl.multiply, operand1=big)
    nisa.tensor_scalar(dst=tile_max, data=tile_max, op0=nl.add, operand0=softmax_max)
    return tile_max


def _attention_const_max_compute(
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    v_hbm: nl.NkiTensor,
    o_hbm: nl.NkiTensor,
    q_layout: AttentionLayout,
    k_layout: AttentionLayout,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: float,
    key_bound: Optional[nl.NkiTensor] = None,
) -> None:
    """Core attention computation — agnostic to LNC sharding.

    Operates on the local slice of Q/O assigned to this core.
    K/V are the full (or head-sliced) tensors.

    Args:
        q_hbm: [N_local, d, Sq_local] or [N_local, Sq_local, d] bf16 @ HBM
        k_hbm: [N_local, d, Sk] or [N_local, Sk, d] bf16 @ HBM
        v_hbm: [N_local, Sk, d] bf16 @ HBM
        o_hbm: [N_local, Sq_local, d] bf16 @ HBM (output)
        softmax_max: pre-signed max for the exp path (float or [_P, 1] SBUF tensor)
        softmax_scale: scaling factor for Q
        key_bound: None, or [N_local, Sq, 1] int32 @ HBM (module docstring)
    """
    Nq, _, d = _extract_dims(q_hbm, q_layout)

    kernel_assert(nl.tile_size.psum_num_banks >= 6, f"need at least 6 PSUM banks, have {nl.tile_size.psum_num_banks}")
    _MM1_BUFFER_BANKS = 2
    _MAX_O_AUG_BANKS = min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, 8)

    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    max_sq_block_size = (
        _MAX_O_AUG_BANKS * nl.tile_size.gemm_stationary_fmax if has_bf16_psum else nl.tile_size.gemm_moving_fmax
    )

    key_index = None
    if key_bound != None:
        _, Sk, _ = _extract_dims(k_hbm, k_layout)
        num_k_tiles = div_ceil(Sk, _P)
        key_index = nl.ndarray(shape=(_P, num_k_tiles), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=key_index, pattern=[[_P, num_k_tiles]], offset=0, channel_multiplier=1)

    for head_idx in range(Nq):
        head_max = softmax_max
        if key_bound != None:
            head_max = _key_bound_max(key_bound[head_idx], key_index, softmax_max)
        q_loader = _make_q_loader(
            q_hbm[head_idx],
            q_layout=q_layout,
            max_sq_block_size=max_sq_block_size,
            softmax_scale=softmax_scale if _has_vector_engine_exp() else None,
        )
        o_tiles = nt.tiles(o_hbm[head_idx], tile_size=(_O_TILE_SIZE, d))
        o_block_start = 0

        for q_block_idx in range(q_loader.num_blocks):
            q_block, sq_block_size = q_loader.load(q_block_idx)

            # O accumulator: one [Sq_tile, d+1] PSUM tile per Sq slice.
            # Columns 0..d-1 hold P @ V, column d accumulates sum_exp for normalization.
            num_output_tiles = div_ceil(sq_block_size, _O_TILE_SIZE)
            kernel_assert(
                num_output_tiles <= _MAX_O_AUG_BANKS,
                "Q block too large: O_aug tiles exceed available PSUM banks.",
            )
            o_aug_psums = nt.psum_pool(
                tile_size=(_O_TILE_SIZE, d + 1),
                element_shape=(sq_block_size, d + 1),
            )
            k_stream = _KVBlockStream(k_hbm[head_idx], layout=k_layout)
            v_stream = _KVBlockStream(v_hbm[head_idx], layout=AttentionLayout.SEQ_MAJOR, transpose=False)

            attention_const_max_core(
                k_stream=k_stream,
                v_stream=v_stream,
                q_block=q_block,
                o_aug_psums=o_aug_psums,
                softmax_max=head_max,
                softmax_scale=softmax_scale,
                per_k_tile_max=key_bound != None,
            )

            # Normalize each output tile by its accumulated sum_exp, then store to HBM.
            for output_tile_idx in range(num_output_tiles):
                o_out = _normalize(o_aug_psums[output_tile_idx].data)
                o_tiles[o_block_start + output_tile_idx, 0].store(o_out)
            o_block_start += num_output_tiles


def attention_const_max_core(
    k_stream: "_KVBlockStream",
    v_stream: "_KVBlockStream",
    q_block: nl.NkiTensor,
    o_aug_psums: nt.NDSlice,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: Optional[float] = None,
    per_k_tile_max: bool = False,
) -> None:
    """Compute scaled dot-product attention for one query block over all KV tiles.

    Args:
        k_stream (_KVBlockStream): Key tiles to attend over.
        v_stream (_KVBlockStream): Value tiles to attend over.
        q_block (nl.NkiTensor): [d, sq_block_size] query block.
        o_aug_psums (nt.NDSlice): Output accumulator, one [sq_tile_size, d + 1]
            tile per query slice, holding the unnormalized output and the softmax
            denominator. Written in place.
        softmax_max (Union[float, nl.NkiTensor]): Max subtracted for numerical
            stability. Scalar or [P_MAX, 1] tensor; with ``per_k_tile_max``, a
            [P_MAX, num_k_tiles] tensor whose column t serves K tile t.
        softmax_scale (Optional[float]): Query-key scaling factor.
        per_k_tile_max (bool): Difflet: ``softmax_max`` holds one column per K tile.

    Returns:
        None: Results are written in place into ``o_aug_psums``.

    Pipeline (P = prologue_depth):
        Prologue  (kv_tile_idx 0..P-1):        v_copy, MM1, exp
        Steady-state (kv_tile_idx P..total-1):  v_copy, MM1, exp, MM2(i-P)
        Epilogue  (drain last P):               MM2

    """
    d = _P
    kernel_assert(
        (k_stream.num_blocks, k_stream.num_tiles, k_stream.last_tile_extent)
        == (v_stream.num_blocks, v_stream.num_tiles, v_stream.last_tile_extent),
        "K/V stream geometry must match",
    )
    num_blocks = k_stream.num_blocks
    sq_block_size = q_block.shape[1]
    num_output_tiles = o_aug_psums.shape[0]
    total_tiles = k_stream.num_tiles
    last_tile_idx = total_tiles - 1
    last_tile_extent = k_stream.last_tile_extent

    scores_dtype = nl.bfloat16 if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    copy_engine = nisa.engine.scalar if _has_vector_engine_exp() else nisa.engine.vector

    # Number of MM1+exp tiles computed before first MM2 fires. Gives TensorE
    # a buffer of ready probabilities so it can always issue MM2 without waiting.
    prologue_depth = min(8, total_tiles)

    # Rotating V_aug buffers: need prologue_depth+1 (v_copy runs at MM1 time,
    # MM2 consumes prologue_depth iterations later; +1 to avoid WAR)
    num_v_aug_buffers = prologue_depth + 1
    v_aug_all = nl.ndarray(shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)

    num_prob_buffers = prologue_depth + 1
    probs_all = nl.ndarray(shape=(_P, num_prob_buffers, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)

    kv_tile_idx = 0  # flat index across all KV blocks
    softmax_max_is_scalar = isinstance(softmax_max, (int, float))

    for block_idx in range(num_blocks):
        k_block = k_stream.load(block_idx)[0]
        v_block = v_stream.load(block_idx)
        tiles_in_block = k_block.shape[0]

        with nl.no_reorder():
            for local_tile in range(tiles_in_block):
                v_buf = kv_tile_idx % num_v_aug_buffers
                prob_buf = kv_tile_idx % num_prob_buffers
                tile_extent = last_tile_extent if kv_tile_idx == last_tile_idx else _P

                # v_copy(i) — capture V tile now while block buffer is valid
                nisa.tensor_copy(
                    dst=v_aug_all[0:tile_extent, v_buf, :d],
                    src=v_block[local_tile, 0].data[0:tile_extent, :],
                    engine=copy_engine,
                )

                # MM1(i): K[D, r]^T @ Q[D, Sq] -> scores[r, Sq]. The free axis of
                # stationary K becomes the PSUM partition axis, so a final r < 128
                # naturally produces a partial-partition score tile.
                scores = nl.ndarray(shape=(_P, sq_block_size), dtype=scores_dtype, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=scores[0:tile_extent, :],
                    stationary=k_block[local_tile].data[:, 0:tile_extent],
                    moving=q_block,
                )

                # exp(i) — fused evict + scale + subtract-max + exp
                if softmax_max_is_scalar:
                    tile_softmax_max = softmax_max
                elif per_k_tile_max:
                    tile_softmax_max = softmax_max[0:tile_extent, kv_tile_idx : kv_tile_idx + 1]
                else:
                    tile_softmax_max = softmax_max[0:tile_extent, :]
                if _has_vector_engine_exp():
                    nisa.exponential(
                        dst=probs_all[0:tile_extent, prob_buf, :],
                        src=scores[0:tile_extent, :],
                        max_value=tile_softmax_max,
                    )
                else:
                    nisa.activation(
                        dst=probs_all[0:tile_extent, prob_buf, :],
                        data=scores[0:tile_extent, :],
                        op=nl.exp,
                        scale=softmax_scale,
                        bias=tile_softmax_max,
                    )

                # MM2(i-P) — only after prologue is filled
                prologue_filled = kv_tile_idx >= prologue_depth
                if prologue_filled:
                    consume_idx = kv_tile_idx - prologue_depth
                    consume_v_buf = consume_idx % num_v_aug_buffers
                    consume_prob_buf = consume_idx % num_prob_buffers
                    should_accumulate = consume_idx != 0
                    cur_v_aug = v_aug_all[:, consume_v_buf, :]
                    consume_extent = last_tile_extent if consume_idx == last_tile_idx else _P
                    for ot_idx in range(num_output_tiles):
                        sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
                        nisa.nc_matmul(
                            dst=o_aug_psums[ot_idx].data,
                            stationary=probs_all[
                                0:consume_extent,
                                consume_prob_buf,
                                nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size),
                            ],
                            moving=cur_v_aug[0:consume_extent, :],
                            accumulate=should_accumulate,
                        )

                kv_tile_idx += 1

    # === Epilogue: drain remaining prologue_depth MM2s ===
    kernel_assert(kv_tile_idx == total_tiles, f"stream yielded {kv_tile_idx} KV tiles, expected {total_tiles}")
    with nl.no_reorder():
        for i in range(prologue_depth):
            drain_idx = total_tiles - prologue_depth + i
            drain_v_buf = drain_idx % num_v_aug_buffers
            drain_prob_buf = drain_idx % num_prob_buffers
            should_accumulate = drain_idx != 0
            drain_v_aug = v_aug_all[:, drain_v_buf, :]
            drain_extent = last_tile_extent if drain_idx == last_tile_idx else _P
            for ot_idx in range(num_output_tiles):
                sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
                nisa.nc_matmul(
                    dst=o_aug_psums[ot_idx].data,
                    stationary=probs_all[
                        0:drain_extent,
                        drain_prob_buf,
                        nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size),
                    ],
                    moving=drain_v_aug[0:drain_extent, :],
                    accumulate=should_accumulate,
                )
