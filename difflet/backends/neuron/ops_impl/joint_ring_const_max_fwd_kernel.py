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
# Difflet: derived from ring_attention_const_max_fwd (vllm-omni-neuron v0.29.0.0.2.0, forked as
# ring_const_max_fwd_kernel.py), whose helpers this driver imports. Changes: joint (image + text)
# attention, with queries of L_img + L_txt rows, a replicated text K/V block attended once before the
# ring under a run-time key mask, and the text key norms folded into the softmax bound.

"""Ring context-parallel joint attention with a const-max softmax.

Per rank of a CP group of size c: queries are this rank's L_img image rows followed by all L_txt text
rows; keys are the replicated text block (attended once, never rotated) and every rank's image block
(rotated around the ring). With one shift c_i per query row for every block,

    c_i = scale · ‖q_i‖ · max_j ‖k_j‖ ,   O_i = Σ_b Σ_{k∈b} e^{s q_i·k − c_i} v_k / Σ_b Σ_{k∈b} e^{s q_i·k − c_i}

the text step and the ring steps fold into one SBUF accumulator by plain addition. max_j ‖k_j‖ covers
the valid text keys (local: text is replicated) and all ranks' image keys (all_reduce(max) over the CP
group), so the text maximum is folded into the local one before the reduce.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.modular_allocator import ModularAllocator
from difflet.backends.neuron.ops_impl.ring_const_max_attention_kernel import _attention_const_max_partial
from difflet.backends.neuron.ops_impl.ring_const_max_fwd_kernel import (
    _P,
    _assemble_c_row,
    _copy_batch_sharded,
    _head_chunks,
    _heads_per_chunk,
    _intersect_runs,
    _issue_kv_permute,
    _run_last_step_and_normalize,
    _run_partial_step,
    _shard_batch_runs,
    _transpose_qk_pipelined,
    _zero_kv_pad,
)


def _fold_max_into(dst_hbm, src_hbm, runs):
    """dst[h] = max(dst[h], src[h]) for this core's owned heads; both [batch_heads, 1] fp32 @ HBM."""
    for run in runs:
        rows = slice(run.start, run.start + run.count)
        a = nl.ndarray((run.count, 1), dtype=nl.float32, buffer=nl.sbuf)
        b = nl.ndarray((run.count, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a, src=dst_hbm[rows])
        nisa.dma_copy(dst=b, src=src_hbm[rows])
        nisa.tensor_tensor(dst=a, data1=a, data2=b, op=nl.maximum)
        nisa.dma_copy(dst=dst_hbm[rows], src=a)


def _run_text_step(q_transposed, k_txt_t, v_txt, o_aug_acc_sb, softmax_scale, num_token_groups, head_dim,
                   runs, c_row, text_key_mask):
    """The text block, once per head chunk, before the ring: initializes the accumulator."""
    acc_slot = 0
    for run in runs:
        heads = slice(run.start, run.start + run.count)
        _attention_const_max_partial(
            q_transposed[heads],
            k_txt_t[heads],
            v_txt[heads],
            o_aug_acc_sb,
            c_row[heads],
            softmax_scale=softmax_scale,
            aug_col_base=acc_slot * num_token_groups * (head_dim + 1),
            is_accumulate_step=False,
            key_mask_hbm=None if text_key_mask is None else text_key_mask[heads],
        )
        acc_slot += run.count


@nki.jit
def joint_ring_attention_const_max_fwd(
    q,
    k,
    v,
    k_txt,
    v_txt,
    text_key_mask=None,
    softmax_scale=None,
    replica_groups=None,
    num_workers=1,
    kv_prefetch_depth: int = 2,
):
    """Ring CP joint attention, const-max softmax, K/V permutes overlapped with the attention.

    Args:
        q: [b, L_img + L_txt, h, 128] token-major: this rank's image rows, then all text rows.
        k, v: [b, L_img, h, 128] token-major: this rank's image block (rotated around the ring).
        k_txt, v_txt: [b, L_txt, h, 128] token-major: the text block, identical on every rank.
            Masked keys must have zeroed K and V rows.
        text_key_mask: optional [b*h, L_txt] in v's dtype, 1 for a valid text key, 0 for padding.
        softmax_scale: defaults to 1/sqrt(128).
        replica_groups: the CP groups (global ranks); num_workers: the CP group size.
        kv_prefetch_depth: ring steps ahead to launch the K/V permutes.

    Returns:
        o: [b, h, L_img + L_txt, 128], in q's row order.
    """
    kernel_assert(nisa.get_nc_version() > nisa.nc_version.gen2, "joint ring const-max not supported on trn1")
    batch, q_len, heads, head_dim = q.shape
    _, img_len, k_heads, _ = k.shape
    _, txt_len, txt_heads, _ = k_txt.shape
    kernel_assert(heads == k_heads == txt_heads, "expects equal q, k and text heads")
    kernel_assert(head_dim == _P, f"head_dim must be 128, got {head_dim}")
    kernel_assert(q_len == img_len + txt_len, f"q rows {q_len} != image {img_len} + text {txt_len}")
    if replica_groups is None:
        replica_groups = ()

    batch_heads = batch * heads
    num_token_groups = div_ceil(q_len, _P)  # query groups: the accumulator and the bound
    img_padded = div_ceil(img_len, _P) * _P  # the ring's K/V buffers, as in the plain kernel
    kv_pad_rows = img_padded - img_len
    softmax_scale = softmax_scale or (1.0 / float(head_dim**0.5))

    q = q.reshape((batch, q_len, heads * head_dim))
    k = k.reshape((batch, img_len, heads * head_dim))
    v = v.reshape((batch, img_len, heads * head_dim))
    k_txt = k_txt.reshape((batch, txt_len, heads * head_dim))
    v_txt = v_txt.reshape((batch, txt_len, heads * head_dim))
    o = nl.ndarray((batch_heads, q_len, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

    program_ndims = nl.program_ndim()
    shard_id = nl.program_id(0) if program_ndims == 1 else 0
    num_shards = nl.num_programs(0) if program_ndims == 1 else 1
    num_heads_per_shard = batch_heads // num_shards
    head_offset = shard_id * num_heads_per_shard
    has_remainder = (batch_heads % num_shards) != 0
    runs = _shard_batch_runs(batch_heads, num_heads_per_shard, head_offset, has_remainder, shard_id)

    prefetch_depth = min(max(kv_prefetch_depth, 1), max(num_workers - 1, 1))
    num_kv_buf = prefetch_depth + 1
    k_bufs, v_bufs = [], []
    for buf_idx in range(num_kv_buf):
        k_bufs.append(nl.ndarray((batch_heads, head_dim, img_padded), dtype=k.dtype, buffer=nl.shared_hbm,
                                 name=f"jcm_k_buf_{buf_idx}"))
        v_bufs.append(nl.ndarray((batch_heads, img_padded, head_dim), dtype=v.dtype, buffer=nl.shared_hbm,
                                 name=f"jcm_v_buf_{buf_idx}"))
    k_txt_t = nl.ndarray((batch_heads, head_dim, txt_len), dtype=k.dtype, buffer=nl.shared_hbm, name="jcm_k_txt_t")
    v_txt_buf = nl.ndarray((batch_heads, txt_len, head_dim), dtype=v.dtype, buffer=nl.shared_hbm, name="jcm_v_txt")
    replica_group = ReplicaGroup(replica_groups)

    heads_per_chunk = _heads_per_chunk(num_token_groups, head_dim)
    max_owned = div_ceil(batch_heads, num_shards)
    acc_heads = min(heads_per_chunk, max_owned)
    chunk_width = batch_heads if max_owned <= heads_per_chunk else heads_per_chunk

    allocator = ModularAllocator(initial_address=0)
    o_aug_acc_sb = allocator.alloc_sbuf_tensor(shape=(_P, acc_heads * num_token_groups * (head_dim + 1)),
                                               dtype=nl.float32)

    local_k_maxsq = nl.ndarray((batch_heads, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="jcm_local_k_maxsq")
    txt_k_maxsq = nl.ndarray((batch_heads, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="jcm_txt_k_maxsq")
    reduce_k_max = num_workers > 1
    if reduce_k_max:
        global_k_maxsq = nl.ndarray((batch_heads, 1), dtype=nl.float32, buffer=nl.shared_hbm,
                                    name="jcm_global_k_maxsq")
    c_row = nl.ndarray((batch_heads, num_token_groups * _P), dtype=nl.bfloat16, buffer=nl.shared_hbm,
                       name="jcm_c_row")
    num_head_iters = num_heads_per_shard + (1 if (has_remainder and shard_id == 0) else 0)
    q_sumsq_ckpt = allocator.get_current_address()
    q_sumsq_sb = allocator.alloc_sbuf_tensor(shape=(_P, num_token_groups), dtype=nl.float32,
                                             block_dim=[num_head_iters], num_free_tiles=[num_head_iters])

    # ── Prologue: image K, then text K (its max folded into the local max), before the reduce ──
    _transpose_qk_pipelined(k_bufs[0], k, batch_heads, head_dim, img_len, num_heads_per_shard, head_offset,
                            has_remainder, shard_id, allocator, heads, k_maxsq_out=local_k_maxsq,
                            dst_stride=img_padded)
    _transpose_qk_pipelined(k_txt_t, k_txt, batch_heads, head_dim, txt_len, num_heads_per_shard, head_offset,
                            has_remainder, shard_id, allocator, heads, k_maxsq_out=txt_k_maxsq)
    _fold_max_into(local_k_maxsq, txt_k_maxsq, runs)
    if reduce_k_max:
        if num_shards > 1:
            nisa.core_barrier(local_k_maxsq, (0, 1))
        ncc.all_reduce(dsts=[global_k_maxsq], srcs=[local_k_maxsq], op=nl.maximum, replica_group=replica_group)
        k_maxsq = global_k_maxsq
    else:
        k_maxsq = local_k_maxsq

    q_transposed = nl.ndarray((batch_heads, head_dim, q_len), dtype=q.dtype, buffer=nl.shared_hbm, name="jcm_q_t")
    _transpose_qk_pipelined(q_transposed, q, batch_heads, head_dim, q_len, num_heads_per_shard, head_offset,
                            has_remainder, shard_id, allocator, heads, q_sumsq_sb=q_sumsq_sb)
    _copy_batch_sharded(v_bufs[0][:, :img_len, :], v, batch_heads, num_heads_per_shard, head_offset,
                        has_remainder, shard_id, heads, head_dim, img_len)
    _copy_batch_sharded(v_txt_buf, v_txt, batch_heads, num_heads_per_shard, head_offset, has_remainder,
                        shard_id, heads, head_dim, txt_len)
    if kv_pad_rows > 0:
        _zero_kv_pad(k_bufs[0], v_bufs[0], img_len, img_padded, head_dim, runs, allocator)
    if num_shards > 1:
        nisa.core_barrier(q_transposed, (0, 1))
        nisa.core_barrier(k_bufs[0], (0, 1))
        nisa.core_barrier(v_bufs[0], (0, 1))

    _assemble_c_row(q_sumsq_sb, k_maxsq, c_row, softmax_scale, q_len, runs, allocator)
    allocator.set_current_address(q_sumsq_ckpt)

    # ── Per head chunk: the text step initializes the accumulator, every ring step accumulates ──
    for chunk in _head_chunks(batch_heads, chunk_width):
        chunk_start, chunk_end = chunk.start, chunk.end
        chunk_runs = _intersect_runs(runs, chunk_start, chunk_end)

        if num_workers > 1:
            for prefetch_step in range(1, prefetch_depth + 1):
                _issue_kv_permute(
                    k_bufs[(prefetch_step - 1) % num_kv_buf][chunk_start:chunk_end],
                    k_bufs[prefetch_step % num_kv_buf][chunk_start:chunk_end],
                    v_bufs[(prefetch_step - 1) % num_kv_buf][chunk_start:chunk_end],
                    v_bufs[prefetch_step % num_kv_buf][chunk_start:chunk_end],
                    replica_group,
                )

        _run_text_step(q_transposed, k_txt_t, v_txt_buf, o_aug_acc_sb, softmax_scale, num_token_groups, head_dim,
                       chunk_runs, c_row, text_key_mask)

        if num_workers == 1:
            _run_last_step_and_normalize(q_transposed, k_bufs[0], v_bufs[0], o_aug_acc_sb, o, None, softmax_scale,
                                         num_token_groups, q_len, head_dim, False, nl.float32, True, chunk_runs,
                                         allocator, c_row, kv_pad_rows=kv_pad_rows)
        else:
            _run_partial_step(q_transposed, k_bufs[0], v_bufs[0], o_aug_acc_sb, softmax_scale, num_token_groups,
                              head_dim, True, chunk_runs, c_row, kv_pad_rows=kv_pad_rows)
            step_idx = 0
            for ring_step in nl.sequential_range(1, num_workers):
                step_idx += 1
                consume_idx = step_idx % num_kv_buf
                produce_step = step_idx + prefetch_depth
                if produce_step <= num_workers - 1:
                    _issue_kv_permute(
                        k_bufs[(produce_step - 1) % num_kv_buf][chunk_start:chunk_end],
                        k_bufs[produce_step % num_kv_buf][chunk_start:chunk_end],
                        v_bufs[(produce_step - 1) % num_kv_buf][chunk_start:chunk_end],
                        v_bufs[produce_step % num_kv_buf][chunk_start:chunk_end],
                        replica_group,
                    )
                if step_idx == num_workers - 1:
                    _run_last_step_and_normalize(q_transposed, k_bufs[consume_idx], v_bufs[consume_idx],
                                                 o_aug_acc_sb, o, None, softmax_scale, num_token_groups, q_len,
                                                 head_dim, False, nl.float32, True, chunk_runs, allocator, c_row,
                                                 kv_pad_rows=kv_pad_rows)
                else:
                    _run_partial_step(q_transposed, k_bufs[consume_idx], v_bufs[consume_idx], o_aug_acc_sb,
                                      softmax_scale, num_token_groups, head_dim, True, chunk_runs, c_row,
                                      kv_pad_rows=kv_pad_rows)

    return o.reshape((batch, heads, q_len, head_dim))
