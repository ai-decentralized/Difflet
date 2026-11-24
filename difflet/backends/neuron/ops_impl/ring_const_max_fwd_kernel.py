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
# Difflet: forked from aws-neuron/vllm-omni-neuron v0.29.0.0.2.0,
# vllm_omni_neuron/kernels/nkilib/experimental/attention/ring_attention_const_max_fwd.py, sha256 b226d960dcdeabf59dffa5238007ba093db41d8c9d1ebe6e96e3d8c08a56f6f3.
# Changes: imports only (the core utilities and neurotile from the installed nkilib, the base
# kernel from its Difflet fork); the kernel code is unchanged.

"""Ring constant-max attention forward — non-causal.

This kernel is the ring-attention driver for `attention_const_max`. The base per-step
attention is K-stationary (scores are produced as [Sk-partition, Sq-free]), which removes
the P-transpose that `attention_cte` does inside its compute. That transpose is a
`dma_transpose`, and `dma_transpose` cannot run concurrently with a collective on the
hardware — so removing it is what lets the KV-rotation collective-permute overlap the
attention compute.

Because the base kernel has no online-max pass, the softmax max is supplied up front: a runtime
per-ROW Cauchy–Schwarz L2 bound on the scaled scores,

    c_i = softmax_scale · ‖q_i‖₂ · max_j‖k_j‖₂

one value per query row — the tightest granularity, so every row's exp stays well-scaled and no
quiet row is over-subtracted by a loud one. Since q_i·k_j ≤ ‖q_i‖·‖k_j‖ the exponent stays ≤ 0 and
the exp can never overflow. max_j‖k_j‖ spans the ring's GLOBAL K (an all_reduce(max) across the CP
group); the per-token ‖q_i‖ are local, since query rows do not rotate. Both norm passes are fused
into the prologue transpose, off the tiles it already loads.

c_i varies along the score tile's free axis, so it cannot ride the exp's per-partition max_value
and is subtracted explicitly on the Vector engine before the exp.

Both c_i and the global key max are STATIC across ring steps (query rows do not rotate through the ring),
so the cross-step reduction is pure addition — no online-softmax correction factors, no running max:

    sum_prev += sum_curr
    o_prev   += o_curr        (both unnormalized)

with a single final normalize o = o_prev / sum_prev.
"""

from dataclasses import dataclass

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.modular_allocator import ModularAllocator
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from difflet.backends.neuron.ops_impl.ring_const_max_attention_kernel import _attention_const_max_partial

_P = nl.tile_size.pmax  # 128: partition dimension (also the head dim and the 128-token group size)
_PSUM_BANK_SIZE = nl.tile_size.psum_bank_fmax_bytes  # 2048: per-bank byte stride for psum address free-offset


@dataclass
class _HeadRange(nl.NKIObject):
    """A contiguous range of global head indices [start, end), i.e. `count` heads from `start`.

    Used for both the head runs an LNC shard owns and the global head chunks the driver loops over,
    so callers can read whichever of `count` / `end` fits.
    """

    start: int  # first global head index in the range
    count: int  # number of heads in the range
    end: int = 0  # one past the last global head index

    def __post_init__(self):
        # Field, not @property: NKI lowering does not fold property getters.
        self.end = self.start + self.count


def _copy_batch_sharded(
    dst, src, bs, num_bs_per_shard, bs_offset, has_remainder, shard_id, src_heads, head_dim, seqlen
):
    """Copy every head of a token-major HBM tensor into a head-major HBM buffer, LNC-sharded.

    Each NC copies only its assigned heads (odd-head-count remainder handled by shard 0),
    parallelizing the DMA work under LNC2. dst is [bs, seqlen, head_dim]; src is
    [batch, seqlen, src_heads*head_dim] — every head interleaved on the token row, so one head is a
    column range repeating each token row rather than a contiguous plane, and its read is one
    token-strided access pattern.
    """
    row_stride = src_heads * head_dim  # one whole token row
    for batch_local_idx in range(num_bs_per_shard):
        head_idx = batch_local_idx + bs_offset
        nisa.dma_copy(
            dst=dst.select(dim=0, index=head_idx),
            src=src.select(dim=0, index=head_idx // src_heads).ap(
                pattern=[[row_stride, seqlen], [1, head_dim]], offset=(head_idx % src_heads) * head_dim
            ),
        )
    if has_remainder and shard_id == 0:
        last_head = bs - 1
        nisa.dma_copy(
            dst=dst.select(dim=0, index=last_head),
            src=src.select(dim=0, index=last_head // src_heads).ap(
                pattern=[[row_stride, seqlen], [1, head_dim]], offset=(last_head % src_heads) * head_dim
            ),
        )


def _zero_kv_pad(k_buf, v_buf, seqlen, seq_padded, head_dim, runs, allocator):
    """Zero the pad tokens [seqlen, seq_padded) of one K and V ring buffer, for this shard's heads.

    K is d-major [head_dim, seq_padded] so the pad is a column slice; V is seq-major
    [seq_padded, head_dim] so it is a row slice. Both are written from one zeroed SBUF tile, which is
    at most (128, 128) since seq_padded - seqlen < 128.
    """
    scratch_addr = allocator.get_current_address()
    zeros = allocator.alloc_sbuf_tensor(shape=(_P, head_dim), dtype=k_buf.dtype)
    nisa.memset(zeros, 0.0)
    pad = seq_padded - seqlen
    for run in runs:
        for head_idx in range(run.start, run.end):
            nisa.dma_copy(dst=k_buf[head_idx][:, seqlen:seq_padded], src=zeros[:head_dim, :pad])
            nisa.dma_copy(dst=v_buf[head_idx][seqlen:seq_padded, :], src=zeros[:pad, :head_dim])
    allocator.set_current_address(scratch_addr)


def _issue_kv_permute(cur_k, nxt_k, cur_v, nxt_v, replica_group):
    """Issue the K and V ring collective-permutes for one ring step.

    Reads cur_k/cur_v (the just-received buffers) and writes nxt_k/nxt_v.
    """
    ncc.collective_permute_implicit(
        srcs_by_channel=[[cur_k]],
        dsts_by_channel=[[nxt_k]],
        replica_group=replica_group,
    )
    ncc.collective_permute_implicit(
        srcs_by_channel=[[cur_v]],
        dsts_by_channel=[[nxt_v]],
        replica_group=replica_group,
    )


def _shard_batch_runs(batch_heads, num_heads_per_shard, head_offset, has_remainder, shard_id):
    """Contiguous `_HeadRange` runs of heads this LNC shard owns.

    `batch_heads` is the flattened batch*heads dimension; each index is one attention head instance.
    Mirrors _copy_batch_sharded: each shard owns a contiguous block of num_heads_per_shard heads at
    head_offset, and shard 0 additionally owns the odd-batch_heads remainder head.

    Examples as (start, count) pairs (batch_heads = batch*heads;
    num_heads_per_shard = batch_heads // num_shards):
      LNC1 (num_shards=1):                        -> [(0, batch_heads)]   one run, all heads
      LNC2, batch_heads=10 (even): shard 0        -> [(0, 5)]             heads 0..4
                                   shard 1        -> [(5, 5)]             heads 5..9
      LNC2, batch_heads=5  (odd):  shard 0        -> [(0, 2), (4, 1)]     heads 0,1 + remainder head 4
                                   shard 1        -> [(2, 2)]             heads 2,3
    """
    runs = []
    if num_heads_per_shard > 0:
        runs.append(_HeadRange(head_offset, num_heads_per_shard))
    if has_remainder and shard_id == 0:
        runs.append(_HeadRange(batch_heads - 1, 1))
    return runs


# Per-partition SBUF (bytes) reserved for everything OTHER than the o_acc/sum_acc accumulators:
# the K/V double-buffered stream, normalize scratch, and allocator fragmentation.
_ACC_SBUF_RESERVE_BYTES = 40 * 1024


def _heads_per_chunk(num_token_groups, head_dim):
    """How many heads' resident accumulators fit the SBUF budget (at least 1).

    The cross-ring-step accumulators (o_acc + sum_acc) stay SBUF-resident for the whole ring, at a
    per-partition cost of `num_token_groups * (head_dim + 1)` fp32 words per head (o_acc:
    num_token_groups*head_dim, sum_acc: num_token_groups). Holding all `batch_heads` heads at once
    OOMs at long seqlen, so processes heads in chunks that fit `budget` = usable
    per-partition SBUF (sbuf_fmax_bytes, ~208 KB trn2 / ~240 KB trn3) minus a fixed reserve, running
    the full ring per chunk.

    Returns floor(budget / bytes_per_head), floored at 1 so a chunk always holds >= 1 head.

    Example: num_token_groups=42 (seqlen=5376), head_dim=128 -> bytes_per_head = 42*4*129 ≈ 21.7 KB;
    budget ≈ 168 KB (trn2) -> 7 heads/chunk.
    """
    bytes_per_head = (
        num_token_groups * 4 * (head_dim + 1)
    )  # o_acc num_token_groups*head_dim*4 + sum_acc num_token_groups*4
    budget = nl.tile_size.sbuf_fmax_bytes - _ACC_SBUF_RESERVE_BYTES
    return max(1, budget // bytes_per_head)


def _head_chunks(batch_heads, chunk_width):
    """Split the global head range [0, batch_heads) into contiguous `_HeadRange` chunks of width
    `chunk_width` (the last one is short when chunk_width does not divide batch_heads).
    """
    chunks = []
    for chunk_start in range(0, batch_heads, chunk_width):
        chunks.append(_HeadRange(chunk_start, min(chunk_width, batch_heads - chunk_start)))
    return chunks


def _intersect_runs(runs, chunk_start, chunk_end):
    """Clip this core's owned head runs to the head range [chunk_start, chunk_end)."""
    clipped_runs = []
    for run in runs:
        clipped_start = max(run.start, chunk_start)
        clipped_end = min(run.end, chunk_end)
        if clipped_end > clipped_start:
            clipped_runs.append(_HeadRange(clipped_start, clipped_end - clipped_start))
    return clipped_runs


_TP_GROUP = 2  # sub-tiles coalesced into one load DMA (and one store DMA).
_TP_DEPTH = 4  # pipeline depth over coalesced groups: _TP_DEPTH group-buffers + _TP_DEPTH*_TP_GROUP
# PSUM banks (<=8) so load(group t+1..) overlaps transpose/evict/store(group t).
# _TP_DEPTH*_TP_GROUP == 8 banks. _TP_GROUP=2/_TP_DEPTH=4 balances load coalescing (2x fewer DMAs)
# with deep overlap; a bigger _TP_GROUP starves depth (only 8 banks) and leaves the compute exposed.


def _transpose_one_batch_pipelined(
    dst_batch,
    src_batch,
    head_dim,
    seqlen,
    loaded_tiles,
    transposed_out,
    psum_banks,
    subtiles_per_group,
    k_maxsq_out_batch=None,
    squared_buf=None,
    running_max_sumsq=None,
    per_token_sumsq=None,
    q_sumsq_sb=None,
    dst_stride=None,
    src_row_stride=None,
    src_col_offset=0,
):
    """Transpose one head out of src_batch -> dst_batch (head_dim, dst_stride) on the PE,
    coalesced + pipelined.

    src_batch is the enclosing batch's [seqlen, src_row_stride] view; this head occupies the
    head_dim columns at src_col_offset in every token row. src_row_stride defaults to head_dim
    (one head per row, i.e. the head is a contiguous plane).

    Each iteration handles one coalesced group of `subtiles_per_group` consecutive 128-token sub-tiles:
      * ONE strided load DMA brings all sub-tiles into `loaded_tiles` [_P, subtiles_per_group*head_dim]
        — sub-tile s at columns [s*head_dim:(s+1)*head_dim]. One 3D-strided read (token, sub-tile, dim)
        fetches them at once; its innermost run is this head's head_dim columns of one token row.
      * `subtiles_per_group` nc_transposes rotate each sub-tile [chunk, head_dim] -> [head_dim, chunk]
        into its own PSUM bank, each evicted (on the Scalar engine) into its column slice of the
        group's SBUF tile.
      * ONE contiguous store DMA writes [head_dim, subtiles_per_group*128] to
        dst_batch[:, group_start_token : group_start_token + subtiles_per_group*128].
    Groups pipeline via buffer slot (group_idx % pipeline_depth), so load(g+1) overlaps
    transpose/store(g). The coalesced loop covers seqlen // group_cols full groups; the trailing
    seqlen % group_cols tokens (< group_cols, for any seqlen) are peeled after the loop as up to
    subtiles_per_group single-sub-tile DMAs — whole 128-tiles plus a final sub-128 tail — each
    clamping every seq-indexed extent to its real row count r (128, or < 128 for the tail).

    Fused per-token L2² (the softmax bound) off the SAME loaded tile the transpose consumes.
    The square/reduce run on the Vector engine while the transpose (PE) and evict (Scalar) run;
    Two modes:
      * K (k_maxsq_out_batch given): fold each sub-tile's per-token L2² into running_max_sumsq; after
        the group loop partition-reduce to this head's scalar max_j‖k_j‖² and store it (only the
        largest key norm enters the bound).
      * Q (q_sumsq_sb given): keep the FULL per-token ‖q_i‖² — reduce each sub-tile straight into its
        column of the head's resident SBUF tile q_sumsq_sb [_P, num_token_groups], which _assemble_c_row
        reads directly (no HBM round-trip). The bound needs ‖q_i‖ for every query row.
    sqrt is deferred to _assemble_c_row (monotone).
    """
    pipeline_depth = len(loaded_tiles)
    group_cols = subtiles_per_group * _P  # columns spanning one coalesced group (== subtiles_per_group*head_dim)
    # dst may be WIDER than seqlen (caller padded it to a 128-multiple), so its row stride is
    # independent of the token count actually transposed.
    dst_stride = seqlen if dst_stride == None else dst_stride
    # Token stride of the source. head_dim when this head is a contiguous plane; wider when all heads
    # share the token row, in which case consecutive tokens of this head no longer touch.
    src_row_stride = head_dim if src_row_stride == None else src_row_stride
    num_coalesced_groups = seqlen // group_cols
    if k_maxsq_out_batch is not None:
        nisa.memset(running_max_sumsq, 0.0)  # sum-of-squares >= 0, so 0.0 is a valid running-max floor
    for group_idx in range(num_coalesced_groups):
        buf = group_idx % pipeline_depth
        group_start_token = group_idx * group_cols

        src_group = src_batch.ap(
            pattern=[[src_row_stride, _P], [_P * src_row_stride, subtiles_per_group], [1, head_dim]],
            offset=src_col_offset + group_start_token * src_row_stride,
        )
        nisa.dma_copy(
            dst=loaded_tiles[buf][:_P, :group_cols],
            src=src_group,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )

        # Transpose each sub-tile into its OWN PSUM bank (distinct banks => the PE transposes don't
        # serialize on one accumulation group), evicting each straight into its column slice of the
        # group's contiguous SBUF tile. The evict runs on the SCALAR engine so it doesn't contend
        # with the fused maxsq's Vector work (square+reduce).
        for subtile_idx in range(subtiles_per_group):
            nisa.nc_transpose(
                psum_banks[buf][subtile_idx][:head_dim, :_P],
                loaded_tiles[buf][:_P, subtile_idx * head_dim : subtile_idx * head_dim + head_dim],
            )
            nisa.tensor_copy(
                transposed_out[buf][:head_dim, subtile_idx * _P : subtile_idx * _P + _P],
                psum_banks[buf][subtile_idx][:head_dim, :_P],
                engine=nisa.engine.scalar,
            )

        # Fused per-token L2² off the loaded group (Vector). Square the whole [_P, group_cols] group
        # once, then sum-over-head_dim per sub-tile: each sub-tile is an independent 128-token group over its head_dim cols.
        nisa.tensor_tensor(
            dst=squared_buf[:_P, :group_cols],
            data1=loaded_tiles[buf][:_P, :group_cols],
            data2=loaded_tiles[buf][:_P, :group_cols],
            op=nl.multiply,
        )
        if q_sumsq_sb is not None:
            # Q: sum-over-head_dim for ALL sub-tiles in ONE reduce over a
            # [_P, subtiles_per_group, head_dim] view of the squared group (axis=2 == head_dim),
            # writing this group's per-token columns at once. One larger reduce instead of
            # `subtiles_per_group` tiny ones.
            first_token_group = subtiles_per_group * group_idx
            squared_view = squared_buf.reshape_dim(1, [subtiles_per_group, head_dim])
            nisa.tensor_reduce(
                dst=q_sumsq_sb[:_P, first_token_group : first_token_group + subtiles_per_group],
                op=nl.add,
                data=squared_view,
                axis=[2],
                keepdims=False,
            )
        else:
            for subtile_idx in range(subtiles_per_group):
                # K: fold per-token ‖k_j‖² into the running per-partition max.
                nisa.tensor_reduce(
                    dst=per_token_sumsq,
                    op=nl.add,
                    data=squared_buf[:_P, subtile_idx * head_dim : subtile_idx * head_dim + head_dim],
                    axis=[1],
                )
                nisa.tensor_tensor(
                    dst=running_max_sumsq, data1=running_max_sumsq, data2=per_token_sumsq, op=nl.maximum
                )

        # Contiguous store: the group's transposed sub-tiles sit side by side along dst's free axis.
        dst_group = dst_batch.ap(pattern=[[dst_stride, head_dim], [1, group_cols]], offset=group_start_token)
        nisa.dma_copy(dst=dst_group, src=transposed_out[buf][:head_dim, :group_cols])

    # Remainder: the trailing seqlen % group_cols tokens (< group_cols) that don't fill a coalesced
    # group. Peeled as up to subtiles_per_group single-sub-tile DMAs — whole 128-tiles plus a final
    # sub-128 tail. A whole tile and the partial tail can't share one strided load (their partition
    # counts differ), so each sub-tile loads/transposes/stores on its own, clamping every seq-indexed
    # extent to its real row count r (128 for a whole tile, < 128 for the tail) so padded rows are
    # never loaded, transposed, stored, or folded into the norm.
    rem = seqlen - num_coalesced_groups * group_cols
    base_group = num_coalesced_groups * subtiles_per_group  # first token-group column after the body
    for rem_subtile_idx in range((rem + _P - 1) // _P):
        r = min(_P, rem - rem_subtile_idx * _P)  # real tokens in this sub-tile (128, or the tail's < 128)
        buf = (num_coalesced_groups + rem_subtile_idx) % pipeline_depth
        token_start = num_coalesced_groups * group_cols + rem_subtile_idx * _P
        src_sub = src_batch.ap(
            pattern=[[src_row_stride, r], [1, head_dim]], offset=src_col_offset + token_start * src_row_stride
        )
        nisa.dma_copy(
            dst=loaded_tiles[buf][:r, :head_dim],
            src=src_sub,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.sync,
        )
        nisa.nc_transpose(psum_banks[buf][0][:head_dim, :r], loaded_tiles[buf][:r, :head_dim])
        nisa.tensor_copy(
            transposed_out[buf][:head_dim, :r],
            psum_banks[buf][0][:head_dim, :r],
            engine=nisa.engine.scalar,
        )
        nisa.tensor_tensor(
            dst=squared_buf[:r, :head_dim],
            data1=loaded_tiles[buf][:r, :head_dim],
            data2=loaded_tiles[buf][:r, :head_dim],
            op=nl.multiply,
        )
        if q_sumsq_sb is not None:
            col = base_group + rem_subtile_idx
            if r < _P:
                # Partial tail column: zero the whole column first so padded rows r.._P-1 are 0 —
                # they hold no real query position and must not reach _assemble_c_row as a norm. A
                # memset of only [r:_P] would start at a runtime partition offset, which the
                # hardware rejects.
                nisa.memset(q_sumsq_sb[:, col : col + 1], 0.0)
            nisa.tensor_reduce(
                dst=q_sumsq_sb[:r, col : col + 1],
                op=nl.add,
                data=squared_buf[:r, :head_dim],
                axis=[1],
            )
        else:
            # K: fold only the real rows into the running per-partition max; padded partitions keep
            # their prior (valid, <= max) value, so the later partition-reduce stays correct.
            nisa.tensor_reduce(dst=per_token_sumsq[:r, :], op=nl.add, data=squared_buf[:r, :head_dim], axis=[1])
            nisa.tensor_tensor(
                dst=running_max_sumsq[:r, :],
                data1=running_max_sumsq[:r, :],
                data2=per_token_sumsq[:r, :],
                op=nl.maximum,
            )
        dst_sub = dst_batch.ap(pattern=[[dst_stride, head_dim], [1, r]], offset=token_start)
        nisa.dma_copy(dst=dst_sub, src=transposed_out[buf][:head_dim, :r])

    if k_maxsq_out_batch is not None:
        # partition max -> scalar at partition 0; store this head's max‖·‖² (in-place is safe).
        nisa.tensor_partition_reduce(running_max_sumsq, nl.max, running_max_sumsq)
        nisa.dma_copy(dst=k_maxsq_out_batch, src=running_max_sumsq[0:1, :])
    # Q mode leaves the per-token ‖q_i‖² in the resident q_sumsq_sb tile — no store; _assemble_c_row
    # reads it directly.


def _transpose_qk_pipelined(
    dst,
    src,
    batch_heads,
    head_dim,
    seqlen,
    num_heads_per_shard,
    head_offset,
    has_remainder,
    shard_id,
    allocator,
    src_heads,
    k_maxsq_out=None,
    q_sumsq_sb=None,
    dst_stride=None,
):
    """Pipelined PE transpose of HBM src (batch, seqlen, src_heads*head_dim) ->
    HBM dst (batch_heads, head_dim, seqlen), per head.

    src is token-major: all `src_heads` heads of a batch share each token row, so global head index
    `idx` lives in batch `idx // src_heads` at column base `(idx % src_heads) * head_dim`. dst is
    head-major, one d-major plane per head.

    The baseline single-buffered every 128-chunk, so its load->transpose->evict->store chain ran fully
    serial. This version coalesces _TP_GROUP sub-tiles per load/store DMA (fewer, bigger transfers) and
    pipelines groups (depth _TP_DEPTH) so DMAs overlap PE/Vector work.
    PSUM banks are partitioned into _TP_DEPTH groups of _TP_GROUP contiguous banks

    All SBUF scratch is drawn from the shared `allocator` (checkpoint at entry, reset at exit).
    The transpose is a prologue-only pass: its scratch is dead before ring step 0, so it never coexists
    with the resident accumulator.

    Softmax bound: compute each owned head's per-token L2² off the already-loaded transpose
    tiles. K (k_maxsq_out given) writes the scalar max‖·‖² to k_maxsq_out[head]; Q (q_sumsq_sb given)
    writes the full per-token ‖q_i‖² vector into the caller's resident SBUF tile q_sumsq_sb[iteration],
    one tile per owned head in loop order (consumed directly by _assemble_c_row).
    """
    # _transpose_one_batch_pipelined always coalesces _TP_GROUP consecutive 128-token sub-tiles into
    # ONE load/store DMA (2x fewer DMAs), pipelined _TP_DEPTH deep. The coalesced loop covers
    # seqlen // (_TP_GROUP*_P) full groups; the trailing seqlen % (_TP_GROUP*_P) tokens (any seqlen,
    # not just a 256-multiple) are peeled after the loop as up to _TP_GROUP single-sub-tile DMAs —
    # whole 128-tiles plus a sub-128 tail. So all coalescing except the final (< _TP_GROUP*_P) group
    # is preserved regardless of seqlen; only the small remainder pays the per-128-DMA cost.
    subtiles_per_group, pipeline_depth = _TP_GROUP, _TP_DEPTH
    group_cols = subtiles_per_group * _P

    # Everything below is transient scratch: bump-allocate from `allocator` and free it all at exit.
    scratch_addr = allocator.get_current_address()

    # Bound scratch: a single square buffer, plus (K) the running per-partition max + per-sub-tile
    # sum. The Q per-token accumulator (q_sumsq_sb) is caller-owned and resident (fed to _assemble_c_row).
    running_max_sumsq = per_token_sumsq = None
    squared_buf = allocator.alloc_sbuf_tensor(shape=(_P, group_cols), dtype=nl.float32)
    if k_maxsq_out is not None:
        running_max_sumsq = allocator.alloc_sbuf_tensor(shape=(_P, 1), dtype=nl.float32)
        per_token_sumsq = allocator.alloc_sbuf_tensor(shape=(_P, 1), dtype=nl.float32)

    # One tile per pipeline slot: `block_dim=[pipeline_depth]` hands back a flat list of
    # pipeline_depth tiles laid out consecutively from the current allocator address.
    loaded_tiles = allocator.alloc_sbuf_tensor(shape=(_P, group_cols), dtype=dst.dtype, block_dim=[pipeline_depth])
    transposed_out = allocator.alloc_sbuf_tensor(
        shape=(head_dim, group_cols), dtype=dst.dtype, block_dim=[pipeline_depth]
    )
    # subtiles_per_group*pipeline_depth distinct PSUM banks (<=8, either 2*4 or 1*8): buffer slot
    # `buf`'s sub-tile `subtile_idx` at bank buf*subtiles_per_group+subtile_idx, so a group's
    # transposes run on independent banks. PSUM is addressed a whole bank at a time, so the banks are
    # built here rather than drawn from `allocator` (which packs SBUF tiles end to end).
    psum_banks = []
    for buf in range(pipeline_depth):
        group_banks = []
        for subtile_idx in range(subtiles_per_group):
            group_banks.append(
                nl.ndarray(
                    (head_dim, _P),
                    dtype=dst.dtype,
                    buffer=nl.psum,
                    address=(0, (buf * subtiles_per_group + subtile_idx) * _PSUM_BANK_SIZE),
                )
            )
        psum_banks.append(group_banks)

    src_row_stride = src_heads * head_dim  # one whole token row of the source

    # Each NC owns a contiguous block of heads (odd-batch_heads remainder head by shard 0).
    for local_head_idx in range(num_heads_per_shard):
        idx = local_head_idx + head_offset
        # K mode writes this head's scalar max into k_maxsq_out; Q mode fills its resident per-token
        # tile. Only one of the two is passed, so the other stays None.
        k_maxsq_slice = k_maxsq_out[idx : idx + 1] if k_maxsq_out is not None else None
        q_sumsq_tile = q_sumsq_sb[local_head_idx] if q_sumsq_sb is not None else None
        _transpose_one_batch_pipelined(
            dst.select(dim=0, index=idx),
            src.select(dim=0, index=idx // src_heads),
            head_dim,
            seqlen,
            loaded_tiles,
            transposed_out,
            psum_banks,
            subtiles_per_group,
            k_maxsq_out_batch=k_maxsq_slice,
            squared_buf=squared_buf,
            running_max_sumsq=running_max_sumsq,
            per_token_sumsq=per_token_sumsq,
            q_sumsq_sb=q_sumsq_tile,
            dst_stride=dst_stride,
            src_row_stride=src_row_stride,
            src_col_offset=(idx % src_heads) * head_dim,
        )
    if has_remainder and shard_id == 0:
        head_idx = batch_heads - 1
        k_maxsq_slice = k_maxsq_out[head_idx : head_idx + 1] if k_maxsq_out is not None else None
        q_sumsq_tile = q_sumsq_sb[num_heads_per_shard] if q_sumsq_sb is not None else None
        _transpose_one_batch_pipelined(
            dst.select(dim=0, index=head_idx),
            src.select(dim=0, index=head_idx // src_heads),
            head_dim,
            seqlen,
            loaded_tiles,
            transposed_out,
            psum_banks,
            subtiles_per_group,
            k_maxsq_out_batch=k_maxsq_slice,
            squared_buf=squared_buf,
            running_max_sumsq=running_max_sumsq,
            per_token_sumsq=per_token_sumsq,
            q_sumsq_sb=q_sumsq_tile,
            dst_stride=dst_stride,
            src_row_stride=src_row_stride,
            src_col_offset=(head_idx % src_heads) * head_dim,
        )

    # Free all transpose scratch — dead once the transposed HBM + per-token norms are written.
    allocator.set_current_address(scratch_addr)


def _assemble_c_row(
    q_sumsq_sb,
    k_maxsq,
    c_row_hbm,
    softmax_scale,
    seqlen,
    runs,
    allocator,
):
    """Build the per-ROW softmax-max bound c_i and store it to c_row_hbm in natural token order.

    THE MATH. For each head and each query row i:

        c_i = softmax_scale · ‖q_i‖ · max_j‖k_j‖
            = sqrt( softmax_scale² · k_maxsq · ‖q_i‖² )

    A Cauchy–Schwarz upper bound on the scaled score of every (query i, key j) pair, since
    q_i·k_j ≤ ‖q_i‖·‖k_j‖ — so the exponent stays ≤ 0 and the exp never overflows. One value per
    row is the tightest granularity available from measured norms, so every row's exp stays
    well-scaled: no over-subtraction of the quiet rows a loud row would otherwise inflate.

    INPUTS (both from the prologue transpose, no extra HBM read):
        q_sumsq_sb[t] = ‖q_i‖² tile [128, num_token_groups] for owned-head iteration t (tile[r, g] =
            ‖q_{g*128+r}‖²), resident SBUF; k_maxsq[head] = global max_j‖k_j‖² (scalar per head).

    LAYOUT FLIP (partition→free). c_tile is partition-major [128, num_token_groups] (tile[r, g] = c_i
    for i = g*128 + r), while the base kernel needs c_i on the score tile's FREE axis in natural token
    order. The flip happens on chip: a PE transpose turns c_tile into [num_token_groups, 128], where
    partition g holds group g's 128 CONSECUTIVE tokens, so each partition stores as one contiguous run
    of c_row_hbm[head]. The compute kernel then reads a contiguous [1, Sq] row back per head and
    replicates it across partitions.

    c_i is STATIC across the ring (query rows don't rotate), so it is assembled once here. Scratch is
    drawn from the shared allocator, reused across owned heads (`runs`), freed on exit.

    STORED IN bf16. Softmax is invariant to the choice of c, so rounding c_i shifts every prob in row i
    by one common factor that cancels in o = ΣPV/ΣP; the LSE reads back this same rounded c, so it is
    consistent there too. Rounding can only lower c by ~0.4%, which cannot cause exp overflow. bf16
    halves the per-Q-block broadcast DMA and the subtract's SBUF read in the hot loop.
    """
    num_token_groups = div_ceil(seqlen, _P)

    scratch_addr = allocator.get_current_address()
    # k_scale_sq = softmax_scale²·k_maxsq[head], broadcast to all 128 partitions so the activation
    # below scales every token's ‖q‖² by it. c_tile holds this head's per-token c_i [128, num_grps].
    k_scale_sq = allocator.alloc_sbuf_tensor(shape=(_P, 1), dtype=nl.float32)
    c_tile = allocator.alloc_sbuf_tensor(shape=(_P, num_token_groups), dtype=c_row_hbm.dtype)
    # Token-order view of c_tile (see LAYOUT FLIP): c_token[g, r] = c_{g*128+r}. PSUM bank 0 is free
    # here — the transpose prologue's banks are all drained before this runs.
    c_token = allocator.alloc_sbuf_tensor(shape=(_P, _P), dtype=c_row_hbm.dtype)
    c_psum = nl.ndarray((_P, _P), dtype=c_row_hbm.dtype, buffer=nl.psum, address=(0, 0))
    tile_iter = 0  # owned-head iteration; matches the prologue's q_sumsq_sb tile order
    for run in runs:
        for head_idx in range(run.start, run.end):
            q_sumsq_head = q_sumsq_sb[tile_iter]
            tile_iter += 1

            nisa.dma_copy(dst=k_scale_sq[0:1, :], src=k_maxsq[head_idx : head_idx + 1, :])
            nisa.tensor_scalar(
                dst=k_scale_sq[0:1, :],
                data=k_scale_sq[0:1, :],
                op0=nl.multiply,
                operand0=float(softmax_scale) ** 2,
            )
            stream_shuffle_broadcast(src=k_scale_sq, dst=k_scale_sq)

            # c_i = sqrt(k_scale_sq · ‖q_i‖²) over all [128, num_grps] in one Scalar-engine op
            # (activation computes op(scale·data), scale = the per-partition-broadcast k_scale_sq).
            nisa.activation(dst=c_tile, op=nl.sqrt, data=q_sumsq_head, scale=k_scale_sq)

            # Transpose to token order, then store each group's 128 tokens as one contiguous run.
            # Writing c_tile out directly with a partition-stride-1 AP instead makes every element its
            # own 2-byte DMA descriptor (num_token_groups*128 per head), which measured ~7us of prologue
            # DMA-queue time per head — ahead of the first Q-block load that waits on it.
            for group_base in range(0, num_token_groups, _P):
                groups = min(_P, num_token_groups - group_base)
                nisa.nc_transpose(c_psum[:groups, :_P], c_tile[:_P, group_base : group_base + groups])
                nisa.tensor_copy(dst=c_token[:groups, :_P], src=c_psum[:groups, :_P], engine=nisa.engine.scalar)
                nisa.dma_copy(
                    dst=c_row_hbm[head_idx].ap(pattern=[[_P, groups], [1, _P]], offset=group_base * _P),
                    src=c_token[:groups, :_P],
                )

    allocator.set_current_address(scratch_addr)


# Token-groups the normalize writes before it stores them, and how many such chunks are in flight.
# The output store is COALESCED over a chunk (one strided DMA for _NORM_STORE_GROUPS whole token-groups)
# because a store per token-group puts that many doorbells and completion waits on the in-order Sync
# queue, right in front of the NEXT head's K/V stream loads — which are also Sync, and which gate that
# head's first MM1, so the Tensor engine waits out the whole run. _NORM_CHUNKS_IN_FLIGHT rotating chunk
# buffers keep the next chunk's Vector writes from waiting on the previous chunk's store, the same WAR
# that serialized this pass when each group had a single shared tile.
# Costs _NORM_CHUNKS_IN_FLIGHT * _NORM_STORE_GROUPS * head_dim * out_dtype bytes per partition
# (4 KB at head_dim 128, bf16).
_NORM_STORE_GROUPS = 8
_NORM_CHUNKS_IN_FLIGHT = 2

# Floor applied to sum_exp before the reciprocal and the LSE log, guarding a row whose probabilities
# all underflowed (0/0). It also sets how far a softmax bound may overshoot a row's true max before it
# does damage: at more than -ln(_SUM_CLAMP) = 85 nats of overshoot the row's whole sum floors here and
# the row is silently rescaled. So the floor sits as low as fp32 allows — the only constraint is that
# the reciprocal below stays finite (1/1e-37 = 1e37 against fp32's 3.4e38). The next floor past this
# one is bf16 probability underflow at e^-87.3, which no clamp can move.
_SUM_CLAMP = 1e-37


def _normalize_one_batch(
    o_aug_acc_sb,
    aug_col_base,
    o_out_batch,
    lse_out_batch,
    head_dim,
    num_token_groups,
    seqlen,
    training,
    lse_dtype,
    allocator,
    c_row_b,
):
    """Normalize one head from the resident SBUF accumulator: o = o_acc / sum_acc.

    o_aug_acc_sb is the resident SBUF accumulator (128 partitions). This head's token-group g lives at
    o_aug_acc_sb[:, aug_col_base + g*(head_dim+1) : +(head_dim+1)] — head_dim output cols then 1 sum col.
    Optionally lse = c_i + log(sum_acc), where c_i is the per-row bound the exp subtracted.
    Token-group g maps to column g of the (128, num_token_groups) LSE tile; query pos i = g*128 + r
    maps to tile[r, g].
    """
    # Scratch drawn once from the shared allocator, freed on exit. This pass runs interleaved with
    # the last ring step's base-kernel compute; those (compiler-managed) buffers are placed around
    # this manual region by the compiler.
    #
    # Every tile a DMA reads is rotated, never shared: with one shared tile, group g+1's Vector write
    # waits for group g's store DMA to drain, which serializes the whole per-head pass AND parks the
    # in-order Vector queue on a DMA semaphore — blocking the next head's scores-c subtracts behind it
    # and idling the Tensor engine for the pass. sum_clamped rotates for the same reason on the LSE
    # path, where the log reads it from Scalar.
    scratch_addr = allocator.get_current_address()
    num_chunks = div_ceil(num_token_groups, _NORM_STORE_GROUPS)
    chunks_in_flight = min(num_chunks, _NORM_CHUNKS_IN_FLIGHT)
    rot = min(num_token_groups, _NORM_STORE_GROUPS * chunks_in_flight)
    sum_clamped = allocator.alloc_sbuf_tensor(
        shape=(_P, 1), dtype=nl.float32, block_dim=[num_token_groups], num_free_tiles=[rot]
    )
    inv_sum = allocator.alloc_sbuf_tensor(
        shape=(_P, 1), dtype=nl.float32, block_dim=[num_token_groups], num_free_tiles=[rot]
    )
    # One chunk buffer holds _NORM_STORE_GROUPS groups side by side along the free axis, so the whole
    # chunk stores in one strided DMA.
    o_out_sb = allocator.alloc_sbuf_tensor(
        shape=(_P, _NORM_STORE_GROUPS * head_dim),
        dtype=o_out_batch.dtype,
        block_dim=[num_chunks],
        num_free_tiles=[chunks_in_flight],
    )
    # LSE is already one contiguous (128, num_token_groups) tile in HBM, so the whole head's columns
    # are filled here and stored with a single DMA after the loop.
    lse_all = allocator.alloc_sbuf_tensor(shape=(_P, num_token_groups), dtype=lse_dtype) if training else None

    # Reload this head's per-token c_i [128, num_token_groups] from c_row_hbm via the same token-order
    # strided AP _assemble_c_row stored it with (tile[r, g] = c_{g*128+r}); each group's LSE column
    # adds its own c column (c_i varies per query row).
    c_row_tile = None
    if training:
        c_row_tile = allocator.alloc_sbuf_tensor(shape=(_P, num_token_groups), dtype=c_row_b.dtype)
        nisa.dma_copy(dst=c_row_tile, src=c_row_b.ap(pattern=[[1, _P], [_P, num_token_groups]], offset=0))

    for chunk_idx in range(num_chunks):
        first_group = chunk_idx * _NORM_STORE_GROUPS
        groups_in_chunk = min(_NORM_STORE_GROUPS, num_token_groups - first_group)
        o_chunk = o_out_sb[chunk_idx]
        # Groups in this chunk whose 128 rows are all real. Only a trailing partial group (seqlen not a
        # 128-multiple) is excluded, and it is always the last group of the last chunk.
        whole_groups = min(groups_in_chunk, max(0, seqlen // _P - first_group))

        for local_idx in range(groups_in_chunk):
            group_idx = first_group + local_idx
            token_start = group_idx * _P
            rows = min(_P, seqlen - token_start)  # final query group is partial when seqlen % 128 != 0
            base = aug_col_base + group_idx * (head_dim + 1)
            o_view = o_aug_acc_sb[:rows, base : base + head_dim]
            sum_view = o_aug_acc_sb[:rows, base + head_dim : base + head_dim + 1]
            sum_buf = sum_clamped[group_idx]
            inv_buf = inv_sum[group_idx]
            o_slot = o_chunk[:, local_idx * head_dim : (local_idx + 1) * head_dim]

            # Clamp once against genuine all-underflow, at the fp32 floor rather than anywhere near the
            # working range: a loud head's legitimate peaked-row sum can sit around e^-25 (~1e-11), which
            # a 1e-9 floor would overwrite and rescale the row by ~100x. sum_clamped feeds both the
            # reciprocal (output scale) and the log (LSE), so the clamp runs once. See _SUM_CLAMP.
            nisa.tensor_scalar(dst=sum_buf[:rows, :], data=sum_view, op0=nl.maximum, operand0=_SUM_CLAMP)
            nisa.reciprocal(dst=inv_buf[:rows, :], data=sum_buf[:rows, :])

            # Scale straight from the SBUF accumulator into this group's slot of the chunk buffer.
            nisa.tensor_scalar(dst=o_slot[:rows, :], data=o_view, op0=nl.multiply, operand0=inv_buf[:rows, :])

            if training:
                lse_col = lse_all[:, group_idx : group_idx + 1]
                if rows < _P:
                    # Partial final group: zero the whole LSE column first so its padding rows (no real
                    # query position) are 0, matching the zero-padded reference (the DMA stores all 128
                    # partitions). A memset of only [rows:_P] would start at a runtime partition offset
                    # (rows), which the hardware rejects; the [:rows] writes below fill the real rows.
                    nisa.memset(lse_col, 0.0)
                # lse = c_i + log(sum_acc), true by shift-invariance. Column g of the
                # (128, num_token_groups) LSE tile. Reuse the already-clamped sum
                # (log(_SUM_CLAMP) ≈ -85 for an all-underflow row, not log(0) = -inf).
                nisa.activation(dst=lse_col[:rows, :], op=nl.log, data=sum_buf[:rows, :])
                nisa.tensor_tensor(
                    dst=lse_col[:rows, :],
                    data1=lse_col[:rows, :],
                    data2=c_row_tile[:rows, group_idx : group_idx + 1],
                    op=nl.add,
                )

        # One strided DMA for the chunk's whole groups: element (r, g, c) of the chunk buffer lands at
        # o_out_batch[(first_group + g)*128 + r, c], so partitions stride by head_dim and each group
        # strides by a whole 128-token block.
        if whole_groups > 0:
            nisa.dma_copy(
                dst=o_out_batch.ap(
                    pattern=[[head_dim, _P], [_P * head_dim, whole_groups], [1, head_dim]],
                    offset=first_group * _P * head_dim,
                ),
                src=o_chunk[:_P, : whole_groups * head_dim],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        # A trailing partial group cannot join that pattern (fewer real rows), so it stores on its own.
        if whole_groups < groups_in_chunk:
            tail_group = first_group + whole_groups
            tail_start = tail_group * _P
            tail_rows = seqlen - tail_start
            nisa.dma_copy(
                dst=o_out_batch[tail_start : tail_start + tail_rows, :],
                src=o_chunk[:tail_rows, whole_groups * head_dim : (whole_groups + 1) * head_dim],
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )

    if training:
        # Whole head's LSE in one DMA: lse_out_batch is the same (128, num_token_groups) layout.
        nisa.dma_copy(dst=lse_out_batch, src=lse_all, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    # Free this head's normalize scratch before the next head's fold+normalize.
    allocator.set_current_address(scratch_addr)


def _partial_attention_into_acc(
    q_transposed,
    k_buf,
    v_buf,
    head_start,
    head_count,
    acc_slot,
    o_aug_acc_sb,
    num_token_groups,
    head_dim,
    softmax_scale,
    is_accumulate_step,
    c_row_hbm,
    kv_pad_rows=0,
):
    """Fold one partial attention over heads [head_start, head_start+head_count) into the resident
    SBUF accumulator, starting at compacted slot `acc_slot` (init when is_accumulate_step=False,
    += when True).

    Single source of truth for the accumulator column math shared by the partial and last ring
    steps. The partial step folds a whole owned run in one call (head_count == run length); the last
    step calls it per head (head_count == 1) so it can interleave the normalize+write between heads.
    """
    head_slice = slice(head_start, head_start + head_count)
    _attention_const_max_partial(
        q_transposed[head_slice],
        k_buf[head_slice],
        v_buf[head_slice],
        o_aug_acc_sb,
        c_row_hbm[head_slice],
        softmax_scale=softmax_scale,
        aug_col_base=acc_slot * num_token_groups * (head_dim + 1),
        is_accumulate_step=is_accumulate_step,
        kv_pad_rows=kv_pad_rows,
    )


def _run_last_step_and_normalize(
    q_transposed,
    k_buf,
    v_buf,
    o_aug_acc_sb,
    o_out,
    lse_out,
    softmax_scale,
    num_token_groups,
    seqlen,
    head_dim,
    training,
    lse_dtype,
    is_accumulate_step,
    runs,
    allocator,
    c_row_hbm,
    kv_pad_rows=0,
):
    """Last ring step: fold each head's attention into the accumulator, then immediately
    normalize + write that head — interleaved per head.

    A separate whole-tensor normalize pass runs fully exposed after all compute (the DMA
    tail). By interleaving per head, head h's normalize+output-write (Vector + DMA, reading
    a scratch tile) overlaps head h+1's attention matmul (PE), so only the final head's
    write stays exposed. is_accumulate_step is False only when this is also ring step 0
    (num_workers==1); otherwise True.

    The accumulator is COMPACTED: acc_slot packs only the heads THIS core owns in the chunk, from
    column 0 in `runs` order (0, 1, ...). Input/output HBM tensors stay indexed by the global head
    head_idx (they span all heads). On LNC1 acc_slot equals the global index; on LNC>1 it drops the
    other cores' heads so the accumulator is sized per-core (acc_heads), not per-chunk (chunk_width).

    `allocator` is the driver's shared SBUF allocator, threaded into _normalize_one_batch so its
    per-head scratch is freed between heads.
    """
    acc_slot = 0  # compacted accumulator slot: this core's owned heads, packed from 0
    for run in runs:
        for head_idx in range(run.start, run.end):
            _partial_attention_into_acc(
                q_transposed,
                k_buf,
                v_buf,
                head_idx,
                1,  # last step folds one head at a time so its normalize+write interleaves
                acc_slot,
                o_aug_acc_sb,
                num_token_groups,
                head_dim,
                softmax_scale,
                is_accumulate_step,
                c_row_hbm,
                kv_pad_rows,
            )
            _normalize_one_batch(
                o_aug_acc_sb,
                acc_slot * num_token_groups * (head_dim + 1),  # merged accumulator column base (chunk-local)
                o_out[head_idx],
                lse_out[head_idx] if training else None,
                head_dim,
                num_token_groups,
                seqlen,
                training,
                lse_dtype,
                allocator,
                c_row_hbm[head_idx],
            )
            acc_slot += 1


def _run_partial_step(
    q_transposed,
    k_buf,
    v_buf,
    o_aug_acc_sb,
    softmax_scale,
    num_token_groups,
    head_dim,
    is_accumulate_step,
    runs,
    c_row_hbm,
    kv_pad_rows=0,
):
    """Per-step attention folding its output straight into the resident SBUF accumulator.

    is_accumulate_step=False initializes the accumulator (ring step 0); True folds (+=) later
    steps. The accumulator stays SBUF-resident across all steps — no per-step HBM round-trip
    and no separate cross-step reduction pass.

    The accumulator is COMPACTED per core: acc_slot packs this core's owned heads from column 0 in
    `runs` order (a run of `count` heads occupies acc_slot .. acc_slot+count). Q/K/V HBM inputs
    stay indexed by the global head range. On LNC1 acc_slot equals the global index.
    """
    acc_slot = 0  # compacted accumulator slot: this core's owned heads, packed from 0
    for run in runs:
        # Fold the whole owned run in one call (the base kernel loops its `count` heads internally).
        _partial_attention_into_acc(
            q_transposed,
            k_buf,
            v_buf,
            run.start,
            run.count,
            acc_slot,
            o_aug_acc_sb,
            num_token_groups,
            head_dim,
            softmax_scale,
            is_accumulate_step,
            c_row_hbm,
            kv_pad_rows,
        )
        acc_slot += run.count


@nki.jit
def ring_attention_const_max_fwd(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    replica_groups: tuple = None,
    num_workers: int = 1,
    softmax_scale: float = None,
    training: bool = False,
    lse_dtype: nki.dtype = nl.float32,
    kv_prefetch_depth: int = 2,
):
    """Fixed-max ring attention forward (non-causal) built on `attention_const_max`.

    The per-step attention is K-stationary (no P-transpose), so the KV-rotation
    collective-permute overlaps the compute. The softmax max is the per-ROW runtime bound
    c_i = softmax_scale · ‖q_i‖ · max_j‖k_j‖ (Cauchy–Schwarz L2 bound), one value per query row,
    computed in the prologue and static across the ring. The ring reduction is pure addition.

    Dimensions:
        b: batch, h: heads (MHA; q_h == k_h), d: head dim (must be 128),
        seqlen: per-rank sequence length (arbitrary; a non-128-multiple final query/KV group is
            peeled into a sub-128 tail in the transpose prologue and the base kernel, and the
            per-token norm + LSE handle its partial final group).

    Args:
        q (nl.NkiTensor): Query, token-major [b, seqlen, h, d]. Transposed to d-major internally.
        k (nl.NkiTensor): Key, token-major [b, seqlen, h, d]. Transposed to d-major internally into
            the ring send buffer, so the collective rotates the already-transposed K.
        v (nl.NkiTensor): Value, token-major [b, seqlen, h, d].
        replica_groups (tuple): Replica groups for the CP collective.
        num_workers (int): Ring size (context-parallel degree).
        softmax_scale (float): Softmax scale. Default 1/sqrt(d).
        training (bool): Emit LSE = c_i + log(sum_exp).
        lse_dtype (nki.dtype): LSE output dtype. Default fp32.
        kv_prefetch_depth (int): Ring steps ahead to launch the K/V permutes. Clamped to
            [1, num_workers-1].

    Numerics of the bound. Output is mathematically invariant to c — a uniform over-estimate rescales
    every probability in a row by one common factor that cancels in o = SUM(PV)/SUM(P) — so slack in
    the bound is free in relative terms. It stops being free at two hard floors: sum_exp reaching the
    normalize's clamp (see _SUM_CLAMP, slack ~85 nats) and the probabilities underflowing bf16
    (~87), either of which silently zeroes a query row. Taking c per ROW rather than over any wider
    span is what keeps the slack small: the Cauchy-Schwarz bound runs ~11.3*alpha*beta against a true
    row max of ~3.9*alpha*beta, and that gap grows with the per-token norm spread QK-norm over the
    full model hidden dim produces — so a single loud token must not be allowed to set the bound for
    its quieter neighbours. c is held in bf16 and the `shifted = scores - c` scratch in fp16, which
    keeps ~2 GB of SBUF traffic per WAN-480p call off every engine; both are numerically equivalent
    to fp32 here, since rounding c scales a row's probabilities by one common factor that cancels
    (and the LSE reads back the same rounded c), while fp16 keeps the exp's absolute exponent error
    below what the bf16 PSUM scores already carry.

    Returns:
        o (nl.NkiTensor): [b, h, seqlen, d], attention output.
        lse (nl.NkiTensor): [b, h, 128, ceil(seqlen/128)], log-sum-exp (if training).

    Notes:
        - Non-causal only. Requires Trainium2 or later.
        - MHA only (q_h == k_h; broadcast before calling).
        - Supports LNC1 and LNC2 (sharded on batch*heads).
    """
    kernel_assert(nisa.get_nc_version() > nisa.nc_version.gen2, "ring_attention_const_max_fwd not supported on trn1")

    # q/k/v are token-major [b, seqlen, h, d] — the qkv-projection + RoPE producer layout, with every
    # head of a token adjacent. Q/K are transposed to d-major internally.
    batch, seqlen, q_heads, head_dim = q.shape
    _, k_seqlen, k_heads, k_head_dim = k.shape

    kernel_assert(q_heads == k_heads, "expects q_heads == kv_heads, broadcast before calling")
    kernel_assert(head_dim == _P, f"head_dim must be 128, got {head_dim}")
    kernel_assert(
        k_seqlen == seqlen and k_head_dim == head_dim,
        f"q and k must be token-major [b, seqlen, h, d]; got q={q.shape}, k={k.shape}",
    )

    if replica_groups is None:
        replica_groups = ()

    # Flatten (batch, heads) -> batch_heads. Each index is one independent attention head instance.
    batch_heads = batch * q_heads
    # 128-token query groups. MUST match the base kernel's groups_per_head = div_ceil(Sq, 128): the
    # driver passes num_token_groups as the per-head accumulator stride, and the base kernel derives
    # its column offsets from groups_per_head — a mismatch would corrupt the accumulator.
    num_token_groups = div_ceil(seqlen, _P)
    # The ring's OWN K/V buffers are padded up to a whole number of 128-token groups so the base
    # kernel's KV loop is uniform (all full 128-row tiles) even at a non-128-multiple seqlen — no
    # peeled sub-128 tail tile, whose load sits outside the stream's double-buffered prefetch. The
    # pad tokens are zeroed, and the base kernel zeroes their [V|1] ones column, so they contribute
    # exactly nothing to either Σ P·V or Σ P·1. Costs kv_pad_rows extra tokens of collective traffic
    # (<128 out of seqlen) and nothing at all when seqlen is already a 128-multiple.
    seq_padded = num_token_groups * _P
    kv_pad_rows = seq_padded - seqlen
    softmax_scale = softmax_scale or (1.0 / float(head_dim**0.5))

    # Collapse (heads, d) into one token row: [batch, seqlen, heads*head_dim]. A head is then a
    # head_dim-wide column range repeating every token row, which the staging DMAs select with a
    # token-strided access pattern — so the framework never has to materialize a [b, h, seqlen, d]
    # permutation of q/k/v just to feed this kernel. Q/K are staged once up front so the per-step
    # attention and the collective always see the (batch_heads, head_dim, seqlen) layout.
    q = q.reshape((batch, seqlen, q_heads * head_dim))
    k = k.reshape((batch, seqlen, k_heads * head_dim))
    v = v.reshape((batch, seqlen, q_heads * head_dim))

    # Final outputs
    o = nl.ndarray((batch_heads, seqlen, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)
    lse = None
    if training:
        lse = nl.ndarray((batch_heads, _P, num_token_groups), dtype=lse_dtype, buffer=nl.shared_hbm)

    # LNC head sharding
    program_ndims = nl.program_ndim()
    shard_id = nl.program_id(0) if program_ndims == 1 else 0
    num_shards = nl.num_programs(0) if program_ndims == 1 else 1
    num_heads_per_shard = batch_heads // num_shards
    head_offset = shard_id * num_heads_per_shard
    has_remainder = (batch_heads % num_shards) != 0
    # Contiguous head runs this LNC shard owns
    runs = _shard_batch_runs(batch_heads, num_heads_per_shard, head_offset, has_remainder, shard_id)

    # Collective-permute KV buffer ring
    prefetch_depth = min(max(kv_prefetch_depth, 1), max(num_workers - 1, 1))
    num_kv_buf = prefetch_depth + 1
    k_bufs = []
    v_bufs = []
    for buf_idx in range(num_kv_buf):
        k_bufs.append(
            nl.ndarray(
                (batch_heads, head_dim, seq_padded), dtype=k.dtype, buffer=nl.shared_hbm, name=f"cm_k_buf_{buf_idx}"
            )
        )
        v_bufs.append(
            nl.ndarray(
                (batch_heads, seq_padded, head_dim), dtype=v.dtype, buffer=nl.shared_hbm, name=f"cm_v_buf_{buf_idx}"
            )
        )

    replica_group = ReplicaGroup(replica_groups)

    # Resident SBUF accumulator (unnormalized output + raw sum_exp, merged). Every ring step's
    # partial attention folds its [Σ P·V | Σ P·1] straight from PSUM into this tile (init on step 0,
    # += after), so there is NO per-step HBM round-trip and NO separate cross-step reduction pass.
    # Layout mirrors the PSUM source: 128 partitions (query rows within a group) ×
    # (batch_heads·num_token_groups) groups, each group carrying (head_dim+1) contiguous columns —
    # head_dim output columns followed by 1 sum column — so the whole fold is a single PSUM→SBUF op.
    # fp32, not q.dtype: it is the cross-ring-step running sum of the unnormalized Σ P·V (and Σ P·1)
    # partials; a bf16 accumulator would re-round that running sum every step.
    #
    # Head chunking: the accumulator holds only the heads THIS NC core owns, sized to fit the SBUF
    # budget (batch_heads·num_token_groups·(head_dim+1)·4 bytes/partition, OOMs at long seqlen), so the
    # driver runs the full ring over one chunk of heads at a time. Two DISTINCT quantities drive it:
    #   * chunk_width — the GLOBAL head-chunk width. Drives _head_chunks and the per-chunk KV
    #     collective, which every LNC core must co-trace over the identical [chunk_start:chunk_end]
    #     slice (a per-rank op).
    #   * acc_heads — the PER-CORE accumulator width. On LNC2 it is ~half of a chunk's global heads,
    #     so sizing by chunk_width would over-allocate and cause OOM.
    # _heads_per_chunk returns how many OWNED accumulators fit the budget, so compare it against the
    # per-core OWNED head count (max_owned).
    heads_per_chunk = _heads_per_chunk(num_token_groups, head_dim)  # SBUF budget cap
    max_owned = div_ceil(batch_heads, num_shards)  # max heads any single core owns
    # Resident accumulator: bounded by BOTH the SBUF budget and this core's owned heads.
    acc_heads = min(heads_per_chunk, max_owned)
    # chunk_width is the GLOBAL chunk stride (drives the co-traced KV collective); acc_heads is
    # per-core. One chunk over all heads when every core's owned accumulators fit the budget;
    # otherwise chunk at the budget. They differ only on LNC>1 (on LNC1 max_owned == batch_heads).
    chunk_width = batch_heads if max_owned <= heads_per_chunk else heads_per_chunk

    # Self-maintained SBUF: the driver owns its allocations explicitly on one bump allocator —
    # the persistent buffer (o_aug_acc) is allocated first and lives the whole ring; every transient
    # (transpose scratch, c-row assembly, per-head normalize) is bump-allocated ABOVE it and freed on
    # exit by its helper (see the set_current_address calls).
    allocator = ModularAllocator(initial_address=0)
    o_aug_acc_sb = allocator.alloc_sbuf_tensor(
        shape=(_P, acc_heads * num_token_groups * (head_dim + 1)), dtype=nl.float32
    )

    # L2 bound tensors:
    #   q_sumsq_sb — the FULL per-token ‖q_i‖², resident in SBUF as one [128, num_token_groups] tile per
    #     owned head (not reduced). Written by the Q transpose prologue, read directly by _assemble_c_row.
    #   local_k_maxsq / global_k_maxsq — per-head max-over-tokens ‖k‖² (one scalar per head); the global
    #     one exists only when there is a ring to reduce it over.
    #   c_row — assembled c_i = scale·‖q_i‖·max_j‖k_j‖, one value per query row.
    #
    # c_i is a full [seqlen] vector per head — too large for a resident 128-partition broadcast tile, so
    # keep it in HBM (token order) and reload a contiguous per-Q-block slice onto the free axis in the
    # base kernel. Assembled once, static across the ring (query rows don't rotate). The row extent is
    # padded to a whole number of 128-token groups: _assemble_c_row stores (and the LSE reload reads) the
    # full [128, num_token_groups] tile with one strided AP, which spans num_token_groups*128 >= seqlen
    # elements. Only [0, seqlen) is ever read as a bound.
    local_k_maxsq = nl.ndarray((batch_heads, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="cm_local_k_maxsq")
    # The global key max needs a collective; with no ring this rank's own max already is the global one.
    reduce_k_max = num_workers > 1
    if reduce_k_max:
        global_k_maxsq = nl.ndarray((batch_heads, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="cm_global_k_maxsq")
    c_row = nl.ndarray(
        (batch_heads, num_token_groups * _P),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
        name="cm_c_row",
    )
    # Resident per-token ‖q‖² tiles, one per owned head. The Q transpose prologue writes them and
    # _assemble_c_row reads them directly — q is local (never rotated), so no HBM round-trip, which
    # removes the reload's anti-dependency on the prologue and lets c assembly overlap the
    # transpose. Freed right after c assembly (before the ring) so the ring scratch reuses it.
    num_head_iters = num_heads_per_shard + (1 if (has_remainder and shard_id == 0) else 0)
    q_sumsq_ckpt = allocator.get_current_address()
    q_sumsq_sb = allocator.alloc_sbuf_tensor(
        shape=(_P, num_token_groups),
        dtype=nl.float32,
        block_dim=[num_head_iters],
        num_free_tiles=[num_head_iters],
    )

    # ── Prologue: transpose Q/K once, stage local K/V into the ring's first buffer ──
    # The transpose is pipelined (depth _TP_DEPTH) so its DMA loads/stores overlap the PE/Vector
    # work. The per-token ‖·‖² is computed INSIDE the transpose off the tiles it already loads: Q writes
    # the full per-token vector into the resident q_sumsq_sb tiles, K max-folds to a scalar
    # (k_maxsq_out=...). The square/reduce run on the otherwise-idle Vector engine while the transpose
    # runs on PE + Scalar. Q's norms stay local; K's max is all_reduced below.
    #
    # ORDER: K is transposed FIRST so its per-head max‖k‖² (local_k_maxsq) is ready as early as
    # possible, and the global-K-max all_reduce (when there is one) is triggered BEFORE the Q transpose
    # and V copy. Lets its latency overlap the Q transpose + V copy.
    _transpose_qk_pipelined(
        k_bufs[0],
        k,
        batch_heads,
        head_dim,
        seqlen,
        num_heads_per_shard,
        head_offset,
        has_remainder,
        shard_id,
        allocator,
        k_heads,
        k_maxsq_out=local_k_maxsq,
        dst_stride=seq_padded,
    )

    # Trigger the global-K-max all_reduce as early as possible (right after K's max is ready) so its
    # launch latency overlaps the Q transpose + V copy below. Q norms stay local (no reduce).
    if reduce_k_max:
        if num_shards > 1:
            # The reduce reads every head's row, so all cores must have written theirs first.
            nisa.core_barrier(local_k_maxsq, (0, 1))
        ncc.all_reduce(dsts=[global_k_maxsq], srcs=[local_k_maxsq], op=nl.maximum, replica_group=replica_group)
        k_maxsq = global_k_maxsq
    else:
        k_maxsq = local_k_maxsq

    # Q transpose + V copy overlap the all_reduce's launch latency.
    q_transposed = nl.ndarray((batch_heads, head_dim, seqlen), dtype=q.dtype, buffer=nl.shared_hbm, name="cm_q_t")
    _transpose_qk_pipelined(
        q_transposed,
        q,
        batch_heads,
        head_dim,
        seqlen,
        num_heads_per_shard,
        head_offset,
        has_remainder,
        shard_id,
        allocator,
        q_heads,
        q_sumsq_sb=q_sumsq_sb,
    )
    _copy_batch_sharded(
        v_bufs[0][:, :seqlen, :],
        v,
        batch_heads,
        num_heads_per_shard,
        head_offset,
        has_remainder,
        shard_id,
        q_heads,
        head_dim,
        seqlen,
    )
    # Zero the K/V pad tokens once. They then ride the ring: each collective-permute moves the full
    # padded extent, so every later buffer inherits the zeros. HBM is uninitialized, and garbage here
    # would reach the exp as an unbounded score (Inf/NaN), which no masking downstream can repair.
    if kv_pad_rows > 0:
        _zero_kv_pad(k_bufs[0], v_bufs[0], seqlen, seq_padded, head_dim, runs, allocator)

    # Barrier before the ring's cross-core reads: the hoisted KV permute reads the full k_bufs[0]/
    # v_bufs[0], and each core reads q_transposed. local_k_maxsq needs no barrier here: a core reads
    # only the rows of the heads it owns and wrote itself (the reduce, which reads all of them, is
    # barriered above).
    if num_shards > 1:
        nisa.core_barrier(q_transposed, (0, 1))
        nisa.core_barrier(k_bufs[0], (0, 1))
        nisa.core_barrier(v_bufs[0], (0, 1))

    # ── Assemble the bound c once (needs the global K max + local q_sumsq, both ready now). ──
    _assemble_c_row(
        q_sumsq_sb,
        k_maxsq,
        c_row,
        softmax_scale,
        seqlen,
        runs,
        allocator,
    )
    allocator.set_current_address(q_sumsq_ckpt)  # free the resident q_sumsq tiles before the ring

    # ── Head-chunk loop: run the FULL ring for one chunk of heads at a time ──
    # The chunk-sized accumulator (o_aug_acc_sb) is reused across chunks, bounding SBUF.
    # Chunk boundaries [chunk_start, chunk_end) are identical on every LNC core, so both cores co-trace
    # each chunk's KV collective over the same k_bufs[i][chunk_start:chunk_end] slice.
    # Each core computes only its owned heads in the chunk.
    #
    # NOTE (perf, multi-chunk LNC>1 only): chunks are GLOBAL head ranges but core ownership is
    # CONTIGUOUS (core 0 = first half of heads, core 1 = second half), so a chunk landing entirely
    # inside one core's half leaves the OTHER core with no owned heads: it co-traces that chunk's
    # collective (a rendezvous) but does no attention compute — it waits. Example (batch_heads=10,
    # LNC2, chunk_width=3): chunk [0,3) → only core 0 works, chunk [6,9) → only core 1 works. Each
    # core still does its 5 heads over the whole kernel (total work is balanced), but the cores partly
    # serialize across chunks instead of running in parallel, so the ring portion degenerates toward
    # LNC1 throughput (the prologue stays 2-way sharded). This regime is reached only when a core's
    # owned accumulators would OOM as a single chunk (very long seqlen) — a partly-serialized run beats
    # not running.
    #
    # TODO(perf): balance the multi-chunk LNC>1 case with round-robin head ownership
    # (head % num_shards == shard_id) WHEN chunking, so every global chunk of width >= 2 holds both
    # cores' heads. The collective is unaffected — it rotates the whole global [chunk_start:chunk_end]
    # slice regardless of ownership; only _shard_batch_runs (emit a strided owned set), the staging
    # loops (_transpose_qk_pipelined / _copy_batch_sharded iterate the owned index set), and the
    # accumulator compaction (already runs-driven) change. Keep contiguous ownership for the
    # single-chunk common case (staging locality, and no imbalance there). Not yet reachable by any
    # test (LNC2 multi-chunk needs a longer per-rank seqlen than the LNC1 chunked test).
    for chunk in _head_chunks(batch_heads, chunk_width):
        chunk_start = chunk.start
        chunk_end = chunk.end
        chunk_runs = _intersect_runs(runs, chunk_start, chunk_end)

        # ── Ring step 0: local K/V into the accumulators directly ──
        # Hoist the first prefetch_depth permutes (this chunk's [chunk_start:chunk_end] slice) to
        # overlap step 0.
        if num_workers > 1:
            for prefetch_step in range(1, prefetch_depth + 1):
                _issue_kv_permute(
                    k_bufs[(prefetch_step - 1) % num_kv_buf][chunk_start:chunk_end],
                    k_bufs[prefetch_step % num_kv_buf][chunk_start:chunk_end],
                    v_bufs[(prefetch_step - 1) % num_kv_buf][chunk_start:chunk_end],
                    v_bufs[prefetch_step % num_kv_buf][chunk_start:chunk_end],
                    replica_group,
                )

        if num_workers == 1:
            # Single ring step is also the last: fold (init) + normalize interleaved per head,
            # so each head's output-write hides behind the next head's attention compute.
            _run_last_step_and_normalize(
                q_transposed,
                k_bufs[0],
                v_bufs[0],
                o_aug_acc_sb,
                o,
                lse,
                softmax_scale,
                num_token_groups,
                seqlen,
                head_dim,
                training,
                lse_dtype,
                False,
                chunk_runs,
                allocator,
                c_row,
                kv_pad_rows=kv_pad_rows,
            )
        else:
            # Ring step 0 initializes the SBUF accumulator (is_accumulate_step=False).
            _run_partial_step(
                q_transposed,
                k_bufs[0],
                v_bufs[0],
                o_aug_acc_sb,
                softmax_scale,
                num_token_groups,
                head_dim,
                False,
                chunk_runs,
                c_row,
                kv_pad_rows=kv_pad_rows,
            )

            # ── Ring steps 1..num_workers-1: partial attention folded into the accumulator ──
            step_idx = 0
            for ring_step in nl.sequential_range(1, num_workers):
                step_idx += 1
                consume_idx = step_idx % num_kv_buf

                # Launch the permute for step step_idx + prefetch_depth (if that step exists) before
                # the attention, so it overlaps compute. Buffer math uses the Python counter.
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
                    # Last ring step: fold (+=) + normalize interleaved per head. A separate final
                    # normalize pass would run fully exposed after all compute (the DMA output tail);
                    # interleaving hides each head's output-write behind the next head's attention.
                    _run_last_step_and_normalize(
                        q_transposed,
                        k_bufs[consume_idx],
                        v_bufs[consume_idx],
                        o_aug_acc_sb,
                        o,
                        lse,
                        softmax_scale,
                        num_token_groups,
                        seqlen,
                        head_dim,
                        training,
                        lse_dtype,
                        True,
                        chunk_runs,
                        allocator,
                        c_row,
                        kv_pad_rows=kv_pad_rows,
                    )
                else:
                    # Fold this step's output straight into the resident SBUF accumulator (+=).
                    _run_partial_step(
                        q_transposed,
                        k_bufs[consume_idx],
                        v_bufs[consume_idx],
                        o_aug_acc_sb,
                        softmax_scale,
                        num_token_groups,
                        head_dim,
                        True,
                        chunk_runs,
                        c_row,
                        kv_pad_rows=kv_pad_rows,
                    )

    o = o.reshape((batch, q_heads, seqlen, head_dim))
    if training:
        lse = lse.reshape((batch, q_heads, _P, num_token_groups))
        return o, lse
    return o
