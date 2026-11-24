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
# vllm_omni_neuron/kernels/nkilib/experimental/attention/attention_const_max.py, sha256 a334a4d6cf82f529faf1d133fc73e6cd331f46ef2e5a1acacb439213dd2fd43e.
# Changes: imports (the core utilities and neurotile from the installed nkilib, the base kernel
# from its Difflet fork); an optional run-time key mask (key_mask_hbm), for joint attention.

"""
Constant-Max Attention Kernel

Computes the UNNORMALIZED partial attention

    [Σ P·V | Σ P·1]     where    P = exp(Q @ K.T * softmax_scale - c)

and folds it into a caller-owned SBUF accumulator. The two matmuls are referred to throughout as:
  - MM1: Q @ K.T (scores)
  - MM2: P @ [V | 1]  (weighted value accumulation, plus the sum_exp column)

`c` is the per-ROW softmax bound c_i = softmax_scale·‖q_i‖·max_j‖k_j‖, supplied by the caller in
``c_row_hbm`` in natural token order. Taking the max as given rather than deriving it online
eliminates:
  1. The DMA transpose between MM1 and MM2, which serializes with collective
     communication (CC) and blocks CC overlap.
  2. The online softmax max-reduction pass.

With no online max the scores stay K-stationary ([Sk-partition, Sq-free]), which is what lets a
ring driver's KV-rotation collective overlap this compute. The accumulator is SBUF-resident across
ring steps and the caller normalizes once at the end, so there is no HBM round-trip per step and no
separate cross-step reduction pass.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.experimental import neurotile as nt

# Tiling constants
_P = nl.tile_size.pmax  # 128: partition dimension
_SK_BLOCK = 2048  # KV streaming block size (16 tiles of 128)
_O_TILE_SIZE = nl.tile_size.gemm_stationary_fmax  # 128: output tile size along Sq (MM2 stationary)
_KV_TILES_PER_BLOCK = _SK_BLOCK // _P  # 16: number of K/V tiles per KV block

# PSUM bank allocation:
#   2 banks for MM1 scores, up to min(psum_num_banks - 2, 8) banks for O_aug accumulation.
#
# Q and O blocks share the same Sq-dimension size. O tiles are staged in PSUM and
# accumulate across all KV blocks. The number of available O_aug banks is derived at
# trace time from nl.tile_size.psum_num_banks. Since probabilities (MM1 output / MM2 stationary)
# have 128 on the free axis, the maximum O block is _MAX_O_AUG_BANKS × gemm_stationary_fmax Sq elements.
#
# On trn2 (gen3), MM1 produces f32 PSUM and the moving operand (Q) is limited to
# gemm_moving_fmax = 512, so we choose max Q/O block = 512. On trn3 (gen4), MM1
# produces bf16 PSUM with no such limit, so the full O_aug capacity is used.
#
# Q stays resident in SBUF while we stream all KV blocks, prefetching the next block
# behind the current block's compute.


def _attention_const_max_partial(
    q_hbm,
    k_hbm,
    v_hbm,
    o_aug_acc_sb,
    c_row_hbm,
    softmax_scale=None,
    aug_col_base=0,
    is_accumulate_step=False,
    kv_pad_rows=0,
    key_mask_hbm=None,
):
    """Partial (unnormalized) attention folding into a resident SBUF accumulator.

    Runs the K-stationary MM1 → exp(scores − c) → MM2([V|1]) pipeline and folds each output
    tile's [Σ P·V | Σ P·1] straight from PSUM into the caller's resident SBUF accumulator —
    is_accumulate_step selects init (ring step 0) vs += (later steps), so there is no HBM
    round-trip and no separate cross-step reduction pass. Does NO LNC sharding of its own: the
    caller passes the head slice this shard owns.

    Args:
        q_hbm: [N, d, Sq] bf16 @ HBM.
        k_hbm: [N, d, Sk] bf16 @ HBM (arbitrary Sk; a trailing sub-128 group is handled).
        v_hbm: [N, Sk, d] bf16 @ HBM.
        o_aug_acc_sb: resident SBUF accumulator this step folds into. Layout mirrors the PSUM
            source: each (head, group) tile occupies (d+1) contiguous columns — d output columns
            followed by 1 sum column.
        c_row_hbm: [N, Sq] bf16 @ HBM holding the per-ROW bound c_i = softmax_scale·‖q_i‖·max_j‖k_j‖
            in natural token order. Each Q-block loads its contiguous [1, sq_block] slice onto the
            free axis and broadcasts it across the 128 partitions.
        softmax_scale: scaling factor for Q. Defaults to 1/sqrt(d).
        aug_col_base: base column of this call's first (head, group) tile (stride d+1 between tiles).
        is_accumulate_step: True folds (+=); False initializes (first ring step).
        kv_pad_rows: Number of trailing Sk tokens that are caller-supplied ZERO padding (used when
            the caller rounded Sk up to a 128-multiple to keep the KV loop uniform). Those rows are
            masked out of the sums exactly; 0 means Sk is entirely real tokens.
        key_mask_hbm: optional [N, Sk] @ HBM, 1 for a valid key and 0 for a masked one, known only
            at run time (Difflet: joint attention's text padding). It replaces the [V|1] ones column,
            so a masked key adds nothing to Σ P·1; the caller zeroes masked keys' V rows (Σ P·V) and
            K rows (their norms then stay out of the bound).

    Pseudocode:
        for each head:
            for each Q-block:
                Q = load(q_hbm[head, Q-block]) * softmax_scale
                c = load(c_row_hbm[head, Q-block])       # per-row bound, broadcast over partitions
                for each KV-block:
                    for each kv tile:
                        S = Q @ K_tile.T                 # MM1: scores
                        P = exp(S - c)                   # evict + subtract + exp
                        O_aug += P @ [V_tile | 1]        # MM2: output + sum_exp
                fold O_aug into o_aug_acc_sb
    """
    Nq, d, Sq = q_hbm.shape
    Sk = k_hbm.shape[2]
    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    # Groups (128-row Sq output tiles) per head — the accumulator stride between heads.
    groups_per_head = (Sq + _O_TILE_SIZE - 1) // _O_TILE_SIZE

    kernel_assert(nl.tile_size.psum_num_banks >= 6, f"need at least 6 PSUM banks, have {nl.tile_size.psum_num_banks}")
    _MM1_BUFFER_BANKS = 2
    _MAX_O_AUG_BANKS = min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, 8)

    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    max_sq_block_size = _MAX_O_AUG_BANKS * _O_TILE_SIZE if has_bf16_psum else nl.tile_size.gemm_moving_fmax

    # Sk need not be a multiple of 128. The whole-128 prefix streams through the pipeline in
    # _SK_BLOCK blocks; the trailing sub-128 tokens are peeled into a single partial tile the
    # pipeline consumes last, because a multi-tile V load cannot carry a sub-128 partition
    # remainder. Every Sk-indexed extent is clamped to each tile's real row count downstream.
    sk_whole = (Sk // _P) * _P
    r_tail = Sk % _P
    total_kv_tiles = (Sk + _P - 1) // _P

    for head_idx in range(Nq):
        # Q is [d, Sq] per head, tiled along Sq at max_sq_block_size; each Q block loaded once and
        # kept resident while all KV blocks stream through it.
        q_block_grid = nt.tiles(q_hbm[head_idx], tile_size=(_P, max_sq_block_size))

        for q_block_idx in range(q_block_grid.shape[1]):
            # Load Q block into SBUF — stays resident for all KV iterations. Q is pre-scaled here
            # rather than folding softmax_scale into the exp, because the bound is
            # c = softmax_scale·‖q‖·max‖k‖: the scores it is subtracted from must already carry
            # the scale.
            q_loaded = q_block_grid[0, q_block_idx].load()
            q_block = q_loaded.data
            nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=softmax_scale)

            sq_block_size = q_loaded.element_shape[1]

            # c_row_hbm holds c_i in natural token order, so this Q-block's contiguous
            # [1, sq_block] slice IS c_i along the score tile's free (Sq) axis.
            # Replicate it down all 128 partitions with ONE partition-stride-0 DMA.
            #
            # Every partition must hold the IDENTICAL c for a query column: softmax is invariant to c only
            # then, since a column whose partitions disagree scales its Sk terms inconsistently and the
            # factor stops cancelling in o = ΣPV/ΣP. 128 reads of the same immutable HBM row give exactly
            # that, and they cost no Vector time. Deriving all 128 partitions from one loaded row with a
            # stream shuffle instead is the same guarantee, but it put ~3.4us of serialized Vector work on
            # every Q-block boundary AHEAD of the Q pre-scale in the in-order Vector queue — and the
            # pre-scale gates this block's first MM1, so the Tensor engine waited out both.
            sq0 = q_block_idx * max_sq_block_size
            per_row_bias = nl.ndarray(shape=(_P, sq_block_size), dtype=c_row_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=per_row_bias,
                src=c_row_hbm[head_idx].ap(pattern=[[0, _P], [1, sq_block_size]], offset=sq0),
            )

            # Allocate output accumulator in PSUM — one tile per Sq output slice, each on its own
            # bank. Accumulate across all KV blocks, then fold into the SBUF accumulator.
            num_output_tiles = (sq_block_size + _O_TILE_SIZE - 1) // _O_TILE_SIZE
            kernel_assert(
                num_output_tiles <= _MAX_O_AUG_BANKS,
                "Q block too large: O_aug tiles exceed available PSUM banks.",
            )
            # Output accumulator width is d+1: columns 0..d-1 hold the weighted output (P @ V), and
            # column d accumulates sum_exp (P @ 1) for the caller's normalize.
            o_aug_psums = nt.psum_pool(
                tile_size=(_O_TILE_SIZE, d + 1),
                element_shape=(sq_block_size, d + 1),
            )
            # Continuous software pipeline across all KV tiles keeps TensorE fed (no per-block
            # boundary bubbles). nt.blocks handles a non-_SK_BLOCK-multiple prefix: the last block
            # has fewer tiles.
            k_stream = None
            v_stream = None
            if sk_whole > 0:
                k_view = nt.blocks(
                    k_hbm[head_idx, :, :sk_whole],
                    tile_size=(_P, d),
                    block_size=(1, _KV_TILES_PER_BLOCK),
                )
                v_view = nt.blocks(
                    v_hbm[head_idx, :sk_whole, :],
                    tile_size=(_P, d),
                    block_size=(_KV_TILES_PER_BLOCK, 1),
                )
                k_stream = k_view[0, :].stream(buffer_count=2)
                v_stream = v_view[:, 0].stream(buffer_count=2)

            # The sub-128 tail as its own single partial tile — a 1-tile grid passes the neurotile
            # load, and it indexes exactly like a block-stream tile grid.
            k_tail = None
            v_tail = None
            if r_tail > 0:
                k_tail = nt.tiles(k_hbm[head_idx, :, sk_whole:Sk], tile_size=(_P, d))[0, :].load()
                v_tail = nt.tiles(v_hbm[head_idx, sk_whole:Sk, :], tile_size=(_P, d))[:, 0].load()

            _process_kv_tiles_pipelined(
                k_stream=k_stream,
                v_stream=v_stream,
                q_block=q_block,
                o_aug_psums=o_aug_psums,
                per_row_bias=per_row_bias,
                k_tail=k_tail,
                v_tail=v_tail,
                total_kv_tiles=total_kv_tiles,
                kv_pad_rows=kv_pad_rows,
                key_mask=None if key_mask_hbm is None else key_mask_hbm[head_idx],
            )

            # Fold each output tile's accumulated o_aug = [Σ P·V | Σ P·1] straight from PSUM into the
            # resident SBUF accumulator (init on the first ring step, += after).
            o_block_start = q_block_idx * (max_sq_block_size // _O_TILE_SIZE)
            for output_tile_idx in range(num_output_tiles):
                g = head_idx * groups_per_head + (o_block_start + output_tile_idx)
                _accumulate_partial(
                    o_aug_psums[output_tile_idx].data,
                    o_aug_acc_sb,
                    aug_col_base + g * (d + 1),
                    is_accumulate_step,
                )


def _process_kv_tiles_pipelined(
    k_stream: Optional[nt.BlockStream],
    v_stream: Optional[nt.BlockStream],
    q_block: nl.NkiTensor,
    o_aug_psums: nt.NDSlice,
    per_row_bias: nl.NkiTensor,
    total_kv_tiles: int,
    k_tail: Optional[nt.NDSlice] = None,
    v_tail: Optional[nt.NDSlice] = None,
    kv_pad_rows: int = 0,
    key_mask=None,
) -> None:
    """Pipelined MM1→exp→MM2 across all KV tiles with a prologue/epilogue split.

    The bound c_i varies along the free (Sq) axis, so it cannot ride the exp's per-partition
    max_value. It is subtracted explicitly (Vector) into an fp16 tile, which the exp then reads.
    The exp is issued via nisa.activation (Scalar engine) so the Vector engine does only the
    subtract and the two pipeline across tiles; the [V|1] copy goes to whichever of the two is
    lighter (see copy_engine below), and all of them pipeline against the Tensor engine's MMs.

    Pipeline (P = prologue_depth):
        Prologue  (kv_tile_idx 0..P-1):        v_copy, MM1, exp
        Steady-state (kv_tile_idx P..total-1):  v_copy, MM1, exp, MM2(i-P)
        Epilogue  (drain last P):               MM2

    A trailing partial tile (the peeled sub-128 KV tail) has r < 128 valid Sk rows. Every
    Sk-indexed extent — the V copy, MM1 output partitions, exp, and the MM2 contraction on both
    operands — is clamped to that tile's real row count r (== v_block[i, 0].data.shape[0]), so
    padded positions contribute nothing to Σ P·V or Σ P·1 and no explicit tail mask is needed.

    Args:
        k_stream/v_stream: block streams over the whole-128 Sk prefix. None when Sk < 128.
        per_row_bias: [_P, sq_block_size] per-row bound c_i replicated down all 128 partitions
            (uniform per free column = per query row).
        k_tail/v_tail: Optional pre-loaded single-tile grids holding the sub-128 Sk tail
            (K tile (128, r_tail), V tile (r_tail, d)). Consumed as the pipeline's last tile.
        kv_pad_rows: Trailing partitions of the LAST tile that are caller-supplied ZERO padding
            (Sk already rounded up to a 128-multiple). Their ones-column is zeroed so they add
            nothing to Σ P·1; Σ P·V is already zero because the padded V tokens are zero.
        total_kv_tiles: ceil(Sk / 128) — the exact flat tile count, including the tail tile. Sets
            the pipeline depth and the epilogue drain range; the last prefix block may be partial,
            so this cannot be derived from num_blocks.
    """
    d = _P
    num_blocks = k_stream.count if k_stream != None else 0
    sq_block_size = q_block.shape[1]
    num_output_tiles = o_aug_psums.shape[0]

    scores_dtype = nl.bfloat16 if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    # V-augmentation copy engine: route to whichever engine is NOT running exp so the copy overlaps.
    # NEVER GpSimd. The compiler triggers the ring's KV collective-permute from GpSimd, and that queue
    # is in-order, so a per-tile copy stream there defers every permute trigger until the previous ring
    # step's copies drain: the driver's kv_prefetch_depth hoist stops taking effect on hardware and the
    # hops go one-per-step (measured ~400-620us of channel idle per hop instead of ~2us). Where the
    # collective cannot hide behind per-core compute that directly stalls the next step's MM1: moving
    # this copy off GpSimd is worth +3.4 MFU points at trn3 cp4 lnc1 (60.9 -> 64.3) and +5.9 at cp4
    # lnc2 (54.1 -> 60.0) on WAN 480p.
    # The subtract runs on Vector and the exp on Scalar, so pick the lighter of those two. The
    # subtract's cost tracks the `shifted` precision, and `shifted` is fp16, which leaves Vector the
    # lighter of the pair -- so the copy goes there. It is ~3.4x cheaper there than on GpSimd.
    copy_engine = nisa.engine.vector

    # Number of MM1+exp tiles computed before first MM2 fires. Gives TensorE
    # a buffer of ready probabilities so it can always issue MM2 without waiting. Bounded by the
    # exact tile count (not num_blocks * tiles-per-block, which over-counts a partial last block
    # and would make the epilogue drain tiles that were never issued).
    prologue_depth = min(8, total_kv_tiles)

    # Rotating V_aug buffers: need prologue_depth+1 (v_copy runs at MM1 time,
    # MM2 consumes prologue_depth iterations later; +1 to avoid WAR)
    num_v_aug_buffers = prologue_depth + 1
    v_aug_all = nl.ndarray(shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)

    num_prob_buffers = prologue_depth + 1
    probs_all = nl.ndarray(shape=(_P, num_prob_buffers, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)

    # `shifted = scores - c_i`, consumed by the exp. fp16 (not bf16) because the exp needs ABSOLUTE
    # precision in the exponent: bf16's 8-bit mantissa leaves ~0.004*|shifted| (~4% on a prob) and
    # fails accuracy, while fp16's 11 bits leave ~0.0005*|shifted| — below the error the bf16 PSUM
    # scores already carry, and finest exactly where |shifted| ~ 0 (the probs that dominate the sum).
    # Halving this tile vs fp32 also halves the subtract's write and the exp's read, the SBUF traffic
    # that was stretching both past 1 cycle/element and slowing the Tensor engine with it.
    # Lifetime is one tile (subtract → exp), so double-buffer — lets subtract(i+1) overlap exp(i).
    # (Deepening this to one buffer per in-flight tile was tried to chase a head-0 miscompute: it made
    # the failure MORE frequent and total rather than partial, so the defect is not this aliasing.)
    num_shifted_buffers = 2
    shifted_all = nl.ndarray(
        shape=(_P, num_shifted_buffers, sq_block_size),
        dtype=nl.float16,
        buffer=nl.sbuf,
    )

    # Per-tile valid-Sk-row slice, recorded as each tile is issued so the deferred MM2 (fired
    # prologue_depth iterations later) and the epilogue drain can reproduce that tile's extent.
    # Every whole-128 prefix tile gets the full _P-partition slice (the extent of every tile `rows`
    # indexes), so the prefix emits exactly the accesses it would for a 128-multiple Sk; only the
    # peeled tail tile is clamped.
    full_rows = slice(0, _P)
    tile_rows = []

    kv_tile_idx = 0  # flat index across all KV tiles

    # Tile sources: the whole-128 prefix blocks, then (when Sk is not a 128-multiple) the peeled
    # sub-128 tail as a final single-tile source, so it rides the same pipeline rather than
    # forcing a separate drain and refill.
    num_sources = num_blocks + (1 if k_tail is not None else 0)

    for source_idx in range(num_sources):
        is_tail = source_idx == num_blocks
        if is_tail:
            k_block = k_tail
            v_block = v_tail
            tiles_in_block = 1
        else:
            k_block = k_stream.load(source_idx)[0]
            v_block = v_stream.load(source_idx)
            tiles_in_block = k_block.shape[0]

        with nl.no_reorder():
            for local_tile in range(tiles_in_block):
                v_buf = kv_tile_idx % num_v_aug_buffers
                prob_buf = kv_tile_idx % num_prob_buffers
                # Rows valid in this tile: all 128 for a prefix tile, r_tail for the tail tile.
                rows = slice(0, v_block[local_tile, 0].data.shape[0]) if is_tail else full_rows
                tile_rows.append(rows)

                # v_copy(i) — capture V tile now while block buffer is valid
                nisa.tensor_copy(
                    dst=v_aug_all[rows, v_buf, :d],
                    src=v_block[local_tile, 0].data,
                    engine=copy_engine,
                )

                # Padded-KV mask. When the caller padded Sk up to a 128-multiple with ZEROED tokens,
                # the last tile's trailing kv_pad_rows partitions are padding. Their V data columns
                # are zero, so Σ P·V is already exact — but the [V|1] ones column would still fold
                # those rows' exp into Σ P·1. Zeroing the ones column there makes BOTH terms exactly
                # zero, so no masked bias is needed. Both memsets start at partition 0 (a non-zero
                # partition start offset is rejected), hence zero-the-column-then-rewrite-the-real-rows
                # rather than memsetting the pad rows.
                if kv_pad_rows > 0 and kv_tile_idx == total_kv_tiles - 1:
                    nisa.memset(v_aug_all[:, v_buf, d : d + 1], 0.0)
                    nisa.memset(v_aug_all[0 : _P - kv_pad_rows, v_buf, d : d + 1], 1.0)

                # Run-time key mask (Difflet): this tile's mask slice becomes its ones column, one key
                # per partition, so a masked key (V row zeroed by the caller) adds nothing to either sum.
                if key_mask is not None:
                    num_rows = rows.stop - rows.start
                    nisa.dma_copy(
                        dst=v_aug_all[rows, v_buf, d : d + 1],
                        src=key_mask.ap(pattern=[[1, num_rows], [1, 1]], offset=kv_tile_idx * _P),
                    )

                # MM1(i) — the K tile's free extent is this tile's row count, so the matmul
                # writes exactly that many score partitions.
                scores = nl.ndarray(shape=(_P, sq_block_size), dtype=scores_dtype, buffer=nl.psum)
                nisa.nc_matmul(dst=scores[rows, :], stationary=k_block[local_tile].data, moving=q_block)

                # exp(i) — subtract the per-row bound, then exponentiate.
                # c_i varies along the free (Sq) axis, so it can't ride the exp's per-partition
                # max_value. Subtract it explicitly (Vector; scores PSUM - c SBUF into an fp16 tile,
                # see shifted_all), then exp on the SCALAR engine (nisa.activation) so the Vector
                # engine does ONLY the subtract and the two pipeline across tiles (the [V|1] copy
                # rides the lighter of the two). Running exp on Vector too is ~1.5x slower.
                # (Folding -c into the score bank ahead of an accumulating MM1 would remove this
                # pass, but nc_matmul cannot target an fp16 PSUM dst and a bf16 bank would round
                # scores-c, whose absolute error ~0.004*c is what the per-row bound exists to avoid.)
                shifted_buf = kv_tile_idx % num_shifted_buffers
                nisa.tensor_tensor(
                    dst=shifted_all[rows, shifted_buf, :],
                    data1=scores[rows, :],
                    data2=per_row_bias[rows, :],
                    op=nl.subtract,
                    engine=nisa.engine.vector,
                )
                nisa.activation(
                    dst=probs_all[rows, prob_buf, :],
                    data=shifted_all[rows, shifted_buf, :],
                    op=nl.exp,
                    scale=1.0,
                    bias=0.0,
                )

                # MM2(i-P) — only after prologue is filled
                prologue_filled = kv_tile_idx >= prologue_depth
                if prologue_filled:
                    consume_idx = kv_tile_idx - prologue_depth
                    _issue_mm2(
                        o_aug_psums=o_aug_psums,
                        probs_all=probs_all,
                        v_aug_all=v_aug_all,
                        tile_idx=consume_idx,
                        rows=tile_rows[consume_idx],
                        num_prob_buffers=num_prob_buffers,
                        num_v_aug_buffers=num_v_aug_buffers,
                        num_output_tiles=num_output_tiles,
                    )

                kv_tile_idx += 1

    # === Epilogue: drain remaining prologue_depth MM2s ===
    with nl.no_reorder():
        for i in range(prologue_depth):
            drain_idx = total_kv_tiles - prologue_depth + i
            _issue_mm2(
                o_aug_psums=o_aug_psums,
                probs_all=probs_all,
                v_aug_all=v_aug_all,
                tile_idx=drain_idx,
                rows=tile_rows[drain_idx],
                num_prob_buffers=num_prob_buffers,
                num_v_aug_buffers=num_v_aug_buffers,
                num_output_tiles=num_output_tiles,
            )


def _issue_mm2(
    o_aug_psums,
    probs_all,
    v_aug_all,
    tile_idx,
    rows,
    num_prob_buffers,
    num_v_aug_buffers,
    num_output_tiles,
) -> None:
    """MM2 for one KV tile: P @ [V|1] accumulated into the O_aug PSUM tiles (tiled along Sq).

    Both operands clamp their contraction (Sk) to `rows`, so a peeled sub-128 tail tile's padded
    rows add nothing to Σ P·V or Σ P·1. Shared by the steady-state and the epilogue drain, which
    issue the same MM2 for different tile indices.
    """
    prob_buf = tile_idx % num_prob_buffers
    v_aug = v_aug_all[:, tile_idx % num_v_aug_buffers, :]
    should_accumulate = tile_idx != 0
    for ot_idx in range(num_output_tiles):
        sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
        nisa.nc_matmul(
            dst=o_aug_psums[ot_idx].data,
            stationary=probs_all[rows, prob_buf, nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size)],
            moving=v_aug[rows, :],
            accumulate=should_accumulate,
        )


def _accumulate_partial(o_aug_psum, o_aug_acc_sb, aug_col, is_accumulate_step):
    """Fold one output tile's [Σ P·V | Σ P·1] from PSUM into a resident SBUF accumulator.

    Replaces the store-to-HBM + reload-and-add cross-step reduction with a single PSUM→SBUF op per
    tile: init (tensor_copy) on the first ring step, add (tensor_tensor) after.

    o_aug_psum: [sq_tile_size, d+1] PSUM — cols 0..d-1 are Σ P·V, col d is Σ P·1 (sum_exp).
    o_aug_acc_sb: [_P, ...] SBUF accumulator (this run's view); this tile lands at
        o_aug_acc_sb[:, aug_col:aug_col+d+1] (d output cols followed by 1 sum col).
    is_accumulate_step: True => += (fold), False => = (initialize).
    """
    sq_tile_size = o_aug_psum.shape[0]
    d = o_aug_psum.shape[1] - 1
    aug_dst = o_aug_acc_sb[:sq_tile_size, aug_col : aug_col + d + 1]
    if is_accumulate_step:
        nisa.tensor_tensor(dst=aug_dst, data1=aug_dst, data2=o_aug_psum, op=nl.add)
    else:
        nisa.tensor_copy(dst=aug_dst, src=o_aug_psum)
