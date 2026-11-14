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
# torch-neuronx 2.14.3, nki 0.7.0, neuronx-cc 2.28.7631), core/attention/attention_cte.py, sha256
# 608f0c0320564df85ef3f038bc749591279e7fe5c22480d1dcdb6b3361a2c734 (MXFP8 attention plan, Task 3).
# Pruned to the subset Difflet calls (Step 6): every branch of a feature ``_check_difflet_subset``
# refuses is gone, the subset's instruction stream unchanged. Added: Difflet's entry point
# ``mx_attention_fwd``, the MX block membership ``qk_perm`` / ``pv_perm``, MM2 on ``nc_matmul_mx``
# (``pv_mx``: ``_mx_load_v``, ``_mx_quantize_p``), MM1 on ``nc_matmul_mx`` with row tiling
# (``qk_mx``: ``_mx_load_q``, ``_mx_load_k``, ``_mx_swizzle_quantize``).

"""
Attention Kernel For Context Encoding (Prefill), Difflet's subset of nkilib's attention_cte

ALGORITHM OVERVIEW:

Base Attention Algorithm:
The kernel computes: Output = softmax(scale * Q @ K^T) @ V

Breaking this down into steps:
  1. Compute attention scores: S = scale * Q @ K^T  (matmul, called MM1)
  2. Apply the optional key bounds to S
  3. Compute row-wise softmax: P = softmax(S) = exp(S - max(S)) / sum(exp(S - max(S)))
  4. Compute final output: O = P @ V  (matmul, called MM2)

THE SUBSET:

- Non-causal attention on q, k, v ``[batch, seqlen, d]`` bf16 (attention_cte's ``tp_q=True,
  tp_k=True`` layout), head dim 128, one k/v batch per q batch, output ``[batch, seqlen_q, d]``.
- Sequence packing (bound_min/bound_max provided): query i attends keys
  ``bound_min[i] <= j < bound_max[i]``, masked with range_select; that path takes scale 1.0.
- ``pv_mx``: MM2 on ``nc_matmul_mx``, P and V MX-quantized per 512-key tile (MXFP8 attention
  plan, Task 3).
- ``qk_mx``: MM1 on ``nc_matmul_mx``, Q and K MX-quantized on load along the head dim, each large
  tile's four 512-key tiles in the four row quadrants of the PE array (plan Task 4).
- ``allow_vector_exp`` as attention_cte has it, but default False: the exp on the Scalar Engine
  (rulings BR19 / BR20).

Everything else attention_cte offers is refused (``_check_difflet_subset``).

IMPLEMENTATION DETAILS AND LOOP STRUCTURE:

Level 1: LNC2 Sharding (on Trn2+)
  - Shards computation across 2 NeuronCores (LNC=2)
  - Primary sharding: Divides batch dimension evenly
  - Secondary sharding (for odd batch): Last batch item sharded on seqlen_q (50%/50%)
    * Falls back to single core for short sequences (< 1024 tokens)

Level 2: Batch Loop
  - Iterates over batch items assigned to this core
  - Each batch item processes independently

Level 3: Section Loop (Flash Attention for long sequences)
  - For K/V length > 10K tokens: divide into 8K-token sections
  - Process one section at a time to fit in SBUF memory
  - Maintains running statistics (max, sum) across sections
  - Final output computed using flash attention rescaling
  - For short sequences: single section contains all K/V
  - Check https://arxiv.org/abs/2205.14135 and https://arxiv.org/abs/2307.08691
    for more details about flash attention.

Level 4: Group Loop (Q sequence processing)
  - Q sequence divided into groups of 128 tokens (_Q_GRP_SZ)
  - Each group processes independently within a section
  - Software pipelining: overlaps operations across groups (i, i+1, i+2)
    * Group i:   PV computation, writeback
    * Group i+1: Exp computation
    * Group i+2: Q load, QK computation
  - Uses modular allocation for efficient buffer reuse

The kernel uses float32 for softmax and bfloat16 for other operations.
"""

from dataclasses import dataclass
from typing import Any

import nki
import nki.isa as nisa
import nki.language as nl
import nki.language.opcode as _nl_opcode

from nkilib.core.utils.kernel_assert import assert_shape, kernel_assert
from nkilib.core.utils.kernel_helpers import (
    PSUM_BANK_SIZE,
    _psum_num_banks,
    div_ceil,
    get_verified_program_sharding_info,
    is_trn3_b1,
)
from nkilib.core.utils.modular_allocator import ModularAllocator

# Physical rotation depth of mm1_partial_max.
_MM1_PARTIAL_MAX_STAGES = 2
_FLOAT32_MIN = -3.4028235e38  # used for initialization and masking

"""
Kernel constraints (based on tested range, values outside range might work in practice)
"""
_MAX_BS = 512  # max tested batch size
_MAX_HEAD_DIM = 512  # max supported head dim (d)


"""
Sharding, tile size and threshold related constants
"""
_MIN_SEQLEN_FOR_LNC2_SHARDING = 1024  # if odd batch, then shard on sequence if len above this
_SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT = 0.5  # When sharding LNC2 on seqlen, pass 50% of seqlen to each shard
_Q_GRP_SZ = 128
_D_TILE_SZ = 128  # Head dim tile size for K partition dim and Q loading
_V_D_TILE_SZ = 512  # Head dim tile size for V free dim (MM2)
_V_TILE_SZ = 128  # V tile size for loading and MM2 operations
_K_TILE_SZ = 512  # K tile size for loading and MM1+masking operations
_EXP_TILE_SZ = 512  # Tile size for exp instructions (must equal _K_TILE_SZ)
_LARGE_TILE_SZ = 2048  # Larger tile size for allocations/pipelining (4 x 512 tiles)
_NUM_K_TILES_IN_LARGE_TILE = _LARGE_TILE_SZ // _K_TILE_SZ  # 512-key K/exp sub-tiles per large tile

# PSUM banks MM1 rotates through. One [128, 512] fp32 score tile fills one bank exactly, so four
# banks means only four score tiles in flight; because the Vector row-max evict that frees a bank
# is slower per tile than MM1 itself, MM1 stalls once per large tile. Banks 5 and 6 are unused
# when MM2 accumulates in PSUM with a single d-tile, so MM1 can have six.
_MM1_PSUM_BANKS_WIDE = (0, 1, 2, 3, 5, 6)
_BF16_ITEMSIZE = 2
_SBUF_COMPILER_RESERVE = 8 * 1024
_FLASH_ATTENTION_THRESHOLD = 10 * 1024  # Use flash attention above this K/V length
_FLASH_ATTENTION_SECTION_LENGTH = 8 * 1024  # Section size when using flash attention


def _sbuf_scratch_budget() -> int:
    """Per-partition SBUF this kernel may claim, net of what the compiler reserves."""
    return nl.tile_size.total_available_sbuf_size - _SBUF_COMPILER_RESERVE


@nki.jit
def mx_attention_fwd(q, k, v, *, scale, bound_min=None, bound_max=None, pv_mx: bool, qk_mx: bool,
                     allow_vector_exp: bool = False):
    """Difflet's entry: non-causal attention in the ``tp_q=True, tp_k=True`` layout (q, k, v
    ``[batch, seqlen, d]`` bf16), optionally key-bounded, as ``ops_impl/attention.py`` calls
    attention_cte on both paths. ``pv_mx`` / ``qk_mx``: MM2 / MM1 on ``nc_matmul_mx``; both False
    is attention_cte unchanged at the same ``allow_vector_exp``. That is attention_cte's switch,
    default False here: the exp on the Scalar Engine, off the Vector Engine that binds shipped
    attention_cte (MXFP8 attention plan, rulings BR19 / BR20)."""
    return _attention_cte(q=q, k=k, v=v, scale=scale, causal_mask=False, tp_q=True, tp_k=True,
                          bound_min=bound_min, bound_max=bound_max, pv_mx=pv_mx, qk_mx=qk_mx,
                          allow_vector_exp=allow_vector_exp)


# DMA-to-PE lowering uses shared transpose scratch. Allow the compiler to
# rotate addresses and legalize buffer reuse in this mode.
if _psum_num_banks() == 7:
    mx_attention_fwd = mx_attention_fwd.enable_address_rotation().enable_non_ssa_legalization()


# The MX block membership of the two matmuls, as ``difflet.quant.mx`` perms (block 0's logical
# indices, then block 1's, ...), for the CPU reference. Both operands of each matmul reach
# ``nc_matmul_mx`` in the group-of-4 layout (nkilib's ``swizzle_quantize_mx``: partition g, byte j
# of the x4 operand holds contraction index 4g + j), and ``quantize_mx`` blocks 8 partitions x 4
# bytes, so block b is indices 32b..32b+31. Locked on the device by
# scripts/mx_attention/membership.py (rerun on MM2's own loaders, ruling BR16, and on MM1's, Task 4).
def qk_perm(d: int):
    """MM1's blocks along the head dim (Q and K)."""
    import torch

    return torch.arange(d)


def pv_perm(tile: int):
    """MM2's blocks along a 512-key tile (P and V)."""
    import torch

    return torch.arange(tile)


# Why MM2's loaders take the routes below (neuronx-cc 2.28, on the device, 3b micro-probes): a DMA
# transpose cannot write the group-of-4 order (its output steps must be 32-byte aligned,
# NCC_IBIR155, and its transposing dimensions contiguous, BIR verifier checkDMATranspose), and
# ``quantize_mx`` takes no strided source (NCC_IBIR164), so each group of 4 must sit contiguous in
# SBUF before it is quantized.
def _mx_load_v(v, data, scale, stage, packed, row_offset, num_keys):
    """V's MM2 operand for one 512-key tile (``pv_mx``): rows ``[row_offset, row_offset +
    num_keys)`` of the HBM ``v`` (flat ``[rows, d]``) -> ``data`` ``[128, d]`` float8_e4m3fn_x4 +
    ``scale`` ``[128, d]`` uint8, partition g byte j = key 4g + j, keys at or past ``num_keys``
    zero. One contiguous DMA puts four rows on each partition (``stage[g, j*d + n] = v[4g + j,
    n]``), GpSimd regroups them (``packed[g, 4n + j]``), then ``quantize_mx``. Once per tile and
    section, shared by every query group."""
    d = data.shape[1]
    full, rest = num_keys // 4, num_keys % 4
    if num_keys < 4 * stage.shape[0]:
        nisa.memset(stage, value=0.0, engine=nisa.gpsimd_engine)
    if full > 0:
        nisa.dma_copy(dst=stage.ap(pattern=[[4 * d, full], [1, 4 * d]], offset=0),
                      src=v.ap(pattern=[[4 * d, full], [1, 4 * d]], offset=row_offset * d))
    if rest > 0:
        nisa.dma_copy(dst=stage.ap(pattern=[[4 * d, 1], [1, rest * d]], offset=full * 4 * d),
                      src=v.ap(pattern=[[4 * d, 1], [1, rest * d]], offset=(row_offset + 4 * full) * d))
    nisa.tensor_copy(dst=packed.ap(pattern=[[4 * d, 128], [4, d], [1, 4]], offset=0),
                     src=stage.ap(pattern=[[4 * d, 128], [1, d], [d, 4]], offset=0),
                     engine=nisa.gpsimd_engine)
    nisa.quantize_mx(dst=data, src=packed, dst_scale=scale)


def _mx_quantize_p(data, scale, psum, stage, src, src_offset, num_q, num_keys):
    """P's MM2 operand for one 512-key tile (``pv_mx``): the bf16 exp tile ``src[:num_q,
    src_offset : src_offset + 512]`` (queries on partitions) -> ``data`` ``[128, 128]``
    float8_e4m3fn_x4 + ``scale`` ``[128, 128]`` uint8, partition g byte j = key 4g + j. Keys at or
    past ``num_keys`` (never written by the exp) are zeroed first, so a partial tile is a 512-key
    tile padded with zeros. nkilib's ``swizzle_quantize_mx`` path on explicit buffers: the tile
    viewed as fp32 pairs, two stride-2 TensorE transposes into ``psum`` (``[128, 256]`` fp32), a
    ScalarE copy to ``stage`` (``[128, 256]`` fp32), ``quantize_mx`` of it viewed as bf16."""
    if num_keys < 512:
        nisa.memset(src[:num_q, nl.ds(src_offset + num_keys, 512 - num_keys)], value=0.0,
                    engine=nisa.gpsimd_engine)
    src32 = src.view(nl.float32)
    o = src_offset // 2
    for h in range(2):
        nisa.nc_transpose(dst=psum.slice(dim=1, start=h, end=2 * num_q, step=2),
                          data=src32[:num_q, :].slice(dim=1, start=o + h, end=o + 256, step=2))
    nisa.tensor_copy(dst=stage, src=psum, engine=nisa.scalar_engine)
    nisa.quantize_mx(dst=data, src=stage.view(nl.bfloat16), dst_scale=scale)


# MM1 (``qk_mx``) contracts over the head dim: at d = 128 an x4 operand spans 32 partitions, so one
# nc_matmul_mx fills a quarter of the PE rows, and four run at once in the row quadrants
# (tile_size (32, 128), tile_position (32 t, 0); operands and scales at partition 32 t). The four
# share one query group (the stationary, repeated in every quadrant) and take a large tile's four
# 512-key tiles (the moving operands, tile t in quadrant t): their score tiles are the large
# tile's, in its four PSUM banks (scripts/mx_attention/row_tiling.py, plan Task 4 Step 1).
_QUADRANT = 32  # PE rows per row tile, = the x4 operand's partitions at d = 128
_SCALE_ROWS = _QUADRANT // 8  # a quadrant's MX scales sit in its first 4 partitions


def _mx_swizzle_quantize(data, scale, psum, stage, src):
    """Four quadrants of MM1's x4 operand at once (``qk_mx``): ``src`` ``[128 rows, 4 * 128]``
    bf16, block t in columns ``128 t ..`` -> ``data`` x4 + ``scale`` uint8, ``[128, 128]`` each
    (``data``/``scale`` may be a 128-column slice of a wider tile), partition ``32 t + g`` element
    ``r`` = block t's row ``r``, head-dim indices ``4g .. 4g + 3``. nkilib's ``swizzle_quantize_mx``
    on explicit buffers, the four blocks in the same two stride-2 TensorE transposes of the fp32
    view (into ``psum``, ``[128, 256]`` fp32), one ScalarE copy to ``stage`` (``[128, 256]``
    fp32) and one ``quantize_mx``."""
    src32 = src.view(nl.float32)
    for h in range(2):
        nisa.nc_transpose(dst=psum.slice(dim=1, start=h, end=256, step=2),
                          data=src32.slice(dim=1, start=h, end=256, step=2))
    nisa.tensor_copy(dst=stage, src=psum, engine=nisa.scalar_engine)
    nisa.quantize_mx(dst=data, src=stage.view(nl.bfloat16), dst_scale=scale)


def _mx_load_q(q, data, scale, psum, stage, nat, row_offset, num_rows):
    """MM1's stationary for one query group (``qk_mx``): rows ``[row_offset, row_offset +
    num_rows)`` of the HBM ``q`` (flat ``[rows, d]``) -> ``data`` ``[128, 128]`` x4 + ``scale``,
    quantized along the head dim and repeated in the four quadrants (quadrant t's partition ``32 t
    + g``, element r = query r's indices ``4g .. 4g + 3``). One DMA reads the rows four times into
    ``nat`` ``[128, 4 d]`` (queries on partitions; rows past ``num_rows`` zero, so a partial group
    still gives the stationary all 128 columns the row tiling needs), then
    ``_mx_swizzle_quantize``."""
    d = q.shape[-1]
    if num_rows < nat.shape[0]:
        nisa.memset(nat, value=0.0, engine=nisa.gpsimd_engine)
    nisa.dma_copy(dst=nat.ap(pattern=[[4 * d, num_rows], [d, 4], [1, d]], offset=0),
                  src=q.ap(pattern=[[d, num_rows], [0, 4], [1, d]], offset=row_offset * d))
    _mx_swizzle_quantize(data, scale, psum, stage, nat)


def _mx_load_k(k, data, scale, psum, stage, nat, row_offset, num_keys):
    """MM1's moving operands for one 2048-key large tile (``qk_mx``): keys ``[row_offset,
    row_offset + num_keys)`` of the HBM ``k`` (flat ``[rows, d]``) -> ``data`` ``[128, 512]`` x4 +
    ``scale``, the large tile's 512-key tile t in quadrant t (partition ``32 t + g``, element n =
    key ``512 t + n``'s indices ``4g .. 4g + 3``), keys at or past ``num_keys`` zero. Per 128-key
    chunk c (``nat[c]``, ``stage[c]``, ``psum[c % 2]``): one DMA brings rows ``128 c ..`` of the
    four tiles (``nat`` ``[128, 4 d]``, tile t in columns ``d t ..``), then
    ``_mx_swizzle_quantize`` into columns ``128 c ..``."""
    d = k.shape[-1]
    for c in range(4):
        rows = [min(max(num_keys - (_K_TILE_SZ * t + 128 * c), 0), 128) for t in range(4)]
        if min(rows) == 128:
            nisa.dma_copy(dst=nat[c].ap(pattern=[[4 * d, 128], [d, 4], [1, d]], offset=0),
                          src=k.ap(pattern=[[d, 128], [_K_TILE_SZ * d, 4], [1, d]],
                                   offset=(row_offset + 128 * c) * d))
        else:
            nisa.memset(nat[c], value=0.0, engine=nisa.gpsimd_engine)
            for t in range(4):
                if rows[t] > 0:
                    nisa.dma_copy(dst=nat[c].ap(pattern=[[4 * d, rows[t]], [1, d]], offset=t * d),
                                  src=k.ap(pattern=[[d, rows[t]], [1, d]],
                                           offset=(row_offset + _K_TILE_SZ * t + 128 * c) * d))
        _mx_swizzle_quantize(data[:, nl.ds(128 * c, 128)], scale[:, nl.ds(128 * c, 128)],
                             psum[c % 2], stage[c], nat[c])


#: attention_cte's arguments outside the subset, refused by name whatever their value.
_REFUSED = (
    "k_prior", "v_prior", "prior_used_len",  # prefix caching
    "sink",  # attention sink
    "sliding_window",  # sliding-window attention
    "tp_out",  # transposed output
    "cache_softmax", "softmax_dtype", "skip_output_normalization",  # training / ring outputs
    "out_o_hbm", "out_neg_max_hbm", "out_sum_hbm",
    "mm_out_dtype",  # bf16 PSUM
    "cp_offset", "global_cp_deg", "cp_strided_q_slicing", "cp_striped_input",  # context parallel
    "position_bias", "bias_layout", "bias_band_params",  # position bias
    "k_cache_sbuf", "v_cache_sbuf", "k_prior_sbuf", "v_prior_sbuf", "init_sbuf_addr",  # SBUF I/O
    "k_scale_sb",  # delayed fp8 K dequantization
    "kv_used_len",  # dynamic active-KV length
    "block_size", "kvp_rank_id", "kvp_group_size", "kvp_seg_block_offset", "kvp_cp_offset_int",
    "kvp_k_threshold_sb", "kvp_seg_block_offset_int",  # round-robin KV parallel
)


def _check_difflet_subset(q, tp_q, tp_k, causal_mask, refused):
    """Refuse every attention_cte feature outside Difflet's subset, whose branches the fork no
    longer has (plan Task 3 Step 6), instead of silently computing something else: causal masking
    (``causal_mask=True``), the transposed q / k layouts (``tp_q`` / ``tp_k`` False), head dims
    other than 128, and any argument of ``_REFUSED`` (or any other name) passed at all: prefix
    caching (``k_prior``, ``v_prior``, ``prior_used_len``), the attention sink (``sink``), sliding
    windows (``sliding_window``), the transposed output (``tp_out``), the softmax statistics and the
    unnormalized output (``cache_softmax``, ``softmax_dtype``, ``skip_output_normalization``,
    ``out_o_hbm``, ``out_neg_max_hbm``, ``out_sum_hbm``), bf16 PSUM (``mm_out_dtype``), context
    parallelism (``cp_offset``, ``global_cp_deg``, ``cp_strided_q_slicing``,
    ``cp_striped_input``), position bias (``position_bias``, ``bias_layout``,
    ``bias_band_params``), SBUF inputs (``k_cache_sbuf``, ``v_cache_sbuf``, ``k_prior_sbuf``,
    ``v_prior_sbuf``, ``init_sbuf_addr``), delayed fp8 K dequantization (``k_scale_sb``), a dynamic
    KV length (``kv_used_len``) and round-robin KV parallelism (``block_size``, ``kvp_*``).
    ``_attention_cte`` also refuses non-bf16 q / k and grouped-query k/v (a k/v batch other than
    q's)."""
    kernel_assert(not causal_mask, "Difflet's attention_cte fork refuses causal_mask")
    kernel_assert(tp_q and tp_k, f"Difflet's attention_cte fork refuses tp_q / tp_k False, got {tp_q=}, {tp_k=}")
    kernel_assert(not refused, f"Difflet's attention_cte fork refuses {', '.join(sorted(refused))}")
    d = q.shape[2]
    kernel_assert(d == 128, f"Difflet's attention_cte fork takes head dim 128 only, got {d}")


def _attention_cte(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    scale: float = 1.0,
    causal_mask: bool = False,
    tp_q: bool = True,
    tp_k: bool = True,
    bound_min=None,
    bound_max=None,
    pv_mx: bool = False,
    qk_mx: bool = False,
    allow_vector_exp: bool = False,
    **refused,
):
    """Attention on one call's batch: validation, the batch loop and the LNC sharding.

      q, k, v: ``[batch, seqlen, d]`` bf16 in HBM, head dim 128.
      scale: softmax scale; must be 1.0 with bounds (pre-scale q, the range_select path has no scale).
      bound_min, bound_max: optional int32 ``[batch, seqlen_q, 1]``: query i attends keys
        ``bound_min[i] <= j < bound_max[i]``.
      pv_mx: MM2 on nc_matmul_mx: P and V MX-quantized per 512-key tile.
      qk_mx: MM1 on nc_matmul_mx: Q and K MX-quantized on load along the head dim, row-tiled.
      allow_vector_exp: attention_cte's switch: allow the Vector Engine exp on trn3. False (the
        fork's default; attention_cte's is True) uses the Scalar Engine.
      causal_mask, tp_q, tp_k, **refused: accepted only at the subset's values (``False``, ``True``,
        ``True``, nothing); see ``_check_difflet_subset``.
    """
    _check_difflet_subset(q, tp_q, tp_k, causal_mask, refused)

    kernel_assert(
        (bound_min is None) == (bound_max is None), "bound_min and bound_max must both be set or both be None"
    )
    # Sequence packing is active when per-query KV bounds are provided
    is_sequence_packed = bound_min is not None

    seqlen_q, seqlen_k, d, d_v, out_shape = _check_input_and_return_shape(q, k, v)
    if is_sequence_packed:
        batch_size = out_shape[0]
        kernel_assert(
            bound_min.shape == (batch_size, seqlen_q, 1),
            f"bound_min shape must be (batch, seqlen_q, 1)=({batch_size}, {seqlen_q}, 1), got {bound_min.shape}",
        )
        kernel_assert(
            bound_max.shape == (batch_size, seqlen_q, 1),
            f"bound_max shape must be (batch, seqlen_q, 1)=({batch_size}, {seqlen_q}, 1), got {bound_max.shape}",
        )

    bs = q.shape[0]
    kernel_assert(bs > 0, f"Batch size must be positive, got {bs}")
    kernel_assert(
        bs <= _MAX_BS,
        f"attention_cte kernel is not tested for batch size above {_MAX_BS}, got {bs}.",
    )
    kernel_assert(k.shape[0] == bs, f"Difflet's attention_cte fork refuses grouped-query k/v, got "
                  f"q batch {bs}, k/v batch {k.shape[0]}")

    result = nl.ndarray(shape=out_shape, dtype=q.dtype, buffer=nl.shared_hbm)

    ac = AttnConfig(
        seqlen_q=seqlen_q,
        seqlen_k_active=seqlen_k,
        d=d,
        d_v=d_v,
        scale=scale,
        dtype=q.dtype,
        is_sequence_packed=is_sequence_packed,
        allow_vector_exp=allow_vector_exp,
        pv_mx=pv_mx,
        qk_mx=qk_mx,
    )
    if pv_mx:
        kernel_assert(d_v == _D_TILE_SZ, f"pv_mx takes d_v {_D_TILE_SZ}, got {d_v=}")
        kernel_assert(_psum_num_banks() == 8, "pv_mx transposes P through PSUM bank 7")
    if qk_mx:
        kernel_assert(_psum_num_banks() == 8 and d_v <= _V_D_TILE_SZ,
                      "qk_mx transposes Q and K through PSUM bank 7 (6 with pv_mx): eight banks, "
                      "MM2 in bank 4 alone")

    grid_ndim, num_shard, shard_id = get_verified_program_sharding_info("attention_cte", max_sharding=2)

    # Shard on batch size while it is divisible (if not sharded, num_shard = 1, shard_id = 0)
    num_bs_per_shard = bs // num_shard
    bs_offset = shard_id * num_bs_per_shard

    for batch_idx in range(num_bs_per_shard):
        _attention_cte_impl(q, k, v, result, batch_idx + bs_offset, ac, bound_min=bound_min, bound_max=bound_max)

    has_remainder = (bs % num_shard) != 0
    last_batch = bs - 1

    # shard on seqlen_q for the remainder bs
    if has_remainder:
        if seqlen_q >= _MIN_SEQLEN_FOR_LNC2_SHARDING:
            total_grps = div_ceil(seqlen_q, _Q_GRP_SZ)
            batch_0_grp = int(total_grps * _SEQLEN_SHARDING_SPLIT_FACTOR_DEFAULT)
            batch_1_grp = total_grps - batch_0_grp

            batch_length = shard_id * batch_1_grp + (1 - shard_id) * batch_0_grp  # shard_id is 0 or 1
            _attention_cte_impl(
                q,
                k,
                v,
                result,
                last_batch,
                ac,
                shard_seqlen_q_start=shard_id * batch_0_grp,
                shard_seqlen_q_length=batch_length,
                bound_min=bound_min,
                bound_max=bound_max,
            )
        else:
            # Have core 0 do all the work
            if shard_id == 0:
                _attention_cte_impl(q, k, v, result, last_batch, ac, bound_min=bound_min, bound_max=bound_max)

    return result


@dataclass
class AttnConfig(nl.NKIObject):
    """High-level attention configuration set at kernel entry point."""

    # Sequence dimensions
    seqlen_q: int = None
    seqlen_k_active: int = None
    d: int = None
    d_v: int = None

    scale: float = None
    dtype: Any = None

    # sequence packing
    is_sequence_packed: bool = None

    allow_vector_exp: bool = False  # Difflet: default False (attention_cte's is True)
    pv_mx: bool = False  # Difflet: MM2 on nc_matmul_mx, P and V MX-quantized per 512-key tile
    qk_mx: bool = False  # Difflet: MM1 on nc_matmul_mx, Q and K MX-quantized along d, row-tiled


def _attention_cte_impl(
    q,
    k_active,
    v_active,
    o,
    batch_id,
    ac: AttnConfig,
    shard_seqlen_q_start=-1,
    shard_seqlen_q_length=-1,
    bound_min: Any = None,
    bound_max: Any = None,
):
    """
    Internal implementation function for attention computation.

    This function processes a single batch and handles the core attention computation
    with flash attention optimization, sectioning, and the optional key bounds.

    Args:
      q, k_active, v_active: Input tensors
      o: Output tensor
      batch_id: Batch index
      ac: High-level attention configuration
      shard_seqlen_q_start, shard_seqlen_q_length: Seqlen sharding parameters
      bound_min, bound_max: Optional bound tensors for sequence packing

    High-level logic:
    For large enough K/V length, we divide the K/V into sections of 8k.

    For each section:
      a. Load K and V to SBUF
      b. Loop over Q (groups) - each group has seqlen 128 (_Q_GRP_SZ)
      c. Within each group:
        i.   Load Q
        ii.  Compute QK^T (MM1) and max
        iii. Compute exponential and transpose
        iv.  Compute PV (MM2)
        v.   Write to output

    Handling multiple sections:
      We keep running max, sum, etc. buffers that we keep updating as we go through
      sections and use these to update the output using flash attention.

    Pipelining:
      To maximize utilization of the hw engines, we use software pipelining to
      inform scheduler decisions. This means we manually interleave iterations of
      the group loop to make sure certain operations of group i+2 should start even
      as group i is being processed. In addition we use modulo allocation for tensors
      to enable pipelining across Q groups and K/V tiles within a section.
    """
    is_seqlen_sharded = shard_seqlen_q_start >= 0

    # Compute all tile parameters including section length and number of sections
    atp = _compute_tile_parameters(ac, is_seqlen_sharded)

    # Update shard length if not sharded
    if not is_seqlen_sharded:
        shard_seqlen_q_start = 0
        shard_seqlen_q_length = atp.num_grps
    atp.shard_grp_start = shard_seqlen_q_start
    atp.shard_grp_count = shard_seqlen_q_length

    # Initialize allocator and buffer container
    allocator = ModularAllocator(initial_address=0)
    bufs = AttnInternalBuffers()

    # Allocate shared utilities (zero bias)
    bufs.zero_bias_tensor = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)
    nisa.memset(bufs.zero_bias_tensor, 0.0)

    # Setup range select bounds for the key bounds (sequence packing)
    _setup_range_select_bounds(ac, atp, bufs, allocator, bound_min, bound_max, batch_id=batch_id)

    # Allocate running statistics (persistent across sections)
    bufs.mm1_running_max = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
    # nisa.exponential (Vector, B1 only) subtracts max_value, whereas the nisa.activation fallback
    # adds a bias, so the Vector exp path needs the max un-negated. It is allocated here, beside the
    # running max it mirrors, because it must be indexed per Q group and persist across sections: a
    # local nl.ndarray inside _exp_impl aliases across software-pipeline stages.
    bufs.mm1_running_max_pos = (
        allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
        if (ac.allow_vector_exp and is_trn3_b1())
        else None
    )
    bufs.exp_running_sum = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
    bufs.exp_sum_reciprocal = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)

    # Mark allocator checkpoint for section-level reset
    sbuf_addr_outer = allocator.get_current_address()

    # Loop over k/v sections
    for section_idx in range(atp.num_sections):
        sp = SectionParams(section_idx=section_idx, section_offset=atp.section_len * section_idx)

        # Allocate the internal buffers
        allocator.set_current_address(sbuf_addr_outer)
        _allocate_attention_buffers(allocator, ac, atp, bufs)
        sbuf_addr = allocator.get_current_address()

        # Load K and V for the section from HBM
        if ac.qk_mx:
            _load_k_tile_mx(k_active, bufs, batch_id, sp, atp)
        else:
            _load_k_tile(k_active, bufs.k_sb, batch_id, sp, atp.num_k_tiles_per_section)
        if ac.pv_mx:
            _load_v_tile_mx(v_active, bufs, batch_id, sp, atp)
        else:
            _load_v_tile(v_active, bufs.v_sb, batch_id, sp, atp.num_v_tiles_per_section)

        # Start with Q load and actual compute
        if shard_seqlen_q_length <= 1:
            # no pipelining when there's only 1 group
            _load_q_impl(shard_seqlen_q_start, atp, bufs, q, batch_id, sbuf_addr)
            _qk_and_max_impl(shard_seqlen_q_start, ac, atp, sp, bufs)
            _update_max_impl(shard_seqlen_q_start, atp, sp, bufs)
            _exp_impl(shard_seqlen_q_start, ac, atp, sp, bufs)
            _pv_impl(shard_seqlen_q_start, ac, atp, sp, bufs)
            _write_back_impl(shard_seqlen_q_start, ac, atp, sp, bufs, o, batch_id)

        else:
            # Do software pipelining. We have a group loop and some initial/final calls
            # outside the loop.
            _load_q_impl(shard_seqlen_q_start, atp, bufs, q, batch_id, sbuf_addr)
            _qk_and_max_impl(shard_seqlen_q_start, ac, atp, sp, bufs)
            _update_max_impl(shard_seqlen_q_start, atp, sp, bufs)
            _exp_impl(shard_seqlen_q_start, ac, atp, sp, bufs)

            _load_q_impl(shard_seqlen_q_start + 1, atp, bufs, q, batch_id, sbuf_addr)
            _qk_and_max_impl(shard_seqlen_q_start + 1, ac, atp, sp, bufs)
            _update_max_impl(shard_seqlen_q_start + 1, atp, sp, bufs)
            shard_seqlen_q_end = shard_seqlen_q_start + shard_seqlen_q_length

            for grp_i in range(shard_seqlen_q_start, shard_seqlen_q_end - 2):  # for each block of seq_q
                # We try to perform software pipelining where the following operations are overlapped:
                # grp_i   :  PV, write_back
                # grp_i+1 :  EXP
                # grp_i+2 :  Load Q, QK+Max
                _load_q_impl(grp_i + 2, atp, bufs, q, batch_id, sbuf_addr)
                _exp_impl(grp_i + 1, ac, atp, sp, bufs)
                _fused_qkmax_and_pv_impl(grp_i, ac, atp, sp, bufs)
                _write_back_impl(grp_i, ac, atp, sp, bufs, o, batch_id)
                _update_max_impl(grp_i + 2, atp, sp, bufs)

            _pv_impl(shard_seqlen_q_end - 2, ac, atp, sp, bufs)

            _write_back_impl(shard_seqlen_q_end - 2, ac, atp, sp, bufs, o, batch_id)
            _exp_impl(shard_seqlen_q_end - 1, ac, atp, sp, bufs)
            _pv_impl(shard_seqlen_q_end - 1, ac, atp, sp, bufs)
            _write_back_impl(shard_seqlen_q_end - 1, ac, atp, sp, bufs, o, batch_id)


@dataclass
class AttnInternalBuffers(nl.NKIObject):
    """Container for all SBUF and PSUM tensor buffers used in attention computation."""

    # SBUF tensors to load q/k/v into
    q_sb = None
    k_sb = None
    v_sb = None

    # SBUF/PSUM tensors for computation

    # QK and max
    mm1_psum = None  # output of MM1 on PSUM
    mm1_copy_sb = None  # unused in the subset (attention_cte's causal-diagonal landing buffer)
    mm1_affine_select_output = None  # unused in the subset (attention_cte's affine_select output)
    mm1_masked = None  # Masked and scaled (if scale != 1.0) output after MM1 (in SBUF)
    mm1_partial_max = None  # tile-wise max after MM1
    mm1_section_max = None  # max for section after MM1
    mm1_running_max = None  # (persistent across sections) Running max across sections for output after MM1
    mm1_running_max_pos = None  # (persistent across sections) mm1_running_max un-negated, for the Vector-Engine exp
    prev_mm1_running_max = None  # Previous running max across sections for output after MM1 (used to hold value temporarily before section update)
    flash_attn_correction_factor = None  # Correction factor for flash attn (exp(prev_max-curr_max))

    # Exp
    exp_sb = None  # Output of exp
    exp_partial_sum = None  # Exp-sum per tile
    exp_section_sum = None  # Exp-sum for section
    exp_tp_sb = None  # Transposed output of Exp (input to MM2)
    exp_running_sum = None  # (persistent across sections) Running sum across sections after exp
    prev_exp_running_sum = (
        None  # Previous running max across sections after exp (used to hold value temporarily before section update)
    )
    exp_sum_reciprocal = None  # (persistent across sections) Reciprocal of exp-sum, calculated in the last section

    # PV
    mm2_psum = None  # output of MM2 on PSUM
    mm2_sb = None  # accumulate output of MM2 (mm2_psum) into SBUF
    mm2_prev_output = None  # output from previous section, loaded from HBM to SBUF
    mm2_accum_flash_attn = None  # Accumulated and scaled by flash_attn_correction_factor output of MM2 across sections
    mm2_final = None  # Output in final section, scaled by exp_sum_reciprocal

    # Shared/utility tensors
    zero_bias_tensor = None  # zeros, used for initialization
    range_sel_lbs = None  # lower bound for range_select (sequence packing)
    range_sel_ubs = None  # upper bound for range_select (sequence packing)

    # MM2 on nc_matmul_mx (ac.pv_mx); v_sb then holds V's x4 data per 512-key tile
    v_scale_sb = None  # V's MX scales per 512-key tile
    v_stage = None  # V rows as loaded, four per partition
    v_packed = None  # V regrouped into the group-of-4 layout, quantize_mx's source
    p_mx = None  # P's x4 data per (Q group, large tile, 512-key tile); exp_tp_sb is its fp32 stage
    p_scale = None  # P's MX scales, same indexing
    p_psum = None  # two [128, 256] fp32 halves of PSUM bank 7 for P's transposes

    # MM1 on nc_matmul_mx (ac.qk_mx); k_sb and q_sb are not allocated
    k_mx = None  # K's x4 data per large tile, its 512-key tile t in quadrant t
    k_scale = None  # K's MX scales, same indexing
    k_nat = None  # K rows as loaded per (large tile, 128-key chunk): the four tiles side by side
    k_stage = None  # K's swizzled fp32 stage between PSUM and quantize_mx, per chunk
    q_mx = None  # Q's x4 data per query group, repeated in the four quadrants
    q_scale = None  # Q's MX scales, same indexing
    q_nat = None  # Q rows as loaded, four copies side by side
    q_stage = None  # Q's swizzled fp32 stage
    qk_psum = None  # two [128, 256] fp32 halves of one PSUM bank for Q's and K's transposes


@dataclass
class SectionParams(nl.NKIObject):
    section_idx = None  # Index of section (controls flash attention accumulation)
    section_offset = None  # Offset of section


@dataclass
class AttnTileParams(nl.NKIObject):
    """Tile and buffer sizing parameters computed during implementation.

    Contains derived parameters specific to buffer allocation and tiling.
    """

    seqlen_k: int = None  # key length

    # Partition/tile sizes
    sb_p: int = None  # SBUF partition size (128)

    # Group parameters
    num_grps: int = None
    shard_grp_start: int = 0  # first Q group this shard owns (seqlen sharding shifts it off 0)
    shard_grp_count: int = None  # Q groups this shard owns; set to num_grps unless seqlen sharded
    num_q_grps_per_load: int = None  # load multiple q groups for better DMA efficiency
    can_pack_q_load: bool = None  # whether Q loads can be packed into num_q_grps_per_load

    # Tile counts per section
    num_large_tiles_per_section: int = None
    num_k_tiles_per_section: int = None
    num_v_tiles_per_section: int = None

    # Exp parameters
    exp_inst_elems: int = None  # exp tile size
    num_exp_insts_per_large_tile: int = None

    # Transpose and MM2 parameters
    # After transpose the scores are laid out as (128,4,128) which effectively stores 4 KxQ tiles
    # of 128x128 (recall K tile size is 512)
    num_tps_in_mm2_grp: int = None  # Number of transpose/MM2 per MM2 group (4)
    mm2_grp_sz: int = None  # Total free dim for MM2 group (4*128 = 512)

    # Masking with range_select: the key bounds (sequence packing)
    dynamic_sel_mask: bool = None
    mm2_accum_in_psum: bool = None  # accumulate MM2 across large tiles in PSUM, not SBUF
    # PSUM banks MM1 rotates its score tiles through. (0, 1, 2, 3) is the original
    # behaviour; six banks let MM1 run further ahead of the Vector evict.
    mm1_psum_banks: tuple = None
    qk_tp_bank: int = None  # qk_mx: the PSUM bank of Q's and K's transposes, outside MM1's

    # Section
    section_len: int = None  # Length of section, typically 8k if multiple sections, else same as k seqlen
    num_sections: int = None  # Number of sections


def _compute_tile_parameters(
    ac: AttnConfig,
    is_seqlen_sharded: bool,
) -> AttnTileParams:
    """
    Compute all tile and partition parameters for attention computation.

    Args:
      ac: High-level attention configuration
      is_seqlen_sharded: Whether Q sequence is sharded

    Returns:
      AttnTileParams: Complete tile parameter configuration
    """
    atp = AttnTileParams()
    atp.dynamic_sel_mask = ac.is_sequence_packed
    atp.seqlen_k = ac.seqlen_k_active

    # Partition size
    atp.sb_p = nl.tile_size.pmax
    # assert that _Q_GRP_SZ = _V_TILE_SZ = atp.sb_p (= 128) since that is an implict assumption in the code
    # and updating it requires careful updates.
    kernel_assert(
        _Q_GRP_SZ == atp.sb_p,
        f"Internal error: expect Q group size to match SBUF partition dimension, got {_Q_GRP_SZ=}, {atp.sb_p=}",
    )
    kernel_assert(
        _V_TILE_SZ == atp.sb_p,
        f"Internal error: expect V tile size to match SBUF partition dimension, got {_V_TILE_SZ=}, {atp.sb_p=}",
    )

    # Group configuration
    atp.num_grps = div_ceil(ac.seqlen_q, atp.sb_p)
    # Whole range by default; the seqlen-sharded path narrows these to the shard it is tracing.
    atp.shard_grp_start = 0
    atp.shard_grp_count = atp.num_grps
    atp.can_pack_q_load = not is_seqlen_sharded

    num_q_grps_per_load_dtype = 4 if ac.dtype == nl.float32 else 8  # fewer groups for float32 for SBUF memory
    atp.num_q_grps_per_load = min(num_q_grps_per_load_dtype if atp.can_pack_q_load else 1, atp.num_grps)
    kernel_assert(
        atp.num_q_grps_per_load > 0,
        f"num_q_grps_per_load must be positive, got {atp.num_q_grps_per_load}. "
        f"This occurs when num_grps={atp.num_grps} is 0 or negative. "
        f"Please check that batch_size, num_heads, and sequence length parameters are positive integers.",
    )

    total_seqlen_k = atp.seqlen_k
    d_v = ac.d_v

    # head dim D tile size (must be computed before section_len for SBUF scaling)
    atp.num_d_tiles = div_ceil(ac.d, _D_TILE_SZ)
    atp.d_tile_size_par_dim = min(ac.d, _D_TILE_SZ)
    atp.d_tile_size_free_dim = min(d_v, _V_D_TILE_SZ)
    atp.num_d_tiles_free_dim = div_ceil(d_v, _V_D_TILE_SZ)

    use_flash_attn = total_seqlen_k > (_FLASH_ATTENTION_THRESHOLD // atp.num_d_tiles)
    if use_flash_attn:
        scaled_section_len = _FLASH_ATTENTION_SECTION_LENGTH // atp.num_d_tiles
        atp.section_len = min(total_seqlen_k, scaled_section_len)
    else:
        atp.section_len = total_seqlen_k

    kernel_assert(atp.section_len > 0, f"section_len must be positive, got {atp.section_len}")
    atp.num_sections = div_ceil(total_seqlen_k, atp.section_len)

    if not use_flash_attn:
        kernel_assert(
            atp.num_sections == 1,
            "Logic fault, must only have 1 section if not using flash_attn",
        )

    # Tile counts per section
    atp.num_large_tiles_per_section = div_ceil(atp.section_len, _LARGE_TILE_SZ)
    atp.num_k_tiles_per_section = div_ceil(atp.section_len, _K_TILE_SZ)
    # pv_mx: V is loaded and quantized per 512-key tile, MM2's contraction on nc_matmul_mx
    atp.num_v_tiles_per_section = div_ceil(atp.section_len, _K_TILE_SZ if ac.pv_mx else _V_TILE_SZ)

    # K/V tile sizes for exp and transpose/MM2
    atp.exp_inst_elems = _EXP_TILE_SZ
    atp.num_exp_insts_per_large_tile = _LARGE_TILE_SZ // atp.exp_inst_elems
    atp.num_tps_in_mm2_grp = _K_TILE_SZ // atp.sb_p  # 512 // 128 = 4
    atp.mm2_grp_sz = _K_TILE_SZ

    # MM2 owns PSUM banks 4..7. Giving each d-tile its own bank -- instead of each large
    # tile -- lets every large tile of a Q group accumulate into the same bank, so the sum
    # over large tiles happens on the PE array and only the final result is evicted to
    # SBUF. Needs one bank per d-tile; anything wider keeps the per-large-tile banks and
    # the SBUF accumulation.
    num_mm2_d_tiles = atp.num_d_tiles_free_dim
    atp.mm2_accum_in_psum = num_mm2_d_tiles <= _psum_num_banks() - 4

    # Hand MM1 six PSUM banks instead of four (see _MM1_PSUM_BANKS_WIDE). Only when MM2
    # confines itself to bank 4 -- it accumulates in PSUM and needs exactly one d-tile bank -- and
    # all eight banks are usable. Every other shape keeps the original four banks.
    atp.mm1_psum_banks = (
        _MM1_PSUM_BANKS_WIDE
        if (atp.mm2_accum_in_psum and num_mm2_d_tiles == 1 and _psum_num_banks() == 8)
        else tuple(range(_NUM_K_TILES_IN_LARGE_TILE))
    )
    if ac.qk_mx:
        # Q's and K's transposes take bank 7, or with pv_mx (P's transposes hold 7) bank 6 from MM1,
        # which keeps five: any four consecutive tiles of the rotation still land on distinct banks.
        atp.qk_tp_bank = 6 if ac.pv_mx else 7
        atp.mm1_psum_banks = tuple(b for b in atp.mm1_psum_banks if b != atp.qk_tp_bank)
        kernel_assert(len(atp.mm1_psum_banks) >= _NUM_K_TILES_IN_LARGE_TILE and 4 not in atp.mm1_psum_banks,
                      f"qk_mx: a large tile's four score tiles need four MM1 banks, got {atp.mm1_psum_banks}")

    return atp


def _setup_range_select_bounds(
    ac: AttnConfig,
    atp: AttnTileParams,
    bufs: AttnInternalBuffers,
    allocator: ModularAllocator,
    bound_min: Any,
    bound_max: Any,
    batch_id: int = 0,
):
    """The key bounds' range_select operands (sequence packing): ``range_sel_lbs`` = bound_min
    and ``range_sel_ubs`` = min(seqlen_k, bound_max) per query, ``[128, num_grps]`` fp32. Nothing
    without bounds."""
    if not atp.dynamic_sel_mask:
        return
    bufs.range_sel_ubs = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
    nisa.iota(
        bufs.range_sel_ubs[...],
        pattern=[[atp.sb_p, atp.num_grps]],
        channel_multiplier=0,
        offset=ac.seqlen_k_active,
    )
    bufs.range_sel_lbs = allocator.alloc_sbuf_tensor(shape=bufs.range_sel_ubs.shape, dtype=bufs.range_sel_ubs.dtype)
    local_allocator = ModularAllocator(allocator._current_address)
    tmp_buffer = local_allocator.alloc_sbuf_tensor(shape=bufs.range_sel_ubs.shape, dtype=bufs.range_sel_ubs.dtype)
    bound_offset = batch_id * atp.sb_p * atp.num_grps
    nisa.dma_copy(
        dst=bufs.range_sel_lbs[...],
        src=bound_min.ap(pattern=[[1, atp.sb_p], [atp.sb_p, atp.num_grps]], offset=bound_offset),
    )
    nisa.dma_copy(
        dst=tmp_buffer[...],
        src=bound_max.ap(pattern=[[1, atp.sb_p], [atp.sb_p, atp.num_grps]], offset=bound_offset),
    )
    nisa.tensor_tensor(dst=bufs.range_sel_ubs, data1=bufs.range_sel_ubs, data2=tmp_buffer, op=nl.minimum)


def _allocate_attention_buffers(
    allocator: ModularAllocator,
    ac: AttnConfig,
    atp: AttnTileParams,
    bufs: AttnInternalBuffers,
):
    """
    Allocate all SBUF and PSUM buffers needed for attention computation.

    Modifies bufs in-place by allocating and assigning all computation buffers.

    We use the modular allocator with num_free_tiles chosen in order to achieve
    multi-buffering and avoid anti-dependencies. The degree of multi-buffering
    along the Q group/KV tile axis is chosen based on experimentation.
    """

    # Define the partition and free dimension for the two matmuls
    mm1_p, mm1_n = atp.sb_p, nl.tile_size.psum_fmax
    mm2_p, mm2_n = atp.sb_p, atp.d_tile_size_free_dim

    p_k, n_k = atp.d_tile_size_par_dim, _K_TILE_SZ  # d is reduction dim for MM1
    if ac.qk_mx:
        # the section's K, quantized once per large tile and shared by every query group
        nlt = atp.num_large_tiles_per_section
        k_free = _LARGE_TILE_SZ // 4  # x4 elements: a 512-key tile per quadrant
        bufs.k_mx = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, k_free), dtype=nl.float8_e4m3fn_x4, block_dim=[nlt], num_free_tiles=[nlt]
        )
        bufs.k_scale = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, k_free), dtype=nl.uint8, block_dim=[nlt], num_free_tiles=[nlt]
        )
        bufs.k_nat = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, 4 * ac.d), dtype=nl.bfloat16, block_dim=[nlt, 4], num_free_tiles=[1, 2]
        )
        bufs.k_stage = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, 2 * atp.sb_p), dtype=nl.float32, block_dim=[nlt, 4], num_free_tiles=[1, 2]
        )
    else:
        bufs.k_sb = allocator.alloc_sbuf_tensor(
            shape=(p_k, n_k),
            dtype=nl.bfloat16,
            block_dim=[atp.num_k_tiles_per_section, atp.num_d_tiles],
            num_free_tiles=[atp.num_k_tiles_per_section, atp.num_d_tiles],
            align_to=32,  # align for dma transpose
        )

    if ac.pv_mx:
        v_tiles = atp.num_v_tiles_per_section
        bufs.v_sb = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, ac.d_v), dtype=nl.float8_e4m3fn_x4, block_dim=[v_tiles], num_free_tiles=[v_tiles]
        )
        bufs.v_scale_sb = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, ac.d_v), dtype=nl.uint8, block_dim=[v_tiles], num_free_tiles=[v_tiles]
        )
        bufs.v_stage = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, 4 * ac.d_v), dtype=nl.bfloat16, block_dim=[v_tiles], num_free_tiles=[2]
        )
        bufs.v_packed = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, 4 * ac.d_v), dtype=nl.bfloat16, block_dim=[v_tiles], num_free_tiles=[2]
        )
    else:
        p_v, n_v = atp.sb_p, atp.d_tile_size_free_dim  # d is free dim for MM2
        bufs.v_sb = allocator.alloc_sbuf_tensor(
            shape=(p_v, n_v),
            dtype=nl.bfloat16,
            block_dim=[atp.num_v_tiles_per_section, atp.num_d_tiles_free_dim],
            num_free_tiles=[atp.num_v_tiles_per_section, atp.num_d_tiles_free_dim],
        )

    if ac.qk_mx:
        # one query group at a time; its stationary keeps all 128 columns (row tiling needs > 64)
        q_blocks, q_free = [atp.num_grps], [2]
        bufs.q_nat = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, 4 * ac.d), dtype=nl.bfloat16, block_dim=q_blocks, num_free_tiles=q_free
        )
        bufs.q_stage = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, 2 * atp.sb_p), dtype=nl.float32, block_dim=q_blocks, num_free_tiles=q_free
        )
        bufs.q_mx = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, _Q_GRP_SZ), dtype=nl.float8_e4m3fn_x4, block_dim=q_blocks, num_free_tiles=q_free
        )
        bufs.q_scale = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, _Q_GRP_SZ), dtype=nl.uint8, block_dim=q_blocks, num_free_tiles=q_free
        )
        bufs.qk_psum = [
            nl.ndarray(
                (atp.sb_p, 2 * atp.sb_p),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, atp.qk_tp_bank * PSUM_BANK_SIZE + half * PSUM_BANK_SIZE // 2),
            )
            for half in range(2)
        ]
    else:
        bufs.q_sb = allocator.alloc_sbuf_tensor(
            shape=(atp.d_tile_size_par_dim, atp.sb_p * atp.num_q_grps_per_load),
            dtype=nl.bfloat16,
            block_dim=[div_ceil(atp.num_grps, atp.num_q_grps_per_load), atp.num_d_tiles],
            num_free_tiles=[2, atp.num_d_tiles],
            align_to=32,  # align for dma transpose
        )

    bufs.flash_attn_correction_factor = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    # buffer to hold the partial row-wise maximum from mm1
    bufs.mm1_partial_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.num_k_tiles_per_section),
        dtype=nl.float32,
        block_dim=[_MM1_PARTIAL_MAX_STAGES],
        num_free_tiles=[_MM1_PARTIAL_MAX_STAGES],
        align_to=4,
    )

    bufs.mm1_section_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.exp_partial_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, div_ceil(atp.section_len, atp.exp_inst_elems)),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.exp_section_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.prev_mm1_running_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.prev_exp_running_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1),
        dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.mm2_prev_output = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.d_tile_size_free_dim),
        dtype=nl.float32,
        block_dim=[atp.num_grps, atp.num_d_tiles_free_dim],
        num_free_tiles=[2, atp.num_d_tiles_free_dim],
    )

    bufs.mm2_accum_flash_attn = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.d_tile_size_free_dim),
        dtype=nl.float32,
        block_dim=[atp.num_grps, atp.num_d_tiles_free_dim],
        num_free_tiles=[2, atp.num_d_tiles_free_dim],
    )

    bufs.mm2_final = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.d_tile_size_free_dim),
        dtype=nl.float32,
        block_dim=[atp.num_grps, atp.num_d_tiles_free_dim],
        num_free_tiles=[2, atp.num_d_tiles_free_dim],
    )

    bufs.mm2_sb = allocator.alloc_sbuf_tensor(
        shape=(mm2_p, atp.d_tile_size_free_dim),
        dtype=nl.float32,
        block_dim=[atp.num_grps, atp.num_d_tiles_free_dim],
        num_free_tiles=[2, atp.num_d_tiles_free_dim],
    )

    bufs.mm1_masked = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ),
        dtype=nl.float32,
        block_dim=[atp.num_grps, atp.num_large_tiles_per_section],
        num_free_tiles=[2, atp.num_large_tiles_per_section],
    )

    # mm1_psum PSUM allocation
    bufs.mm1_psum = []
    for grp_idx in range(atp.num_grps):
        grp_row = []
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            tile_row = []
            for k_tile_idx in range(4):
                mm1_psum_tile = nl.ndarray(
                    (mm1_p, mm1_n),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    # Key the bank off the tile's position in the whole section, not off
                    # k_tile_idx alone, so consecutive large tiles land on different banks and MM1
                    # can run ahead of the evict instead of waiting for bank 0 to be freed.
                    address=(
                        0,
                        atp.mm1_psum_banks[
                            (large_tile_idx * _NUM_K_TILES_IN_LARGE_TILE + k_tile_idx) % len(atp.mm1_psum_banks)
                        ]
                        * PSUM_BANK_SIZE,
                    ),
                )
                tile_row.append(mm1_psum_tile)
            grp_row.append(tile_row)
        bufs.mm1_psum.append(grp_row)

    # attention_cte's causal-diagonal buffers: dead in the subset, still allocated on the unmasked
    # path so the SBUF layout (and the exp_sb staging decided below) stays attention_cte's.
    # shortcut: freeing them may let exp_sb take a second stage; try it with a timed probe.
    if not atp.dynamic_sel_mask:
        bufs.mm1_copy_sb = allocator.alloc_sbuf_tensor(
            shape=(mm1_p, mm1_n),
            dtype=nl.float32,
            block_dim=[atp.num_grps, atp.num_large_tiles_per_section, 4],
            num_free_tiles=[1, 1, 2],
        )

        bufs.mm1_affine_select_output = allocator.alloc_sbuf_tensor(
            shape=(mm1_p, mm1_n),
            dtype=nl.float32,
            block_dim=[atp.num_grps, atp.num_large_tiles_per_section, 4],
            num_free_tiles=[1, 1, 2],
        )

    bufs.exp_tp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.mm2_grp_sz),
        dtype=nl.bfloat16,
        block_dim=[
            atp.num_grps,
            atp.num_large_tiles_per_section,
            atp.num_tps_in_mm2_grp,
        ],
        num_free_tiles=[2, atp.num_large_tiles_per_section, atp.num_tps_in_mm2_grp],
        align_to=32,  # align for dma transpose
    )

    if ac.pv_mx:
        # exp_tp_sb ([128, 512] bf16 = [128, 256] fp32) is P's stage between PSUM and quantize_mx
        p_blocks = [atp.num_grps, atp.num_large_tiles_per_section, atp.num_tps_in_mm2_grp]
        p_free = [2, atp.num_large_tiles_per_section, atp.num_tps_in_mm2_grp]
        bufs.p_mx = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, atp.sb_p), dtype=nl.float8_e4m3fn_x4, block_dim=p_blocks, num_free_tiles=p_free
        )
        bufs.p_scale = allocator.alloc_sbuf_tensor(
            shape=(atp.sb_p, atp.sb_p), dtype=nl.uint8, block_dim=p_blocks, num_free_tiles=p_free
        )
        bufs.p_psum = [
            nl.ndarray(
                (atp.sb_p, 2 * atp.sb_p),
                dtype=nl.float32,
                buffer=nl.psum,
                address=(0, 7 * PSUM_BANK_SIZE + half * PSUM_BANK_SIZE // 2),
            )
            for half in range(2)
        ]

    # mm2_psum allocation. MM2 uses banks 4.. (MM1 owns 0..3 above).
    #
    # When atp.mm2_accum_in_psum, the bank is chosen by d-tile and every large tile of a Q
    # group is handed the same tiles, so the MM2 matmuls accumulate across large tiles in
    # place and _pv_large_tile_impl evicts once. Otherwise the bank is chosen by large tile and
    # each large tile is accumulated into SBUF on the Vector Engine.
    bufs.mm2_psum = []
    for grp_idx in range(atp.num_grps):
        grp_row = []
        shared_tiles = None
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            if shared_tiles is not None:
                grp_row.append(shared_tiles)
                continue
            tmp = []
            for d_idx in range(atp.num_d_tiles_free_dim):
                bank_idx = d_idx if atp.mm2_accum_in_psum else large_tile_idx
                mm2_psum_tile = nl.ndarray(
                    (mm2_p, mm2_n),
                    dtype=nl.float32,
                    buffer=nl.psum,
                    address=(0, ((4 + (bank_idx % (_psum_num_banks() - 4))) * PSUM_BANK_SIZE)),
                )
                tmp.append(mm2_psum_tile)
            grp_row.append(tmp)
            if atp.mm2_accum_in_psum:
                shared_tiles = tmp
        bufs.mm2_psum.append(grp_row)

    # Allocated last so the second stage -- the exp run-ahead over the next Q group -- is taken
    # only when the partition has room, since it does not fit a whole non-flash section at d=128.
    exp_sb_stage_bytes = atp.num_large_tiles_per_section * _LARGE_TILE_SZ * _BF16_ITEMSIZE
    free_bytes = _sbuf_scratch_budget() - allocator.get_current_address()
    exp_sb_grp_stages = 2 if free_bytes >= 2 * exp_sb_stage_bytes else 1
    bufs.exp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ),
        dtype=nl.bfloat16,
        block_dim=[atp.num_grps, atp.num_large_tiles_per_section],
        num_free_tiles=[exp_sb_grp_stages, atp.num_large_tiles_per_section],
    )


def _check_input_and_return_shape(q, k, v) -> tuple:
    """Validate input tensor shapes (bf16, the tp_q / tp_k layout) and compute the output shape.

    Returns (seqlen_q, seqlen_k, d, d_v, out_shape).
    """
    kernel_assert(str(q.dtype) == str(nl.bfloat16) and str(k.dtype) == str(nl.bfloat16),
                  f"Difflet's attention_cte fork takes bf16 q / k only, got {q.dtype}, {k.dtype}")
    batch_size, seqlen_q, d = q.shape
    batch_size_kv, seqlen_k = k.shape[0], k.shape[1]
    assert_shape(k, (batch_size_kv, seqlen_k, d), "k")
    d_v = v.shape[2]
    kernel_assert(
        d_v <= _MAX_HEAD_DIM,
        f"we do not support v head_dim > {_MAX_HEAD_DIM}, got v head dim {d_v}",
    )
    assert_shape(v, (batch_size_kv, seqlen_k, d_v), "v")
    return (seqlen_q, seqlen_k, d, d_v, (batch_size, seqlen_q, d_v))


def _load_q_tile(q, out, batch_id, grp_i, seqlen_offset, sbuf_addr, grps_per_load=1):
    """Load Q tile from HBM to SBUF transposed (q = (bs, seqlen, d)).

    out[grp_i // grps_per_load][d_tile] has shape (d_tile_size, 128 * grps_per_load)
    """
    local_allocator = ModularAllocator(initial_address=sbuf_addr)
    _, seqlen, actual_d = q.shape
    d_tile_size = out[0][0].shape[0]
    num_d_tiles = len(out[0])

    for d_tile in range(num_d_tiles):
        d = min(d_tile_size, actual_d - d_tile * d_tile_size)
        d_offset = d_tile * d_tile_size
        dst_buf = out[grp_i // grps_per_load][d_tile]

        if grps_per_load > 1:
            # Use DMA transpose
            num_p = min(seqlen - seqlen_offset, _Q_GRP_SZ * grps_per_load)
            nisa.dma_transpose(
                dst=dst_buf.ap([[_Q_GRP_SZ * grps_per_load, d], [1, 1], [1, 1], [1, num_p]]),
                src=q.ap(
                    [[actual_d, num_p], [1, 1], [1, 1], [1, d]],
                    offset=batch_id * seqlen * actual_d + seqlen_offset * actual_d + d_offset,
                ),
            )
        else:
            # Use NC transpose
            loaded = local_allocator.alloc_sbuf_tensor(shape=(_Q_GRP_SZ, d), dtype=nl.bfloat16)
            psum_buf = nl.ndarray((d, _Q_GRP_SZ), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0))

            num_p = min(seqlen - seqlen_offset, _Q_GRP_SZ)
            # q shape: (bs, seqlen, actual_d), loading q[batch_id, seqlen_offset:seqlen_offset+num_p, d_offset:d_offset+d]
            # Pattern stride: actual_d between seqlen positions, d elements per position
            # Offset: batch_id*seqlen*actual_d + seqlen_offset*actual_d + d_offset
            loaded_dst_pat = loaded.ap(pattern=[[d, num_p], [1, d]], offset=0)
            q_src_pat = q.ap(
                pattern=[[actual_d, num_p], [1, d]],
                offset=batch_id * seqlen * actual_d + seqlen_offset * actual_d + d_offset,
            )
            nisa.dma_copy(dst=loaded_dst_pat, src=q_src_pat)

            nisa.nc_transpose(psum_buf[:d, :num_p], loaded[:num_p, :d])
            num_f = min(seqlen - seqlen_offset, _Q_GRP_SZ)
            nisa.tensor_copy(dst_buf[:d, :num_f], psum_buf[:d, :num_f])


def _load_k_tile(k, out, batch_id, sp: SectionParams, num_tiles):
    """Load K tiles from HBM to SBUF in _K_TILE_SZ (512)-element chunks by DMA transpose.

    k has shape (bs, seqlen, d). Return shape of out[i] is (d, _K_TILE_SIZE) where
    i = 0..num_k_tiles_per_section
    """
    _, seqlen, actual_d = k.shape
    if num_tiles > 0:
        d_tile_size, n = out[0][0].shape
        num_d_tiles = len(out[0])

    kernel_assert(n == _K_TILE_SZ, f"expect to load in tile of size {_K_TILE_SZ=}")
    for tile in range(num_tiles):
        seqlen_offset = sp.section_offset + tile * n
        if seqlen_offset >= seqlen:
            # since we always use section_len/512 tiles even for last section
            # we might exit early
            return
        for d_tile in range(num_d_tiles):
            d = min(d_tile_size, actual_d - d_tile * d_tile_size)
            d_offset = d_tile * d_tile_size
            dst_buf = out[tile][d_tile]
            num_p = min(seqlen - seqlen_offset, n)
            nisa.dma_transpose(
                dst=dst_buf.ap([[n, d], [1, 1], [1, 1], [1, num_p]]),
                src=k.ap(
                    [[actual_d, num_p], [1, 1], [1, 1], [1, d]],
                    offset=batch_id * seqlen * actual_d + seqlen_offset * actual_d + d_offset,
                ),
            )


def _load_v_tile(v, out, batch_id, sp: SectionParams, num_tiles):
    """Load V tiles from HBM to SBUF in _V_TILE_SZ (128)-element chunks.

    - v of shape (bs, seqlen, d).
    - out[i] has shape (_V_TILE_SZ, d) where i = 0..num_v_tiles_per_section
    """
    _, seqlen, actual_d = v.shape
    if num_tiles > 0:
        p, d_tile_size = out[0][0].shape
        num_d_tiles = len(out[0])

    for tile in range(num_tiles):
        seqlen_offset = sp.section_offset + p * tile
        num_p = min(seqlen - seqlen_offset, p)
        if num_p > 0:
            for d_tile in range(num_d_tiles):
                d = min(d_tile_size, actual_d - d_tile * d_tile_size)
                d_offset = d_tile * d_tile_size
                out_dst_pat = out[tile][d_tile].ap(pattern=[[d_tile_size, num_p], [1, d]], offset=0)
                v_src_pat = v.ap(
                    pattern=[[actual_d, num_p], [1, d]],
                    offset=batch_id * seqlen * actual_d + seqlen_offset * actual_d + d_offset,
                )
                nisa.dma_copy(dst=out_dst_pat, src=v_src_pat)


def _load_v_tile_mx(v, bufs, batch_id, sp: SectionParams, atp: AttnTileParams):
    """pv_mx: the section's V, quantized per 512-key tile on load (``_mx_load_v``); the last
    tile's missing keys are zero."""
    _, seqlen, _ = v.shape
    for tile in range(atp.num_v_tiles_per_section):
        start = sp.section_offset + tile * _K_TILE_SZ
        if start < seqlen:
            _mx_load_v(
                v,
                bufs.v_sb[tile],
                bufs.v_scale_sb[tile],
                bufs.v_stage[tile],
                bufs.v_packed[tile],
                row_offset=batch_id * seqlen + start,
                num_keys=min(seqlen - start, _K_TILE_SZ),
            )


def _load_k_tile_mx(k, bufs, batch_id, sp: SectionParams, atp: AttnTileParams):
    """qk_mx: the section's K, quantized per large tile on load (``_mx_load_k``); the last tile's
    missing keys are zero."""
    _, seqlen, _ = k.shape
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        start = sp.section_offset + large_tile_idx * _LARGE_TILE_SZ
        if start < seqlen:
            _mx_load_k(
                k,
                bufs.k_mx[large_tile_idx],
                bufs.k_scale[large_tile_idx],
                bufs.qk_psum,
                bufs.k_stage[large_tile_idx],
                bufs.k_nat[large_tile_idx],
                row_offset=batch_id * seqlen + start,
                num_keys=min(seqlen - start, _LARGE_TILE_SZ),
            )


def _load_q_impl(grp_i, atp: AttnTileParams, bufs: AttnInternalBuffers, q, batch_id, sbuf_addr):
    """Load Q group from HBM to SBUF, every num_q_grps_per_load groups; with qk_mx, quantize
    every group on its own (``_mx_load_q``)."""
    if bufs.q_mx is not None:
        _, seqlen, _ = q.shape
        start = grp_i * _Q_GRP_SZ
        _mx_load_q(q, bufs.q_mx[grp_i], bufs.q_scale[grp_i], bufs.qk_psum[grp_i % 2], bufs.q_stage[grp_i],
                   bufs.q_nat[grp_i], row_offset=batch_id * seqlen + start,
                   num_rows=min(seqlen - start, _Q_GRP_SZ))
        return
    if grp_i % atp.num_q_grps_per_load == 0:
        _load_q_tile(q, bufs.q_sb, batch_id, grp_i, grp_i * _Q_GRP_SZ, sbuf_addr,
                     grps_per_load=atp.num_q_grps_per_load)


def _qk_and_max_impl(grp_i, ac: AttnConfig, atp: AttnTileParams, sp: SectionParams, bufs: AttnInternalBuffers):
    """Compute QK^T matmul (MM1) and find row-wise maximum for this Q group, applying the key
    bounds if present."""
    nisa.memset(
        bufs.mm1_partial_max[grp_i % _MM1_PARTIAL_MAX_STAGES][:, : atp.num_k_tiles_per_section],
        value=_FLOAT32_MIN,
    )

    for large_tile_idx in range(atp.num_large_tiles_per_section):
        _qk_and_max_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)


def _update_max_impl(grp_i, atp: AttnTileParams, sp: SectionParams, bufs: AttnInternalBuffers):
    """Update running maximum across sections and compute flash attention correction factor."""
    # Step 1: Compute section max.
    #
    # On the first section mm1_section_max and mm1_running_max hold the same value, so reduce
    # straight into the running max and skip a GpSimd copy between this reduce and the exp that
    # consumes the max. Only section_idx > 0 needs the two kept apart, because there the section
    # max is combined with the surviving running max.
    reduce_into_running = sp.section_idx == 0 or atp.num_sections == 1
    nisa.tensor_reduce(
        bufs.mm1_running_max[:, grp_i] if reduce_into_running else bufs.mm1_section_max[grp_i][:, 0],
        nl.maximum,
        bufs.mm1_partial_max[grp_i % _MM1_PARTIAL_MAX_STAGES],
        1,
        negate=True,
    )

    # Step 2: compute and store running max, and flash attention correction factor
    if atp.num_sections != 1:
        if sp.section_idx == 0:
            nisa.memset(bufs.flash_attn_correction_factor[grp_i][...], value=0.0)
        if sp.section_idx > 0:
            nisa.activation(
                bufs.prev_mm1_running_max[grp_i][...],
                nl.copy,
                bufs.mm1_running_max[:, grp_i],
                scale=-1.0,
                bias=bufs.zero_bias_tensor,
            )
            nisa.tensor_tensor(
                bufs.mm1_running_max[:, grp_i],
                bufs.mm1_running_max[:, grp_i],
                bufs.mm1_section_max[grp_i],
                op=nl.minimum,
            )
            nisa.activation(
                bufs.flash_attn_correction_factor[grp_i][:, 0],
                nl.exp,
                bufs.prev_mm1_running_max[grp_i],
                bias=bufs.mm1_running_max[:, grp_i],
                scale=1.0,
            )


@dataclass
class KTileInfo(nl.NKIObject):
    """Trace-time position of one K tile (shared by the QK and exp passes)."""

    seqlen_k: int  # key length
    k_start_pos: int  # start of the tile
    unmasked: bool  # tile starts within the keys
    num_valid_k: int = 0  # valid keys from the tile start to the end of the keys

    def __post_init__(self):
        # Field, not @property: NKI eager/nki_hop lowering does not fold property getters.
        self.num_valid_k = self.seqlen_k - self.k_start_pos


def _classify_tile(seqlen_offset, atp: AttnTileParams):
    """A K tile starting at key ``seqlen_offset``: has keys when it starts before the end."""
    return KTileInfo(atp.seqlen_k, seqlen_offset, seqlen_offset < atp.seqlen_k)


def _exp_tile_selection(large_tile_idx, exp_tile_idx, atp: AttnTileParams, sp: SectionParams):
    """Decide whether one exp tile is processed, and how many valid keys it has.

    Returns (should_process_tile, num_valid_k_remaining):
      - should_process_tile: False when the tile starts past the end of the keys, True otherwise.
      - num_valid_k_remaining: seqlen_k - k_start_pos, the count of valid keys from the tile start
        to the end. The caller caps it to the exp tile width for num_f.
    """
    seqlen_offset = sp.section_offset + large_tile_idx * _LARGE_TILE_SZ + exp_tile_idx * atp.exp_inst_elems
    tile_info = _classify_tile(seqlen_offset, atp)
    return tile_info.unmasked, tile_info.num_valid_k


def _exp_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Compute exp(QK - row max) for one Q group for one k-v section, accumulate the softmax row-sum, and
    transpose the result for MM2 (the PV matmul), or with ``pv_mx`` quantize it for MM2.

    Args:
        grp_i: Q-group index (which 128-query block to process).
        ac (AttnConfig): attention config  (shapes, scale ...)
        atp (AttnTileParams): derived tiling/sizing parameters (number Q groups , num tiles ...).
        sp (SectionParams): current flash-attention section and its offsets.
        bufs (AttnInternalBuffers): SBUF and PSUM working buffers; reads mm1_masked and
            mm1_running_max, writes exp_sb / exp_tp_sb / exp_partial_sum.
    """
    q_seqlen_offset = grp_i * atp.sb_p
    num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
    nisa.memset(bufs.exp_partial_sum[grp_i][...], value=0.0, engine=nisa.gpsimd_engine)

    # Exp tiles must stay the width of the K tiles the masking wrote.
    kernel_assert(
        atp.exp_inst_elems == _K_TILE_SZ,
        f"exp tile size must match the K tile size, got {atp.exp_inst_elems=}, {_K_TILE_SZ=}",
    )

    # trn3 B1 has the Vector-Engine nisa.exponential, which subtracts max_value directly, whereas
    # nisa.activation on the Scalar Engine adds a bias. mm1_running_max is stored pre-negated
    # (mm1_section_max is reduced with negate=True), so the Scalar path uses it as-is and the Vector
    # path needs it flipped back to a true max. Hoisted out of the tile loop: it is per Q group.
    use_vector_exp = bufs.mm1_running_max_pos is not None
    if use_vector_exp:
        # Per Q group, not per tile: mm1_running_max[:, grp_i] is settled before this group's exp runs.
        exp_max_tile = bufs.mm1_running_max_pos[:num_p, grp_i : grp_i + 1]
        nisa.tensor_scalar(
            exp_max_tile,
            bufs.mm1_running_max[:num_p, grp_i : grp_i + 1],
            op0=nl.multiply,
            operand0=-1.0,
        )

    # Precompute each exp tile's selection once (shared by the loop below) and chain the fused
    # sum: reset on the first executed tile, accumulate on the rest, single read-out on the last.
    exp_selections = []
    exp_first_tile = None
    exp_last_tile = None
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        for exp_tile_idx in range(atp.num_exp_insts_per_large_tile):
            should_process_tile, num_valid_k_remaining = _exp_tile_selection(large_tile_idx, exp_tile_idx, atp, sp)
            exp_selections.append((should_process_tile, num_valid_k_remaining))
            if should_process_tile:
                current_exp_tile = (large_tile_idx, exp_tile_idx)
                exp_first_tile = exp_first_tile or current_exp_tile
                exp_last_tile = current_exp_tile

    # nisa.exponential only reaches the 2x/4x modes its fused accumulator requires on an even
    # element count, and the Vector and Scalar accumulators are distinct hardware, so the chain
    # cannot switch engines partway: one odd tile sends this whole group to the Scalar fallback.
    if use_vector_exp and any(
        min(num_valid_k, atp.exp_inst_elems) % 2 for selected, num_valid_k in exp_selections if selected
    ):
        use_vector_exp = False

    for large_tile_idx in range(atp.num_large_tiles_per_section):
        for exp_tile_idx in range(atp.num_exp_insts_per_large_tile):
            should_process_tile, num_valid_k_remaining = exp_selections[
                large_tile_idx * atp.num_exp_insts_per_large_tile + exp_tile_idx
            ]

            if not should_process_tile:
                continue
            num_f = min(num_valid_k_remaining, atp.exp_inst_elems)
            # Step 1: Compute exponential — chained sum (reset on the first executed
            # tile, accumulate on the rest, single read-out on the last).
            is_first_exp = (large_tile_idx, exp_tile_idx) == exp_first_tile
            is_last_exp = (large_tile_idx, exp_tile_idx) == exp_last_tile
            exp_reduce_res = (
                bufs.exp_partial_sum[grp_i][
                    :num_p,
                    large_tile_idx * atp.num_exp_insts_per_large_tile + exp_tile_idx,
                ]
                if is_last_exp
                else None
            )
            exp_reduce_cmd = nisa.reduce_cmd.reset_reduce if is_first_exp else nisa.reduce_cmd.reduce
            if use_vector_exp:
                nisa.exponential(
                    dst=bufs.exp_sb[grp_i][large_tile_idx][:num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)],
                    src=bufs.mm1_masked[grp_i][large_tile_idx][:num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)],
                    max_value=exp_max_tile,
                    reduce_res=exp_reduce_res,
                    reduce_cmd=exp_reduce_cmd,
                )
            else:
                nisa.activation(
                    bufs.exp_sb[grp_i][large_tile_idx][:num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)],
                    op=nl.exp,
                    data=bufs.mm1_masked[grp_i][large_tile_idx][
                        :num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)
                    ],
                    reduce_op=nl.add,
                    reduce_cmd=exp_reduce_cmd,
                    reduce_res=exp_reduce_res,
                    bias=bufs.mm1_running_max[:num_p, grp_i],
                )

            if ac.pv_mx:  # Step 2 (pv_mx): P's MX operand for this 512-key tile
                _mx_quantize_p(
                    bufs.p_mx[grp_i][large_tile_idx][exp_tile_idx],
                    bufs.p_scale[grp_i][large_tile_idx][exp_tile_idx],
                    bufs.p_psum[exp_tile_idx % 2],
                    bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].view(nl.float32),
                    bufs.exp_sb[grp_i][large_tile_idx],
                    src_offset=exp_tile_idx * atp.exp_inst_elems,
                    num_q=num_p,
                    num_keys=num_f,
                )
                continue

            # Step 2: Perform DMA transpose
            num_f_outer = num_f // atp.sb_p
            num_f_inner = num_f % atp.sb_p
            # split dma_transpose into two parts to satisfy API since we have both Q and K sequence masking
            # Focusing on exp_tp_sb which is arranged as [128, 4, 128] where each of the 4 [128, 128] blocks
            # share the same Q seqlen (on free dim) and cover 4 tiles of K seqlen (partition dim)
            # First region, we have num_f_outer [128, 128] blocks each having full partition dim (K) and each
            # accessing num_p (<128) on the free dim (Q).
            # Second region, we handle the remaining K (num_f_inner) - here we have the (num_f_outer+1)th [128,128]
            # block being utilized with num_f_inner access on partition dim and num_p on the free dim.

            # Example: num_f_outer = 3, num_f_inner = 33, num_p = 100
            # Region 1: AP: [[512, 128], [128, 3], [1, 100]] => a, b, c = np.mgrid[0:128, 0:3, 0:100]
            # Region 2: AP: [[512, 33], [128, 1], [1, 100]]  => a, b, c = np.mgrid[0:33, 0:1, 0:100] with offset 128 * 3

            # NOTE: we add the [1,1] because we need 4 dims for dma_transpose

            # Case 1: handle 0:128x
            if num_f_outer >= 1:
                nisa.dma_transpose(
                    dst=bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                        [
                            [atp.mm2_grp_sz, atp.sb_p],
                            [1, 1],
                            [atp.sb_p, num_f_outer],
                            [1, num_p],
                        ]
                    ),
                    src=bufs.exp_sb[grp_i][large_tile_idx].ap(
                        [
                            [_LARGE_TILE_SZ, num_p],
                            [1, 1],
                            [atp.sb_p, num_f_outer],
                            [1, atp.sb_p],
                        ],
                        offset=exp_tile_idx * atp.mm2_grp_sz,
                    ),
                )

            # Case 2: handle num_f - 128x
            if num_f_inner > 0:
                nisa.dma_transpose(
                    dst=bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                        [
                            [atp.mm2_grp_sz, num_f_inner],
                            [1, 1],
                            [atp.sb_p, 1],
                            [1, num_p],
                        ],
                        offset=num_f_outer * atp.sb_p,
                    ),
                    src=bufs.exp_sb[grp_i][large_tile_idx].ap(
                        [
                            [_LARGE_TILE_SZ, num_p],
                            [1, 1],
                            [atp.sb_p, 1],
                            [1, num_f_inner],
                        ],
                        offset=exp_tile_idx * atp.mm2_grp_sz + num_f_outer * atp.sb_p,
                    ),
                )


def _pv_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Compute score@value matmul (P@V, MM2) for this Q group."""
    for _d_tile in range(atp.num_d_tiles_free_dim):
        nisa.memset(bufs.mm2_sb[grp_i][_d_tile][...], value=0.0)

    for large_tile_idx in range(atp.num_large_tiles_per_section):
        _pv_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)


def _fused_qkmax_and_pv_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Fused implementation computing QK+max for group i+2 while computing PV for group i (software pipelining)."""
    qkmax_grp = grp_i + 2

    nisa.memset(
        bufs.mm1_partial_max[qkmax_grp % _MM1_PARTIAL_MAX_STAGES][:, : atp.num_k_tiles_per_section],
        value=_FLOAT32_MIN,
        engine=nisa.gpsimd_engine,
    )

    for large_tile_idx in range(atp.num_large_tiles_per_section):
        _pv_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)
        _qk_and_max_large_tile_impl(qkmax_grp, large_tile_idx, ac, atp, sp, bufs)


def _write_back_impl(
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
    o,
    batch_id,
):
    """Finalize output using flash attention: accumulate across sections, apply softmax normalization, write to HBM."""
    # Step 1: Compute/update exp-sum and its reciprocal
    q_seqlen_offset = grp_i * atp.sb_p
    # Same aliasing as the running max above: on the first section exp_section_sum and
    # exp_running_sum hold the same value, so reduce straight into the running sum and skip the
    # GpSimd copy. Only section_idx > 0 needs them apart, where the section
    # sum is folded into the running sum through the flash correction factor.
    reduce_into_running = sp.section_idx == 0 or atp.num_sections == 1
    nisa.tensor_reduce(
        bufs.exp_running_sum[:, grp_i] if reduce_into_running else bufs.exp_section_sum[grp_i][...],
        nl.add,
        bufs.exp_partial_sum[grp_i],
        axis=1,
    )
    is_last_section = sp.section_idx == atp.num_sections - 1
    if atp.num_sections != 1:
        if sp.section_idx > 0:
            nisa.tensor_copy(
                bufs.prev_exp_running_sum[grp_i][...],
                bufs.exp_running_sum[:, grp_i],
            )
            nisa.tensor_scalar(
                bufs.exp_running_sum[:, grp_i],
                bufs.prev_exp_running_sum[grp_i][:, 0],
                nl.multiply,
                bufs.flash_attn_correction_factor[grp_i],
                op1=nl.add,
                operand1=bufs.exp_section_sum[grp_i],
            )
        if is_last_section:
            nisa.reciprocal(
                bufs.exp_sum_reciprocal[:, grp_i],
                bufs.exp_running_sum[:, grp_i],
            )
    else:
        # Single section: the reduce above already landed in exp_running_sum.
        nisa.reciprocal(bufs.exp_sum_reciprocal[:, grp_i], bufs.exp_running_sum[:, grp_i])

    num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

    # Step 2: one section: scale by the reciprocal, write to HBM; first of several: write to HBM;
    # a later one: load the previous output, apply the flash correction and accumulate, write to
    # HBM (scaled by the reciprocal in the last section).
    if atp.num_sections == 1:
        _scale_reciprocal_write_back_impl(bufs.mm2_sb[grp_i], grp_i, ac, atp, bufs, o, batch_id, num_p)
    elif sp.section_idx == 0:
        _write_back_o_impl(bufs.mm2_sb[grp_i], grp_i, ac, atp, o, batch_id, num_p)
    else:
        # Load previous output, scale by flash_attn_correction_factor and accumulate
        d_tile_size = atp.d_tile_size_free_dim
        for d_tile in range(atp.num_d_tiles_free_dim):
            d = min(d_tile_size, ac.d_v - d_tile * d_tile_size)
            d_offset = d_tile * d_tile_size
            prev_dst_pat = bufs.mm2_prev_output[grp_i][d_tile].ap(pattern=[[d, num_p], [1, d]], offset=0)
            o_src_pat = o.ap(
                pattern=[[ac.d_v, num_p], [1, d]],
                offset=batch_id * ac.seqlen_q * ac.d_v + grp_i * atp.sb_p * ac.d_v + d_offset,
            )
            nisa.dma_copy(dst=prev_dst_pat, src=o_src_pat)
            nisa.scalar_tensor_tensor(
                bufs.mm2_accum_flash_attn[grp_i][d_tile][:num_p, :d],
                data=bufs.mm2_prev_output[grp_i][d_tile][:num_p, :d],
                op0=nl.multiply,
                operand0=bufs.flash_attn_correction_factor[grp_i][:num_p, 0],
                op1=nl.add,
                operand1=bufs.mm2_sb[grp_i][d_tile][:num_p, :d],
            )
        if is_last_section:
            _scale_reciprocal_write_back_impl(bufs.mm2_accum_flash_attn[grp_i], grp_i, ac, atp, bufs, o, batch_id, num_p)
        else:
            _write_back_o_impl(bufs.mm2_accum_flash_attn[grp_i], grp_i, ac, atp, o, batch_id, num_p)


def _scale_reciprocal_write_back_impl(
    src_bufs,
    grp_i,
    ac: AttnConfig,
    atp: AttnTileParams,
    bufs: AttnInternalBuffers,
    o,
    batch_id,
    num_p,
):
    """Write back o for the final section after multiplication by reciprocal, iterating over d-tiles."""
    d_tile_size = atp.d_tile_size_free_dim
    for d_tile in range(atp.num_d_tiles_free_dim):
        d = min(d_tile_size, ac.d_v - d_tile * d_tile_size)
        nisa.tensor_scalar(
            bufs.mm2_final[grp_i][d_tile][:num_p, :d],
            src_bufs[d_tile][:num_p, :d],
            nl.multiply,
            bufs.exp_sum_reciprocal[:num_p, grp_i],
            engine=nisa.engine.vector,
        )

    _write_back_o_impl(bufs.mm2_final[grp_i], grp_i, ac, atp, o, batch_id, num_p)


def _write_back_o_impl(src_bufs, grp_i, ac: AttnConfig, atp: AttnTileParams, o, batch_id, num_p):
    """Write source buffers to output (o), iterating over d-tiles.

    Args:
      src_bufs: List of source buffers in SBUF, one per d-tile (src_bufs[d_tile])
      grp_i: Q group index
      ac: Attention configuration
      atp: Tile parameters
      o: Output tensor (HBM)
      batch_id: Batch index
      num_p: Number of partition elements (seqlen_q tile size)
    """
    d_tile_size = atp.d_tile_size_free_dim
    for d_tile in range(atp.num_d_tiles_free_dim):
        d = min(d_tile_size, ac.d_v - d_tile * d_tile_size)
        d_offset = d_tile * d_tile_size
        o_dst_pat = o.ap(
            pattern=[[ac.d_v, num_p], [1, d]],
            offset=batch_id * ac.seqlen_q * ac.d_v + grp_i * atp.sb_p * ac.d_v + d_offset,
        )
        src_pat = src_bufs[d_tile].ap(pattern=[[d, num_p], [1, d]], offset=0)
        nisa.dma_copy(dst=o_dst_pat, src=src_pat)


def _k_tile_selection(qkmax_grp, large_tile_idx, k_tile_idx, ac: AttnConfig, atp: AttnTileParams, sp: SectionParams):
    """Decide whether one K tile is processed by the QK + row-max pass, and return its position.

    Returns ``(tile_info, should_process)``:
      - tile_info (KTileInfo): seqlen_k, k_start_pos and num_valid_k, from ``_classify_tile``.
      - should_process: True if the tile has work; False when it is beyond the section's K tiles,
        or out of range (Q group past seqlen_q, or k_start past the end of the keys).
    """
    q_seqlen_offset = qkmax_grp * atp.sb_p
    # Resolve the tile's absolute key position: index within the large tile -> within the section ->
    # global K-tile index -> key offset.
    k_tile_idx_in_section = large_tile_idx * _NUM_K_TILES_IN_LARGE_TILE + k_tile_idx
    k_tile_idx_global = atp.num_k_tiles_per_section * sp.section_idx + k_tile_idx_in_section
    tile_info = _classify_tile(k_tile_idx_global * _K_TILE_SZ, atp)
    # Process only if the tile is in-section, has keys, and the Q group is in range.
    in_section = k_tile_idx_in_section < atp.num_k_tiles_per_section
    should_process_tile = in_section and tile_info.unmasked and q_seqlen_offset < ac.seqlen_q
    return tile_info, should_process_tile


def _emit_row_max_evict(dst, data, scale, reduce_res, reduce_cmd):
    """Evict MM1 scores scaled, with a fused row-max onto the Vector Engine's accumulator
    (attention_cte keeps this evict on Vector: its Scalar ``activate2`` form is slower)."""
    nisa.tensor_scalar_reduce(
        dst,
        data=data,
        op0=nl.multiply,
        operand0=scale,
        reduce_op=nl.maximum,
        reduce_res=reduce_res,
        reduce_cmd=reduce_cmd,
    )


def _row_max_chain_endpoints(k_tile_selections, ac: AttnConfig):
    """Return the first and last executed k-tile index in this large tile whose row-max reduces can
    be chained on the Vector accumulator (reset on first, accumulate on the rest, read out once on
    last), or None to fall back to per-tile reduces (also None when fewer than two tiles are
    executed, and with the key bounds: attention_cte never validated the chain for sequence
    packing).

    k_tile_selections is the per-k-tile ``(KTileInfo, should_process)`` list for the large tile (computed
    once by the caller and shared with the emit loop); the executed tiles (should_process) are exactly the
    ones the emit loop processes, so (first, last) bound them.
    """
    if ac.is_sequence_packed:
        return None

    chain = None
    num_executed = 0
    for k_tile_idx in range(_NUM_K_TILES_IN_LARGE_TILE):
        _, should_process = k_tile_selections[k_tile_idx]
        if not should_process:
            continue
        num_executed = num_executed + 1
        chain = (chain[0] if chain is not None else k_tile_idx, k_tile_idx)
    if num_executed < 2:
        return None
    return chain


def _qk_and_max_large_tile_impl(
    qkmax_grp,
    large_tile_idx,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Compute QK^T matmul (MM1) and find row-wise maximum for this Q group and large (2048) K tile,
    applying the key bounds if present."""

    q_seqlen_offset = qkmax_grp * atp.sb_p
    # Valid query rows in this Q group (partition dim); loop-invariant across k-tiles.
    num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

    # perform matmul and masking in 512 (_K_TILE_SZ) tile increment on the seqlen dimension
    # Compute each k-tile's selection once, shared by _row_max_chain_endpoints and the loop below.
    k_tile_selections = []
    for k_tile_idx in range(_NUM_K_TILES_IN_LARGE_TILE):
        k_tile_selections.append(_k_tile_selection(qkmax_grp, large_tile_idx, k_tile_idx, ac, atp, sp))
    # Row-max chaining: chain this large tile's executed reduces on the Vector accumulator, reading out
    # once (on the last chained tile) instead of per tile. None = fall back to per-tile reduces.
    row_max_chain = _row_max_chain_endpoints(k_tile_selections, ac)
    for k_tile_idx in range(_NUM_K_TILES_IN_LARGE_TILE):
        tile, should_process = k_tile_selections[k_tile_idx]

        if not should_process:
            continue
        # Extract relevant tensor tiles for convenience
        mm1_psum_tile = bufs.mm1_psum[qkmax_grp][large_tile_idx][k_tile_idx]
        mm1_masked_tile = bufs.mm1_masked[qkmax_grp][large_tile_idx]
        mm1_partial_max_tile = bufs.mm1_partial_max[qkmax_grp % _MM1_PARTIAL_MAX_STAGES]

        k_tile_idx_in_section = large_tile_idx * _NUM_K_TILES_IN_LARGE_TILE + k_tile_idx
        num_f = min(tile.num_valid_k, _K_TILE_SZ)

        # Step 1: MM1 matmul
        if ac.qk_mx:
            # row tiling: tile k_tile_idx of the large tile in quadrant k_tile_idx, the group's Q
            # (all 128 columns; rows past num_p are never evicted) as its stationary
            row = k_tile_idx * _QUADRANT
            nisa.nc_matmul_mx(
                dst=mm1_psum_tile[:, :num_f],
                stationary=bufs.q_mx[qkmax_grp][nl.ds(row, _QUADRANT), :],
                moving=bufs.k_mx[large_tile_idx][nl.ds(row, _QUADRANT), nl.ds(0, num_f)],
                stationary_scale=bufs.q_scale[qkmax_grp][nl.ds(row, _SCALE_ROWS), :],
                moving_scale=bufs.k_scale[large_tile_idx][nl.ds(row, _SCALE_ROWS), nl.ds(0, num_f)],
                tile_position=(row, 0),
                tile_size=(_QUADRANT, _Q_GRP_SZ),
            )
        else:
            k_tile_to_use = bufs.k_sb[k_tile_idx_in_section]
            for d_tile in range(atp.num_d_tiles):
                d_chunk = min(atp.d_tile_size_par_dim, ac.d - d_tile * atp.d_tile_size_par_dim)
                nisa.nc_matmul(
                    mm1_psum_tile[:num_p, :num_f],
                    bufs.q_sb[qkmax_grp // atp.num_q_grps_per_load][d_tile][
                        :d_chunk,
                        nl.ds((qkmax_grp % atp.num_q_grps_per_load) * _Q_GRP_SZ, num_p),
                    ],
                    k_tile_to_use[d_tile][:d_chunk, :num_f],
                )

        # Step 2: Masking + scale + max reduce
        # Row-max chain: reset on the first chained tile, accumulate (reduce) on the rest, and read out
        # only on the last; when not chaining, every tile resets and reads out (per-tile reduce).
        row_max_reduce_cmd = (
            nisa.reduce_cmd.reduce
            if (row_max_chain is not None and k_tile_idx != row_max_chain[0])
            else nisa.reduce_cmd.reset_reduce
        )
        row_max_reduce_res = (
            None
            if (row_max_chain is not None and k_tile_idx != row_max_chain[1])
            else mm1_partial_max_tile[:num_p, k_tile_idx_in_section]
        )

        if atp.dynamic_sel_mask:  # the key bounds (sequence packing): bound_min <= k < bound_max
            bound0 = bufs.range_sel_lbs[:num_p, nl.ds(qkmax_grp, 1)]
            bound1 = bufs.range_sel_ubs[:num_p, nl.ds(qkmax_grp, 1)]
            kernel_assert(ac.scale == 1.0, "range_select path doesn't support scale != 1.0")
            nisa.range_select(
                mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                on_true_tile=mm1_psum_tile[:num_p, :num_f],
                on_false_value=_FLOAT32_MIN,
                comp_op0=nl.greater_equal,
                comp_op1=nl.less,
                bound0=bound0[:num_p, :1],
                bound1=bound1[:num_p, :1],
                reduce_op=_nl_opcode.maximum,
                reduce_res=row_max_reduce_res,
                reduce_cmd=row_max_reduce_cmd,
                range_start=tile.k_start_pos,
            )
        else:  # no masking
            _emit_row_max_evict(
                dst=mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
                data=mm1_psum_tile[:num_p, :num_f],
                scale=ac.scale,
                reduce_res=row_max_reduce_res,
                reduce_cmd=row_max_reduce_cmd,
            )


@dataclass
class Mm2TileInfo(nl.NKIObject):
    """Trace-time position of one MM2 (P@V) matmul within a large tile."""

    mm2_grp_i: int  # index of the K-seqlen group inside the large tile
    mm2_i: int  # index of the V tile inside that group
    v_tile_idx: int  # index of the V tile within the section
    num_p: int  # valid keys in this V tile (contraction dim)
    num_f: int  # valid query rows in this Q group


def _pv_mm2_tile_selection(pv_grp, large_tile_idx, ac: AttnConfig, atp: AttnTileParams, sp: SectionParams):
    """Decide which MM2 (P@V) matmuls one (Q group, large tile) pair runs, and where they read from.

    Trace-time only: it emits no instructions. ``_pv_large_tile_impl`` calls it for its own large tile
    to drive the matmul loop, and again for the *later* large tiles so it can tell whether it holds the
    last contribution to the shared MM2 PSUM accumulator and therefore owns the eviction.

    Returns ``(tiles, psum_valid)``:
      - tiles (list[Mm2TileInfo]): one entry per matmul with keys, in emission order.
      - psum_valid: True if the accumulator this large tile contributes to is valid to read, i.e. the
        large tile is in range and at least one matmul was selected.
    """
    q_seqlen_offset = pv_grp * atp.sb_p
    num_mm2_grps_in_large_tile = _LARGE_TILE_SZ // atp.mm2_grp_sz
    # pv_mx: one nc_matmul_mx contracts the whole 512-key group (128 partitions x 4 packed keys)
    v_tile_sz = atp.mm2_grp_sz if ac.pv_mx else _V_TILE_SZ
    num_mm2_per_grp = atp.mm2_grp_sz // v_tile_sz
    num_mm2_per_large_tile = num_mm2_per_grp * num_mm2_grps_in_large_tile

    seqlen_k = atp.seqlen_k
    k_seqlen_mask = sp.section_offset + large_tile_idx * _LARGE_TILE_SZ < seqlen_k

    tiles = []
    for mm2_grp_i in range(num_mm2_grps_in_large_tile):
        k_start_pos_512_tile = sp.section_offset + large_tile_idx * _LARGE_TILE_SZ + mm2_grp_i * atp.mm2_grp_sz

        for mm2_i in range(num_mm2_per_grp):
            v_tile_idx = large_tile_idx * num_mm2_per_large_tile + mm2_grp_i * num_mm2_per_grp + mm2_i
            k_start_pos = k_start_pos_512_tile + mm2_i * v_tile_sz

            num_p = min(seqlen_k - k_start_pos, v_tile_sz)
            num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

            if v_tile_idx < atp.num_v_tiles_per_section and num_p > 0 and num_f > 0:
                tiles.append(
                    Mm2TileInfo(
                        mm2_grp_i=mm2_grp_i,
                        mm2_i=mm2_i,
                        v_tile_idx=v_tile_idx,
                        num_p=num_p,
                        num_f=num_f,
                    )
                )

    return tiles, (k_seqlen_mask and len(tiles) > 0)


def _pv_large_tile_impl(
    pv_grp,
    large_tile_idx,
    ac: AttnConfig,
    atp: AttnTileParams,
    sp: SectionParams,
    bufs: AttnInternalBuffers,
):
    """Perform MM2 (P@V) matmul for the Q grp and large (2048) V tile."""

    q_seqlen_offset = pv_grp * atp.sb_p
    mm2_psum_tile = bufs.mm2_psum[pv_grp][large_tile_idx]
    mm2_tiles, psum_valid = _pv_mm2_tile_selection(pv_grp, large_tile_idx, ac, atp, sp)

    if atp.mm2_accum_in_psum:
        # Every large tile of this Q group was handed the same PSUM tiles, so the matmuls below
        # accumulate on top of the earlier large tiles' partial sums on the PE array. Only the last
        # large tile that contributes evicts, and it evicts a complete sum -- so the Vector Engine
        # runs one copy per (Q group, d-tile) instead of one copy plus one add per large tile.
        evict = psum_valid and not any(
            _pv_mm2_tile_selection(pv_grp, later_idx, ac, atp, sp)[1]
            for later_idx in range(large_tile_idx + 1, atp.num_large_tiles_per_section)
        )
        accumulate_into_sb = False
    else:
        evict = psum_valid
        accumulate_into_sb = large_tile_idx > 0

    # Loop over d-tiles as outermost: without mm2_accum_in_psum all d-tiles share one PSUM bank, so
    # each must be evicted before the next is computed.
    for d_tile in range(atp.num_d_tiles_free_dim):
        # Step 1: Perform matmul and accumulate in PSUM for each K-seqlen group
        for tile in mm2_tiles:
            if ac.pv_mx:
                # One matmul per 512 keys, zero-padded past the last key (num_p is not needed).
                # All 128 stationary columns: rows past num_f are never evicted.
                nisa.nc_matmul_mx(
                    dst=mm2_psum_tile[d_tile][:, : atp.d_tile_size_free_dim],
                    stationary=bufs.p_mx[pv_grp][large_tile_idx][tile.mm2_grp_i],
                    moving=bufs.v_sb[tile.v_tile_idx],
                    stationary_scale=bufs.p_scale[pv_grp][large_tile_idx][tile.mm2_grp_i],
                    moving_scale=bufs.v_scale_sb[tile.v_tile_idx],
                )
                continue
            exp_tp_sb_tile = bufs.exp_tp_sb[pv_grp][large_tile_idx][tile.mm2_grp_i]
            nisa.nc_matmul(
                mm2_psum_tile[d_tile][: tile.num_f, : atp.d_tile_size_free_dim],
                exp_tp_sb_tile[: tile.num_p, nl.ds(tile.mm2_i * _V_TILE_SZ, tile.num_f)],
                bufs.v_sb[tile.v_tile_idx][d_tile][: tile.num_p, : atp.d_tile_size_free_dim],
            )

        # Step 2: Evict PSUM to SBUF for this d-tile
        if evict:
            num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            dst = bufs.mm2_sb[pv_grp][d_tile][:num_p, :]
            psum = mm2_psum_tile[d_tile][:num_p, :]

            if accumulate_into_sb:
                nisa.tensor_tensor(dst, dst, psum, nl.add)
            else:
                nisa.tensor_copy(dst, psum)
