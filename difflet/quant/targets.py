"""Which linears each DiT quantizes. Torch-free.

Only the per-token projections go to MXFP8: attention q/k/v and the MLP up/down. They hold
about 60% of Qwen-Image's parameters and most of its linear-layer compute. Everything that
conditions the blocks stays bf16: the per-block modulation (``img_mod.1``/``txt_mod.1``: a
third of the parameters, but applied to one ``[B, D]`` vector, so no compute to win and the
scale/shift/gate it produces is precision-sensitive), the embedders, ``img_in``/``txt_in``,
``norm_out`` and ``proj_out``.

The attention output projections (``to_out.0``, ``to_add_out``) also stay bf16, against the
ModelOpt / FastVideo convention, on measurement (trn3, tp4, 1024x1024, 50 steps, mxfp8 vs
bf16; ``docs/verification/2026-10-31-qwen-image-fp8-native-trn3.md`` 6d): with them quantized
the step is 267.7 ms and the image 29.3 dB PSNR; with them in bf16, 260.2 ms and 31.6 dB.
They sit right before the residual add, so their error lands unattenuated, and the compiler
schedules the block worse with a custom op there (the single-core probe: +1.3 ms per block).

LTX-2 (diffusers' ``LTX2VideoTransformer3DModel``) runs a video stream (4096 wide: 32 heads x
128) and an audio stream (2048 wide: 32 x 64) through 48 blocks; each block has, per stream, a
self-attention, a text cross-attention and a 4x MLP (``ff`` 16384 wide, ``audio_ff`` 8192), and
one cross-attention each way between the streams. The targets are the linears whose input is
the video or the text stream: the video ``attn1`` / ``attn2`` q/k/v, the video queries of
``audio_to_video_attn``, the video keys and values of ``video_to_audio_attn``, and the video
MLP. The audio stream's linears stay bf16 by default: ``audio_attn1``, ``audio_attn2.to_q``,
``audio_ff``, the audio keys and values of ``audio_to_video_attn`` and the audio queries of
``video_to_audio_attn`` are fed by the audio stream, 126 tokens at the default shape, too few
rows for the kernel to win anything; ``audio_attn2.to_k`` / ``to_v`` are fed by the text stream
(the audio caption projection: 1024 rows, K 2048) but kept with the audio stream, so a device
variant through ``DIFFLET_QUANT_TARGETS`` decides that stream as a whole. Every
``to_out.0`` stays bf16 as above; the per-block modulation is a parameter table, not a linear;
the embedders, ``proj_in`` / ``audio_proj_in``, the caption projections, the ``av_cross_attn_*``
scale-shift and gate embedders and ``proj_out`` / ``audio_proj_out`` are bf16 top-level layers.
At tp > 1 every ``to_out.0`` (K 4096 or 2048) and both MLP-downs (16384 and 8192) are
row-parallel, quantized per rank K range: the default row-parallel target, ``ff.net.2``, keeps
a K per rank that is a multiple of 128 at every tp its 32 heads divide; an override that
quantizes a 2048-wide ``to_out.0`` has one up to tp 16.

Wan 2.1-T2V-14B and Wan 2.2-T2V-A14B (diffusers' ``WanTransformer3DModel`` names) run 40 blocks
of 5120 (40 heads x 128), each a self-attention, a text cross-attention (4096-wide text at 512
tokens) and a 13824-wide FFN; Wan 2.2's two experts share that block graph, so one table serves
both (one spec, two quantized directories). The targets are the text cross-attention's q/k/v
and the FFN (the self-attention's q/k/v stay bf16 for quality, an override's in the half-split
rotary layout, see ``WAN_TARGETS``), written in **checkpoint-key spelling**: the neuron module
(``difflet/models/wan/modeling_wan.py``) names its FFN ``ffn.net_in`` / ``net_out`` where the
checkpoint has ``ffn.net.0.proj`` / ``net.2``, and the application applies the spec through its
``checkpoint_key`` (``quantize_module_(key_fn=...)``), so the patterns match the weights the
writer quantizes and the modules the loader swaps alike. Every ``to_out.0`` stays bf16 as
above; the per-block modulation is ``scale_shift_table``, a parameter, not a linear; the
``condition_embedder`` and ``proj_out`` are bf16 top-level layers. At tp > 1 ``to_out.0``
(K 5120) and ``ffn.net.2`` (K 13824) are row-parallel: ``net.2``'s K per rank is a multiple of
128 at tp 1, 2 and 4 (3456) but not at tp 8 (1728), which the writer refuses naming the layer.

HunyuanVideo 1.0 (diffusers' ``HunyuanVideoTransformer3DModel`` names, which the neuron module
keeps and the checkpoint uses) runs 20 dual-stream and 40 single-stream blocks of 3072 (24 heads
x 128) with a 12288-wide MLP. A dual block keeps the image stream (10,240 tokens at the default
320x512x61, 118,800 at 720x1280x129) and the text stream (256 tokens) apart outside the joint
attention; a single block runs both as one sequence. The table started as FLUX's, one pattern
per device-bisection group (G1 the dual blocks' image q/k/v, whose outputs feed the q/k RMSNorm,
the rotary op and the attention kernel; G2 their text q/k/v, 256 rows; G3 / G4 the image and
text MLPs; G5 the single blocks' joint q/k/v; G6 their MLP-up ``proj_mlp``; G7 their output
projection's two halves). The device's leave-one-out kept G1 and G5 bf16 for quality (the table
comment below), so the default targets are the text q/k/v, both dual MLPs, ``proj_mlp`` and the
``proj_out`` halves. Every column target has K 3072. The dual blocks' output projections stay
bf16 as above, and so do the per-block modulation (``norm1.linear``, ``norm1_context.linear``,
``norm.linear``: applied to one ``[B, D]`` vector, as Qwen-Image's), the token refiner
(``context_embedder``: its own attention and MLP on the text, before the blocks), the
embedders, ``norm_out`` and ``proj_out``. The single blocks' ``proj_out`` is one checkpoint
``Linear(3072 + 12288, 3072)`` that the neuron module runs as two row-parallel linears, each on
its own input sharding (``proj_out_attn`` / ``proj_out_mlp``). HunyuanVideo's spec splits it as
FLUX's does (``SPLITS_BY_MODEL``), so the quantized checkpoint stores the halves as tensors of
their own, read by name, and the targets name the halves, each quantized per rank K range of
its own; a target on the fused name matches no weight the writer stores, and the writer
refuses it naming the halves. At tp > 1 the MLP-downs (``ff.net.2``, ``ff_context.net.2``) and
``proj_out_mlp``, all K 12288, are row-parallel with a K per rank of 6144, 3072 and 1536 at
tp 2, 4 and 8, whole 512-wide kernel tiles; ``proj_out_attn`` (K 3072) has 1536, 768 and 384,
multiples of 128 with a partial last tile at tp 4 and 8 (768 = 512 + 256), as an override that
quantizes ``to_out.0`` / ``to_add_out`` (K 3072) would. The token refiner's
``attn.to_out.0`` and ``ff.net.2`` are row layers of the same K: the row-parallel set names
them for overrides, since the FP8 twin of a row layer shards its weight along K whatever the
set says, and a weight the writer laid out as one K range is read in other block memberships
than it was scaled in wherever a rank's range is not whole tiles (``attn.to_out.0`` at tp 4
and 8).

HunyuanVideo-1.5 (diffusers' ``HunyuanVideo15Transformer3DModel``: the neuron module is
diffusers' class, so its module names are the checkpoint keys and ``checkpoint_key`` is the
identity) runs 54 dual-stream blocks of 2048 (16 heads x 128) with an 8192-wide MLP and no
single-stream blocks, so no ``proj_out`` to split. A block keeps the image stream (49,290 tokens
at the default 480x848x121) and the text stream (1,985: the image-embedding, ByT5 and Qwen2.5-VL
tokens) apart outside the joint attention. The table holds one pattern per device-bisection
group: G1 the image q/k/v (``to_q/k/v``, whose outputs feed the q/k RMSNorm, the rotary op and
the attention kernel) and G3 the image MLP (``ff``): 270 linears, 2.49B of the 8.33B parameters
(30%). The text stream's q/k/v (G2, ``add_*_proj``) and MLP (G4, ``ff_context``) stay bf16:
with them the step ran 2.7x bf16's (the table comment below). The quantization saves memory,
not time: the hub DiT is fp32, so the quantized checkpoint is written from the bf16 copy
``compile`` makes, and a target holds 1.28 bytes per weight resident against bf16's 2 (fp8, its
stored scales and the kernel's scale layout). The output projections (``to_out.0``,
``to_add_out``) stay bf16 as above, and so do the per-block modulation (``norm1.linear``,
``norm1_context.linear``: 2048 -> 12288, 2.72B parameters, applied to one ``[B, D]`` vector),
the token refiner (``context_embedder``), the glyph projection (``context_embedder_2``),
``image_embedder``, ``time_embed``, ``norm_out`` and ``proj_out``. At tp > 1 the image MLP-down
(``ff.net.2``, K 8192) is row-parallel with a K per rank of 8192 / tp, whole 512-wide kernel
tiles at every tp the 16 heads allow (512 at tp 16), so the default set refuses no tp. The text
MLP-down (``ff_context.net.2``, the same K), the output projections (K 2048) and the token
refiner's ``attn.to_out.0`` and ``ff.net.2`` are row layers too: the row-parallel set names them
for overrides, as HunyuanVideo's does (``to_out.0`` has 256 per rank at tp 8, a partial tile).

Patterns are regexes full-matched against diffusers' module names (the ``transformer.`` trace
prefix is stripped by :meth:`QuantSpec.matches`).
"""

from __future__ import annotations

from difflet.quant.spec import QuantSpec

_BLOCK = r"transformer_blocks\.\d+"

QWEN_IMAGE_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)",
    rf"{_BLOCK}\.(img_mlp|txt_mlp)\.net\.0\.proj",
    rf"{_BLOCK}\.(img_mlp|txt_mlp)\.net\.2",
)

#: Qwen-Image modules a reader may expect quantized that are deliberately not.
QWEN_IMAGE_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",
    rf"{_BLOCK}\.(img_mod|txt_mod)\.1",
    r"time_text_embed\..*",
    r"img_in",
    r"txt_in",
    r"norm_out\.linear",
    r"proj_out",
)

#: The targets whose weight the TP layer shards along K (row-parallel): under ``mxfp8`` each
#: rank's K range is quantized with its own block layout (``difflet/quant/mx.py``).
QWEN_IMAGE_ROW_PARALLEL: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",  # not a default target; kept for overrides
    rf"{_BLOCK}\.(img_mlp|txt_mlp)\.net\.2",
)

# FLUX.1 (diffusers' FluxTransformer2DModel names, which ``difflet/models/flux/modeling_flux.py``
# keeps): the 19 dual-stream blocks' q/k/v for both streams and both MLPs (``ff``, ``ff_context``),
# the 38 single-stream blocks' q/k/v and MLP-up (``proj_mlp``). The output projections stay bf16
# by default as for Qwen-Image: the dual blocks' ``to_out.0`` / ``to_add_out``; the single
# blocks' ``proj_out`` is quantized as its two halves (below). FLUX's spec splits that ``proj_out`` (``SINGLE_BLOCK_SPLIT_LINEARS``),
# so its quantized checkpoint stores ``proj_out_attn`` / ``proj_out_mlp``, the two row-parallel
# linears the neuron model runs, as tensors of their own, read by name; untargeted they are the
# fused weight's column ranges in the source's dtype. The default quantizes both halves, each per
# rank K range of its own, on measurement (trn3, tp4, 1024x1024, 50 steps;
# docs/verification/2026-11-07-mxfp8-video-models.md section 6): 183.4 ms per step against
# 222.3 in bf16 (0.82x) and 191.1 without them, and no more drift from bf16 over 4 prompts x
# 3 seeds than without them (mean 24.7 against 24.2 dB; two bf16 runs at different TP degrees
# are 29.3 dB apart at 50 steps, so the absolute verdict is the CPU fp32 reference's).
_SINGLE = r"single_transformer_blocks\.\d+"

#: FLUX's and HunyuanVideo's single-stream ``proj_out``: one checkpoint ``Linear(dim + mlp_dim,
#: dim)`` over ``cat([attn, mlp])`` that the neuron models run as two row-parallel linears, each
#: on its own input sharding. A spec that splits it stores ``proj_out_attn`` / ``proj_out_mlp``
#: as tensors of their own (``checkpoint.py``), so each half can be quantized per rank K range.
SINGLE_BLOCK_SPLIT_LINEARS: tuple[str, ...] = (rf"{_SINGLE}\.proj_out",)

FLUX_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)",
    rf"{_BLOCK}\.(ff|ff_context)\.net\.0\.proj",
    rf"{_BLOCK}\.(ff|ff_context)\.net\.2",
    rf"{_SINGLE}\.attn\.(to_q|to_k|to_v)",
    rf"{_SINGLE}\.proj_mlp",
    rf"{_SINGLE}\.proj_out_(attn|mlp)",
)

#: The single blocks' ``proj_out`` halves as targets (row-parallel: each quantized per rank K
#: range of the half), for ``targets=`` / ``DIFFLET_QUANT_TARGETS`` overrides.
FLUX_SPLIT_TARGETS: tuple[str, ...] = (rf"{_SINGLE}\.proj_out_(attn|mlp)",)

FLUX_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",
    rf"{_SINGLE}\.proj_out",
    rf"{_BLOCK}\.norm1(_context)?\.linear",
    rf"{_SINGLE}\.norm\.linear",
    r"time_text_embed\..*",
    r"context_embedder",
    r"x_embedder",
    r"norm_out\.linear",
    r"proj_out",
)

FLUX_ROW_PARALLEL: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",  # not default targets; kept for overrides
    rf"{_BLOCK}\.(ff|ff_context)\.net\.2",
    *FLUX_SPLIT_TARGETS,  # not default targets; kept for overrides
)

# LTX-2 (diffusers' LTX2VideoTransformer3DModel names, which
# ``difflet/models/ltx_2/modeling_ltx_2.py`` keeps under the ``transformer.`` trace prefix): the
# linears fed by the video or the text stream.
_LTX_2_ATTNS = r"(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)"

LTX_2_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn1\.(to_q|to_k|to_v)",
    rf"{_BLOCK}\.attn2\.(to_q|to_k|to_v)",
    rf"{_BLOCK}\.audio_to_video_attn\.to_q",
    rf"{_BLOCK}\.video_to_audio_attn\.(to_k|to_v)",
    rf"{_BLOCK}\.ff\.net\.0\.proj",
    rf"{_BLOCK}\.ff\.net\.2",
)

#: LTX-2 modules a reader may expect quantized that are deliberately not: the output
#: projections, the audio stream's linears (``audio_attn2.to_k`` / ``to_v`` are text-fed but
#: kept with the stream; a ``DIFFLET_QUANT_TARGETS`` variant decides it on the device) and the
#: bf16 top level.
LTX_2_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.{_LTX_2_ATTNS}\.to_out\.0",
    rf"{_BLOCK}\.(audio_attn1|audio_attn2)\.(to_q|to_k|to_v)",
    rf"{_BLOCK}\.audio_to_video_attn\.(to_k|to_v)",
    rf"{_BLOCK}\.video_to_audio_attn\.to_q",
    rf"{_BLOCK}\.audio_ff\.net\.0\.proj",
    rf"{_BLOCK}\.audio_ff\.net\.2",
    r"proj_in",
    r"audio_proj_in",
    r"caption_projection\..*",
    r"audio_caption_projection\..*",
    r"time_embed\..*",
    r"audio_time_embed\..*",
    r"av_cross_attn_.*",
    r"proj_out",
    r"audio_proj_out",
)

LTX_2_ROW_PARALLEL: tuple[str, ...] = (
    rf"{_BLOCK}\.{_LTX_2_ATTNS}\.to_out\.0",  # not default targets; kept for overrides
    rf"{_BLOCK}\.ff\.net\.2",
    rf"{_BLOCK}\.audio_ff\.net\.2",  # not a default target; kept for overrides
)

# Wan 2.1 and Wan 2.2-A14B (diffusers' WanTransformer3DModel names, the checkpoint's spelling of
# the FFN): the text cross-attention's q/k/v and the two FFN linears, in the 917d087 order (the
# targets key the quantized directory, so its directories are reused). The self-attention's
# q/k/v stay bf16 for quality, not speed (trn3, tp4, 480x832, 50 steps; the speed in
# docs/verification/2026-11-07-mxfp8-video-models.md section 8, the two-expert quality in Task
# H6 of docs/plans/2026-11-08-mxfp8-wan-half-split-rope.md). Their speed is solved: the
# interleaved rotary's pair swap on q/k from the MXFP8 kernel moved 2-byte elements (1480 DMA
# instructions per block, overlapping, their durations summed to 87.8 ms; the block 56.89 ms,
# the step 2263 ms), where on q/k from a bf16 matmul it moves 256-byte head rows (240, summed
# to 0.04 ms). So when any self-attention q or k is quantized, the neuron application reorders
# every self-attention's q/k channels at load and rotates contiguous halves
# (``WanTransformer3DModel.use_half_split_rope_``): the block runs 8.05 ms against this set's
# 9.30, the step at 9 frames 0.77x bf16's against this set's 0.90x. Their quality is not: with
# both Wan 2.2 experts quantized, the mean over 4 prompts x 3 seeds at 9 frames is 27.60 dB /
# SSIM 0.860 from bf16 with them and 29.04 / 0.895 with this set (-1.44 dB, -0.034; better on
# 3 of the 12 clips, the worst 17.90 against 28.50 dB), past the 1 dB / 0.01 the rule allows,
# where Wan 2.1 alone loses 0.46 dB / 0.003. One table serves both, so they stay bf16; an
# override (``DIFFLET_QUANT_TARGETS``) that quantizes a self-attention q or k still gets the
# half-split layout, and bf16 q/k keep the interleaved one, faster for them (bf16 runs 483.6 ms
# per step in the half split against 431.9 interleaved).
_WAN_BLOCK = r"blocks\.\d+"

WAN_TARGETS: tuple[str, ...] = (
    rf"{_WAN_BLOCK}\.attn2\.(to_q|to_k|to_v)",
    rf"{_WAN_BLOCK}\.ffn\.net\.0\.proj",
    rf"{_WAN_BLOCK}\.ffn\.net\.2",
)

#: Wan modules a reader may expect quantized that are deliberately not: the self-attention's
#: q/k/v (measured above), the output projections, the modulation tables (parameters, not
#: linears: they match no module, named so the table reads complete), the embedders and the
#: output layer.
WAN_KEPT_BF16: tuple[str, ...] = (
    rf"{_WAN_BLOCK}\.attn1\.(to_q|to_k|to_v)",
    rf"{_WAN_BLOCK}\.(attn1|attn2)\.to_out\.0",
    rf"{_WAN_BLOCK}\.scale_shift_table",
    r"condition_embedder\..*",
    r"scale_shift_table",
    r"proj_out",
)

WAN_ROW_PARALLEL: tuple[str, ...] = (
    rf"{_WAN_BLOCK}\.(attn1|attn2)\.to_out\.0",  # not default targets; kept for overrides
    rf"{_WAN_BLOCK}\.ffn\.net\.2",
)

# HunyuanVideo 1.0 (diffusers' HunyuanVideoTransformer3DModel names, the checkpoint keys): one
# pattern per device-bisection group, G1-G7 in table order. The default is the leave-one-out arm
# V15 (trn3, tp4, 320x512x61, 50 steps, seed 42; docs/verification/2026-11-07-mxfp8-video-models.md,
# the HunyuanVideo section): G1 (dual image q/k/v) and G5 (single joint q/k/v) stay bf16. With
# every group the step is 606.0 ms against 660.0 in bf16, but the clip is 21.2 dB / SSIM 0.830 /
# LPIPS 0.170 from bf16 (latents cosine 0.984; two bf16 runs at tp2 and tp4 are 36.6 dB apart).
# No single group carries the loss (dropping G1, G5, G4 or G6 alone recovers 7-9 dB, G2 nothing),
# and V15 is the only arm that clears the floor-relative screen: 31.09 dB / 0.938 / 0.050 at
# 634.9 ms (0.96x). The dual block's device time is 11.29 ms without G1 against 13.02 with it
# (bf16 11.45); the single block's 9.65 without G5 against 9.56 (bf16 10.31).
HUNYUAN_VIDEO_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(add_q_proj|add_k_proj|add_v_proj)",  # G2 dual text q/k/v (256 rows)
    rf"{_BLOCK}\.ff\.net\.(0\.proj|2)",  # G3 dual image MLP
    rf"{_BLOCK}\.ff_context\.net\.(0\.proj|2)",  # G4 dual text MLP (256 rows)
    rf"{_SINGLE}\.proj_mlp",  # G6 single joint MLP-up
    rf"{_SINGLE}\.proj_out_(attn|mlp)",  # G7 single proj_out halves (row-parallel)
)

#: HunyuanVideo modules a reader may expect quantized that are deliberately not: the image-stream
#: q/k/v of both block kinds (G1, G5: quality, measured above), the dual blocks' output
#: projections, the modulation, the single blocks' fused ``proj_out`` (stored as its halves,
#: which are targets), the token refiner and the embedders, and the output layer.
HUNYUAN_VIDEO_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_q|to_k|to_v)",
    rf"{_SINGLE}\.attn\.(to_q|to_k|to_v)",
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",
    rf"{_BLOCK}\.norm1(_context)?\.linear",
    rf"{_SINGLE}\.norm\.linear",
    rf"{_SINGLE}\.proj_out",
    r"context_embedder\..*",
    r"time_text_embed\..*",
    r"norm_out\.linear",
    r"proj_out",
)

#: Every row layer of the neuron module with a key of its own in the quantized checkpoint, the
#: single blocks' ``proj_out`` halves included (the spec stores them as tensors of their own).
HUNYUAN_VIDEO_ROW_PARALLEL: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",  # not default targets; kept for overrides
    rf"{_BLOCK}\.(ff|ff_context)\.net\.2",
    rf"{_SINGLE}\.proj_out_(attn|mlp)",
    # not default targets; kept for overrides
    r"context_embedder\.token_refiner\.refiner_blocks\.\d+\.(attn\.to_out\.0|ff\.net\.2)",
)

# HunyuanVideo-1.5 (diffusers' HunyuanVideo15Transformer3DModel names, the checkpoint keys): one
# pattern per device-bisection group in G order, G1 the image q/k/v and G3 the image MLP, the
# strings F6's arms carried in DIFFLET_QUANT_TARGETS (they key the quantized directory, so the
# default resolves those arms'). A memory saving by decision (the 480p_t2v DiT is 16.65 GB in
# bf16, 14.86 GB resident with these quantized). Measured (trn3, tp2 x CFG2, 50 steps, seed 42;
# docs/plans/2026-11-07-mxfp8-hunyuan-video-15.md, F6 Step 7): at 480x848x9 the step is 557 ms
# in bf16 and 1528 with both streams' q/k/v and MLPs (G2 the text q/k/v, G4 the text MLP). Each
# group alone is at or below bf16 (G1 546, G2 554, G3 528, G4 556 ms), but the slowdowns do not
# add up: G1 + G2 + G3 is 1514 and G1 + G3 511. At 480x848x121 this set runs 10,314 / 10,313 ms
# against 10,241 in bf16 (1.007x, the two runs bit-identical), 6.97 against 7.80 GiB of DiT
# weights per rank, peak HBM 9.21 + 8.45 GiB (allocator + runtime) against 10.04 + 7.52. The clip
# is 12.76 dB / SSIM 0.190 from bf16, against a chaotic bf16 floor of 15.90 / 0.312 (tp4 against
# tp2; the frames show the same composition): the floor-relative screen fails on SSIM (0.190
# against 0.212), and quality is pending phase G (diffusers on the CPU in fp32). The plan never
# makes a set that fails the screen the default; this one is the user's decision (2026-11-08,
# option A), taken with these numbers.
HUNYUAN_VIDEO_15_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_q|to_k|to_v)",  # G1 image q/k/v
    rf"{_BLOCK}\.ff\.net\.(0\.proj|2)",  # G3 image MLP
)

#: HunyuanVideo-1.5 modules a reader may expect quantized that are deliberately not: the text
#: stream's q/k/v and MLP (G2, G4, for speed: the bisection above dropped G4 when all four
#: groups ran 1528 ms against bf16's 557, then G2 when G1 + G2 + G3 still ran 1514, 2.7x
#: bf16's, against 511 without it), the output projections, the modulation, the token refiner
#: and the glyph, image and time embedders, and the output layer.
HUNYUAN_VIDEO_15_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(add_q_proj|add_k_proj|add_v_proj)",
    rf"{_BLOCK}\.ff_context\.net\.(0\.proj|2)",
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",
    rf"{_BLOCK}\.norm1(_context)?\.linear",
    r"context_embedder\..*",
    r"context_embedder_2\..*",
    r"image_embedder\..*",
    r"time_embed\..*",
    r"norm_out\.linear",
    r"proj_out",
)

#: Every row layer of the neuron module (``_install_tensor_parallel``), the token refiner's
#: included: the FP8 twin of a row layer shards its weight along K whatever this set says.
HUNYUAN_VIDEO_15_ROW_PARALLEL: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",  # not default targets; kept for overrides
    rf"{_BLOCK}\.(ff|ff_context)\.net\.2",
    # not default targets; kept for overrides
    r"context_embedder\.token_refiner\.refiner_blocks\.\d+\.(attn\.to_out\.0|ff\.net\.2)",
)


TARGETS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen_image": QWEN_IMAGE_TARGETS,
    "flux": FLUX_TARGETS,
    "ltx_2": LTX_2_TARGETS,
    "wan": WAN_TARGETS,
    "hunyuan_video": HUNYUAN_VIDEO_TARGETS,
    "hunyuan_video_15": HUNYUAN_VIDEO_15_TARGETS,
}
ROW_PARALLEL_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen_image": QWEN_IMAGE_ROW_PARALLEL,
    "flux": FLUX_ROW_PARALLEL,
    "ltx_2": LTX_2_ROW_PARALLEL,
    "wan": WAN_ROW_PARALLEL,
    "hunyuan_video": HUNYUAN_VIDEO_ROW_PARALLEL,
    "hunyuan_video_15": HUNYUAN_VIDEO_15_ROW_PARALLEL,
}
#: The fused checkpoint linears each model's quantized checkpoint stores as two column ranges
#: (``QuantSpec.split_linears``). A model without an entry keeps every weight whole, and its
#: spec and directory digest are what they were before splits existed.
SPLITS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "flux": SINGLE_BLOCK_SPLIT_LINEARS,
    "hunyuan_video": SINGLE_BLOCK_SPLIT_LINEARS,
}


def targets_for(model: str) -> tuple[str, ...]:
    try:
        return TARGETS_BY_MODEL[model]
    except KeyError:
        raise ValueError(
            f"no FP8 target layers are defined for {model!r}; "
            f"models with FP8 support: {sorted(TARGETS_BY_MODEL)}"
        ) from None


def row_parallel_for(model: str) -> tuple[str, ...]:
    return ROW_PARALLEL_BY_MODEL.get(model, ())


def splits_for(model: str) -> tuple[str, ...]:
    return SPLITS_BY_MODEL.get(model, ())


def default_spec(model: str, **overrides) -> QuantSpec:
    """The mxfp8 recipe for ``model``; ``targets=`` replaces the model's default layer set,
    ``compute=`` and ``tp_degree=`` complete it. The model's splits stay under a ``targets=``
    override: the checkpoint layout does not depend on which layers are quantized, so a set
    without the halves stores them unquantized and the model reads them by name as ever."""
    overrides.setdefault("row_parallel", row_parallel_for(model))
    overrides.setdefault("targets", targets_for(model))
    overrides.setdefault("split_linears", splits_for(model))
    return QuantSpec(**overrides)


__all__ = [
    "FLUX_KEPT_BF16",
    "FLUX_ROW_PARALLEL",
    "FLUX_SPLIT_TARGETS",
    "FLUX_TARGETS",
    "HUNYUAN_VIDEO_15_KEPT_BF16",
    "HUNYUAN_VIDEO_15_ROW_PARALLEL",
    "HUNYUAN_VIDEO_15_TARGETS",
    "HUNYUAN_VIDEO_KEPT_BF16",
    "HUNYUAN_VIDEO_ROW_PARALLEL",
    "HUNYUAN_VIDEO_TARGETS",
    "LTX_2_KEPT_BF16",
    "LTX_2_ROW_PARALLEL",
    "LTX_2_TARGETS",
    "QWEN_IMAGE_KEPT_BF16",
    "QWEN_IMAGE_ROW_PARALLEL",
    "QWEN_IMAGE_TARGETS",
    "ROW_PARALLEL_BY_MODEL",
    "SINGLE_BLOCK_SPLIT_LINEARS",
    "SPLITS_BY_MODEL",
    "TARGETS_BY_MODEL",
    "WAN_KEPT_BF16",
    "WAN_ROW_PARALLEL",
    "WAN_TARGETS",
    "default_spec",
    "row_parallel_for",
    "splits_for",
    "targets_for",
]
