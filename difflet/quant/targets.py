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
and the FFN (the self-attention's q/k/v stay bf16 on measurement, see ``WAN_TARGETS``),
written in **checkpoint-key spelling**: the neuron module
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
attention; a single block runs both as one sequence. The targets are FLUX's, one pattern per
device-bisection group: the dual blocks' image q/k/v (``to_q/k/v``, whose outputs feed the q/k
RMSNorm, the rotary op and the attention kernel), text q/k/v (``add_*_proj``: 256 rows) and both
MLPs (``ff`` on the image rows, ``ff_context`` on the 256 text rows), and the single blocks'
joint q/k/v and MLP-up (``proj_mlp``), each on the image and text rows together. Every column
target has K 3072. The output projections stay bf16 as above, and so do the per-block
modulation (``norm1.linear``, ``norm1_context.linear``, ``norm.linear``: applied to one
``[B, D]`` vector, as Qwen-Image's), the token refiner (``context_embedder``: its own attention
and MLP on the text, before the blocks), the embedders, ``norm_out`` and ``proj_out``. The
single blocks' ``proj_out`` waits for its split: it is one checkpoint ``Linear(3072 + 12288,
3072)`` that the neuron module reads as two row-parallel column ranges (``proj_out_attn`` /
``proj_out_mlp``), and until HunyuanVideo's spec splits it (``SPLITS_BY_MODEL``: phase D's
writer, which Task E6 adopts) the writer would spend minutes on a directory holding it quantized
whole, which every rank then refuses as a pattern that names none of its modules, without
naming the cause; so a spec that names it is refused by name before anything is written
(:func:`fused_proj_out_targets`). At tp > 1 the MLP-downs (``ff.net.2``, ``ff_context.net.2``:
K 12288) are row-parallel with a K per rank of 6144, 3072 and 1536 at tp 2, 4 and 8, whole
512-wide kernel tiles; an override that quantizes ``to_out.0`` / ``to_add_out`` (K 3072) has
1536, 768 and 384, multiples of 128 with a partial last tile at tp 4 and 8. The token refiner's
``attn.to_out.0`` and ``ff.net.2`` are row layers of the same K: the row-parallel set names
them for overrides, since the FP8 twin of a row layer shards its weight along K whatever the
set says, and a weight the writer laid out as one K range is read in other block memberships
than it was scaled in wherever a rank's range is not whole tiles (``attn.to_out.0`` at tp 4
and 8).

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
# by default as for Qwen-Image: the dual blocks' ``to_out.0`` / ``to_add_out`` and the single
# blocks' ``proj_out``. FLUX's spec splits that ``proj_out`` (``SINGLE_BLOCK_SPLIT_LINEARS``),
# so its quantized checkpoint stores ``proj_out_attn`` / ``proj_out_mlp``, the two row-parallel
# linears the neuron model runs, as tensors of their own, read by name; untargeted they are the
# fused weight's column ranges in the source's dtype. The halves can be quantized, each per rank
# K range of its own, but they are override targets (``FLUX_SPLIT_TARGETS``) until the device
# picks the default (``docs/plans/2026-11-07-mxfp8-proj-out-rope.md`` D5): Wan's self-attention
# q/k/v went into its default unmeasured and cost the tp4 block 5x.
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
)

#: The single blocks' ``proj_out`` halves as targets (row-parallel: each quantized per rank K
#: range of the half), for ``targets=`` / ``DIFFLET_QUANT_TARGETS`` overrides.
FLUX_SPLIT_TARGETS: tuple[str, ...] = (rf"{_SINGLE}\.proj_out_(attn|mlp)",)

FLUX_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",
    rf"{_SINGLE}\.proj_out",
    rf"{_SINGLE}\.proj_out_(attn|mlp)",
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
# the FFN): the text cross-attention's q/k/v and the two FFN linears. The self-attention's q/k/v
# stay bf16 on measurement (trn3, tp4, 480x832x9, 50 steps; docs/verification/
# 2026-11-07-mxfp8-video-models.md section 4): with them quantized the block NEFF ran 56 ms
# against 10.5 in bf16 (the step 2.26 s against 0.43), a compiler effect around those custom-op
# outputs (they feed the global q/k norm, the rotary embedding and the attention kernel) that a
# one-core block does not reproduce; without them the step is 385 ms against 432 (0.89x).
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
# pattern per device-bisection group, in this order (the device campaign drops groups by index).
# The set starts out equal to FLUX's but is HunyuanVideo's own: the bisection may drop a group for
# HunyuanVideo alone.
HUNYUAN_VIDEO_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_q|to_k|to_v)",  # G1 dual image q/k/v (RMSNorm + rotary)
    rf"{_BLOCK}\.attn\.(add_q_proj|add_k_proj|add_v_proj)",  # G2 dual text q/k/v (256 rows)
    rf"{_BLOCK}\.ff\.net\.(0\.proj|2)",  # G3 dual image MLP
    rf"{_BLOCK}\.ff_context\.net\.(0\.proj|2)",  # G4 dual text MLP (256 rows)
    rf"{_SINGLE}\.attn\.(to_q|to_k|to_v)",  # G5 single joint q/k/v
    rf"{_SINGLE}\.proj_mlp",  # G6 single joint MLP-up
)

#: HunyuanVideo modules a reader may expect quantized that are deliberately not: the output
#: projections, the modulation, the single blocks' fused ``proj_out`` (and the neuron module's
#: names for its halves), the token refiner and the embedders, and the output layer.
HUNYUAN_VIDEO_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",
    rf"{_BLOCK}\.norm1(_context)?\.linear",
    rf"{_SINGLE}\.norm\.linear",
    rf"{_SINGLE}\.proj_out",  # until HunyuanVideo's spec splits it (Task E6)
    rf"{_SINGLE}\.proj_out_(attn|mlp)",  # the neuron module's names for it
    r"context_embedder\..*",
    r"time_text_embed\..*",
    r"norm_out\.linear",
    r"proj_out",
)

#: Every row layer of the neuron module with a checkpoint key of its own (the single blocks'
#: ``proj_out`` halves join with the split).
HUNYUAN_VIDEO_ROW_PARALLEL: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",  # not default targets; kept for overrides
    rf"{_BLOCK}\.(ff|ff_context)\.net\.2",
    # not default targets; kept for overrides
    r"context_embedder\.token_refiner\.refiner_blocks\.\d+\.(attn\.to_out\.0|ff\.net\.2)",
)

#: Why a HunyuanVideo spec may not name a single block's fused ``proj_out``, after the caller's
#: model or flag and the layer's name.
FUSED_PROJ_OUT_REASON: str = (
    "is read as two column ranges (proj_out_attn / proj_out_mlp) by the neuron module, and this "
    "model's MXFP8 checkpoint does not store it as those halves yet (phase D's split writer, "
    "adopted in roadmap task E6); drop it from the targets"
)


def fused_proj_out_targets(spec: QuantSpec, num_single_layers: int) -> list[str]:
    """The single blocks' fused ``proj_out`` names (``single_transformer_blocks.{i}.proj_out``,
    ``i < num_single_layers``) that ``spec`` quantizes, in block order: the targets a
    HunyuanVideo spec is refused for (``FUSED_PROJ_OUT_REASON``), on the driver before the writer
    runs and in the application. The names are checkpoint keys, so a pattern on the neuron
    module's ``proj_out_attn`` / ``proj_out_mlp`` matches none of them; the writer's dead-pattern
    refusal names that one. FLUX's single blocks have the same names and the same split, but
    FLUX's spec stores the halves (``SPLITS_BY_MODEL``), so its writer refuses the fused name
    itself, naming the halves; only HunyuanVideo calls this, until its spec splits too."""
    names = (f"single_transformer_blocks.{i}.proj_out" for i in range(num_single_layers))
    return [name for name in names if spec.matches(name)]


TARGETS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen_image": QWEN_IMAGE_TARGETS,
    "flux": FLUX_TARGETS,
    "ltx_2": LTX_2_TARGETS,
    "wan": WAN_TARGETS,
    "hunyuan_video": HUNYUAN_VIDEO_TARGETS,
}
ROW_PARALLEL_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen_image": QWEN_IMAGE_ROW_PARALLEL,
    "flux": FLUX_ROW_PARALLEL,
    "ltx_2": LTX_2_ROW_PARALLEL,
    "wan": WAN_ROW_PARALLEL,
    "hunyuan_video": HUNYUAN_VIDEO_ROW_PARALLEL,
}
#: The fused checkpoint linears each model's quantized checkpoint stores as two column ranges
#: (``QuantSpec.split_linears``). A model without an entry keeps every weight whole, and its
#: spec and directory digest are what they were before splits existed.
SPLITS_BY_MODEL: dict[str, tuple[str, ...]] = {"flux": SINGLE_BLOCK_SPLIT_LINEARS}


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
    "FUSED_PROJ_OUT_REASON",
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
    "fused_proj_out_targets",
    "row_parallel_for",
    "splits_for",
    "targets_for",
]
