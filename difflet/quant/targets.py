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
both (one spec, two quantized directories). The targets are the q/k/v of both attentions and
the FFN, written in **checkpoint-key spelling**: the neuron module
(``difflet/models/wan/modeling_wan.py``) names its FFN ``ffn.net_in`` / ``net_out`` where the
checkpoint has ``ffn.net.0.proj`` / ``net.2``, and the application applies the spec through its
``checkpoint_key`` (``quantize_module_(key_fn=...)``), so the patterns match the weights the
writer quantizes and the modules the loader swaps alike. Every ``to_out.0`` stays bf16 as
above; the per-block modulation is ``scale_shift_table``, a parameter, not a linear; the
``condition_embedder`` and ``proj_out`` are bf16 top-level layers. At tp > 1 ``to_out.0``
(K 5120) and ``ffn.net.2`` (K 13824) are row-parallel: ``net.2``'s K per rank is a multiple of
128 at tp 1, 2 and 4 (3456) but not at tp 8 (1728), which the writer refuses naming the layer.

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
# as for Qwen-Image: the dual blocks' ``to_out.0`` / ``to_add_out`` and the single blocks'
# ``proj_out`` (which the neuron model splits into ``proj_out_attn`` / ``proj_out_mlp``, read as
# column ranges of the checkpoint's ``proj_out``: a quantized target there would also need the
# writer to slice the block scales).
_SINGLE = r"single_transformer_blocks\.\d+"

FLUX_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)",
    rf"{_BLOCK}\.(ff|ff_context)\.net\.0\.proj",
    rf"{_BLOCK}\.(ff|ff_context)\.net\.2",
    rf"{_SINGLE}\.attn\.(to_q|to_k|to_v)",
    rf"{_SINGLE}\.proj_mlp",
)

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
# the FFN): the q/k/v of the self- and the text cross-attention and the two FFN linears.
_WAN_BLOCK = r"blocks\.\d+"

WAN_TARGETS: tuple[str, ...] = (
    rf"{_WAN_BLOCK}\.attn1\.(to_q|to_k|to_v)",
    rf"{_WAN_BLOCK}\.attn2\.(to_q|to_k|to_v)",
    rf"{_WAN_BLOCK}\.ffn\.net\.0\.proj",
    rf"{_WAN_BLOCK}\.ffn\.net\.2",
)

#: Wan modules a reader may expect quantized that are deliberately not: the output projections,
#: the modulation tables (parameters, not linears: they match no module, named so the table
#: reads complete), the embedders and the output layer.
WAN_KEPT_BF16: tuple[str, ...] = (
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

TARGETS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen_image": QWEN_IMAGE_TARGETS,
    "flux": FLUX_TARGETS,
    "ltx_2": LTX_2_TARGETS,
    "wan": WAN_TARGETS,
}
ROW_PARALLEL_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen_image": QWEN_IMAGE_ROW_PARALLEL,
    "flux": FLUX_ROW_PARALLEL,
    "ltx_2": LTX_2_ROW_PARALLEL,
    "wan": WAN_ROW_PARALLEL,
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


def default_spec(model: str, **overrides) -> QuantSpec:
    """The mxfp8 recipe for ``model``; ``targets=`` replaces the model's default layer set,
    ``compute=`` and ``tp_degree=`` complete it."""
    overrides.setdefault("row_parallel", row_parallel_for(model))
    overrides.setdefault("targets", targets_for(model))
    return QuantSpec(**overrides)


__all__ = [
    "FLUX_KEPT_BF16",
    "FLUX_ROW_PARALLEL",
    "FLUX_TARGETS",
    "LTX_2_KEPT_BF16",
    "LTX_2_ROW_PARALLEL",
    "LTX_2_TARGETS",
    "QWEN_IMAGE_KEPT_BF16",
    "QWEN_IMAGE_ROW_PARALLEL",
    "QWEN_IMAGE_TARGETS",
    "ROW_PARALLEL_BY_MODEL",
    "TARGETS_BY_MODEL",
    "WAN_KEPT_BF16",
    "WAN_ROW_PARALLEL",
    "WAN_TARGETS",
    "default_spec",
    "row_parallel_for",
    "targets_for",
]
