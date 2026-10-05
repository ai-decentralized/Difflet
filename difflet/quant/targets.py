"""Per-model FP8 target sets: which linears carry fp8 weights.

A target is either a dotted suffix (``to_q`` matches ``blocks.3.attn1.to_q``) or
a glob with ``*`` matched against the full module name
(``single_transformer_blocks.*.proj_out`` matches the block projections but not
the root ``proj_out``). The sets follow FastVideo's FP8 layer set: attention
q/k/v/out (both streams of double-stream blocks) and the FFN in/out
projections. Embedders, modulation / adaLN, ``norm_out``, the root
``proj_out`` and HunyuanVideo's token refiner stay bf16.
"""

from __future__ import annotations

_ATTN = ("to_q", "to_k", "to_v", "to_out.0")
_ATTN_CTX = ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")

# Wan: Difflet's ``ffn.net_in`` / ``ffn.net_out`` (traced + CPU model) and
# diffusers' ``ffn.net.0.proj`` / ``ffn.net.2`` (the HF checkpoint the offline
# quantizer reads). Suffix-matched: Wan's only parallel linears are these.
WAN_TARGETS: tuple[str, ...] = (
    *_ATTN,
    "ffn.net_in",
    "ffn.net_out",
    "ffn.net.0.proj",
    "ffn.net.2",
)

FLUX_TARGETS: tuple[str, ...] = (
    *(f"transformer_blocks.*.attn.{n}" for n in (*_ATTN, *_ATTN_CTX)),
    "transformer_blocks.*.ff.net.0.proj",
    "transformer_blocks.*.ff.net.2",
    "transformer_blocks.*.ff_context.net.0.proj",
    "transformer_blocks.*.ff_context.net.2",
    *(f"single_transformer_blocks.*.attn.{n}" for n in ("to_q", "to_k", "to_v")),
    "single_transformer_blocks.*.proj_mlp",
    # HF fused projection; split into the two device halves at load.
    "single_transformer_blocks.*.proj_out",
    "single_transformer_blocks.*.proj_out_attn",
    "single_transformer_blocks.*.proj_out_mlp",
)

QWEN_IMAGE_TARGETS: tuple[str, ...] = (
    *(f"transformer_blocks.*.attn.{n}" for n in (*_ATTN, *_ATTN_CTX)),
    "transformer_blocks.*.img_mlp.net.0.proj",
    "transformer_blocks.*.img_mlp.net.2",
    "transformer_blocks.*.txt_mlp.net.0.proj",
    "transformer_blocks.*.txt_mlp.net.2",
)

# Same block layout as FLUX (the anchored globs exclude the token refiner), minus the
# double blocks' text-stream q / k projections. With add_q_proj or add_k_proj in fp8 the
# tp4 device graph returns all-NaN, even with the activation clamped and with static
# scales (trn2, 2026-10-04, real weights + real text; tp1 and every other layer type are
# clean, add_v_proj included). Both feed the per-head RMSNorm (norm_added_q / _k); the
# text stream is 256 of ~10.5k tokens, so keeping them bf16 costs ~nothing.
_HV_BF16_TEXT_QK = ("transformer_blocks.*.attn.add_q_proj", "transformer_blocks.*.attn.add_k_proj")
HUNYUAN_VIDEO_TARGETS: tuple[str, ...] = tuple(t for t in FLUX_TARGETS if t not in _HV_BF16_TEXT_QK)

LTX_2_TARGETS: tuple[str, ...] = tuple(
    f"transformer_blocks.*.{attn}.{n}"
    for attn in (
        "attn1",
        "attn2",
        "audio_attn1",
        "audio_attn2",
        "audio_to_video_attn",
        "video_to_audio_attn",
    )
    for n in _ATTN
) + (
    "transformer_blocks.*.ff.net.0.proj",
    "transformer_blocks.*.ff.net.2",
    "transformer_blocks.*.audio_ff.net.0.proj",
    "transformer_blocks.*.audio_ff.net.2",
)

# Spellings that exist only in the HF checkpoint the offline quantizer reads,
# never as a module of the traced device model: Wan's diffusers FFN names and
# the fused single-block proj_out that FLUX / HunyuanVideo split at load.
HF_ONLY_TARGETS: frozenset[str] = frozenset(
    {"ffn.net.0.proj", "ffn.net.2", "single_transformer_blocks.*.proj_out"}
)


def device_targets(targets: "tuple[str, ...] | list[str]") -> list[str]:
    """The targets the device-side convert must find as modules."""
    return [t for t in targets if t not in HF_ONLY_TARGETS]


TARGETS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "wan": WAN_TARGETS,
    "flux": FLUX_TARGETS,
    "qwen_image": QWEN_IMAGE_TARGETS,
    "hunyuan_video": HUNYUAN_VIDEO_TARGETS,
    "ltx_2": LTX_2_TARGETS,
}


def targets_for(model_type: str) -> tuple[str, ...]:
    """The target set of a wired model type; ``ValueError`` names the wired ones.

    ``DIFFLET_FP8_TARGETS`` (comma-separated targets) overrides the set — an
    experiment switch for per-layer speed studies; it changes the spec, so the
    quantized checkpoint gets its own directory.
    """
    import os

    override = os.environ.get("DIFFLET_FP8_TARGETS")
    if override:
        return tuple(t.strip() for t in override.split(",") if t.strip())
    try:
        return TARGETS_BY_MODEL[model_type]
    except KeyError:
        raise ValueError(
            f"FP8 PTQ is not wired for model type {model_type!r}; "
            f"wired: {', '.join(TARGETS_BY_MODEL)}"
        ) from None


__all__ = [
    "FLUX_TARGETS",
    "HUNYUAN_VIDEO_TARGETS",
    "LTX_2_TARGETS",
    "QWEN_IMAGE_TARGETS",
    "TARGETS_BY_MODEL",
    "WAN_TARGETS",
    "targets_for",
]
