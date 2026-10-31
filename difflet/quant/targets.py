"""Which linears each DiT quantizes. Torch-free.

Only the per-token projections go to FP8: attention q/k/v/out and the MLP up/down. They hold
about 67% of Qwen-Image's parameters and essentially all of its linear-layer compute. Everything
that conditions the blocks stays bf16: the per-block modulation (``img_mod.1``/``txt_mod.1``: a
third of the parameters, but applied to one ``[B, D]`` vector, so no compute to win and the
scale/shift/gate it produces is precision-sensitive), the embedders, ``img_in``/``txt_in``,
``norm_out`` and ``proj_out``. This is the layer choice NVIDIA ModelOpt and FastVideo make for
DiTs too.

Patterns are regexes full-matched against diffusers' module names (the ``transformer.`` trace
prefix is stripped by :meth:`QuantSpec.matches`).
"""

from __future__ import annotations

from difflet.quant.spec import QuantSpec

_BLOCK = r"transformer_blocks\.\d+"

QWEN_IMAGE_TARGETS: tuple[str, ...] = (
    rf"{_BLOCK}\.attn\.(to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj)",
    rf"{_BLOCK}\.attn\.(to_out\.0|to_add_out)",
    rf"{_BLOCK}\.(img_mlp|txt_mlp)\.net\.0\.proj",
    rf"{_BLOCK}\.(img_mlp|txt_mlp)\.net\.2",
)

#: Qwen-Image modules a reader may expect quantized that are deliberately not.
QWEN_IMAGE_KEPT_BF16: tuple[str, ...] = (
    rf"{_BLOCK}\.(img_mod|txt_mod)\.1",
    r"time_text_embed\..*",
    r"img_in",
    r"txt_in",
    r"norm_out\.linear",
    r"proj_out",
)

TARGETS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "qwen_image": QWEN_IMAGE_TARGETS,
}


def targets_for(model: str) -> tuple[str, ...]:
    try:
        return TARGETS_BY_MODEL[model]
    except KeyError:
        raise ValueError(
            f"no FP8 target layers are defined for {model!r}; "
            f"models with FP8 support: {sorted(TARGETS_BY_MODEL)}"
        ) from None


def default_spec(model: str, **overrides) -> QuantSpec:
    """The v1 recipe for ``model``: fp8, per-tensor weights, static activations."""
    return QuantSpec(targets=targets_for(model), **overrides)


__all__ = [
    "QWEN_IMAGE_KEPT_BF16",
    "QWEN_IMAGE_TARGETS",
    "TARGETS_BY_MODEL",
    "default_spec",
    "targets_for",
]
