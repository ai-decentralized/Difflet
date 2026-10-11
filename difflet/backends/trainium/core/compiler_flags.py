"""Shared neuronx-cc tensorizer options for the DiT backbones, and the Wan transformer's
virtual core size.

Every backbone passes ``--tensorizer-options='--enable-ccop-compute-overlap …'``; this module
owns the string so the extras are consistent per model and so an experiment can be switched
on for any model from the environment without a code change:

``DIFFLET_TENSORIZER_EXTRA``   appended to every backbone's tensorizer options (space separated).
``DIFFLET_STRIDED_DMA``        ``1`` / ``0`` forces ``--vectorize-strided-dma`` on / off; unset keeps
                               the per-model default in ``STRIDED_DMA_DEFAULTS``.

Wan 2.1 on trn2 tp4 (ported from 38997f9 on the quantization branch): tracing the transformer
with ``NEURON_RT_VIRTUAL_CORE_SIZE=2`` makes ``ops_impl.attention`` use ``attention_cte[2]``,
which runs the NKI attention on both physical cores of the LNC2 pair instead of one. Measured
480x832x9: 573.7 -> 486.1 ms per DiT call together with ``--vectorize-strided-dma`` (2026-10-05);
480x832x81: 9.59 -> 6.12 s per DiT call (2026-10-11, profiler capture). At 81 frames the
attention is ~43% of the single-core graph's wall time, so nearly all of the gain is the
virtual core size (2-layer A/B: -33.8% from VC2, a further -2.9% from strided DMA).

The compile-cache key does not see compiler arguments, so the cache code includes
``tensorizer_cache_inputs(model)`` (additive-only: empty when no extra is active, so every
artifact compiled before this module keeps its key).
"""

from __future__ import annotations

import os

CCOP_OVERLAP_FLAG = "--enable-ccop-compute-overlap"
STRIDED_DMA_FLAG = "--vectorize-strided-dma"

# model_name (as in CacheSpec.model_name / the orchestrators' _MODEL_TYPE) -> default.
STRIDED_DMA_DEFAULTS: dict[str, bool] = {
    "wan": True,
    "flux": False,
    "qwen_image": False,
    "hunyuan_video": False,
    "ltx_2": False,
}

# Backbones that build their --tensorizer-options from this module. Only these get extras (and
# the matching cache-key entry); every other backbone keeps its fixed option string, so an env
# switch can never put an extra into a key whose NEFF was compiled without it.
_CONSUMERS = frozenset({"wan"})

# model_name -> model-specific env var appended after DIFFLET_TENSORIZER_EXTRA (legacy switch).
_MODEL_EXTRA_ENV: dict[str, str] = {"wan": "DIFFLET_WAN_TENSORIZER_EXTRA"}


def strided_dma_enabled(model: str | None) -> bool:
    value = os.environ.get("DIFFLET_STRIDED_DMA")
    if value is None or value == "":
        return STRIDED_DMA_DEFAULTS.get(model or "", False)
    return value.strip() not in ("0", "false", "False", "no", "off")


def tensorizer_extras(model: str | None) -> list[str]:
    """The extra tensorizer flags beyond ``--enable-ccop-compute-overlap``, in order:
    the per-model default (strided DMA), the generic ``DIFFLET_TENSORIZER_EXTRA`` and the
    model-specific env (e.g. ``DIFFLET_WAN_TENSORIZER_EXTRA``). Duplicates are dropped."""
    if (model or "") not in _CONSUMERS:
        return []
    extras: list[str] = []
    if strided_dma_enabled(model):
        extras.append(STRIDED_DMA_FLAG)
    for name in ("DIFFLET_TENSORIZER_EXTRA", _MODEL_EXTRA_ENV.get(model or "")):
        if name:
            extras.extend(os.environ.get(name, "").split())
    seen: list[str] = []
    for flag in extras:
        if flag not in seen:
            seen.append(flag)
    return seen


def tensorizer_options(model: str | None) -> str:
    """The value for ``--tensorizer-options='…'``."""
    return " ".join([CCOP_OVERLAP_FLAG, *tensorizer_extras(model)])


def tensorizer_cache_inputs(model: str | None) -> dict:
    """Cache-key contribution: ``{"tensorizer_extras": [...]}`` when any extra is active,
    ``{}`` otherwise (so artifacts compiled before this module keep their key)."""
    extras = tensorizer_extras(model)
    return {"tensorizer_extras": extras} if extras else {}


# NEURON_RT_VIRTUAL_CORE_SIZE the Wan transformer is traced (and run) with. The CLI orchestrator,
# the resident serving worker and the step benchmark all read it here so the three paths trace
# the same graph. DIFFLET_WAN_VIRTUAL_CORE_SIZE=1 restores the single-core graph (A/B).
WAN_TRANSFORMER_VIRTUAL_CORE_SIZE = 2


def wan_transformer_virtual_core_size() -> int | None:
    """The virtual core size for the Wan transformer graph (None = leave the environment unset)."""
    override = os.environ.get("DIFFLET_WAN_VIRTUAL_CORE_SIZE", "").strip()
    if override:
        value = int(override)
        return value if value > 1 else None
    return WAN_TRANSFORMER_VIRTUAL_CORE_SIZE
