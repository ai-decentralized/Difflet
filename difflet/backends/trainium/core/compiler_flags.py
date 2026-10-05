"""Shared neuronx-cc tensorizer options for the DiT backbones.

Every backbone passes ``--tensorizer-options='--enable-ccop-compute-overlap …'``; this module
owns the string so the extras are consistent per model and so an experiment can be switched
on for any model from the environment without a code change:

``DIFFLET_TENSORIZER_EXTRA``   appended to every backbone's tensorizer options (space separated).
``DIFFLET_STRIDED_DMA``        ``1`` / ``0`` forces ``--vectorize-strided-dma`` on / off; unset keeps
                               the per-model default in ``STRIDED_DMA_DEFAULTS``.

Wan 2.1 measured on trn2 tp4 (2026-10-05, same host / day / compiler): ``--vectorize-strided-dma``
together with ``NEURON_RT_VIRTUAL_CORE_SIZE=2`` cut the DiT step from 573.7 to 486.1 ms (bf16)
and from 585.1 to 473.0 ms (fp8 static), numerics unchanged, so Wan defaults it on. The other
models default it off until they are screened (``docs/verification/2026-10-05-fp8-step-levers.md``).

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
    # LTX-2 480x704x49 tp4 (2026-10-05, with VC2): bf16 459.7 -> 413.7 ms per DiT step.
    "ltx_2": True,
}

# NEURON_RT_VIRTUAL_CORE_SIZE for the models whose orchestrator runs in-process and used to
# follow the caller's environment (Wan has its own switch in its orchestrator; FLUX forces 2 in
# its backbone; Qwen-Image / HunyuanVideo set 2 in their orchestrators). ``None`` = leave the
# environment alone. ``DIFFLET_VIRTUAL_CORE_SIZE=1`` restores the single-core graph.
VIRTUAL_CORE_DEFAULTS: dict[str, int | None] = {
    "ltx_2": 2,
}


def virtual_core_size(model: str | None) -> int | None:
    """The virtual core size a model's DiT graph is traced for (None = environment default)."""
    default = VIRTUAL_CORE_DEFAULTS.get(model or "")
    if default is None:
        return None
    override = os.environ.get("DIFFLET_VIRTUAL_CORE_SIZE", "").strip()
    if override:
        value = int(override)
        return value if value > 1 else None
    return default


def apply_virtual_core_env(model: str | None) -> int | None:
    """Export ``NEURON_RT_VIRTUAL_CORE_SIZE`` for an in-process compile / load of ``model``
    when the model has a default (see ``VIRTUAL_CORE_DEFAULTS``); returns the value set."""
    value = virtual_core_size(model)
    if value is not None:
        os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = str(value)
    elif model in VIRTUAL_CORE_DEFAULTS:
        os.environ.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)  # explicit single-core request
    return value


def virtual_core_cache_inputs(model: str | None) -> dict:
    """Cache-key contribution: ``{"virtual_core_size": N}`` for a model with a default, ``{}``
    otherwise (additive-only, like ``tensorizer_cache_inputs``)."""
    value = virtual_core_size(model)
    return {"virtual_core_size": value} if value is not None else {}

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
