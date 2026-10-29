"""Wan's two DiT-step levers on the neuron backend, Wan only (F6a's proposals 1 and 2,
benchmark/trn2/four_model_native_vs_aot_2026-10-26.md, "Wan2.1 native vs AoT DiT step: cause").

* **The ccop overlap flag.** neuronx-cc's ``--tensorizer-options=--enable-ccop-compute-overlap``
  (``CCOP_OVERLAP_FLAG``), the flag the AoT path compiles with. The Wan applications built for
  the neuron device (the DiT and the device umT5 always; the device VAE decoder only when its
  Wan callers ask, ``ccop_overlap=True``, because Qwen-Image builds the same decoder class and
  must keep its compile flags) call ``apply_ccop_overlap(device, self)`` in ``__init__``, before
  anything compiles: it appends the flag to this process's ``NEURON_CC_FLAGS``, once, after any
  flags the user exported. torch-neuronx reads that variable whenever it builds a kernel and puts
  it in the NEFF cache key (``difflet.backends.neuron.compile.append_cc_flags``), so every
  compile of the process gets the flag and no NEFF compiled without it is reused. The
  application holds the flag: when the last application holding it is released (garbage
  collected, or ``release_ccop_overlap``), the flag added here leaves ``NEURON_CC_FLAGS`` again,
  so a model an API caller builds after its Wan applications are gone compiles without it.
  While a Wan application lives, every compile of the process gets the flag; the CLI runs each
  Wan stage in processes of its own, so no other model's compiles see it there. A user
  ``--tensorizer-options`` in ``NEURON_CC_FLAGS`` wins: the flag is not added then, and the stage
  keys do not name it (``ccop_overlap_applies``).
* **The scatter-free RoPE.** The DiT's self-attentions apply the rotary embedding with
  ``apply_rotary_emb_stacked`` (``difflet/backends/neuron/ops_impl/embeddings.py``): the same
  math as the shared ``apply_rotary_emb``, bit-identical on CPU, but the interleaved result is
  built with ``stack`` + ``flatten`` instead of two strided writes, which torch-neuronx lowers
  to scatters. ``NeuronWanTransformerApplication.build_module`` sets it on the module it builds
  (``WanAttention.rotary_fn``), so the shared helper, the modeling file's default (the AoT and
  TPU paths), Flux and HunyuanVideo are unchanged.

Measured together on Wan2.1 14B at 480x832, 9 frames, TP4 (that document's "Task 6" section):
1065.5 -> 1038.5 ms per DiT step, within the quality gate on two seeds.

Both are on by default. ``DIFFLET_NEURON_WAN_CCOP_OVERLAP=0`` and
``DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE=0`` (or off/false/no) turn each off (``difflet/envs.py``).
The Wan neuron stage keys name what is on (``stage_key_fields``), so a manifest written for
other compiled graphs is not taken for these; with both off the keys are the ones the stages
had before the levers.

Torch-free: the CLI driver keys the stages with it.
"""

from __future__ import annotations

import os
import threading
import weakref
from collections.abc import Mapping
from typing import Any

from difflet import envs

#: AoT's tensorizer option: overlap collectives with compute.
CCOP_OVERLAP_FLAG = "--tensorizer-options=--enable-ccop-compute-overlap"
#: The stage key value of the scatter-free RoPE.
STACKED_ROPE = "stack"


def ccop_overlap_enabled() -> bool:
    return bool(envs.DIFFLET_NEURON_WAN_CCOP_OVERLAP)


def scatter_free_rope_enabled() -> bool:
    return bool(envs.DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE)


def ccop_overlap_applies(env: Mapping[str, str] | None = None) -> bool:
    """Whether a Wan neuron process with ``env`` (``os.environ`` by default) compiles with
    ``CCOP_OVERLAP_FLAG``: the switch on, and the user's ``NEURON_CC_FLAGS`` either naming it
    already or holding no other ``--tensorizer-options`` (``append_cc_flags`` leaves a user's in
    place). The stage keys and ``apply_ccop_overlap`` both decide by it."""
    if not ccop_overlap_enabled():
        return False
    from difflet.backends.neuron.compile import NEURON_CC_FLAGS_ENV, cc_flags_to_add

    source = os.environ if env is None else env
    tokens = (source.get(NEURON_CC_FLAGS_ENV) or "").split()
    return CCOP_OVERLAP_FLAG in tokens or bool(cc_flags_to_add(CCOP_OVERLAP_FLAG, env=source))


class _Hold:
    """The applications holding the flag in this process, and what to restore when the last one
    goes: the ``NEURON_CC_FLAGS`` value before the flag was added and the value set then (None
    while the flag present is not one ``apply_ccop_overlap`` added, e.g. the user's)."""

    lock = threading.RLock()
    holders: dict[int, weakref.finalize] = {}
    before: str | None = None
    set_to: str | None = None


def apply_ccop_overlap(device: Any, owner: Any) -> str | None:
    """On the neuron device, unless switched off, append ``CCOP_OVERLAP_FLAG`` to this process's
    ``NEURON_CC_FLAGS`` (once) on behalf of the application ``owner``, which holds it until it is
    released (``release_ccop_overlap``, or garbage collected); return the value now set, or None
    when nothing was done (another device, or the switch off). A user ``--tensorizer-options``
    is left in place, with a warning (``append_cc_flags``), and nothing is held then."""
    device_type = getattr(device, "type", None) or str(device).split(":", 1)[0]
    if device_type != "neuron" or not ccop_overlap_enabled():
        return None
    from difflet.backends.neuron.compile import NEURON_CC_FLAGS_ENV, append_cc_flags

    with _Hold.lock:
        before = os.environ.get(NEURON_CC_FLAGS_ENV)
        had = CCOP_OVERLAP_FLAG in (before or "").split()
        value = append_cc_flags(CCOP_OVERLAP_FLAG)
        if CCOP_OVERLAP_FLAG not in value.split():
            return value
        if not had:
            _Hold.before, _Hold.set_to = before, value
        key = id(owner)
        if key not in _Hold.holders:
            finalizer = weakref.finalize(owner, _release, key)
            finalizer.atexit = False  # nothing to restore for a process that is exiting
            _Hold.holders[key] = finalizer
    return value


def release_ccop_overlap(owner: Any) -> None:
    """``owner`` no longer holds the flag; once no application does, the flag added by
    ``apply_ccop_overlap`` leaves ``NEURON_CC_FLAGS`` (the value before it, when nothing else
    changed the variable since; else the value without the flag). A flag the user exported
    stays."""
    finalizer = _Hold.holders.get(id(owner))
    if finalizer is not None:
        finalizer()


def _release(key: int) -> None:
    from difflet.backends.neuron.compile import NEURON_CC_FLAGS_ENV

    with _Hold.lock:
        _Hold.holders.pop(key, None)
        if _Hold.holders or _Hold.set_to is None:
            return
        before, set_to = _Hold.before, _Hold.set_to
        _Hold.before = _Hold.set_to = None
        current = os.environ.get(NEURON_CC_FLAGS_ENV)
        if current == set_to:
            restored = before
        else:
            rest = [t for t in (current or "").split() if t != CCOP_OVERLAP_FLAG]
            restored = " ".join(rest) if rest else None
        if restored is None:
            os.environ.pop(NEURON_CC_FLAGS_ENV, None)
        else:
            os.environ[NEURON_CC_FLAGS_ENV] = restored


def _forget_holders() -> None:
    """Drop every hold without touching ``NEURON_CC_FLAGS`` (the unit tests restore the variable
    themselves after each test, and an application of one test collected during a later one
    must not change that test's environment)."""
    with _Hold.lock:
        for finalizer in _Hold.holders.values():
            finalizer.detach()
        _Hold.holders.clear()
        _Hold.before = _Hold.set_to = None


def stage_key_fields(stage: str) -> dict[str, str]:
    """The Wan neuron stage key fields of the levers that are on: ``neuron_cc_flags`` on both
    stages when the stage processes will compile with the flag (``ccop_overlap_applies``: every
    compile of a Wan process gets it), ``rope`` on the transformer stage. The torchrun stages
    inherit this process's environment, so the decision is the one they make."""
    fields: dict[str, str] = {}
    if stage in ("transformer", "vae") and ccop_overlap_applies():
        fields["neuron_cc_flags"] = CCOP_OVERLAP_FLAG
    if stage == "transformer" and scatter_free_rope_enabled():
        fields["rope"] = STACKED_ROPE
    return fields


__all__ = [
    "CCOP_OVERLAP_FLAG",
    "STACKED_ROPE",
    "apply_ccop_overlap",
    "ccop_overlap_applies",
    "ccop_overlap_enabled",
    "release_ccop_overlap",
    "scatter_free_rope_enabled",
    "stage_key_fields",
]
