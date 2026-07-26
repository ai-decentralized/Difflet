"""Keep preloaded allocators away from the AOT compiler.

Scoped to the compile window and reverted afterwards, so the runtime path is
left as the rest of Difflet set it up.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# ``difflet run``/``generate`` re-exec themselves with jemalloc preloaded for a
# ~17% faster warm weight load (``difflet/cli/main.py::_ensure_jemalloc``).
# ``LD_PRELOAD`` is inherited by every subprocess, including the ``neuronx-cc``
# compiler that libneuronxla spawns, and neuronx-cc's native extensions abort
# under jemalloc during garbage collection at fork:
#
#     free(): invalid pointer      (or: free(): invalid size)
#     Fatal Python error: Aborted
#       File ".../neuron_dtypes/__init__.py", line 160 in dtype
#
# main.py tries to avoid this by preloading only for the weight-loading
# commands and excluding the standalone ``compile`` — but ``difflet run`` also
# compiles whenever the cache is cold, so a command-name allowlist cannot be
# the safety boundary. Strip the allocator for the duration of any compile
# instead: the runtime keeps jemalloc and its speedup, the compiler never sees
# it. libneuronxla gives libtcmalloc the same treatment before spawning the
# compiler (``libneuronxla/neuron_cc_wrapper.py::call_neuron_compiler``); its
# filter simply does not cover jemalloc.
_ALLOCATOR_PRELOAD_MARKERS = ("libjemalloc",)


def strip_allocator_preloads(ld_preload: str) -> str:
    """Drop allocator-override entries from an ``LD_PRELOAD`` value.

    Other preloads are preserved, in order.
    """
    return ":".join(
        entry
        for entry in ld_preload.split(":")
        if entry and not any(marker in entry for marker in _ALLOCATOR_PRELOAD_MARKERS)
    )


@contextlib.contextmanager
def without_allocator_preload() -> Iterator[None]:
    """Hide preloaded allocators from subprocesses spawned in this block.

    Re-entrant: nesting this around an inner compile scope is a no-op.

    Assumes compilation never overlaps weight loading. ``os.environ`` is
    process-global, so *anything* spawned inside this block runs without the
    allocator preload, not just ``neuronx-cc``. Difflet is strictly
    compile-then-load today, so in practice the window only ever covers the
    compiler. If compilation is ever pipelined against loading -- or a ``--dp``
    router starts workers mid-compile -- this has to become a per-subprocess
    env override instead, which means patching
    ``libneuronxla.neuron_cc_wrapper.call_neuron_compiler``.

    Verified not to perturb results: a cold-cache FLUX run under this guard
    produces a byte-identical PNG to one built with jemalloc absent entirely.
    """
    original = os.environ.get("LD_PRELOAD")
    if not original:
        yield
        return

    stripped = strip_allocator_preloads(original)
    if stripped == original:
        yield
        return

    logger.debug("Stripping allocator preloads from LD_PRELOAD for neuronx-cc")
    if stripped:
        os.environ["LD_PRELOAD"] = stripped
    else:
        del os.environ["LD_PRELOAD"]
    try:
        yield
    finally:
        os.environ["LD_PRELOAD"] = original
