"""Retry cached compile failures instead of replaying them.

Scoped to the compile window and reverted afterwards, so nothing outside an
AOT compile sees the patched entry points.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# ``neuronx_distributed`` hardcodes ``retry_failed_compilation=False`` when it
# calls into libneuronxla (``trace/model_builder.py``). libneuronxla caches
# *failures*: a crashed compile leaves a ``model.log`` in the cache entry, and
# every later run re-raises that stored error without recompiling
# (``neuron_cc_wrapper.py::compile_cache_entry``):
#
#     if entry.log_exists():
#         if retry_failed_compilation: ...
#         else: raise subprocess.CalledProcessError(-1, "", stderr=error_log)
#
# Any crashed compile therefore wedges a model permanently -- transient or
# not -- and the replayed corpse surfaces as an instant, misleading
# ``died with <Signals.SIGHUP: 1>`` on every subsequent run. Force the retry so
# a cached failure is re-attempted instead of replayed; a genuine compile error
# still propagates, just from a real compile rather than from a stale log.
_PATCHED_COMPILE_FNS = ("neuron_xla_compile", "neuron_xla_wlo_compile")

_RETRY_FLAG = "_difflet_forces_retry"


def _forcing_retry(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        kwargs["retry_failed_compilation"] = True
        return fn(*args, **kwargs)

    setattr(wrapper, _RETRY_FLAG, True)
    return wrapper


@contextlib.contextmanager
def retrying_cached_failures() -> Iterator[None]:
    """Make NxD retry cached compile failures instead of replaying them.

    Re-entrant: nesting this around an inner compile scope is a no-op.

    Re-entrant but not thread-safe. The guard swaps module attributes, which
    are process-global, and the "already patched?" check is a plain read
    followed by a write. Nested scopes are fine -- the inner one sees the flag
    and does nothing -- but two threads entering concurrently could both read
    the original and double-wrap it, leaving a wrapper installed after both
    exit. Difflet compiles components sequentially, so this does not arise; a
    lock would be needed before compiling concurrently in one process.
    """
    try:
        from neuronx_distributed.trace import model_builder
    except ImportError:  # pragma: no cover - Neuron toolchain absent
        yield
        return

    restore: list[tuple[str, object]] = []
    for name in _PATCHED_COMPILE_FNS:
        original = getattr(model_builder, name, None)
        if original is None or getattr(original, _RETRY_FLAG, False):
            # Absent in this toolchain version, or already patched by an
            # enclosing compile scope.
            continue
        setattr(model_builder, name, _forcing_retry(original))
        restore.append((name, original))

    try:
        yield
    finally:
        for name, original in restore:
            setattr(model_builder, name, original)
