"""Children that die with their parent: a Linux ``PR_SET_PDEATHSIG`` ``preexec_fn``.

Shared by the DP router's workers (SIGKILL: a worker owns its devices outright) and the
CUDA ``torchrun`` launch (SIGTERM: torchrun's elastic agent terminates its ranks on
SIGTERM, whereas a SIGKILLed agent would orphan them, each still holding its GPU).
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable

_PR_SET_PDEATHSIG = 1


def die_with_parent(sig: int) -> Callable[[], None]:
    """A ``preexec_fn`` that has the kernel send ``sig`` to the child when its parent dies.

    A no-op off Linux (``prctl`` is Linux-only). If the parent already died between the
    fork and the ``prctl`` (the child was re-parented to init), the child signals itself.
    """

    def _preexec() -> None:
        if not sys.platform.startswith("linux"):
            return
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, int(sig), 0, 0, 0)
        if os.getppid() == 1:  # the parent already died between fork and here
            os.kill(os.getpid(), int(sig))

    return _preexec


__all__ = ["die_with_parent"]
