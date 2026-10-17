"""Driver and rank helpers shared by the staged orchestrators that run on a non-AoT torchrun
backend (neuron): Wan (``wan.py``) and Qwen-Image (``qwen_image.py``).

Every helper is inert unless DIFFLET_BACKEND names such a backend (``torchrun_backend``), so
unset, trainium and tpu keep their stage paths unchanged. Torch-free except ``fallback_tracker``,
which runs in the ranks: the driver checks must not import torch before torchrun fans out.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

from difflet import envs
from difflet.cli import runner


def torchrun_backend() -> str | None:
    """The non-AoT torchrun-MPMD backend DIFFLET_BACKEND names (neuron), else None.

    The launcher's own opt-in rule (``runner._torchrun_mpmd_selected``): exactly when a stage
    runs as torchrun ranks, the DiT stage builds the backend's application through
    DiffletPipeline and the stage cache key names the backend.
    """
    if not runner._torchrun_mpmd_selected():
        return None
    from difflet.backends import get_backend

    return get_backend(envs.DIFFLET_BACKEND).name


#: The TeaCache options as the command line names them, with their argparse destinations.
TEACACHE_FLAGS = (
    ("--teacache-cadence", "teacache_cadence"),
    ("--teacache-online-delta", "teacache_online_delta"),
    ("--teacache-speedup", "teacache_speedup"),
    ("--teacache-calibration", "teacache_calibration"),
)


def torchrun_unsupported_options(args: argparse.Namespace) -> list[str]:
    """The options every torchrun backend's ranks reject in phase 1, as the command line names
    them: TeaCache.

    The ranks keep their own checks (the application's ``_reject_unsupported`` for TeaCache);
    this list lets the driver refuse the same options before torchrun starts tp*cp processes
    that import torch only to fail. A model adds its own (Qwen-Image: request batches,
    --cfg-parallel, several --shapes; Wan: several --shapes, since it runs SP and DP
    replicas). Torch-free.
    """
    return [flag for flag, name in TEACACHE_FLAGS if getattr(args, name, None) is not None]


def batch_options(args: argparse.Namespace) -> list[str]:
    """Request batches (--requests, a data-parallel worker), as the command line names them:
    for a model whose torchrun ranks run no request queue. Torch-free."""
    found = []
    if getattr(args, "requests", None) is not None:
        found.append("--requests")
    if getattr(args, "requests_dir", None) is not None:
        found.append("--requests/--dp (data-parallel worker)")
    return found


def reject_unsupported_torchrun_options(found: list[str], *, runs: str) -> None:
    """On a torchrun backend, ``found`` non-empty: print one message naming the options and exit
    2 before anything is launched (or downloaded, for ``run``). ``runs`` is what the model's
    ranks do support, request batching included ("tensor and context parallelism at one static
    shape, one request per launch"; Wan's DP workers drain a request queue). No-op on unset,
    trainium and tpu, which support these options or reject them where they always did."""
    if not found:
        return
    backend = torchrun_backend()
    if backend is None:
        return
    print(f"Error: the {backend} backend does not support {', '.join(found)} (it runs "
          f"{runs}, no TeaCache); nothing was launched.",
          file=sys.stderr, flush=True)
    raise SystemExit(2)


def root_compile_cache_at_cache_dir(cache_dir: str | None, *, tag: str) -> None:
    """A torchrun backend (neuron) compiles into the NEFF and HLO caches that
    ``runner.run_stage`` exports to its ranks under ``$DIFFLET_COMPILE_CACHE/neuron/``, while
    --cache-dir holds only the stage manifest. With --cache-dir given and DIFFLET_COMPILE_CACHE
    unset or empty, export --cache-dir (absolute) as DIFFLET_COMPILE_CACHE before any stage
    launch, so the compile cache lives under it; an exported DIFFLET_COMPILE_CACHE wins. Read
    from os.environ directly: envs fills in the default when the variable is unset. ``tag``
    prefixes the note (``[wan]``)."""
    if not cache_dir or os.environ.get("DIFFLET_COMPILE_CACHE"):
        return
    backend = torchrun_backend()
    if backend is None:
        return
    root = os.path.abspath(os.path.expanduser(cache_dir))
    os.environ["DIFFLET_COMPILE_CACHE"] = root
    print(f"[{tag}] backend {backend!r} compiles into --cache-dir: "
          f"DIFFLET_COMPILE_CACHE={root} (NEFF and HLO caches under {root}/neuron)",
          flush=True)


def fallback_tracker(pipe):
    """``track_fallbacks()`` for a pipeline on the neuron backend (it synchronizes the device);
    an empty record otherwise."""
    if getattr(getattr(pipe, "backend", None), "name", None) == "neuron":
        from difflet.backends.neuron.runtime import track_fallbacks

        return track_fallbacks()
    return contextlib.nullcontext([])
