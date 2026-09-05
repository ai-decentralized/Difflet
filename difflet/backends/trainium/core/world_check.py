"""Process-world consistency guard for Neuron component loads.

One NxD process has exactly one model-parallel world. Components that share a
process may use different ``tp_degree`` values, but every component whose
``world_size`` is greater than one must declare the *same* world — the world
of the process communicator established by the first component loaded. A
component that claims a smaller world in an already-initialized process
crashes the Neuron runtime at weight init instead of raising:

* HunyuanVideo tp2cp2 ulysses, 2026-08-30: DiT world 4 + VAE world 2 in one
  generate process -> SIGSEGV loading the VAE (fixed by 3923ae9).
* The same run, second attempt: a stale VAE NEFF compiled for world 4 loaded
  by a world-2 config -> ``std::out_of_range`` abort.
* Qwen-Image, 2026-07: TP=4/world=4 components + TP=1/world=1 VAE in one
  resident process, rejected by implicit, explicit-placement and reversed-
  order runtime experiments (docs/design/qwen_trn2_topology/
  03_adaptation_assessment.md). The validated resident topology keeps one
  world and lets TP differ (05_flux_runtime_validation.md).

The only sanctioned exception is a ``world_size == 1`` component: by
convention it runs alone on rank 0 (the Wan/Qwen ``vae`` stage subprocesses,
HunyuanVideo 1.5 ``process`` block-load mode), and
``MultiComponentApplication._component_load_rank_range`` clamps it there.

This module turns both crash signatures into ``NeuronWorldMismatchError``
*before* any weight touches the device. The checks run once per component at
load time; nothing here is on the inference path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NEURON_CONFIG_FILE_NAME = "neuron_config.json"

_TOPOLOGY_DOCS = (
    "docs/design/qwen_trn2_topology/03_adaptation_assessment.md "
    "(rejected: shared-process mixed world) and "
    "docs/design/qwen_trn2_topology/05_flux_runtime_validation.md "
    "(validated: one world, mixed TP)"
)


class NeuronWorldMismatchError(RuntimeError):
    """A component's world_size disagrees with its artifact or with the process world."""


@dataclass(frozen=True)
class ProcessWorld:
    """The model-parallel world this process committed to, and which component set it."""

    world_size: int
    established_by: str


_lock = threading.Lock()
_process_world: ProcessWorld | None = None


def process_world() -> ProcessWorld | None:
    """Return the world this process has committed to, or ``None`` before any load."""
    with _lock:
        return _process_world


def reset_process_world() -> None:
    """Forget the committed process world (tests only; a real process never needs this)."""
    global _process_world
    with _lock:
        _process_world = None


def read_saved_neuron_config(compiled_model_path: str | os.PathLike[str]) -> dict | None:
    """Return the ``neuron_config`` block saved next to a compiled artifact, or ``None``.

    Missing or unreadable files return ``None`` so the caller's own
    "artifact not found" handling stays authoritative.
    """
    path = os.path.join(str(compiled_model_path), NEURON_CONFIG_FILE_NAME)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        return None
    block = saved.get("neuron_config") if isinstance(saved, dict) else None
    return block if isinstance(block, dict) else None


def check_artifact_world(
    component: str,
    *,
    declared_world: int,
    declared_tp: int,
    saved_neuron_config: dict | None,
) -> None:
    """Raise if the compiled artifact was built for a different world/tp than declared.

    The compile path already refuses to reuse such an artifact
    (``MultiComponentApplication._compiled_config_matches``); this is the
    load-path twin, so a stale NEFF fails here instead of aborting inside the
    runtime with ``std::out_of_range``.
    """
    if saved_neuron_config is None:
        return
    mismatches = []
    for key, declared in (("world_size", declared_world), ("tp_degree", declared_tp)):
        saved = saved_neuron_config.get(key)
        if saved is not None and int(saved) != int(declared):
            mismatches.append(f"{key}: artifact={int(saved)} declared={int(declared)}")
    if mismatches:
        raise NeuronWorldMismatchError(
            f"{component}: the compiled artifact does not match this topology "
            f"({'; '.join(mismatches)}). Loading a NEFF under a different rank "
            "layout aborts inside the Neuron runtime (std::out_of_range on device, "
            "2026-08-30). Recompile the component for this topology, or point the "
            "load at the artifact that was compiled for it."
        )


def check_process_world(
    component: str,
    *,
    declared_world: int,
    local_ranks_size: int | None,
) -> None:
    """Commit this process to ``declared_world`` or raise if it already committed to another.

    ``world_size == 1`` components are exempt (standalone-stage convention).
    ``local_ranks_size`` is checked too when the caller passes one: a component
    cannot be spread over more or fewer ranks than it declares, except in the
    one-rank-per-process torchrun mode (``local_ranks_size == 1``), where the
    declared world is the global one.
    """
    global _process_world
    declared = int(declared_world)
    if declared <= 1:
        return
    if local_ranks_size is not None and int(local_ranks_size) > 1 and int(local_ranks_size) != declared:
        raise NeuronWorldMismatchError(
            f"{component}: declares world_size={declared} but is being loaded onto "
            f"local_ranks_size={int(local_ranks_size)} ranks. A component's declared "
            "world must equal the rank range it is loaded on; wire "
            "world_size=parallel.world_size for co-resident components."
        )
    with _lock:
        if _process_world is None:
            _process_world = ProcessWorld(world_size=declared, established_by=component)
            logger.info(
                "Neuron process world established at world_size=%d by %s", declared, component
            )
            return
        if _process_world.world_size != declared:
            raise NeuronWorldMismatchError(
                f"{component}: declares world_size={declared} but this process already "
                f"initialized its Neuron world at world_size={_process_world.world_size} "
                f"(by {_process_world.established_by}). One NxD process has exactly one "
                "model-parallel world; a component with a different world in the same "
                "process crashes the Neuron runtime at weight init (SIGSEGV on device: "
                "HunyuanVideo DiT world 4 + VAE world 2, 2026-08-30). Co-resident "
                "components must declare world_size=parallel.world_size (tp_degree may "
                "differ); a component that needs its own world must run in its own "
                f"stage process. See {_TOPOLOGY_DOCS}."
            )


def check_component_worlds(components: Iterable[tuple[str, int]]) -> int | None:
    """Pre-flight for a multi-component application: all multi-rank components must agree.

    ``components`` is ``(name, world_size)`` per component about to be loaded in
    this process. Returns the shared world (``None`` when every component is a
    standalone world-1 component). Raises before any component is loaded, so a
    mis-wired topology costs zero device time.
    """
    worlds = [(name, int(world)) for name, world in components]
    multi = [(name, world) for name, world in worlds if world > 1]
    distinct = sorted({world for _name, world in multi})
    if len(distinct) > 1:
        listing = ", ".join(f"{name}=w{world}" for name, world in worlds)
        raise NeuronWorldMismatchError(
            "Co-resident components declare different Neuron worlds "
            f"({listing}); one process has exactly one world. Every component "
            "loaded in the same process must declare world_size=parallel.world_size "
            "(tp_degree may differ), or a world_size=1 component must run in its own "
            "stage process. Mixed worlds crash the Neuron runtime at weight init "
            f"(SIGSEGV on device, 2026-08-30). See {_TOPOLOGY_DOCS}."
        )
    return distinct[0] if distinct else None
