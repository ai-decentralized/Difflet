"""Shared NeuronCore visibility parsing and selection."""

from __future__ import annotations

import os
import re

# Last-resort core visibility when nothing else can be established: no
# ``NEURON_RT_VISIBLE_CORES`` in the environment *and* ``neuron-ls`` unavailable
# (no Neuron tools, or a container without driver access). Four cores is the
# trn2.3xlarge shape Difflet is developed on. This used to be the *only* answer
# this module could give, which meant a 64-core trn2.48xlarge silently ran on
# four cores; ``difflet.planner.hardware.detected_core_count`` now supplies the
# real number whenever the driver can be queried.
DEFAULT_NEURON_CORE_IDS: tuple[int, ...] = (0, 1, 2, 3)


def default_neuron_core_ids() -> tuple[int, ...]:
    """The cores to assume when the environment does not pin a visibility list.

    Probes the driver via ``neuron-ls`` and falls back to
    ``DEFAULT_NEURON_CORE_IDS``. The probe result is cached in
    ``difflet.planner.hardware``, so repeated calls cost nothing.
    """

    from difflet.planner.hardware import detected_core_count

    detected = detected_core_count()
    if detected is None or detected < 1:
        return DEFAULT_NEURON_CORE_IDS
    return tuple(range(detected))


def resolve_available_neuron_core_ids(*, required_num_cores: int) -> tuple[int, ...]:
    """Resolve inherited Neuron core visibility, else the host's detected cores."""

    raw = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if raw is None or not raw.strip():
        core_ids = default_neuron_core_ids()
    else:
        resolved: list[int] = []
        for item in raw.split(","):
            token = item.strip()
            if not re.fullmatch(r"\d+(?:-\d+)?", token):
                raise ValueError(
                    "NEURON_RT_VISIBLE_CORES must contain comma-separated core IDs or ranges"
                )
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError("NEURON_RT_VISIBLE_CORES ranges must be ascending")
                resolved.extend(range(start, end + 1))
            else:
                resolved.append(int(token))
        core_ids = tuple(resolved)

    if len(core_ids) != len(set(core_ids)):
        raise ValueError("NEURON_RT_VISIBLE_CORES must not contain duplicate core IDs")
    if len(core_ids) < required_num_cores:
        raise ValueError(
            "NEURON_RT_VISIBLE_CORES does not provide enough cores: "
            f"requires {required_num_cores}, has {len(core_ids)}"
        )
    return core_ids


def select_neuron_core_ids(*, required_num_cores: int) -> tuple[int, ...]:
    """Select the required cores without escaping the inherited allocation."""

    return resolve_available_neuron_core_ids(required_num_cores=required_num_cores)[
        :required_num_cores
    ]
