"""Shared NeuronCore visibility parsing and selection."""

from __future__ import annotations

import os
import re

DEFAULT_NEURON_CORE_IDS: tuple[int, ...] = (0, 1, 2, 3)


def resolve_available_neuron_core_ids(*, required_num_cores: int) -> tuple[int, ...]:
    """Resolve inherited Neuron core visibility, defaulting to the four-core host."""

    raw = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if raw is None or not raw.strip():
        core_ids = DEFAULT_NEURON_CORE_IDS
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
