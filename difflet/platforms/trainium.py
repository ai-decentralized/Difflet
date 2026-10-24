"""The Trainium platform: staged Neuron NEFFs, one process driving every NeuronCore."""

from __future__ import annotations

import importlib.util
import os

from difflet.platforms.base import Platform, PlatformCapabilities


class TrainiumPlatform(Platform):
    name = "trainium"
    display_name = "Trainium"
    capabilities = PlatformCapabilities(
        requires_aot=True,
        single_process_multi_core=True,
        supports_torchrun_mpmd=False,
    )

    @classmethod
    def detect(cls) -> bool:
        return importlib.util.find_spec("torch_neuronx") is not None

    def prepare_runtime(self, parallel) -> None:
        return None

    def resolve_load_rank_range(
        self,
        *,
        start_rank_id: int | None,
        local_ranks_size: int | None,
    ) -> tuple[int | None, int | None]:
        if start_rank_id is not None or local_ranks_size is not None:
            return start_rank_id, local_ranks_size

        world_size = _env_int("WORLD_SIZE")
        rank = _env_int("RANK")
        if world_size is not None and world_size > 1 and rank is not None:
            # Experimental torchrun/PJRT MPMD path: each process owns one
            # local NeuronCore, so loading every rank from every process trips
            # Neuron's invalid-device check. The stable Flux path remains
            # single process + NEURON_RT_NUM_CORES=N.
            return rank, 1

        return None, None

    def layer_markers(self) -> tuple[type, type]:
        from difflet.backends.trainium.core.layer_boundary_marker import (
            ModuleMarkerEndWrapper,
            ModuleMarkerStartWrapper,
        )

        return ModuleMarkerStartWrapper, ModuleMarkerEndWrapper


def create_platform() -> TrainiumPlatform:
    return TrainiumPlatform()


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
