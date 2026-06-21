"""Trainium runtime policy."""

from __future__ import annotations

import os

from difflet.backends.base import BackendCapabilities, BackendRuntime


class TrainiumBackend(BackendRuntime):
    name = "trainium"
    capabilities = BackendCapabilities(
        requires_aot=True,
        single_process_multi_core=True,
        supports_torchrun_mpmd=False,
    )

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


def create_backend() -> TrainiumBackend:
    return TrainiumBackend()


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
