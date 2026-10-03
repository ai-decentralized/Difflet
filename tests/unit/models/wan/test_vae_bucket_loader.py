"""Ensure graph phase changes release device-owning execution objects."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from difflet.backends.trainium.wan.vae import _WanBucketLoader


def test_only_current_partition_models_stay_alive(monkeypatch):
    active = set()
    events = []
    weights = [object()]

    class ExportModel:
        def __init__(self, index):
            self.index = index

        def save_neff(self, path):
            Path(path).write_bytes(str(self.index).encode())

        def save_metaneff(self, path):
            Path(path).write_bytes(b"metadata")

    class ExecutionModel:
        def __init__(self, neff, metadata, local_ranks, world_size):
            assert (metadata, local_ranks, world_size) == (b"metadata", 1, 1)
            self.index = int(neff)

        def initialize(self, state, loaded_weights, start_rank):
            assert state == [] and loaded_weights is weights and start_rank == 0
            active.add(self.index)
            events.append(self.index)
            assert len(active) <= 2

        def forward(self, inputs):
            return inputs

        def __del__(self):
            active.discard(self.index)

    monkeypatch.setattr(torch.classes.neuron, "SPMDModel", ExecutionModel)
    inputs = [
        (torch.zeros(1, 4, 1, 2, 2), torch.zeros(1)),
        (torch.zeros(1, 8, 1, 1, 4, 4), torch.zeros(1)),
        (torch.zeros(1, 4, 1, 2, 2), torch.zeros(1, 8, 2, 2, 2)),
        (torch.zeros(1, 8, 1, 1, 4, 4), torch.zeros(1, 8, 2, 4, 4)),
    ]
    routes = {str([list(t.shape) for t in xs]): ("decoder", i) for i, xs in enumerate(inputs)}
    exported = SimpleNamespace(models=[ExportModel(i) for i in range(4)])
    identity = lambda xs: xs
    nxd = SimpleNamespace(
        input_shape_map=routes,
        models=SimpleNamespace(named_children=lambda: iter([("decoder", exported)])),
        flattener_map=SimpleNamespace(named_children=lambda: iter(
            [(f"decoder_{i}", identity) for i in range(4)]
        )),
        packer=identity,
    )
    loader = _WanBucketLoader(nxd, weights, 0)
    for index in (0, 1, 2, 3, 2, 3, 0, 1):
        result = loader.run(inputs[index])
        assert result[0] is inputs[index][0]
    assert events == [0, 1, 2, 3, 0, 1]
    assert active == {0, 1}
    with pytest.raises(ValueError, match="signature"):
        loader.run((torch.zeros(1),))
    assert active == {0, 1}
    del loader
    assert not active
