"""Coverage for the Megatron-SP sequence collectives (public API + both backends).

The collectives are numerically identity on the CPU backend (``tp == 1``), so the
host assertions are about *dispatch wiring*: that the public surface resolves and
that the Trainium impl forwards the right keyword to the right nxd primitive.
"""

import torch


def test_public_api_exposes_sequence_parallel_collectives():
    from difflet.ops import (
        gather_from_sequence_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )
    from difflet.ops.collectives import gather_from_sequence_parallel_region as mod_gather
    from difflet.ops.collectives import reduce_scatter_to_sequence_parallel_region as mod_rs
    from difflet.ops.collectives import scatter_to_sequence_parallel_region as mod_scatter

    for fn in (
        gather_from_sequence_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
        mod_gather,
        mod_rs,
        mod_scatter,
    ):
        assert callable(fn)


def test_cpu_backend_sequence_collectives_are_identity(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    from difflet.ops import (
        gather_from_sequence_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    x = torch.randn(2, 8, 4)
    # Identity for every boundary op and every dim (no-op at tp==1).
    assert scatter_to_sequence_parallel_region(x, dim=1) is x
    assert gather_from_sequence_parallel_region(x, dim=1) is x
    assert reduce_scatter_to_sequence_parallel_region(x, dim=1) is x
    assert scatter_to_sequence_parallel_region(x, dim=2) is x


def test_cpu_round_trip_preserves_values(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")

    from difflet.ops import (
        gather_from_sequence_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
    )

    x = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4)
    sharded = scatter_to_sequence_parallel_region(x, dim=1)
    gathered = gather_from_sequence_parallel_region(sharded, dim=1)
    out = reduce_scatter_to_sequence_parallel_region(gathered, dim=1)
    assert torch.equal(out, x)


def test_public_dispatch_wrappers_invoke_backend(monkeypatch):
    # Exercise the difflet.ops.collectives module-level wrappers directly (the
    # difflet.ops.__getattr__ path resolves straight to the backend and skips
    # these), so their dispatch bodies are covered.
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    from difflet.ops import collectives

    x = torch.randn(1, 4, 2)
    assert collectives.scatter_to_sequence_parallel_region(x, dim=1) is x
    assert collectives.gather_from_sequence_parallel_region(x, dim=1) is x
    assert collectives.reduce_scatter_to_sequence_parallel_region(x, dim=1) is x


def test_trainium_scatter_forwards_sequence_dimension(monkeypatch):
    from difflet.backends.trainium.ops_impl import collectives

    seen = {}

    def fake(tensor, *, sequence_dimension):
        seen["sequence_dimension"] = sequence_dimension
        return tensor

    monkeypatch.setattr(collectives, "_nxd_scatter_to_sequence_parallel_region", fake)
    x = torch.zeros(1, 6, 2)
    out = collectives.scatter_to_sequence_parallel_region(x, dim=1)
    assert out is x
    assert seen == {"sequence_dimension": 1}


def test_trainium_gather_forwards_gather_dim(monkeypatch):
    from difflet.backends.trainium.ops_impl import collectives

    seen = {}

    def fake(tensor, *, gather_dim):
        seen["gather_dim"] = gather_dim
        return tensor

    monkeypatch.setattr(
        collectives, "gather_from_tensor_model_parallel_region_with_dim", fake
    )
    x = torch.zeros(1, 6, 2)
    out = collectives.gather_from_sequence_parallel_region(x, dim=1)
    assert out is x
    assert seen == {"gather_dim": 1}


def test_trainium_reduce_scatter_forwards_partition_dim(monkeypatch):
    from difflet.backends.trainium.ops_impl import collectives

    seen = {}

    def fake(tensor, *, partition_dim):
        seen["partition_dim"] = partition_dim
        return tensor

    monkeypatch.setattr(
        collectives, "reduce_scatter_to_tensor_model_parallel_region_with_dim", fake
    )
    x = torch.zeros(1, 6, 2)
    out = collectives.reduce_scatter_to_sequence_parallel_region(x, dim=1)
    assert out is x
    assert seen == {"partition_dim": 1}
