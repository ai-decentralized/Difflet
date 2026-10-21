"""difflet/backends/_torch_megatron: Megatron layers shared by TPU and CUDA.

The layers are pure torch; the only backend-specific thing they touch is the
collectives module they are built against. ``build_parallel_layers(collectives)``
returns fresh classes per call so two backends in one process never share state.
"""

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from difflet.backends._torch_megatron.linear import build_parallel_layers


def _identity_collectives(tp_size=1, tp_rank=0):
    return SimpleNamespace(
        gather_tp_dim=lambda t, *, dim: t,
        scatter_tp_dim=lambda t, *, dim: t,
        reduce_tp=lambda t: t,
        get_tp_size=lambda: tp_size,
        get_tp_rank=lambda: tp_rank,
        get_tp_groups=lambda: [list(range(tp_size))],
    )


def test_layers_are_distinct_classes_per_collectives_module():
    a = build_parallel_layers(_identity_collectives())
    b = build_parallel_layers(_identity_collectives())
    assert a.ColumnParallelLinear is not b.ColumnParallelLinear
    assert a.RowParallelLinear is not b.RowParallelLinear
    assert a.ParallelEmbedding is not b.ParallelEmbedding


def test_column_parallel_matches_plain_linear_at_tp1():
    L = build_parallel_layers(_identity_collectives())
    layer = L.ColumnParallelLinear(8, 16, bias=True)
    x = torch.randn(4, 8)
    assert torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias))
    assert layer._difflet_shard == {"weight": 0, "bias": 0}


def test_column_parallel_shards_the_output_dim():
    L = build_parallel_layers(_identity_collectives(tp_size=2))
    layer = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
    assert tuple(layer.weight.shape) == (8, 8)
    assert tuple(layer.bias.shape) == (8,)


def test_row_parallel_shard_metadata_and_bias_after_reduce():
    calls = []
    c = _identity_collectives()
    c.reduce_tp = lambda t: calls.append("reduce") or t
    L = build_parallel_layers(c)
    layer = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
    x = torch.randn(4, 16)
    out = layer(x)
    assert calls == ["reduce"]
    assert layer._difflet_shard == {"weight": 1, "bias": None}
    assert torch.allclose(out, F.linear(x, layer.weight) + layer.bias)


def test_row_parallel_reduce_output_false_skips_the_reduce():
    # The Trainium layer honours reduce_output= and so does the shared one (F5 task 3:
    # Flux's single block merges two row-parallel reduces into one explicit reduce).
    calls = []
    c = _identity_collectives()
    c.reduce_tp = lambda t: calls.append("reduce") or t
    L = build_parallel_layers(c)
    layer = L.RowParallelLinear(16, 8, input_is_parallel=True, reduce_output=False)
    x = torch.randn(4, 16)
    out = layer(x)
    assert calls == []
    assert torch.allclose(out, F.linear(x, layer.weight) + layer.bias)


def test_row_parallel_skip_bias_add_returns_the_bias_unadded():
    L = build_parallel_layers(_identity_collectives())
    layer = L.RowParallelLinear(16, 8, input_is_parallel=True, skip_bias_add=True)
    x = torch.randn(4, 16)
    out, bias = layer(x)
    assert bias is layer.bias
    assert torch.allclose(out, F.linear(x, layer.weight))
    assert layer.tensor_parallel_group is None


def test_indivisible_shapes_are_rejected_at_construction():
    L = build_parallel_layers(_identity_collectives(tp_size=4))
    try:
        L.ColumnParallelLinear(8, 6)
    except ValueError as exc:
        assert "tp=4" in str(exc)
    else:
        raise AssertionError("expected ValueError for 6 % 4")


def test_embedding_shard_metadata_follows_the_mode():
    L = build_parallel_layers(_identity_collectives())
    assert L.ParallelEmbedding(10, 4)._difflet_shard == {"weight": 0}
    assert L.ParallelEmbedding(10, 4, shard_across_embedding=True)._difflet_shard == {"weight": 1}


def test_embedding_matches_plain_embedding_at_tp1():
    L = build_parallel_layers(_identity_collectives())
    emb = L.ParallelEmbedding(10, 4)
    ids = torch.tensor([[1, 3], [9, 0]])
    assert torch.equal(emb(ids), F.embedding(ids, emb.weight))


def test_norm_and_embeddings_match_the_cpu_reference():
    from difflet.backends._torch_megatron import embeddings as E
    from difflet.backends._torch_megatron import norm as N
    from difflet.backends.cpu.ops_impl import embeddings as CE
    from difflet.backends.cpu.ops_impl import norm as CN

    x = torch.randn(2, 5, 8)
    rms = N.RMSNorm(8)
    crms = CN.RMSNorm(8)
    crms.load_state_dict(rms.state_dict())
    assert torch.equal(rms(x), crms(x))
    cos, sin = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.equal(E.apply_rotary_emb(x, cos, sin), CE.apply_rotary_emb(x, cos, sin))


def test_weights_and_checkpoint_import_without_tpu():
    for name in [k for k in sys.modules if k.startswith("difflet.backends.tpu")]:
        del sys.modules[name]
    from difflet.backends._torch_megatron import checkpoint, weights  # noqa: F401

    assert not any(k.startswith("difflet.backends.tpu") for k in sys.modules)
    assert callable(weights.shard_dim) and callable(checkpoint.load_checkpoint_into)
