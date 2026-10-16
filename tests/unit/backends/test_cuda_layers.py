"""CUDA linear / norm / embeddings ops (CUDA plan Phase 2b), mirroring test_tpu_layers.py.

Runs without a GPU: at tp=1 no collective is reached. The layers are the shared
``_torch_megatron`` ones bound to the CUDA collectives.
"""

import pytest
import torch
import torch.nn.functional as F

from difflet.backends.cuda.ops_impl import embeddings as E
from difflet.backends.cuda.ops_impl import linear as L
from difflet.backends.cuda.ops_impl import norm as N
from difflet.backends.cuda.ops_impl import parallel_mesh as pm
from difflet.pipeline.parallel_mesh import MeshSpec


@pytest.fixture(autouse=True)
def _mesh_tp1():
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(MeshSpec(tp=1))
    yield
    pm.destroy_parallel_mesh()


def _reinit(spec):
    pm.destroy_parallel_mesh()
    pm.init_parallel_mesh(spec)


def test_column_parallel_matches_plain_linear_at_tp1():
    layer = L.ColumnParallelLinear(8, 16, bias=True)
    x = torch.randn(4, 8)
    assert torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias))


def test_column_parallel_shards_the_output_dim():
    _reinit(MeshSpec(tp=2))
    layer = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
    assert tuple(layer.weight.shape) == (8, 8)
    assert layer._difflet_shard == {"weight": 0, "bias": 0}


def test_row_parallel_shards_the_input_dim_and_replicates_bias():
    _reinit(MeshSpec(tp=2))
    layer = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
    assert tuple(layer.weight.shape) == (8, 8)
    assert tuple(layer.bias.shape) == (8,)
    assert layer._difflet_shard == {"weight": 1, "bias": None}


def test_indivisible_shapes_are_rejected_at_construction():
    _reinit(MeshSpec(tp=4))
    with pytest.raises(ValueError, match="tp=4"):
        L.ColumnParallelLinear(8, 6)


def test_layers_require_an_initialized_mesh():
    pm.destroy_parallel_mesh()
    with pytest.raises(RuntimeError, match="not initialized"):
        L.ColumnParallelLinear(8, 16)


def test_embedding_matches_plain_embedding_at_tp1():
    emb = L.ParallelEmbedding(10, 4)
    ids = torch.tensor([[1, 3], [9, 0]])
    assert torch.equal(emb(ids), F.embedding(ids, emb.weight))


def test_rmsnorm_computes_variance_in_fp32():
    layer = N.RMSNorm(8)
    x = torch.randn(3, 8)
    expected = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_layer_norm_and_custom_rmsnorm_are_exported():
    assert N.LayerNorm is torch.nn.LayerNorm
    assert N.CustomRMSNorm is N.RMSNorm


def test_rotary_matches_the_cpu_reference_implementation():
    from difflet.backends.cpu.ops_impl import embeddings as CE

    x = torch.randn(2, 5, 8)
    cos, sin = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.equal(E.apply_rotary_emb(x, cos, sin), CE.apply_rotary_emb(x, cos, sin))


def test_public_surface_matches_the_ops_contract():
    from difflet.ops import _EXPORTS

    for module, mod in (("linear", L), ("norm", N), ("embeddings", E)):
        expected = {name for name, (m, _) in _EXPORTS.items() if m == module}
        assert expected <= set(mod.__all__), module
