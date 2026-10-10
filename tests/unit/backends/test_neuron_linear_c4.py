"""Unit tests for the neuron backend's tensor-parallel linear and embedding layers (C4).

CPU only. Single-process tests cover shapes, shard declarations, NxD keyword
semantics and meta construction. The gloo tests run the layers at TP4 in four
CPU processes against TP1, eagerly and compiled (aot_eager, fullgraph).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl import linear as L
from difflet.backends.neuron.ops_impl import parallel_mesh as pm
from difflet.backends.tpu.core.weights import load_sharded_state_dict, shard_dim
from difflet.pipeline.parallel_mesh import MeshSpec
from tests.unit.backends._neuron_gloo import run_ranks
from tests.unit.backends._neuron_toy import ToyTPMLP, reference_mlp, toy_full_weights
from tests.unit.backends._neuron_workers import c4_tp_mlp_worker

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _fresh_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def _spec_only(tp: int) -> None:
    # No process group in this process: the mesh records the spec, layers are
    # sized for it, and any collective at tp > 1 raises.
    pm.init_parallel_mesh(MeshSpec(tp=tp))


# --------------------------------------------------------------------- tp == 1


def test_layers_default_to_tp1_without_a_mesh_in_a_single_process():
    # Unlike TPU (test_tpu_layers.py:72-75), a lone process needs no mesh:
    # model code and CPU references build the layers unsharded.
    col = L.ColumnParallelLinear(8, 16)
    row = L.RowParallelLinear(16, 8)
    emb = L.ParallelEmbedding(32, 8)
    assert col.weight.shape == (16, 8) and col.output_size_per_partition == 16
    assert row.weight.shape == (8, 16) and row.input_size_per_partition == 16
    assert emb.weight.shape == (32, 8) and emb.num_embeddings_per_partition == 32


@pytest.mark.parametrize("gather_output", [True, False])
def test_column_parallel_matches_plain_linear_at_tp1(gather_output):
    layer = L.ColumnParallelLinear(8, 16, gather_output=gather_output)
    x = torch.randn(2, 3, 8)
    assert torch.equal(layer(x), F.linear(x, layer.weight, layer.bias))


@pytest.mark.parametrize("reduce_output", [True, False])
@pytest.mark.parametrize("input_is_parallel", [True, False])
def test_row_parallel_matches_plain_linear_at_tp1(input_is_parallel, reduce_output):
    layer = L.RowParallelLinear(
        16, 8, input_is_parallel=input_is_parallel, reduce_output=reduce_output
    )
    x = torch.randn(2, 3, 16)
    assert torch.equal(layer(x), F.linear(x, layer.weight) + layer.bias)


@pytest.mark.parametrize("shard_across_embedding", [False, True])
def test_embedding_matches_plain_embedding_at_tp1(shard_across_embedding):
    layer = L.ParallelEmbedding(32, 8, shard_across_embedding=shard_across_embedding)
    ids = torch.randint(0, 32, (3, 5))
    assert torch.equal(layer(ids), F.embedding(ids, layer.weight))


def test_row_parallel_skip_bias_add_returns_the_bias_separately():
    # Flux and HunyuanVideo single blocks (modeling_flux.py:666-672,
    # modeling_hunyuan_video.py:686-693) unpack (out, bias) and add the bias
    # after their own merged all-reduce.
    layer = L.RowParallelLinear(
        16, 8, input_is_parallel=True, reduce_output=False, skip_bias_add=True
    )
    x = torch.randn(2, 16)
    out, bias = layer(x)
    assert bias is layer.bias
    assert torch.equal(out, F.linear(x, layer.weight))

    no_bias = L.RowParallelLinear(16, 8, bias=False, input_is_parallel=True, skip_bias_add=True)
    out, bias = no_bias(x)
    assert bias is None
    assert torch.equal(out, F.linear(x, no_bias.weight))


def test_column_parallel_rejects_skip_bias_add():
    with pytest.raises(NotImplementedError, match="skip_bias_add"):
        L.ColumnParallelLinear(8, 16, skip_bias_add=True)


def test_sequence_parallel_inside_the_layer_is_rejected():
    for build in (
        lambda: L.ColumnParallelLinear(8, 16, sequence_parallel_enabled=True),
        lambda: L.RowParallelLinear(16, 8, sequence_parallel_enabled=True),
        lambda: L.ParallelEmbedding(32, 8, sequence_parallel_enabled=True),
    ):
        with pytest.raises(NotImplementedError, match="sequence_parallel_enabled"):
            build()


def test_nxd_keyword_arguments_are_accepted_and_dtype_is_honoured():
    # The keywords modeling_wan.py passes (320-337, 446-468) plus other NxD knobs.
    bf16 = torch.bfloat16
    col = L.ColumnParallelLinear(
        8, 16, gather_output=False, dtype=bf16, reduce_dtype=bf16,
        sequence_parallel_enabled=False, keep_master_weight=False, pad=False,
    )
    row = L.RowParallelLinear(
        16, 8, input_is_parallel=True, dtype=bf16, reduce_dtype=bf16, reduce_output=False
    )
    emb = L.ParallelEmbedding(32, 8, shard_across_embedding=True, pad=False, dtype=bf16)
    assert {p.dtype for m in (col, row, emb) for p in m.parameters()} == {bf16}
    assert row.reduce_output is False and row.skip_bias_add is False


# ------------------------------------------------------ shard metadata (tp=4)


def test_column_parallel_shards_the_output_dim():
    _spec_only(4)
    layer = L.ColumnParallelLinear(8, 16, bias=True)
    assert layer.weight.shape == (4, 8)
    assert layer.bias.shape == (4,)
    assert layer.output_size_per_partition == 4
    assert layer._difflet_shard == {"weight": 0, "bias": 0}
    assert shard_dim(layer, "weight") == 0 and shard_dim(layer, "bias") == 0


def test_row_parallel_shards_the_input_dim_and_replicates_bias():
    _spec_only(4)
    layer = L.RowParallelLinear(8, 16, bias=True)
    assert layer.weight.shape == (16, 2)
    assert layer.input_size_per_partition == 2
    # The bias must NOT be sharded: it is added once, after the all-reduce.
    assert layer.bias.shape == (16,)
    assert layer._difflet_shard == {"weight": 1, "bias": None}
    assert shard_dim(layer, "weight") == 1 and shard_dim(layer, "bias") is None


def test_embedding_shard_metadata_follows_the_mode():
    _spec_only(4)
    vocab = L.ParallelEmbedding(64, 16)
    assert vocab.weight.shape == (16, 16) and vocab.num_embeddings_per_partition == 16
    assert shard_dim(vocab, "weight") == 0
    by_dim = L.ParallelEmbedding(64, 16, shard_across_embedding=True)
    assert by_dim.weight.shape == (64, 4) and by_dim.embedding_dim_per_partition == 4
    assert shard_dim(by_dim, "weight") == 1


def test_indivisible_shapes_are_rejected_at_construction():
    _spec_only(4)
    with pytest.raises(ValueError, match="cannot shard output_size"):
        L.ColumnParallelLinear(8, 10, bias=False)
    with pytest.raises(ValueError, match="cannot shard input_size"):
        L.RowParallelLinear(10, 8, bias=False)
    with pytest.raises(ValueError, match="cannot shard num_embeddings"):
        L.ParallelEmbedding(10, 8)
    with pytest.raises(ValueError, match="cannot shard embedding_dim"):
        L.ParallelEmbedding(8, 10, shard_across_embedding=True)


def test_collectives_refuse_to_run_without_a_process_group():
    # A spec-only mesh at tp=4 must fail loudly rather than return one rank's
    # partial result as if it were the whole answer.
    _spec_only(4)
    cases = [
        (L.ColumnParallelLinear(8, 16, gather_output=True), torch.randn(2, 8)),
        (L.RowParallelLinear(16, 8, input_is_parallel=True), torch.randn(2, 4)),
        (L.RowParallelLinear(16, 8, input_is_parallel=False), torch.randn(2, 16)),
        (L.ParallelEmbedding(32, 8), torch.randint(0, 32, (2, 3))),
        (L.ParallelEmbedding(32, 8, shard_across_embedding=True), torch.randint(0, 32, (2, 3))),
    ]
    for layer, inp in cases:
        with pytest.raises(RuntimeError):
            layer(inp)


def test_layers_build_on_the_meta_device():
    _spec_only(4)
    with torch.device("meta"):
        built = [
            L.ColumnParallelLinear(8, 16),
            L.RowParallelLinear(16, 8),
            L.ParallelEmbedding(32, 8),
        ]
    built.append(L.ColumnParallelLinear(8, 16, device="meta"))
    assert all(p.is_meta for m in built for p in m.parameters())
    assert built[0].weight.shape == (4, 8) and built[1].weight.shape == (8, 4)


def test_shard_declarations_survive_init_empty_weights():
    accelerate = pytest.importorskip("accelerate")
    _spec_only(4)
    with accelerate.init_empty_weights(include_buffers=False):
        mlp = ToyTPMLP(8, 16)
    assert mlp.up.weight.is_meta and mlp.down.weight.is_meta
    assert mlp.up.weight.shape == (4, 8) and mlp.down.weight.shape == (8, 4)
    names = ("up.weight", "up.bias", "down.weight", "down.bias")
    assert [shard_dim(mlp, n) for n in names] == [0, 0, 1, None]


# ------------------------------------------------------------- toy and loader


def test_toy_mlp_at_tp1_is_bitwise_the_reference():
    weights = toy_full_weights(16, 64, seed=0)
    mlp = ToyTPMLP(16, 64)
    load_sharded_state_dict(mlp, weights, tp_size=1, tp_rank=0)
    x = torch.randn(2, 3, 16)
    with torch.no_grad():
        assert torch.equal(mlp(x), reference_mlp(x, weights))


def test_tp4_shards_reassemble_to_the_full_weights():
    # The loader slices by _difflet_shard: concatenating every rank's slice along
    # the declared dim gives back the checkpoint; replicated entries arrive whole.
    _spec_only(4)
    weights = toy_full_weights(16, 64, seed=3)
    declared = {name: shard_dim(ToyTPMLP(16, 64), name) for name in weights}
    assert declared == {"up.weight": 0, "up.bias": 0, "down.weight": 1, "down.bias": None}
    per_rank = []
    for rank in range(4):
        mlp = ToyTPMLP(16, 64)
        load_sharded_state_dict(mlp, weights, tp_size=4, tp_rank=rank)
        per_rank.append(mlp.state_dict())
    for name, full in weights.items():
        if declared[name] is None:
            assert all(torch.equal(sd[name], full) for sd in per_rank), name
        else:
            joined = torch.cat([sd[name] for sd in per_rank], dim=declared[name])
            assert torch.equal(joined, full), name


def test_ops_dispatch_to_the_neuron_layers(monkeypatch):
    from difflet.backends import registry
    from difflet.ops._dispatch import load_backend_attr

    registry._get_backend_by_name.cache_clear()
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    for name in L.__all__:
        assert load_backend_attr("linear", name) is getattr(L, name)
    registry._get_backend_by_name.cache_clear()


# ------------------------------------------------------------ gloo, 4 ranks

_EXACT_CASES = (
    "column_gather_exact",
    "row_scatter_exact",
    "row_skip_bias_add_exact",
    "row_partial_plus_bias_exact",
    "embedding_vocab_exact",
    "embedding_dim_exact",
)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_tp4_matches_tp1_across_four_gloo_ranks(compiled):
    results = run_ranks(c4_tp_mlp_worker, compiled, world_size=4, timeout=300.0)
    assert len(results) == 4
    for rank, res in enumerate(results):
        assert res["uninitialized_mesh_raises"], f"rank {rank}: sized a layer with no mesh"
        assert res["shapes_ok"], f"rank {rank}: ToyTPMLP shards have the wrong shapes"
        inexact = [case for case in _EXACT_CASES if not res[case]]
        assert not inexact, f"rank {rank}: not bit-exact against TP1: {inexact}"
        assert res["mlp_max_abs_err"] <= 1e-5, (
            f"rank {rank}: TP4 MLP vs TP1 max abs err {res['mlp_max_abs_err']:.3e}"
        )
    # The all-reduce must leave every rank with the same bits, or replicated
    # activations drift apart block by block.
    assert len({res["mlp_digest"] for res in results}) == 1


# ------------------------------------------------- Wan through difflet.ops

_WAN_SMOKE = textwrap.dedent(
    """
    import torch
    import torch.nn.functional as F

    from difflet.backends.neuron.ops_impl import linear as L
    from difflet.models.wan.modeling_wan import WanFeedForward
    from difflet.models.wan.umt5.modeling_umt5 import (
        WanUmT5Config,
        WanUmT5DenseGatedActDense,
        WanUmT5EncoderModel,
    )

    ff = WanFeedForward(16, 64, dtype=torch.float32)
    assert type(ff.net_in) is L.ColumnParallelLinear, type(ff.net_in)
    assert type(ff.net_out) is L.RowParallelLinear, type(ff.net_out)
    assert ff.net_out.reduce_output is True
    x = torch.randn(1, 3, 16)
    h = F.gelu(F.linear(x, ff.net_in.weight, ff.net_in.bias), approximate="tanh")
    assert torch.equal(ff(x), F.linear(h, ff.net_out.weight) + ff.net_out.bias)

    cfg = WanUmT5Config(vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_heads=4, num_layers=1)
    kinds = {type(m) for m in WanUmT5EncoderModel(cfg).modules()}
    assert {L.ColumnParallelLinear, L.RowParallelLinear, L.ParallelEmbedding} <= kinds, kinds
    assert WanUmT5DenseGatedActDense(cfg)(torch.randn(1, 3, 16)).shape == (1, 3, 16)
    print("WAN_SMOKE_OK")
    """
)


def test_wan_and_umt5_build_on_the_neuron_layers_at_tp1():
    # Fresh process: modeling_wan binds difflet.ops names at import time.
    for dep in ("diffusers", "transformers"):
        if importlib.util.find_spec(dep) is None:
            pytest.skip(f"{dep} is not installed (C0 installs it)")
    env = dict(
        os.environ,
        DIFFLET_BACKEND="neuron",
        DIFFLET_DISABLE_PREWARM="1",
        PYTHONPATH=str(REPO_ROOT),
    )
    proc = subprocess.run(
        [sys.executable, "-c", _WAN_SMOKE],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0 and "WAN_SMOKE_OK" in proc.stdout, proc.stderr[-4000:]
