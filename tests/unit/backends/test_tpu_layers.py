"""Unit tests for the TPU linear/norm/embeddings/attention ops (Phase 2b-2d).

Runs without a TPU: at tp=1 no collective is reached, so nothing imports
torch_xla. The tp>1 sharding numerics are verified separately on a real 2x2
v5e mesh against a full-weight oracle (see the plan's Phase 2b notes) — the
tests here cover shapes, contracts, and the sharding *metadata* that a
checkpoint loader will depend on.
"""

import pytest
import torch
import torch.nn.functional as F

from difflet.backends.tpu.ops_impl import attention as A
from difflet.backends.tpu.ops_impl import embeddings as E
from difflet.backends.tpu.ops_impl import linear as L
from difflet.backends.tpu.ops_impl import norm as N
from difflet.backends.tpu.ops_impl import parallel_mesh as pm
from difflet.backends.tpu.ops_impl import platform as P
from difflet.backends.tpu.core.weights import shard_dim as L_shard
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


# --------------------------------------------------------------------- linear


def test_column_parallel_matches_plain_linear_at_tp1():
    layer = L.ColumnParallelLinear(8, 16, bias=True)
    x = torch.randn(4, 8)
    assert torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias))


def test_column_parallel_shards_the_output_dim():
    _reinit(MeshSpec(tp=4))
    layer = L.ColumnParallelLinear(8, 16, bias=True)
    assert layer.weight.shape == (4, 8)
    assert layer.bias.shape == (4,)
    assert L_shard(layer, "weight") == 0


def test_row_parallel_shards_the_input_dim_and_replicates_bias():
    _reinit(MeshSpec(tp=4))
    layer = L.RowParallelLinear(8, 16, bias=True)
    assert layer.weight.shape == (16, 2)
    assert L_shard(layer, "weight") == 1
    # Bias must NOT be sharded: it is added once, after the all-reduce.
    assert layer.bias.shape == (16,)
    assert L_shard(layer, "bias") is None


def test_row_parallel_reduce_output_and_skip_bias_add_follow_nxd(monkeypatch):
    """HunyuanVideo's single block sums two ``reduce_output=False`` partials and
    reduces once; ``skip_bias_add`` hands the bias back for after that reduce.
    Ignoring the kwargs reduced twice and added the bias tp times (v5e: cosine
    0.909 vs diffusers). At tp=1 the reduce is identity, so count the calls."""
    _reinit(MeshSpec(tp=1))
    calls = []
    monkeypatch.setattr(L, "reduce_tp", lambda t: calls.append(1) or t)
    x = torch.randn(3, 8)

    plain = L.RowParallelLinear(8, 4, bias=True, input_is_parallel=True)
    ref = F.linear(x, plain.weight, plain.bias)
    assert torch.allclose(plain(x), ref)
    assert len(calls) == 1

    partial = L.RowParallelLinear(8, 4, bias=True, input_is_parallel=True,
                                  reduce_output=False, skip_bias_add=True)
    partial.weight.data.copy_(plain.weight.data)
    partial.bias.data.copy_(plain.bias.data)
    out, bias = partial(x)
    assert len(calls) == 1  # no reduce inside
    assert bias is partial.bias
    assert torch.allclose(out + bias, ref)

    unreduced_biased = L.RowParallelLinear(8, 4, bias=True, input_is_parallel=True,
                                           reduce_output=False)
    unreduced_biased.weight.data.copy_(plain.weight.data)
    unreduced_biased.bias.data.copy_(plain.bias.data)
    assert torch.allclose(unreduced_biased(x), ref)  # NxD: per-rank bias on the partial
    assert len(calls) == 1


def test_indivisible_shapes_are_rejected_at_construction():
    _reinit(MeshSpec(tp=4))
    with pytest.raises(ValueError, match="cannot shard output_size"):
        L.ColumnParallelLinear(8, 10, bias=False)
    with pytest.raises(ValueError, match="cannot shard input_size"):
        L.RowParallelLinear(10, 8, bias=False)


def test_layers_require_an_initialized_mesh():
    pm.destroy_parallel_mesh()
    with pytest.raises(RuntimeError, match="not initialized"):
        L.ColumnParallelLinear(8, 16)


def test_embedding_shard_metadata_follows_the_mode():
    _reinit(MeshSpec(tp=4))
    vocab = L.ParallelEmbedding(64, 16)
    assert vocab.weight.shape == (16, 16)
    assert L_shard(vocab, "weight") == 0

    dim = L.ParallelEmbedding(64, 16, shard_across_embedding=True)
    assert dim.weight.shape == (64, 4)
    assert L_shard(dim, "weight") == 1


def test_embedding_matches_plain_embedding_at_tp1():
    layer = L.ParallelEmbedding(32, 8)
    ids = torch.randint(0, 32, (3, 5))
    assert torch.allclose(layer(ids), F.embedding(ids, layer.weight))


# ------------------------------------------------------------------ attention


def test_attention_scale_none_means_one_not_sdpa_default():
    # The op contract treats scale=None as 1.0. SDPA's own default is
    # 1/sqrt(head_dim), so passing None straight through would silently
    # rescale every score.
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    ref = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)).float(), -1), v)
    assert torch.allclose(A.attention(q, k, v, scale=None), ref, atol=1e-5)


def test_attention_causal_masks_the_future():
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    out = A.attention(q, k, v, scale=1.0, causal=True)
    # Query 0 can only attend to key 0, so its output must equal v[..., 0, :].
    assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-5)


def test_attention_bounds_require_both_ends():
    q = k = v = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="both be provided"):
        A.attention(q, k, v, bound_min=torch.zeros(1, 1, 1, dtype=torch.long))


def test_attention_bounds_and_mask_are_mutually_exclusive():
    q = k = v = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="mutually exclusive"):
        A.attention(
            q, k, v,
            bound_min=torch.zeros(1, 1, 1, dtype=torch.long),
            bound_max=torch.full((1, 1, 1), 4, dtype=torch.long),
            attention_mask=torch.ones(4, 4, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "name",
    ["ring_attention", "joint_ring_attention", "ulysses_attention", "joint_ulysses_attention"],
)
def test_context_parallel_modes_fail_loudly_rather_than_silently(name):
    # A silently-wrong CP implementation is far worse than an explicit error.
    with pytest.raises(NotImplementedError, match="Phase 5"):
        getattr(A, name)()


# ----------------------------------------------------------- norm / platform


def test_rmsnorm_computes_variance_in_fp32():
    layer = N.RMSNorm(8)
    x = torch.randn(3, 8)
    expected = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_rotary_matches_the_shared_reference_implementation():
    # Byte-identical to the CPU/Trainium implementation on purpose; rotary
    # interleaving bugs are silent until numerical validation.
    from difflet.backends.cpu.ops_impl import embeddings as cpu_e

    h = torch.randn(2, 4, 16)
    cos = torch.randn(2, 4, 16)
    sin = torch.randn(2, 4, 16)
    assert torch.equal(E.apply_rotary_emb(h, cos, sin), cpu_e.apply_rotary_emb(h, cos, sin))


def test_platform_target_is_tpu():
    assert P.get_platform_target() is P.hardware.TPU


def test_default_matmul_precision_is_highest():
    # Not cosmetic: XLA's default TPU matmul precision measured ~1e-2 error
    # per matmul on v5e, which makes the Phase 5 parity target unreachable.
    assert P.DEFAULT_MATMUL_PRECISION == "highest"


# --------------------------------------------------- chunked attention memory

def test_chunking_only_kicks_in_above_the_threshold(monkeypatch):
    # The threshold itself is a tuning knob (raised once no_grad fixed the
    # memory picture), so pin it here rather than asserting that some
    # particular model shape does or does not chunk.
    monkeypatch.setattr(A, "DEFAULT_MAX_SCORE_ELEMENTS", 1000)
    assert A._should_chunk(torch.zeros(1, 2, 20, 4), torch.zeros(1, 2, 20, 4)) is False
    assert A._should_chunk(torch.zeros(1, 2, 40, 4), torch.zeros(1, 2, 40, 4)) is True


@pytest.mark.parametrize("queries", [1024, 1500, 2048])
def test_chunked_attention_equals_the_one_shot_result(queries, monkeypatch):
    # Each block is a full independent softmax over all keys, so this must be
    # exactly the unchunked answer — chunking is a memory change, not an
    # approximation. Sizes straddle the chunk boundary to catch a bad tail.
    monkeypatch.setattr(A, "QUERY_CHUNK", 512)
    torch.manual_seed(0)
    q = torch.randn(1, 2, queries, 16)
    k = torch.randn(1, 2, 700, 16)
    v = torch.randn(1, 2, 700, 16)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.25)
    got = A._chunked_attention(q, k, v, scale=0.25, causal=False)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6)


def test_chunked_attention_preserves_causality_across_blocks(monkeypatch):
    # The mask must be built from ABSOLUTE query positions; block-local indices
    # would let block 1 attend to keys it should not see.
    monkeypatch.setattr(A, "QUERY_CHUNK", 4)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 12, 8)
    k = torch.randn(1, 1, 12, 8)
    v = torch.randn(1, 1, 12, 8)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=0.25
    )
    got = A._chunked_attention(q, k, v, scale=0.25, causal=True)
    assert torch.allclose(got, expected, atol=1e-6)


def test_attention_routes_large_inputs_through_chunking(monkeypatch):
    calls = []
    real = A._chunked_attention
    monkeypatch.setattr(A, "_chunked_attention",
                        lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])
    monkeypatch.setattr(A, "DEFAULT_MAX_SCORE_ELEMENTS", 8)
    q = torch.randn(1, 2, 16, 8)
    A.attention(q, q, q, scale=0.25)
    assert calls, "large attention did not take the chunked path"


def test_chunk_threshold_counts_heads_folded_into_the_batch_axis():
    # difflet's Qwen-Image modeling calls attention with (B*heads, S, D), not
    # (B, heads, S, D). Treating the 3-D form as one head under-counted the
    # score matrix 6x and silently skipped chunking on the exact shape the
    # threshold exists for.
    folded = torch.zeros(6, 5120, 8)      # 6 heads x 5120 queries, 3-D
    stacked = torch.zeros(1, 6, 5120, 8)  # same thing, 4-D
    # Both forms must count the same 6 leading elements. Treating the 3-D form
    # as one head under-counted the score matrix 6x, so the threshold silently
    # never fired on the exact shape it exists for.
    assert A._leading_size(folded) == A._leading_size(stacked) == 6
    assert A._should_chunk(folded, folded) == A._should_chunk(stacked, stacked)


def test_chunked_attention_handles_the_folded_head_layout():
    torch.manual_seed(0)
    q = torch.randn(6, 300, 16)
    k = torch.randn(6, 200, 16)
    v = torch.randn(6, 200, 16)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.25)
    got = A._chunked_attention(q, k, v, scale=0.25, causal=False)
    assert torch.allclose(got, expected, atol=1e-6)
