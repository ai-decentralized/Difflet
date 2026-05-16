def test_collective_public_aliases_resolve():
    from nova.ops import get_tp_rank, get_tp_size, reduce_tp
    from nova.ops.collectives import gather_tp_dim, scatter_tp_dim

    assert callable(gather_tp_dim)
    assert callable(reduce_tp)
    assert callable(scatter_tp_dim)
    assert callable(get_tp_rank)
    assert callable(get_tp_size)


def test_mx_public_aliases_resolve_for_cpu_backend(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from nova.ops import dequantize_mx, linear_mx, matmul_mx, quantize_mx
    from nova.ops.mx import matmul_mx as module_matmul_mx

    assert callable(dequantize_mx)
    assert callable(linear_mx)
    assert callable(matmul_mx)
    assert callable(quantize_mx)
    assert callable(module_matmul_mx)


def test_mx_public_aliases_resolve_for_trainium_backend(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "trainium")

    from nova.ops import dequantize_mx, linear_mx, matmul_mx, quantize_mx

    assert callable(dequantize_mx)
    assert callable(linear_mx)
    assert callable(matmul_mx)
    assert callable(quantize_mx)


def test_trainium_gather_tp_dim_uses_nxd_gather_dim_keyword(monkeypatch):
    import torch

    from nova.backends.trainium.ops_impl import collectives

    seen = {}

    def fake_gather(tensor, *, gather_dim):
        seen["gather_dim"] = gather_dim
        return tensor

    monkeypatch.setattr(
        collectives, "gather_from_tensor_model_parallel_region_with_dim", fake_gather
    )

    tensor = torch.arange(24).reshape(2, 3, 4)

    out = collectives.gather_tp_dim(tensor, dim=2)

    assert out is tensor
    assert seen == {"gather_dim": 2}


def test_trainium_scatter_tp_dim_slices_requested_dimension(monkeypatch):
    import torch

    from nova.backends.trainium.ops_impl import collectives

    monkeypatch.setattr(collectives, "get_tensor_model_parallel_size", lambda: 2)
    monkeypatch.setattr(collectives, "get_tensor_model_parallel_rank", lambda: 1)

    tensor = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    out = collectives.scatter_tp_dim(tensor, dim=1)

    assert torch.equal(out, tensor[:, 2:4, :])
    assert out.is_contiguous()


def test_trainium_scatter_tp_dim_returns_input_for_tp_one(monkeypatch):
    import torch

    from nova.backends.trainium.ops_impl import collectives

    monkeypatch.setattr(collectives, "get_tensor_model_parallel_size", lambda: 1)

    tensor = torch.arange(24).reshape(2, 3, 4)

    out = collectives.scatter_tp_dim(tensor, dim=2)

    assert out is tensor


def test_attention_public_alias_resolves_to_trainium_impl(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "trainium")

    from nova.ops import attention, cross_attention

    assert callable(attention)
    assert callable(cross_attention)


def test_apply_rotary_emb_matches_wan_formula(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "trainium")

    import torch

    from nova.ops.embeddings import apply_rotary_emb

    hidden = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.tensor([[[[10.0, 20.0, 30.0, 40.0]]]])
    sin = torch.tensor([[[[0.5, 0.25, 0.125, 0.0625]]]])

    out = apply_rotary_emb(hidden, cos, sin)

    expected = torch.empty_like(hidden)
    expected[..., 0] = hidden[..., 0] * cos[..., 0] - hidden[..., 1] * sin[..., 1]
    expected[..., 1] = hidden[..., 0] * sin[..., 1] + hidden[..., 1] * cos[..., 0]
    expected[..., 2] = hidden[..., 2] * cos[..., 2] - hidden[..., 3] * sin[..., 3]
    expected[..., 3] = hidden[..., 2] * sin[..., 3] + hidden[..., 3] * cos[..., 2]
    assert torch.equal(out, expected)


def test_cpu_backend_registry_resolves(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    from nova.backends import get_backend

    backend = get_backend()
    assert backend.name == "cpu"
    assert backend.capabilities.requires_aot is False


def test_cpu_ops_run_on_regular_torch_tensors(monkeypatch):
    monkeypatch.setenv("NOVA_BACKEND", "cpu")

    import torch

    from nova.ops import RMSNorm, attention, gather_tp_dim

    norm = RMSNorm(4)
    x = torch.ones((1, 2, 4), dtype=torch.float32)
    assert norm(x).shape == x.shape

    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    assert attention(q, k, v, scale=0.5).shape == q.shape
    assert gather_tp_dim(x, dim=1) is x
