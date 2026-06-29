"""Unit tests for difflet.utils.kv_cache_reconstruct_utils (pure torch ops)."""

from types import SimpleNamespace

import pytest
import torch

from difflet.utils import kv_cache_reconstruct_utils as kv


def _mock_config(num_layers=2, num_kv_heads=2, tp_degree=4, batch_size=1, seq_len=3):
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        num_key_value_heads=num_kv_heads,
        neuron_config=SimpleNamespace(
            tp_degree=tp_degree, batch_size=batch_size, seq_len=seq_len
        ),
    )


def _device_cache(num_layers, tp_degree, batch_size, seq_len, head_dim):
    """Build a NeuronDeviceCache: list[tp_degree] of {layer-key: tensor}.

    Each rank holds 1 KV head/rank; index = layer*2 + is_v.
    """
    torch.manual_seed(0)
    cache = []
    for rank in range(tp_degree):
        rank_dict = {}
        for layer in range(num_layers):
            for is_v in (0, 1):
                key = f"kv_mgr.past_key_values.{layer * 2 + is_v}"
                rank_dict[key] = torch.randn(batch_size, 1, seq_len, head_dim)
        cache.append(rank_dict)
    return cache


def test_reconstruct_neuron_cpu_kv_cache():
    torch.manual_seed(0)
    # 2 layers -> 4 entries; even=K, odd=V; each [B, H, S, D].
    params = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.randn(1, 2, 3, 4)) for _ in range(4)]
    )
    k_cache, v_cache = kv.reconstruct_neuron_cpu_kv_cache(params)
    # [B, num_layers, H, S, D]
    assert k_cache.shape == (1, 2, 2, 3, 4)
    assert v_cache.shape == (1, 2, 2, 3, 4)
    assert torch.equal(k_cache[:, 0], params[0])
    assert torch.equal(v_cache[:, 0], params[1])


def test_extract_all_k_or_v_heads_from_rank_selects_k_and_v():
    cache = _device_cache(num_layers=1, tp_degree=1, batch_size=1, seq_len=3, head_dim=4)
    k = kv.extract_all_k_or_v_heads_from_rank(cache, rank_idx=0, layer_idx=0, is_v_head=False)
    v = kv.extract_all_k_or_v_heads_from_rank(cache, rank_idx=0, layer_idx=0, is_v_head=True)
    assert torch.equal(k, cache[0]["kv_mgr.past_key_values.0"])
    assert torch.equal(v, cache[0]["kv_mgr.past_key_values.1"])


def test_reconstruct_base_requires_num_hidden_layers():
    cache = _device_cache(1, 1, 1, 3, 4)
    bad_config = SimpleNamespace()  # no num_hidden_layers
    with pytest.raises(AttributeError):
        kv.reconstruct_neuron_device_kv_cache_base(
            cache, bad_config, num_kv_heads=1, reconstruct_cache_layer_fn=lambda *a: None
        )


def test_reconstruct_base_stacks_layers():
    config = _mock_config(num_layers=2)

    def fake_layer(neuron_cache, inference_config, num_kv_heads, layer_idx, is_v_cache):
        # return a deterministic [B, H, S, D] tensor per layer.
        return torch.full((1, 2, 3, 4), float(layer_idx + (10 if is_v_cache else 0)))

    k, v = kv.reconstruct_neuron_device_kv_cache_base(
        _device_cache(2, 4, 1, 3, 4), config, num_kv_heads=2, reconstruct_cache_layer_fn=fake_layer
    )
    assert k.shape == (1, 2, 2, 3, 4)
    assert v.shape == (1, 2, 2, 3, 4)
    assert torch.all(k[:, 0] == 0) and torch.all(k[:, 1] == 1)
    assert torch.all(v[:, 0] == 10) and torch.all(v[:, 1] == 11)


def test_gqa_replicate_layer_concatenates_unique_heads():
    config = _mock_config(num_kv_heads=2, tp_degree=4, batch_size=1, seq_len=3)
    cache = _device_cache(num_layers=2, tp_degree=4, batch_size=1, seq_len=3, head_dim=4)
    out = kv.reconstruct_cache_layer_gqa_replicate_to_tp_degree(
        cache, config, num_kv_heads=2, layer_idx=0, is_v_cache=False
    )
    # dup_degree = 4 // 2 = 2 -> ranks 0,2 -> 2 heads cat -> [B, 2, S, D]
    assert out.shape == (1, 2, 3, 4)


def test_gqa_replicate_layer_rejects_wrong_shape():
    config = _mock_config(num_kv_heads=2, tp_degree=4, batch_size=1, seq_len=99)
    cache = _device_cache(num_layers=1, tp_degree=4, batch_size=1, seq_len=3, head_dim=4)
    with pytest.raises(ValueError):
        kv.reconstruct_cache_layer_gqa_replicate_to_tp_degree(
            cache, config, num_kv_heads=2, layer_idx=0, is_v_cache=False
        )


def test_reconstruct_gqa_full_pipeline():
    config = _mock_config(num_layers=2, num_kv_heads=2, tp_degree=4, batch_size=1, seq_len=3)
    cache = _device_cache(num_layers=2, tp_degree=4, batch_size=1, seq_len=3, head_dim=4)
    k, v = kv.reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree(cache, config)
    assert k.shape == (1, 2, 2, 3, 4)
    assert v.shape == (1, 2, 2, 3, 4)


def test_reconstruct_gqa_requires_num_key_value_heads():
    config = SimpleNamespace(
        num_hidden_layers=1, neuron_config=SimpleNamespace(tp_degree=4, batch_size=1, seq_len=3)
    )
    cache = _device_cache(1, 4, 1, 3, 4)
    with pytest.raises(AttributeError):
        kv.reconstruct_neuron_device_kv_cache_gqa_replicate_to_tp_degree(cache, config)
