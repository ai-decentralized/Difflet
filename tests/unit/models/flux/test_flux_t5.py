"""CPU-backend unit tests for difflet.models.flux.t5.modeling_t5.

These exercise the torch-native (DIFFLET_BACKEND=cpu) path: real forward passes
through the T5 encoder building blocks with tiny configs. No Neuron hardware is
needed. See the project test recipe: set DIFFLET_BACKEND=cpu before importing
difflet.ops / the modeling module, and reload to rebind onto the cpu backend.
"""

import importlib
import os
from types import SimpleNamespace

import pytest

_PREV_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch  # noqa: E402


def _reload(name):
    return importlib.reload(importlib.import_module(name))


# Reload ONLY the flux modeling module so its `from difflet.ops import ...`
# rebinds onto the cpu backend (env set above), even if an earlier test imported
# it under a different backend. The shared trainium core modules are imported
# normally — reloading them would replace classes other test modules depend on.
import difflet.backends.trainium.core.config as _config_mod  # noqa: E402
t5 = _reload("difflet.models.flux.t5.modeling_t5")

NeuronConfig = _config_mod.NeuronConfig


def _restore_backend_env():
    # Restore the process-wide backend env so collecting other (trainium-only)
    # test modules in the same session is unaffected; the autouse fixture
    # re-applies the cpu backend for each test in this module.
    if _PREV_BACKEND is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = _PREV_BACKEND


_restore_backend_env()


@pytest.fixture(autouse=True)
def _cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    torch.manual_seed(0)
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def _t5_config(is_gated_act=False, dense_act_fn="relu", is_decoder=False, num_layers=2):
    return SimpleNamespace(
        d_model=32,
        d_ff=64,
        d_kv=8,
        num_heads=4,
        num_layers=num_layers,
        dropout_rate=0.0,
        dense_act_fn=dense_act_fn,
        is_gated_act=is_gated_act,
        layer_norm_epsilon=1e-6,
        is_decoder=is_decoder,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=128,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        vocab_size=50,
        tie_word_embeddings=False,
    )


def test_t5_layer_norm_fp32():
    ln = t5.T5LayerNorm(32, eps=1e-6)
    x = torch.randn(2, 3, 32)
    out = ln(x)
    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_t5_layer_norm_half_precision_branch():
    ln = t5.T5LayerNorm(8)
    ln.weight.data = ln.weight.data.to(torch.bfloat16)
    x = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    out = ln(x)
    assert out.dtype == torch.bfloat16


def test_dense_act_dense_forward():
    cfg = _t5_config()
    mod = t5.NeuronT5DenseActDense(cfg)
    out = mod(torch.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)


def test_dense_gated_act_dense_forward():
    cfg = _t5_config(is_gated_act=True, dense_act_fn="gelu_new")
    mod = t5.NeuronT5DenseGatedActDense(cfg)
    out = mod(torch.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)


def test_layer_ff_non_gated():
    cfg = _t5_config(is_gated_act=False)
    ff = t5.NeuronT5LayerFF(cfg)
    assert isinstance(ff.DenseReluDense, t5.NeuronT5DenseActDense)
    x = torch.randn(2, 4, 32)
    out = ff(x)
    assert out.shape == x.shape


def test_layer_ff_gated():
    cfg = _t5_config(is_gated_act=True, dense_act_fn="gelu_new")
    ff = t5.NeuronT5LayerFF(cfg)
    assert isinstance(ff.DenseReluDense, t5.NeuronT5DenseGatedActDense)
    out = ff(torch.randn(2, 4, 32))
    assert out.shape == (2, 4, 32)


def test_relative_position_bucket_bidirectional_and_unidirectional():
    rel = torch.arange(-5, 6).unsqueeze(0)
    b_bi = t5.NeuronT5Attention._relative_position_bucket(
        rel, bidirectional=True, num_buckets=8, max_distance=128
    )
    b_uni = t5.NeuronT5Attention._relative_position_bucket(
        rel, bidirectional=False, num_buckets=8, max_distance=128
    )
    assert b_bi.shape == rel.shape
    assert b_uni.shape == rel.shape
    assert int(b_bi.min()) >= 0
    assert int(b_uni.min()) >= 0


def test_attention_with_relative_bias_and_mask():
    cfg = _t5_config()
    attn = t5.NeuronT5Attention(cfg, has_relative_attention_bias=True)
    x = torch.randn(2, 5, 32)
    mask = torch.zeros(2, 1, 1, 5)
    out = attn(x, mask=mask, output_attentions=True)
    attn_output, present, position_bias, attn_weights = out
    assert attn_output.shape == (2, 5, 32)
    assert present is None  # encoder, not decoder
    # position_bias broadcasts to batch when the mask is added
    assert position_bias.shape == (2, 4, 5, 5)
    assert attn_weights.shape == (2, 4, 5, 5)


def test_attention_without_relative_bias_uses_zero_bias():
    cfg = _t5_config()
    attn = t5.NeuronT5Attention(cfg, has_relative_attention_bias=False)
    x = torch.randn(1, 4, 32)
    out = attn(x)
    assert out[0].shape == (1, 4, 32)


def test_attention_decoder_use_cache_returns_present():
    cfg = _t5_config(is_decoder=True)
    attn = t5.NeuronT5Attention(cfg, has_relative_attention_bias=True)
    x = torch.randn(1, 4, 32)
    out = attn(x, use_cache=True)
    present = out[1]
    assert present is not None
    key, value = present
    assert key.shape == (1, 4, 4, 8)
    # Reuse the cache for a follow-up single token to hit the past_key_value path.
    x2 = torch.randn(1, 1, 32)
    out2 = attn(x2, past_key_value=present, use_cache=True)
    assert out2[0].shape == (1, 1, 32)
    assert out2[1][0].shape[2] == 5


def test_attention_cross_attention_key_value_states():
    cfg = _t5_config()
    attn = t5.NeuronT5Attention(cfg, has_relative_attention_bias=False)
    x = torch.randn(1, 4, 32)
    kv = torch.randn(1, 6, 32)
    out = attn(x, key_value_states=kv)
    assert out[0].shape == (1, 4, 32)


def test_attention_layer_head_mask():
    cfg = _t5_config()
    attn = t5.NeuronT5Attention(cfg, has_relative_attention_bias=True)
    x = torch.randn(1, 4, 32)
    head_mask = torch.ones(4)
    out = attn(x, layer_head_mask=head_mask)
    assert out[0].shape == (1, 4, 32)


def test_attention_compute_bias_default_device():
    cfg = _t5_config()
    attn = t5.NeuronT5Attention(cfg, has_relative_attention_bias=True)
    bias = attn.compute_bias(5, 5)
    assert bias.shape == (1, 4, 5, 5)


def test_attention_past_key_value_wrong_length_raises():
    cfg = _t5_config(is_decoder=True)
    attn = t5.NeuronT5Attention(cfg, has_relative_attention_bias=True)
    x = torch.randn(1, 2, 32)
    with pytest.raises(ValueError):
        attn(x, past_key_value=(torch.randn(1, 4, 2, 8),))


def test_block_decoder_past_key_value_validation():
    cfg = _t5_config(is_decoder=True)
    block = t5.NeuronT5Block(cfg, has_relative_attention_bias=True)
    x = torch.randn(1, 1, 32)
    with pytest.raises(ValueError):
        block(x, past_key_value=(torch.randn(1, 4, 1, 8),))


def test_layer_self_attention_forward():
    cfg = _t5_config()
    layer = t5.NeuronT5LayerSelfAttention(cfg, has_relative_attention_bias=True)
    x = torch.randn(2, 5, 32)
    out = layer(x)
    assert out[0].shape == x.shape


def test_block_forward_encoder():
    cfg = _t5_config()
    block = t5.NeuronT5Block(cfg, has_relative_attention_bias=True)
    x = torch.randn(2, 5, 32)
    mask = torch.zeros(2, 1, 1, 5)
    out = block(x, attention_mask=mask)
    assert out[0].shape == x.shape


def test_stack_forward_return_dict_and_tuple():
    cfg = _t5_config()
    embed = t5.ParallelEmbedding(cfg.vocab_size, cfg.d_model)
    stack = t5.NeuronT5Stack(cfg, embed)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = stack(input_ids=ids, return_dict=True)
    assert out.last_hidden_state.shape == (2, 6, 32)

    out_tuple = stack(
        input_ids=ids,
        return_dict=False,
        output_hidden_states=True,
        output_attentions=True,
    )
    assert isinstance(out_tuple, tuple)
    assert out_tuple[0].shape == (2, 6, 32)


def test_stack_get_set_input_embeddings():
    cfg = _t5_config()
    embed = t5.ParallelEmbedding(cfg.vocab_size, cfg.d_model)
    stack = t5.NeuronT5Stack(cfg, embed)
    assert stack.get_input_embeddings() is embed
    new_embed = t5.ParallelEmbedding(cfg.vocab_size, cfg.d_model)
    stack.set_input_embeddings(new_embed)
    assert stack.get_input_embeddings() is new_embed


def test_stack_requires_inputs():
    cfg = _t5_config()
    stack = t5.NeuronT5Stack(cfg, t5.ParallelEmbedding(cfg.vocab_size, cfg.d_model))
    with pytest.raises(ValueError):
        stack(input_ids=None, inputs_embeds=None)
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    with pytest.raises(ValueError):
        stack(input_ids=ids, inputs_embeds=torch.randn(1, 3, 32))


def test_stack_decoder_use_cache():
    cfg = _t5_config(is_decoder=True)
    embed = t5.ParallelEmbedding(cfg.vocab_size, cfg.d_model)
    stack = t5.NeuronT5Stack(cfg, embed)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = stack(input_ids=ids, use_cache=True, return_dict=True)
    assert out.past_key_values is not None
    assert len(out.past_key_values) == cfg.num_layers


def test_encoder_model_forward_and_accessors():
    cfg = _t5_config()
    model = t5.NeuronT5EncoderModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = model(input_ids=ids)
    assert out.last_hidden_state.shape == (2, 6, 32)
    assert model.get_encoder() is model.encoder
    assert model.get_input_embeddings() is model.shared
    new_embed = t5.ParallelEmbedding(cfg.vocab_size, cfg.d_model)
    model.set_input_embeddings(new_embed)
    assert model.shared is new_embed


def test_encoder_model_rejects_attention_mask():
    cfg = _t5_config()
    model = t5.NeuronT5EncoderModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(AssertionError):
        model(input_ids=ids, attention_mask=torch.ones(1, 4))


def test_t5_inference_config_required_attributes_and_construct():
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    cfg = t5.T5InferenceConfig(
        neuron_config=nc,
        vocab_size=50,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=128,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=False,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        classifier_dropout=0.0,
    )
    required = cfg.get_required_attributes()
    assert "d_model" in required
    for attr in required:
        assert hasattr(cfg, attr)


def test_t5_inference_config_missing_attribute_raises():
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    with pytest.raises(AssertionError):
        t5.T5InferenceConfig(neuron_config=nc, vocab_size=50)


def test_convert_hf_to_neuron_state_dict_clones():
    sd = {"a": torch.randn(2, 2)}
    out = t5.NeuronT5Application.convert_hf_to_neuron_state_dict(sd, None)
    assert torch.equal(out["a"], sd["a"])
    # static no-op tied-weights helper is callable
    assert t5.NeuronT5Application.update_state_dict_for_tied_weights({}) is None
