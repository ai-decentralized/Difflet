"""CPU-backend unit tests for difflet.models.flux.clip.modeling_clip.

Real forward passes through the CLIP text-encoder building blocks on the
torch-native (DIFFLET_BACKEND=cpu) backend with tiny configs.
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


# Reload only the flux modeling module so it rebinds onto the cpu backend; import
# the shared trainium core normally (reloading it would break other test modules).
import difflet.backends.trainium.core.config as _config_mod  # noqa: E402
clip = _reload("difflet.models.flux.clip.modeling_clip")

NeuronConfig = _config_mod.NeuronConfig


def _restore_backend_env():
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


def _clip_config(eos_token_id=2, hidden_size=32, num_attention_heads=4):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_dropout=0.0,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        intermediate_size=64,
        num_hidden_layers=2,
        vocab_size=50,
        max_position_embeddings=16,
        eos_token_id=eos_token_id,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        dropout=0.0,
    )


def test_clip_inference_config_required_attributes():
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    cfg = clip.CLIPInferenceConfig(
        neuron_config=nc,
        _name_or_path="x",
        architectures=["CLIPTextModel"],
        attention_dropout=0.0,
        bos_token_id=0,
        dropout=0.0,
        eos_token_id=2,
        hidden_act="gelu",
        hidden_size=32,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=64,
        layer_norm_eps=1e-5,
        max_position_embeddings=16,
        model_type="clip_text_model",
        num_attention_heads=4,
        num_hidden_layers=2,
        pad_token_id=1,
        projection_dim=32,
        transformers_version="4.0",
        vocab_size=50,
    )
    required = cfg.get_required_attributes()
    assert "hidden_size" in required and "vocab_size" in required
    for attr in required:
        assert hasattr(cfg, attr)


def test_clip_attention_raises_when_not_divisible():
    cfg = _clip_config(hidden_size=30, num_attention_heads=4)
    with pytest.raises(ValueError):
        clip.NeuronCLIPAttention(cfg)


def test_clip_attention_forward_with_masks():
    cfg = _clip_config()
    attn = clip.NeuronCLIPAttention(cfg)
    h = torch.randn(1, 7, 32)
    cam = torch.zeros(1, 1, 7, 7)
    am = torch.zeros(1, 1, 7, 7)
    out, weights = attn(h, attention_mask=am, causal_attention_mask=cam, output_attentions=True)
    assert out.shape == (1, 7, 32)
    assert weights.shape == (1, 4, 7, 7)


def test_clip_attention_forward_no_output_attentions():
    cfg = _clip_config()
    attn = clip.NeuronCLIPAttention(cfg)
    out, weights = attn(torch.randn(1, 5, 32), output_attentions=False)
    assert out.shape == (1, 5, 32)
    assert weights is None


def test_clip_attention_bad_causal_mask_raises():
    cfg = _clip_config()
    attn = clip.NeuronCLIPAttention(cfg)
    h = torch.randn(1, 5, 32)
    with pytest.raises(ValueError):
        attn(h, causal_attention_mask=torch.zeros(1, 1, 4, 4))


def test_clip_attention_bad_attention_mask_raises():
    cfg = _clip_config()
    attn = clip.NeuronCLIPAttention(cfg)
    h = torch.randn(1, 5, 32)
    with pytest.raises(ValueError):
        attn(h, attention_mask=torch.zeros(1, 1, 4, 4))


def test_clip_mlp_forward():
    cfg = _clip_config()
    mlp = clip.NeuronCLIPMLP(cfg)
    out = mlp(torch.randn(1, 5, 32))
    assert out.shape == (1, 5, 32)


def test_clip_encoder_layer_forward():
    cfg = _clip_config()
    layer = clip.NeuronCLIPEncoderLayer(cfg)
    h = torch.randn(1, 6, 32)
    cam = torch.zeros(1, 1, 6, 6)
    out = layer(h, attention_mask=None, causal_attention_mask=cam, output_attentions=True)
    assert out[0].shape == (1, 6, 32)
    assert out[1].shape == (1, 4, 6, 6)


def test_clip_encoder_forward_hidden_states():
    cfg = _clip_config()
    enc = clip.NeuronCLIPEncoder(cfg)
    h = torch.randn(1, 6, 32)
    cam = torch.zeros(1, 1, 6, 6)
    out = enc(
        inputs_embeds=h,
        causal_attention_mask=cam,
        output_attentions=True,
        output_hidden_states=True,
    )
    assert out.last_hidden_state.shape == (1, 6, 32)
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1
    assert len(out.attentions) == cfg.num_hidden_layers


def test_clip_text_embeddings_forward():
    cfg = _clip_config()
    emb = clip.NeuronCLIPTextEmbeddings(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = emb(input_ids=ids)
    assert out.shape == (1, 5, 32)


def test_clip_text_model_forward_eos_2():
    cfg = _clip_config(eos_token_id=2)
    model = clip.NeuronCLIPTextModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    out = model(input_ids=ids)
    assert out.last_hidden_state.shape == (1, 7, 32)
    assert out.pooler_output.shape == (1, 32)


def test_clip_text_transformer_eos_not_2_and_tuple_return():
    cfg = _clip_config(eos_token_id=49)
    transformer = clip.NeuronCLIPTextTransformer(cfg).eval()
    ids = torch.randint(0, 48, (1, 7))
    ids[0, 3] = 49  # ensure an eos token is present
    out = transformer(
        input_ids=ids,
        attention_mask=torch.ones(1, 7),
        output_attentions=True,
        output_hidden_states=True,
        return_dict=False,
    )
    assert isinstance(out, tuple)
    assert out[0].shape == (1, 7, 32)


def test_clip_text_transformer_requires_input_ids():
    cfg = _clip_config()
    transformer = clip.NeuronCLIPTextTransformer(cfg).eval()
    with pytest.raises(ValueError):
        transformer(input_ids=None)


def test_clip_encoder_output_wrapper():
    out = clip.CLIPEncoderOutput({"pooler_output": torch.zeros(1, 4)})
    assert out.pooler_output.shape == (1, 4)


def test_clip_convert_hf_to_neuron_state_dict_renames():
    sd = {"text_model.encoder.layers.0.weight": torch.randn(2, 2)}
    out = clip.NeuronClipApplication.convert_hf_to_neuron_state_dict(dict(sd), None)
    assert "neuron_text_encoder.encoder.layers.0.weight" in out
    assert clip.NeuronClipApplication.update_state_dict_for_tied_weights({}) is None
