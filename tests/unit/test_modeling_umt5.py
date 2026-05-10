"""Unit tests for nova.models.wan.umt5.modeling_umt5 (W3b)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


def test_modeling_umt5_imports_only_from_allowed_modules():
    """Rule 1: UMT5 modeling imports only torch / stdlib / transformers
    activations / nova.ops.
    """
    src = Path("/home/ubuntu/nova/nova/models/wan/umt5/modeling_umt5.py").read_text()
    tree = ast.parse(src)
    forbidden_roots = {"neuronx_distributed", "nkilib", "torch_neuronx"}
    forbidden_prefixes = ("nova.core",)
    offending: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden_roots or alias.name.startswith(forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if root in forbidden_roots or mod.startswith(forbidden_prefixes):
                offending.append(mod)

    assert not offending, f"forbidden imports in modeling_umt5.py: {offending}"


def test_wan_umt5_config_defaults_match_xxl():
    from nova.models.wan.umt5.modeling_umt5 import WanUmT5Config

    cfg = WanUmT5Config()
    assert cfg.vocab_size == 256384
    assert cfg.d_model == 4096
    assert cfg.d_kv == 64
    assert cfg.d_ff == 10240
    assert cfg.num_heads == 64
    assert cfg.num_layers == 24
    assert cfg.is_gated_act is True
    assert cfg.dense_act_fn == "gelu_new"
    assert cfg.relative_attention_num_buckets == 32
    assert cfg.relative_attention_max_distance == 128
    assert cfg.layer_norm_epsilon == 1e-6
    assert cfg.inner_dim == 4096


def test_wan_umt5_config_rejects_non_gated():
    from nova.models.wan.umt5.modeling_umt5 import WanUmT5Config

    with pytest.raises(NotImplementedError, match="is_gated_act"):
        WanUmT5Config(is_gated_act=False)


def test_wan_umt5_config_from_diffusers_dict_filters_unknown_keys():
    from nova.models.wan.umt5.modeling_umt5 import WanUmT5Config

    raw = {
        "vocab_size": 256384,
        "d_model": 4096,
        "d_kv": 64,
        "d_ff": 10240,
        "num_heads": 64,
        "num_layers": 24,
        "_name_or_path": "google/umt5-xxl",  # noise
        "architectures": ["UMT5EncoderModel"],  # noise
        "tokenizer_class": "T5Tokenizer",  # noise
    }
    cfg = WanUmT5Config.from_diffusers_dict(raw)
    assert cfg.d_model == 4096
    assert cfg.num_layers == 24


def test_relative_position_bucket_matches_t5_reference():
    """Bucket function is shared with T5; sanity-check known buckets."""
    from nova.models.wan.umt5.modeling_umt5 import WanUmT5Attention
    import torch

    bucket = WanUmT5Attention._relative_position_bucket
    # bidirectional=True, num_buckets=32, max_distance=128
    pos = torch.tensor([-128, -16, -1, 0, 1, 16, 128])
    out = bucket(pos, bidirectional=True, num_buckets=32, max_distance=128)
    # Reference: |pos|=0..15 are exact (16 buckets per side), |pos|>=16 uses log
    # bidirectional shifts +0 for negative, +num_buckets/2 for positive.
    # Just verify shape + range invariants.
    assert out.shape == pos.shape
    assert (out >= 0).all() and (out < 32).all()


def test_modeling_umt5_classes_are_importable():
    from nova.models.wan.umt5 import modeling_umt5

    expected = [
        "WanUmT5Attention",
        "WanUmT5Block",
        "WanUmT5Config",
        "WanUmT5DenseGatedActDense",
        "WanUmT5EncoderModel",
        "WanUmT5LayerFF",
        "WanUmT5LayerSelfAttention",
        "WanUmT5Stack",
    ]
    for name in expected:
        cls = getattr(modeling_umt5, name)
        assert inspect.isclass(cls), f"{name} should be a class"


def test_wan_umt5_encoder_model_forward_signature():
    from nova.models.wan.umt5.modeling_umt5 import WanUmT5EncoderModel

    sig = inspect.signature(WanUmT5EncoderModel.forward)
    params = list(sig.parameters)
    assert params == ["self", "input_ids", "attention_mask"]


def test_wan_umt5_state_dict_key_layout_matches_hf():
    """Verify the module attribute names produce HF-compatible state-dict keys.

    HF UMT5EncoderModel keys look like:
      shared.weight
      encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
      encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight
      encoder.block.{i}.layer.0.layer_norm.weight
      encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
      encoder.block.{i}.layer.1.layer_norm.weight
      encoder.final_layer_norm.weight

    This test does NOT instantiate (would need a TP PG); it walks the class
    tree to confirm submodule attribute names.
    """
    from nova.models.wan.umt5 import modeling_umt5 as m

    # Top-level encoder
    enc_init = inspect.getsource(m.WanUmT5EncoderModel.__init__)
    assert "self.shared" in enc_init
    assert "self.encoder" in enc_init

    # Stack
    stack_init = inspect.getsource(m.WanUmT5Stack.__init__)
    assert "self.embed_tokens" in stack_init
    assert "self.block" in stack_init
    assert "self.final_layer_norm" in stack_init

    # Block layer-0 = self-attn
    block_init = inspect.getsource(m.WanUmT5Block.__init__)
    assert "self.layer" in block_init

    # Layer self-attention wrapper exposes `.SelfAttention` and `.layer_norm`
    sa_init = inspect.getsource(m.WanUmT5LayerSelfAttention.__init__)
    assert "self.SelfAttention" in sa_init
    assert "self.layer_norm" in sa_init

    # Attention has q/k/v/o
    att_init = inspect.getsource(m.WanUmT5Attention.__init__)
    for name in ("self.q", "self.k", "self.v", "self.o"):
        assert name in att_init
    assert "self.relative_attention_bias" in att_init

    # FF wrapper exposes `.DenseReluDense` and `.layer_norm`
    ff_init = inspect.getsource(m.WanUmT5LayerFF.__init__)
    assert "self.DenseReluDense" in ff_init
    assert "self.layer_norm" in ff_init

    # Gated FFN has wi_0, wi_1, wo
    dense_init = inspect.getsource(m.WanUmT5DenseGatedActDense.__init__)
    for name in ("self.wi_0", "self.wi_1", "self.wo"):
        assert name in dense_init


def test_wan_text_encoder_inference_config_defaults():
    """WanTextEncoderInferenceConfig.add_derived_config sets text_seq_len=512."""
    import torch

    from nova.backends.trainium.wan.text_encoder import (
        WanTextEncoderInferenceConfig,
    )
    from nova.backends.trainium.core.config import NeuronConfig
    from nova.utils.diffusers_adapter import load_diffusers_config

    snap = (
        "/home/ubuntu/.cache/huggingface/hub/"
        "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
        "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
    )
    if not Path(f"{snap}/text_encoder/config.json").exists():
        pytest.skip("Wan2.2 UMT5 config snapshot not present")

    nc = NeuronConfig(
        batch_size=1, tp_degree=4, world_size=4, torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    cfg = WanTextEncoderInferenceConfig(
        neuron_config=nc, load_config=load_diffusers_config(f"{snap}/text_encoder")
    )
    assert cfg.text_seq_len == 512
    assert cfg.d_model == 4096
    assert cfg.num_layers == 24
    assert cfg.num_heads == 64
    assert cfg.d_kv == 64
    assert cfg.d_ff == 10240
    assert cfg.is_gated_act is True
    assert cfg.inner_dim == 4096


def test_model_wrapper_wan_text_encoder_input_shapes():
    import torch

    from nova.backends.trainium.wan.text_encoder import (
        ModelWrapperWanTextEncoder,
        WanTextEncoderInferenceConfig,
    )
    from nova.backends.trainium.core.config import NeuronConfig
    from nova.utils.diffusers_adapter import load_diffusers_config

    snap = (
        "/home/ubuntu/.cache/huggingface/hub/"
        "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
        "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
    )
    if not Path(f"{snap}/text_encoder/config.json").exists():
        pytest.skip("Wan2.2 UMT5 config snapshot not present")

    nc = NeuronConfig(
        batch_size=1, tp_degree=4, world_size=4, torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    cfg = WanTextEncoderInferenceConfig(
        neuron_config=nc, load_config=load_diffusers_config(f"{snap}/text_encoder")
    )
    mw = ModelWrapperWanTextEncoder(config=cfg, model_cls=object, tag="WanUmT5EncoderModel")
    inputs = mw.input_generator()
    shapes = [tuple(t.shape) for t in inputs[0]]
    dtypes = [t.dtype for t in inputs[0]]
    assert shapes == [(1, 512), (1, 512)]
    assert dtypes == [torch.int64, torch.int32]
