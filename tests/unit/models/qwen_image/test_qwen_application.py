"""CPU-only unit coverage for difflet.models.qwen_image.application.

Covers the host-side config/parallelism helpers, the contract validator, the
component declaration, and the ``__call__`` dispatch. The Neuron transformer is
constructed from a tiny diffusers config (no weights, no compile) or replaced
with a lightweight fake for the forward-dispatch branches.
"""

import json
import os
from types import SimpleNamespace


import pytest
import torch

from difflet.models.qwen_image import application as app_mod
from difflet.models.qwen_image.application import (
    NeuronQwenImageApplication,
    QwenImageDiTInputBundle,
    _normalize_dtype,
    create_qwen_image_transformer_config,
    validate_qwen_image_dit_inputs,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def _write_transformer_config(model_dir, **overrides):
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    config = {
        "_class_name": "QwenImageTransformer2DModel",
        "patch_size": 2,
        "in_channels": 64,
        "out_channels": 16,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "guidance_embeds": False,
        "axes_dims_rope": [2, 2, 4],
    }
    config.update(overrides)
    (transformer_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model_dir)


def _bundle(config):
    bs = int(getattr(config.neuron_config, "batch_size", 1))
    text_seq_len = int(getattr(config, "text_seq_len", 1024))
    return QwenImageDiTInputBundle(
        hidden_states=torch.zeros((bs, int(config.image_seq_len), int(config.in_channels))),
        timestep=torch.zeros((bs,)),
        encoder_hidden_states=torch.zeros((bs, text_seq_len, int(config.joint_attention_dim))),
        encoder_hidden_states_mask=torch.ones((bs, text_seq_len), dtype=torch.bool),
        guidance=torch.zeros((bs,)),
    )


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_normalize_dtype_accepts_aliases_and_rejects_unknown():
    assert _normalize_dtype(torch.float16) == torch.float16
    assert _normalize_dtype("bf16") == torch.bfloat16
    assert _normalize_dtype("torch.float32") == torch.float32
    with pytest.raises(ValueError, match="Unsupported Qwen-Image dtype"):
        _normalize_dtype("int8")


def test_as_model_inputs_order():
    config = SimpleNamespace(
        neuron_config=SimpleNamespace(batch_size=1),
        image_seq_len=4,
        in_channels=8,
        joint_attention_dim=8,
        text_seq_len=4,
    )
    bundle = _bundle(config)
    inputs = bundle.as_model_inputs()
    assert len(inputs) == 5
    assert inputs[0] is bundle.hidden_states
    assert inputs[-1] is bundle.guidance


def test_validate_dit_inputs_passes_and_reports_shape_and_dtype():
    config = SimpleNamespace(
        neuron_config=SimpleNamespace(batch_size=1),
        image_seq_len=4,
        in_channels=8,
        joint_attention_dim=8,
        text_seq_len=4,
    )
    bundle = _bundle(config)
    validate_qwen_image_dit_inputs(bundle, config=config, dtype=torch.float32)

    bad_shape = QwenImageDiTInputBundle(
        hidden_states=torch.zeros((1, 5, 8)),
        timestep=bundle.timestep,
        encoder_hidden_states=bundle.encoder_hidden_states,
        encoder_hidden_states_mask=bundle.encoder_hidden_states_mask,
        guidance=bundle.guidance,
    )
    with pytest.raises(ValueError, match="hidden_states"):
        validate_qwen_image_dit_inputs(bad_shape, config=config, dtype=torch.float32)

    bad_dtype = QwenImageDiTInputBundle(
        hidden_states=bundle.hidden_states,
        timestep=bundle.timestep.to(torch.float16),
        encoder_hidden_states=bundle.encoder_hidden_states,
        encoder_hidden_states_mask=bundle.encoder_hidden_states_mask,
        guidance=bundle.guidance,
    )
    with pytest.raises(TypeError, match="timestep"):
        validate_qwen_image_dit_inputs(bad_dtype, config=config, dtype=torch.float32)


def test_create_transformer_config_from_tiny_diffusers_config(tmp_path):
    model_path = _write_transformer_config(tmp_path / "Qwen-Image")
    config = create_qwen_image_transformer_config(
        model_path=model_path,
        world_size=4,
        tp_degree=4,
        dtype=torch.bfloat16,
        height=64,
        width=64,
        text_seq_len=16,
        context_parallel_enabled=True,
        cp_mode="ring",
    )
    assert config.neuron_config.world_size == 4
    assert config.context_parallel_enabled is True
    assert config.cp_mode == "ring"
    assert int(config.in_channels) == 64


# --------------------------------------------------------------------------- #
# idle application (no transformer config -> transformer is None)
# --------------------------------------------------------------------------- #
def _idle_app(tmp_path, **kwargs):
    return NeuronQwenImageApplication(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": None, "width": None},
        **kwargs,
    )


def test_idle_application_defaults_and_no_components(tmp_path):
    app = _idle_app(tmp_path)
    assert app.transformer is None
    assert app.shape == {"height": 1024, "width": 1024, "num_frames": None}
    assert app.components() == []
    assert "compile" in app.no_components_message("compile")
    assert "load" in app.no_components_message("load")
    # unknown action delegates to the base-class message
    assert isinstance(app.no_components_message("other"), str)


def test_idle_application_raises_for_transformer_only_paths(tmp_path):
    app = _idle_app(tmp_path)
    with pytest.raises(NotImplementedError, match="contract requires"):
        app.dit_input_contract()
    with pytest.raises(NotImplementedError, match="forward_dit requires"):
        app.forward_dit(object())
    with pytest.raises(NotImplementedError, match="fused probe"):
        app.teacache_delta(object())
    with pytest.raises(NotImplementedError, match="end-to-end"):
        app()


# --------------------------------------------------------------------------- #
# active application via a tiny real config
# --------------------------------------------------------------------------- #
def test_application_declares_transformer_and_contract(tmp_path):
    model_path = _write_transformer_config(tmp_path / "Qwen-Image")
    app = NeuronQwenImageApplication(
        model_path=model_path,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        shape={"height": 64, "width": 64},
        text_seq_len=8,
    )
    assert app.transformer is not None
    assert [c.name for c in app.components()] == ["transformer"]
    contract = app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 16, 64)
    assert contract["encoder_hidden_states"]["shape"] == (1, 8, 32)
    assert contract["encoder_hidden_states_mask"]["dtype"] is torch.bool
    assert contract["guidance"]["shape"] == (1,)


def test_application_declares_fused_teacache_probe_component(tmp_path):
    model_path = _write_transformer_config(tmp_path / "Qwen-Image")
    app = NeuronQwenImageApplication(
        model_path=model_path,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        shape={"height": 64, "width": 64},
        text_seq_len=8,
        teacache_fused=True,
    )
    assert app.teacache_probe_fused is True
    assert [c.name for c in app.components()] == ["transformer", "teacache_probe"]


def test_call_with_active_transformer_delegates_to_pipeline(tmp_path):
    model_path = _write_transformer_config(tmp_path / "Qwen-Image")
    app = NeuronQwenImageApplication(
        model_path=model_path,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        shape={"height": 64, "width": 64},
        text_seq_len=8,
    )
    # No args -> falls through to the runtime pipeline, which needs conditioning.
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        app()


# --------------------------------------------------------------------------- #
# dispatch branches with a lightweight fake transformer
# --------------------------------------------------------------------------- #
class FakeTransformer:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def __call__(self, *inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return inputs[0]


class FakeProbe:
    def __init__(self):
        self.calls = []

    def teacache_delta(self, *inputs):
        self.calls.append(inputs)
        return torch.tensor(0.5)


def _fake_config(dtype):
    return SimpleNamespace(
        neuron_config=SimpleNamespace(batch_size=1, torch_dtype=dtype),
        image_seq_len=4,
        in_channels=8,
        joint_attention_dim=8,
        text_seq_len=4,
    )


def test_forward_dit_validates_then_calls_transformer(tmp_path):
    app = _idle_app(tmp_path)
    app.dtype = torch.float32
    config = _fake_config(torch.float32)
    app.transformer = FakeTransformer(config)
    bundle = _bundle(config)

    out = app.forward_dit(bundle)
    assert torch.equal(out, bundle.hidden_states)
    assert len(app.transformer.calls) == 1


def test_call_dispatches_bundle_named_and_positional(tmp_path):
    app = _idle_app(tmp_path)
    app.dtype = torch.float32
    config = _fake_config(torch.float32)
    app.transformer = FakeTransformer(config)
    bundle = _bundle(config)

    # single positional bundle
    app(bundle)
    # direct keyword tensors -> bundle
    app(**{
        "hidden_states": bundle.hidden_states,
        "timestep": bundle.timestep,
        "encoder_hidden_states": bundle.encoder_hidden_states,
        "encoder_hidden_states_mask": bundle.encoder_hidden_states_mask,
        "guidance": bundle.guidance,
    })
    # raw positional passthrough to the transformer
    app(bundle.hidden_states, bundle.timestep)
    assert len(app.transformer.calls) == 3


def test_teacache_delta_routes_to_fused_probe(tmp_path):
    app = _idle_app(tmp_path)
    app.dtype = torch.float32
    config = _fake_config(torch.float32)
    app.transformer = FakeTransformer(config)
    app.teacache_probe = FakeProbe()
    app.teacache_probe_fused = True
    bundle = _bundle(config)

    delta = app.teacache_delta(bundle)
    assert float(delta) == pytest.approx(0.5)
    assert len(app.teacache_probe.calls) == 1
