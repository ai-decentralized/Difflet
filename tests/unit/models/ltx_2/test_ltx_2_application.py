"""CPU-only unit coverage for difflet.models.ltx_2.application.

Covers dtype normalization, the dual-stream contract validator, transformer
config construction, the application skeleton (single/segmented/cfg-parallel),
component declaration, and ``__call__`` dispatch. The Neuron transformer is
built from a tiny diffusers config (no weights / no compile) or replaced with a
lightweight fake for the forward-dispatch branches.
"""

import json
import os
from types import SimpleNamespace


import pytest
import torch

from difflet.models.ltx_2.application import (
    LTX2DiTInputBundle,
    NeuronLTX2Application,
    _normalize_dtype,
    create_ltx_2_transformer_config,
    validate_ltx_2_dit_inputs,
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
        "_class_name": "LTX2VideoTransformer3DModel",
        "in_channels": 128,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "cross_attention_dim": 16,
        "vae_scale_factors": [8, 32, 32],
        "pos_embed_max_pos": 20,
        "base_height": 2048,
        "base_width": 2048,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_patch_size": 16,
        "audio_patch_size_t": 1,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 8,
        "audio_cross_attention_dim": 16,
        "audio_scale_factor": 4,
        "audio_sampling_rate": 16000,
        "audio_hop_length": 160,
        "audio_pos_embed_max_pos": 20,
        "num_layers": 1,
        "activation_fn": "gelu-approximate",
        "qk_norm": "rms_norm_across_heads",
        "caption_channels": 32,
        "attention_bias": True,
        "attention_out_bias": True,
        "rope_theta": 10000.0,
        "rope_double_precision": True,
        "causal_offset": 1,
        "timestep_scale_multiplier": 1000,
        "cross_attn_timestep_scale_multiplier": 1000,
        "rope_type": "interleaved",
        "use_prompt_embeddings": True,
        "perturbed_attn": False,
    }
    config.update(overrides)
    (transformer_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model_dir)


def _fake_config(dtype=torch.float32, batch_size=1):
    return SimpleNamespace(
        neuron_config=SimpleNamespace(batch_size=batch_size, torch_dtype=dtype),
        video_seq_len=6,
        in_channels=128,
        audio_seq_len=4,
        audio_in_channels=128,
        text_seq_len=5,
        audio_text_seq_len=5,
        video_text_dim=32,
        audio_text_dim=32,
    )


def _bundle(config):
    bs = int(config.neuron_config.batch_size)
    return LTX2DiTInputBundle(
        hidden_states=torch.zeros((bs, config.video_seq_len, config.in_channels)),
        audio_hidden_states=torch.zeros((bs, config.audio_seq_len, config.audio_in_channels)),
        encoder_hidden_states=torch.zeros((bs, config.text_seq_len, config.video_text_dim)),
        audio_encoder_hidden_states=torch.zeros(
            (bs, config.audio_text_seq_len, config.audio_text_dim)
        ),
        timestep=torch.zeros((bs,)),
        sigma=torch.zeros((bs,)),
        encoder_attention_mask=torch.ones((bs, config.text_seq_len), dtype=torch.bool),
        audio_encoder_attention_mask=torch.ones((bs, config.audio_text_seq_len), dtype=torch.bool),
        video_coords=torch.zeros((bs, 3, config.video_seq_len, 2), dtype=torch.float32),
        audio_coords=torch.zeros((bs, 1, config.audio_seq_len, 2), dtype=torch.float32),
    )


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_normalize_dtype_accepts_aliases_and_rejects_unknown():
    assert _normalize_dtype(torch.bfloat16) == torch.bfloat16
    assert _normalize_dtype("float32") == torch.float32
    assert _normalize_dtype("bfloat16") == torch.bfloat16
    with pytest.raises(ValueError, match="Unsupported LTX-2 dtype"):
        _normalize_dtype("fp8")


def test_as_model_inputs_order():
    config = _fake_config()
    inputs = _bundle(config).as_model_inputs()
    assert len(inputs) == 10
    assert inputs[4].shape == (1,)  # timestep
    assert inputs[8].shape == (1, 3, config.video_seq_len, 2)  # video_coords


def test_validate_dit_inputs_passes_and_flags_shape_and_dtype():
    config = _fake_config()
    bundle = _bundle(config)
    validate_ltx_2_dit_inputs(bundle, config=config, dtype=torch.float32)

    bad_shape = LTX2DiTInputBundle(
        **{**bundle.__dict__, "audio_hidden_states": torch.zeros((1, 99, 128))}
    )
    with pytest.raises(ValueError, match="audio_hidden_states"):
        validate_ltx_2_dit_inputs(bad_shape, config=config, dtype=torch.float32)

    bad_dtype = LTX2DiTInputBundle(
        **{**bundle.__dict__, "video_coords": bundle.video_coords.to(torch.float64)}
    )
    with pytest.raises(TypeError, match="video_coords"):
        validate_ltx_2_dit_inputs(bad_dtype, config=config, dtype=torch.float32)


def test_create_transformer_config_from_tiny_diffusers_config(tmp_path):
    model_path = _write_transformer_config(tmp_path / "LTX-2")
    config = create_ltx_2_transformer_config(
        model_path=model_path,
        world_size=8,
        tp_degree=4,
        dtype=torch.bfloat16,
        height=256,
        width=512,
        num_frames=17,
        text_seq_len=8,
        audio_num_frames=4,
        frame_rate=12.0,
        cfg_parallel_enabled=True,
    )
    assert config.neuron_config.world_size == 8
    assert config.cfg_parallel_enabled is True
    assert config.frame_rate == 12.0
    # audio_text_seq_len defaults to text_seq_len when not supplied
    assert int(config.audio_text_seq_len) == 8


# --------------------------------------------------------------------------- #
# idle application (empty model dir -> transformer is None)
# --------------------------------------------------------------------------- #
def _idle_app(tmp_path, **kwargs):
    return NeuronLTX2Application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="fp32",
        shape={"height": None, "width": None, "num_frames": None},
        **kwargs,
    )


def test_idle_application_defaults_and_no_components(tmp_path):
    app = _idle_app(tmp_path)
    assert app.transformer is None
    assert app.shape == {"height": 512, "width": 768, "num_frames": 121}
    assert app.text_seq_len == 1024
    assert app.components() == []
    assert "compile" in app.no_components_message("compile")
    assert "load" in app.no_components_message("load")
    assert isinstance(app.no_components_message("other"), str)


def test_idle_application_raises_for_transformer_only_paths(tmp_path):
    app = _idle_app(tmp_path)
    with pytest.raises(NotImplementedError, match="contract requires"):
        app.dit_input_contract()
    with pytest.raises(NotImplementedError, match="forward_dit requires"):
        app.forward_dit(object())
    with pytest.raises(NotImplementedError, match="teacache_mod_input requires"):
        app.teacache_mod_input(torch.zeros(1), torch.zeros(1))
    with pytest.raises(NotImplementedError, match="end-to-end"):
        app()


# --------------------------------------------------------------------------- #
# active application via a tiny real config
# --------------------------------------------------------------------------- #
def test_single_mode_application_declares_transformer_and_contract(tmp_path):
    model_path = _write_transformer_config(tmp_path / "LTX-2")
    app = NeuronLTX2Application(
        model_path=model_path,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        shape={"height": 256, "width": 512, "num_frames": 17},
        text_seq_len=8,
        audio_num_frames=4,
        frame_rate=12.0,
    )
    assert app.transformer is not None
    assert [c.name for c in app.components()] == ["transformer"]
    assert app.frame_rate == 12.0
    contract = app.dit_input_contract()
    assert contract["hidden_states"]["shape"] == (1, 3 * 8 * 16, 128)
    assert contract["audio_hidden_states"]["shape"] == (1, 4, 128)
    assert contract["encoder_hidden_states"]["shape"] == (1, 8, 32)
    assert contract["video_coords"]["dtype"] is torch.float32


def test_segmented_mode_uses_component_specs(tmp_path):
    model_path = _write_transformer_config(tmp_path / "LTX-2")
    app = NeuronLTX2Application(
        model_path=model_path,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        shape={"height": 256, "width": 512, "num_frames": 17},
        transformer_mode="segmented",
        text_seq_len=8,
        audio_num_frames=4,
    )
    assert [c.name for c in app.components()] == ["transformer_block"]


def test_cfg_parallel_doubles_world_and_batch(tmp_path):
    model_path = _write_transformer_config(tmp_path / "LTX-2")
    app = NeuronLTX2Application(
        model_path=model_path,
        parallel=DiffletParallelConfig(tp_degree=4, cfg_parallel_enabled=True),
        dtype="bf16",
        shape={"height": 256, "width": 512, "num_frames": 17},
        text_seq_len=8,
        audio_num_frames=4,
    )
    assert app.cfg_parallel_enabled is True
    assert app.transformer.config.neuron_config.world_size == 8
    assert app.transformer.config.neuron_config.batch_size == 2
    assert app.dit_input_contract()["timestep"]["shape"] == (2,)


def test_invalid_transformer_mode_raises(tmp_path):
    model_path = _write_transformer_config(tmp_path / "LTX-2")
    with pytest.raises(ValueError, match="transformer_mode must be"):
        NeuronLTX2Application(
            model_path=model_path,
            parallel=DiffletParallelConfig(tp_degree=4),
            dtype="bf16",
            shape={"height": 256, "width": 512, "num_frames": 17},
            transformer_mode="bogus",
            text_seq_len=8,
            audio_num_frames=4,
        )


# --------------------------------------------------------------------------- #
# dispatch branches with a lightweight fake transformer
# --------------------------------------------------------------------------- #
class FakeTransformer:
    def __init__(self, config):
        self.config = config
        self.calls = []
        self.mod_calls = []

    def __call__(self, *inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return inputs[0], inputs[1]

    def teacache_mod_input(self, hidden_states, timestep):
        self.mod_calls.append((hidden_states, timestep))
        return hidden_states * 2


def test_forward_dit_validates_then_calls_transformer(tmp_path):
    app = _idle_app(tmp_path)
    app.dtype = torch.float32
    config = _fake_config()
    app.transformer = FakeTransformer(config)
    bundle = _bundle(config)

    video, audio = app.forward_dit(bundle)
    assert torch.equal(video, bundle.hidden_states)
    assert torch.equal(audio, bundle.audio_hidden_states)


def test_teacache_mod_input_delegates(tmp_path):
    app = _idle_app(tmp_path)
    config = _fake_config()
    app.transformer = FakeTransformer(config)
    out = app.teacache_mod_input(torch.ones(2), torch.zeros(2))
    assert torch.allclose(out, torch.full((2,), 2.0))


def test_call_dispatches_bundle_named_and_positional(tmp_path):
    app = _idle_app(tmp_path)
    app.dtype = torch.float32
    config = _fake_config()
    app.transformer = FakeTransformer(config)
    bundle = _bundle(config)

    app(bundle)  # single positional bundle
    app(**bundle.__dict__)  # direct keyword tensors
    app(bundle.hidden_states, bundle.audio_hidden_states)  # raw positional passthrough
    assert len(app.transformer.calls) == 3


def test_call_with_active_transformer_delegates_to_pipeline(tmp_path):
    model_path = _write_transformer_config(tmp_path / "LTX-2")
    app = NeuronLTX2Application(
        model_path=model_path,
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype="bf16",
        shape={"height": 256, "width": 512, "num_frames": 17},
        text_seq_len=8,
        audio_num_frames=4,
    )
    # No args -> falls through to the runtime pipeline, which needs conditioning.
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        app()
