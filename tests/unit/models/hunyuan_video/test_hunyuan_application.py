"""CPU coverage for HunyuanVideo application config/helpers and routing.

Exercises the pure host-side helpers (input-bundle contracts, dtype
normalisation, validation) and the parts of ``NeuronHunyuanVideoApplication``
reachable without Trainium runtime / real weights: construction against an
empty model path (no ``config.json`` -> no backbone components), component
declaration, and ``no_components_message`` / ``dit_input_contract`` branches.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.hunyuan_video import application as app


# --------------------------------------------------------------------------- #
# Input bundles
# --------------------------------------------------------------------------- #
def test_dit_input_bundle_as_model_inputs_order():
    bundle = app.HunyuanVideoDiTInputBundle(
        hidden_states=torch.zeros(1),
        timestep=torch.ones(1),
        encoder_hidden_states=torch.full((1,), 2.0),
        encoder_attention_mask=torch.full((1,), 3.0),
        pooled_projections=torch.full((1,), 4.0),
        guidance=torch.full((1,), 5.0),
    )
    values = [float(t[0]) for t in bundle.as_model_inputs()]
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_dit15_input_bundle_as_model_inputs_order():
    bundle = app.HunyuanVideo15DiTInputBundle(
        hidden_states=torch.zeros(1),
        timestep=torch.ones(1),
        encoder_hidden_states=torch.full((1,), 2.0),
        encoder_attention_mask=torch.full((1,), 3.0),
        timestep_r=torch.full((1,), 4.0),
        encoder_hidden_states_2=torch.full((1,), 5.0),
        encoder_attention_mask_2=torch.full((1,), 6.0),
        image_embeds=torch.full((1,), 7.0),
    )
    values = [float(t[0]) for t in bundle.as_model_inputs()]
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


# --------------------------------------------------------------------------- #
# _normalize_dtype
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value, expected",
    [
        (torch.bfloat16, torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("torch.bfloat16", torch.bfloat16),
        ("fp32", torch.float32),
        ("float32", torch.float32),
        ("torch.float32", torch.float32),
    ],
)
def test_normalize_dtype(value, expected):
    assert app._normalize_dtype(value) == expected


def test_normalize_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported"):
        app._normalize_dtype("int8")


# --------------------------------------------------------------------------- #
# validate_hunyuan_video_dit_inputs (1.0)
# --------------------------------------------------------------------------- #
def _config_10():
    return SimpleNamespace(
        neuron_config=SimpleNamespace(batch_size=1),
        text_seq_len=3,
        in_channels=2,
        latent_frames=2,
        latent_height=2,
        latent_width=2,
        text_embed_dim=6,
        pooled_projection_dim=5,
    )


def _bundle_10(dtype=torch.float32):
    return app.HunyuanVideoDiTInputBundle(
        hidden_states=torch.zeros(1, 2, 2, 2, 2, dtype=dtype),
        timestep=torch.zeros(1, dtype=dtype),
        encoder_hidden_states=torch.zeros(1, 3, 6, dtype=dtype),
        encoder_attention_mask=torch.zeros(1, 3, dtype=torch.int64),
        pooled_projections=torch.zeros(1, 5, dtype=dtype),
        guidance=torch.zeros(1, dtype=dtype),
    )


def test_validate_dit_inputs_passes():
    app.validate_hunyuan_video_dit_inputs(_bundle_10(), config=_config_10(), dtype=torch.float32)


def test_validate_dit_inputs_bad_shape():
    bundle = _bundle_10()
    bundle = app.HunyuanVideoDiTInputBundle(
        **{**bundle.__dict__, "pooled_projections": torch.zeros(1, 9)}
    )
    with pytest.raises(ValueError, match="pooled_projections"):
        app.validate_hunyuan_video_dit_inputs(bundle, config=_config_10(), dtype=torch.float32)


def test_validate_dit_inputs_bad_dtype():
    with pytest.raises(TypeError, match="dtype"):
        app.validate_hunyuan_video_dit_inputs(
            _bundle_10(dtype=torch.float32), config=_config_10(), dtype=torch.bfloat16
        )


# --------------------------------------------------------------------------- #
# validate_hunyuan_video15_dit_inputs (1.5)
# --------------------------------------------------------------------------- #
def _config_15():
    return SimpleNamespace(
        neuron_config=SimpleNamespace(batch_size=1),
        text_seq_len=3,
        text_seq_len_2=2,
        image_seq_len=4,
        in_channels=2,
        latent_frames=2,
        latent_height=2,
        latent_width=2,
        text_embed_dim=6,
        text_embed_2_dim=5,
        image_embed_dim=7,
    )


def _bundle_15(dtype=torch.float32):
    return app.HunyuanVideo15DiTInputBundle(
        hidden_states=torch.zeros(1, 2, 2, 2, 2, dtype=dtype),
        timestep=torch.zeros(1, dtype=dtype),
        encoder_hidden_states=torch.zeros(1, 3, 6, dtype=dtype),
        encoder_attention_mask=torch.zeros(1, 3, dtype=torch.int64),
        timestep_r=torch.zeros(1, dtype=dtype),
        encoder_hidden_states_2=torch.zeros(1, 2, 5, dtype=dtype),
        encoder_attention_mask_2=torch.zeros(1, 2, dtype=torch.int64),
        image_embeds=torch.zeros(1, 4, 7, dtype=dtype),
    )


def test_validate_dit15_inputs_passes():
    app.validate_hunyuan_video15_dit_inputs(_bundle_15(), config=_config_15(), dtype=torch.float32)


def test_validate_dit15_inputs_bad_shape():
    bundle = app.HunyuanVideo15DiTInputBundle(
        **{**_bundle_15().__dict__, "image_embeds": torch.zeros(1, 9, 7)}
    )
    with pytest.raises(ValueError, match="1.5 DiT input"):
        app.validate_hunyuan_video15_dit_inputs(bundle, config=_config_15(), dtype=torch.float32)


def test_validate_dit15_inputs_bad_dtype():
    with pytest.raises(TypeError, match="1.5 DiT input"):
        app.validate_hunyuan_video15_dit_inputs(
            _bundle_15(dtype=torch.float32), config=_config_15(), dtype=torch.bfloat16
        )


# --------------------------------------------------------------------------- #
# NeuronHunyuanVideoApplication construction (no config.json -> no components)
# --------------------------------------------------------------------------- #
def _parallel():
    return SimpleNamespace(tp_degree=1, world_size=1, cp_degree=1, cp_mode="gather_kv")


def test_application_builds_without_config_defaults_10():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path,
            parallel=_parallel(),
            dtype="bf16",
            shape={"height": None, "width": None, "num_frames": None},
        )
    assert application.model_version == "1.0"
    assert application.dtype == torch.bfloat16
    # 1.0 default shape
    assert application.shape == {"height": 320, "width": 512, "num_frames": 61}
    assert application.transformer is None
    assert application.vae_decoder is None
    assert application.components() == []
    assert application.pipeline is not None


def test_application_builds_version_15_default_shape():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path,
            parallel=_parallel(),
            dtype=torch.float32,
            shape={},
            model_version="1.5",
        )
    assert application.model_version == "1.5"
    assert application.shape == {"height": 480, "width": 848, "num_frames": 121}


def test_application_explicit_shape_override():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path,
            parallel=_parallel(),
            dtype="bf16",
            shape={"height": 16, "width": 32, "num_frames": 9},
            transformer_subfolder="custom_transformer",
        )
    assert application.shape == {"height": 16, "width": 32, "num_frames": 9}
    assert application.transformer_path.endswith("custom_transformer")


def test_application_no_components_messages():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype="bf16", shape={}
        )
        assert "transformer/config.json" in application.no_components_message("compile")
        assert "compiled component" in application.no_components_message("load")

        application_15 = app.NeuronHunyuanVideoApplication(
            model_path=path,
            parallel=_parallel(),
            dtype="bf16",
            shape={},
            model_version="1.5",
        )
        assert "HunyuanVideo 1.5" in application_15.no_components_message("compile")


def test_application_no_components_message_default_branch():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype="bf16", shape={}
        )
        # an action that is neither 'compile' nor 'load' falls through to the base
        message = application.no_components_message("inspect")
    assert isinstance(message, str)
    assert message


def test_application_dit_input_contract_requires_transformer():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype="bf16", shape={}
        )
    with pytest.raises(NotImplementedError, match="active transformer"):
        application.dit_input_contract()


def test_application_call_without_runtime_raises():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype="bf16", shape={}
        )
    with pytest.raises(NotImplementedError, match="end-to-end"):
        application(torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_application_forward_dit_requires_transformer():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype="bf16", shape={}
        )
    with pytest.raises(NotImplementedError, match="active transformer"):
        application.forward_dit(_bundle_10())


def test_application_teacache_mod_input_requires_transformer():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype="bf16", shape={}
        )
    with pytest.raises(NotImplementedError, match="TeaCache"):
        application.teacache_mod_input(_bundle_10())


def test_application_teacache_delta_requires_fused_probe():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype="bf16", shape={}
        )
    with pytest.raises(NotImplementedError, match="fused probe"):
        application.teacache_delta(_bundle_10())


# --------------------------------------------------------------------------- #
# components() / forward_dit routing with a fake transformer (no NEFF)
# --------------------------------------------------------------------------- #
class _FakeTransformer:
    def __init__(self):
        self.config = _config_10()
        self.calls = []

    def __call__(self, *inputs):
        self.calls.append(inputs)
        return torch.zeros(1)


def _app_with_fake_transformer(path):
    application = app.NeuronHunyuanVideoApplication(
        model_path=path, parallel=_parallel(), dtype=torch.float32, shape={}
    )
    application.transformer = _FakeTransformer()
    return application


def test_components_lists_transformer_spec():
    with tempfile.TemporaryDirectory() as path:
        application = _app_with_fake_transformer(path)
        specs = application.components()
    names = [spec.name for spec in specs]
    assert "transformer" in names


def test_components_uses_component_specs_and_probe_and_vae():
    spec_cls = app.ComponentSpec

    class _SpecTransformer:
        def component_specs(self, prefix):
            return [spec_cls(name=f"{prefix}.block0", component=object())]

    class _SpecVae:
        def component_specs(self, prefix):
            return [spec_cls(name=f"{prefix}.decoder", component=object())]

    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path, parallel=_parallel(), dtype=torch.float32, shape={}
        )
        application.transformer = _SpecTransformer()
        application.teacache_probe = object()
        application.vae_decoder = _SpecVae()
        names = [spec.name for spec in application.components()]
    assert "transformer.block0" in names
    assert "teacache_probe" in names
    assert "vae_decoder.decoder" in names


def test_dit_input_contract_10_shapes():
    with tempfile.TemporaryDirectory() as path:
        application = _app_with_fake_transformer(path)
        contract = application.dit_input_contract()
    assert set(contract) == {
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "pooled_projections",
        "guidance",
    }
    assert contract["hidden_states"]["shape"] == (1, 2, 2, 2, 2)
    assert contract["encoder_attention_mask"]["dtype"] == torch.int64


def test_forward_dit_validates_and_dispatches():
    with tempfile.TemporaryDirectory() as path:
        application = _app_with_fake_transformer(path)
        application.forward_dit(_bundle_10(dtype=torch.float32))
    assert application.transformer.calls  # transformer was called with 6 inputs
    assert len(application.transformer.calls[0]) == 6


def test_call_with_single_bundle_arg_routes_to_forward_dit():
    with tempfile.TemporaryDirectory() as path:
        application = _app_with_fake_transformer(path)
        application(_bundle_10(dtype=torch.float32))
    assert len(application.transformer.calls) == 1


def test_call_with_direct_kwargs_builds_bundle():
    with tempfile.TemporaryDirectory() as path:
        application = _app_with_fake_transformer(path)
        bundle = _bundle_10(dtype=torch.float32)
        application(
            hidden_states=bundle.hidden_states,
            timestep=bundle.timestep,
            encoder_hidden_states=bundle.encoder_hidden_states,
            encoder_attention_mask=bundle.encoder_attention_mask,
            pooled_projections=bundle.pooled_projections,
            guidance=bundle.guidance,
        )
    assert len(application.transformer.calls) == 1


def test_call_with_positional_args_forwards_to_transformer():
    with tempfile.TemporaryDirectory() as path:
        application = _app_with_fake_transformer(path)
        application(torch.zeros(1), torch.zeros(1))
    # positional path forwards raw args straight through (no bundle validation)
    assert application.transformer.calls[-1] == (torch.zeros(1), torch.zeros(1))


def test_call_15_with_direct_kwargs_builds_bundle():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path,
            parallel=_parallel(),
            dtype=torch.float32,
            shape={},
            model_version="1.5",
        )

        class _Fake15Transformer:
            def __init__(self):
                self.config = _config_15()
                self.calls = []

            def __call__(self, *inputs):
                self.calls.append(inputs)
                return torch.zeros(1)

        application.transformer = _Fake15Transformer()
        bundle = _bundle_15(dtype=torch.float32)
        application(
            hidden_states=bundle.hidden_states,
            timestep=bundle.timestep,
            encoder_hidden_states=bundle.encoder_hidden_states,
            encoder_attention_mask=bundle.encoder_attention_mask,
            timestep_r=bundle.timestep_r,
            encoder_hidden_states_2=bundle.encoder_hidden_states_2,
            encoder_attention_mask_2=bundle.encoder_attention_mask_2,
            image_embeds=bundle.image_embeds,
        )
        assert len(application.transformer.calls) == 1
        assert len(application.transformer.calls[0]) == 8


def test_dit_input_contract_15_shapes():
    with tempfile.TemporaryDirectory() as path:
        application = app.NeuronHunyuanVideoApplication(
            model_path=path,
            parallel=_parallel(),
            dtype=torch.float32,
            shape={},
            model_version="1.5",
        )

        application.transformer = SimpleNamespace(config=_config_15())
        contract = application.dit_input_contract()
    assert "image_embeds" in contract
    assert "encoder_hidden_states_2" in contract
    assert contract["image_embeds"]["shape"] == (1, 4, 7)


# --------------------------------------------------------------------------- #
# TeaCache probe sub-app: opt-out for callers that never run adaptive modes
# --------------------------------------------------------------------------- #
class _FakeBackbone:
    def __init__(self, *, model_path, config):
        self.model_path = model_path
        self.config = config


class _FakeProbe(_FakeBackbone):
    pass


class _FakeFusedProbe(_FakeBackbone):
    pass


def _app_with_probe_branch(monkeypatch, tmp_path, **kwargs):
    import sys
    import types
    import warnings

    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app, "_load_diffusers_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        app, "create_hunyuan_video_backbone_config",
        lambda **kw: SimpleNamespace(neuron_config=SimpleNamespace(world_size=1, tp_degree=1)),
    )
    backbone = types.ModuleType("difflet.backends.trainium.hunyuan_video.backbone")
    backbone.NeuronHunyuanVideoBackboneApplication = _FakeBackbone
    probe = types.ModuleType("difflet.backends.trainium.hunyuan_video.teacache_probe")
    probe.NeuronHunyuanVideoTeacacheProbeApplication = _FakeProbe
    probe.NeuronHunyuanVideoTeacacheProbeFusedApplication = _FakeFusedProbe
    monkeypatch.setitem(sys.modules, backbone.__name__, backbone)
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scheduler config is absent in tmp_path
        return app.NeuronHunyuanVideoApplication(
            model_path=str(tmp_path), parallel=_parallel(), dtype=torch.float32,
            shape={}, enable_vae_decoder=False, **kwargs,
        )


def test_probe_is_built_by_default_and_listed_as_a_component(monkeypatch, tmp_path):
    application = _app_with_probe_branch(monkeypatch, tmp_path)
    assert isinstance(application.transformer, _FakeBackbone)
    assert isinstance(application.teacache_probe, _FakeProbe)
    assert application.teacache_probe_fused is False
    assert [spec.name for spec in application.components()] == ["transformer", "teacache_probe"]


def test_probe_opt_out_builds_no_probe_component(monkeypatch, tmp_path):
    # The CLI's plain and probe-free runs opt out: no probe NEFF is ever
    # compiled or loaded for them (this used to be done by nulling the
    # attribute after construction).
    application = _app_with_probe_branch(monkeypatch, tmp_path, enable_teacache_probe=False)
    assert isinstance(application.transformer, _FakeBackbone)
    assert application.teacache_probe is None
    assert application.teacache_probe_fused is False
    assert [spec.name for spec in application.components()] == ["transformer"]


def test_fused_probe_still_selected_when_enabled(monkeypatch, tmp_path):
    application = _app_with_probe_branch(monkeypatch, tmp_path, teacache_fused=True)
    assert isinstance(application.teacache_probe, _FakeFusedProbe)
    assert application.teacache_probe_fused is True
