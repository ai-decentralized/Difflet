"""Canonical subset keys for HV 1.0 probes; full canonical keys for HV 1.5."""

from __future__ import annotations

import json

import pytest
import torch

from difflet.backends.trainium.core.application_base import (
    NeuronApplicationBase,
    checkpoint_missing_weights,
)
from difflet.backends.trainium.core.config import NeuronConfig


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    torch.manual_seed(0)


def _aliased_names(wrapper):
    instance = wrapper.get_model_instance()
    instance.load_module()
    module, aliases = instance.get(0)
    return (
        module,
        aliases,
        {name for name, param in module.named_parameters() if any(param is a for a in aliases)},
    )


# =========================================================== HunyuanVideo 1.0


def _hv1_config(tmp_path):
    from difflet.models.hunyuan_video.application import create_hunyuan_video_backbone_config

    transformer_dir = tmp_path / "hv" / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideoTransformer3DModel",
                "in_channels": 2,
                "out_channels": 2,
                "num_attention_heads": 2,
                "attention_head_dim": 6,
                "num_layers": 1,
                "num_single_layers": 1,
                "num_refiner_layers": 1,
                "mlp_ratio": 2.0,
                "patch_size": 2,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "guidance_embeds": True,
                "text_embed_dim": 6,
                "pooled_projection_dim": 5,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 2],
            }
        )
    )
    return create_hunyuan_video_backbone_config(
        model_path=str(tmp_path / "hv"),
        world_size=1,
        tp_degree=1,
        dtype=torch.float32,
        height=32,
        width=32,
        num_frames=5,
        text_seq_len=8,
    )


def _hv1_hf_state_dict(backbone, config) -> dict:
    """Diffusers-shaped checkpoint: single-block proj_out still fused."""
    sd = {k: v.clone() for k, v in backbone.state_dict().items()}
    for i in range(config.num_single_layers):
        attn_w = sd.pop(f"single_transformer_blocks.{i}.proj_out_attn.weight")
        attn_b = sd.pop(f"single_transformer_blocks.{i}.proj_out_attn.bias")
        mlp_w = sd.pop(f"single_transformer_blocks.{i}.proj_out_mlp.weight")
        sd[f"single_transformer_blocks.{i}.proj_out.weight"] = torch.cat([attn_w, mlp_w], dim=1)
        sd[f"single_transformer_blocks.{i}.proj_out.bias"] = attn_b
    return sd


def _hv1_modules():
    from difflet.backends.trainium.hunyuan_video import teacache_probe as probe_app
    from difflet.backends.trainium.hunyuan_video import teacache_probe_model as probe_model
    from difflet.backends.trainium.hunyuan_video.backbone import (
        NeuronHunyuanVideoBackboneApplication,
    )
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    return (
        probe_app,
        probe_model,
        NeuronHunyuanVideoBackboneApplication,
        HunyuanVideoTransformer3DModel,
    )


def test_hv1_probe_models_construct_only_the_prefix():
    probe_app, probe_model, backbone_app, transformer_cls = _hv1_modules()
    assert backbone_app._model_cls is transformer_cls
    assert not issubclass(probe_model.HunyuanVideoTeacacheProbeFusedModel, transformer_cls)
    assert not issubclass(probe_model.HunyuanVideoTeacacheProbeModel, transformer_cls)
    assert probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication._model_cls is (
        probe_model.HunyuanVideoTeacacheProbeFusedModel
    )
    assert probe_app.NeuronHunyuanVideoTeacacheProbeApplication._model_cls is (
        probe_model.HunyuanVideoTeacacheProbeModel
    )


def test_hv1_state_declarations():
    probe_app, _, backbone_app, _ = _hv1_modules()
    assert probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication.state_tensor_names == {
        "prev_mod"
    }
    # The v1 probe takes prev_mod_input as an INPUT, so it adds no state.
    assert probe_app.NeuronHunyuanVideoTeacacheProbeApplication.state_tensor_names == frozenset()
    assert backbone_app.state_tensor_names == frozenset()
    assert NeuronApplicationBase.state_tensor_names == frozenset()


def test_hv1_parameter_names_are_canonical_subset_plus_state(tmp_path):
    probe_app, probe_model, _, transformer_cls = _hv1_modules()
    config = _hv1_config(tmp_path)
    backbone = transformer_cls(config)
    fused = probe_model.HunyuanVideoTeacacheProbeFusedModel(
        config,
        seq_len=probe_app._probe_seq_len(config),
        inner_dim=int(config.inner_dim),
        batch_size=1,
    )
    plain = probe_model.HunyuanVideoTeacacheProbeModel(config)

    backbone_names = set(backbone.state_dict())
    fused_state = probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication.state_tensor_names
    expected = {
        k
        for k in backbone_names
        if k.startswith(probe_app._HunyuanVideoPrefixApplication.weight_prefixes)
    }
    assert expected < backbone_names
    assert set(fused.state_dict()) - fused_state == expected
    assert set(fused.state_dict()) - backbone_names == fused_state
    assert set(plain.state_dict()) == expected
    assert not any(k.startswith("model.") for k in fused.state_dict())


def test_hv1_converters_filter_to_the_required_subset(tmp_path):
    probe_app, probe_model, backbone_app, transformer_cls = _hv1_modules()
    fused_app = probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication
    plain_app = probe_app.NeuronHunyuanVideoTeacacheProbeApplication

    config = _hv1_config(tmp_path)
    backbone = transformer_cls(config)
    hf = _hv1_hf_state_dict(backbone, config)
    backbone_keys = set(backbone_app.convert_hf_to_neuron_state_dict(dict(hf), config))
    expected = set(probe_model.HunyuanVideoTeacacheProbeModel(config).state_dict())
    assert expected < backbone_keys
    assert set(fused_app.convert_hf_to_neuron_state_dict(dict(hf), config)) == expected
    assert set(plain_app.convert_hf_to_neuron_state_dict(dict(hf), config)) == expected
    assert "single_transformer_blocks.0.proj_out_attn.weight" in backbone_keys
    assert not any(k.startswith("model.") for k in backbone_keys)


def test_hv1_backbone_checkpoint_serves_every_probe_weight(tmp_path):
    probe_app, probe_model, backbone_app, transformer_cls = _hv1_modules()
    config = _hv1_config(tmp_path)
    backbone = transformer_cls(config)
    converted = backbone_app.convert_hf_to_neuron_state_dict(
        _hv1_hf_state_dict(backbone, config), config
    )
    fused = probe_model.HunyuanVideoTeacacheProbeFusedModel(
        config,
        seq_len=probe_app._probe_seq_len(config),
        inner_dim=int(config.inner_dim),
        batch_size=1,
    )
    plain = probe_model.HunyuanVideoTeacacheProbeModel(config)
    state = probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication.state_tensor_names

    assert checkpoint_missing_weights(fused, converted, state) == set()
    assert checkpoint_missing_weights(fused, converted) == {"prev_mod"}
    assert checkpoint_missing_weights(plain, converted) == set()
    # The wrapper-era layout ("model." prefix) is reported as missing.
    nested = {f"model.{k}": v for k, v in converted.items()}
    assert "x_embedder.proj.weight" in checkpoint_missing_weights(fused, nested, state)


def test_hv1_aliased_tensors_are_exactly_the_declared_state(tmp_path):
    probe_app, probe_model, _, transformer_cls = _hv1_modules()
    config = _hv1_config(tmp_path)

    fused_wrapper = probe_app.ModelWrapperHunyuanVideoTeacacheProbeFused(
        config,
        probe_model.HunyuanVideoTeacacheProbeFusedModel,
        tag="probe",
        compiler_args="",
        priority_model_idx=0,
    )
    module, aliases, aliased = _aliased_names(fused_wrapper)
    assert isinstance(module, probe_model.HunyuanVideoTeacachePrefix)
    assert aliased == probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication.state_tensor_names
    assert set(aliases.values()) == {1}

    plain_wrapper = probe_app.ModelWrapperHunyuanVideoTeacacheProbe(
        config,
        probe_model.HunyuanVideoTeacacheProbeModel,
        tag="probe",
        compiler_args="",
        priority_model_idx=0,
    )
    module, aliases, aliased = _aliased_names(plain_wrapper)
    assert isinstance(module, probe_model.HunyuanVideoTeacachePrefix)
    assert aliases == {} and aliased == set()


# =========================================================== HunyuanVideo 1.5

def _hv15_config(tmp_path):
    from difflet.backends.trainium.hunyuan_video.backbone15 import (
        HunyuanVideo15BackboneInferenceConfig,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    transformer_dir = tmp_path / "hv15" / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Transformer3DModel",
                "in_channels": 65,
                "out_channels": 32,
                "num_attention_heads": 4,
                "attention_head_dim": 8,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "mlp_ratio": 2.0,
                "patch_size": 1,
                "patch_size_t": 1,
                "qk_norm": "rms_norm",
                "text_embed_dim": 12,
                "text_embed_2_dim": 10,
                "image_embed_dim": 6,
                "rope_theta": 256.0,
                "rope_axes_dim": [2, 2, 4],
                "target_size": 640,
                "task_type": "t2v",
            }
        )
    )
    return HunyuanVideo15BackboneInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1, tp_degree=1, world_size=1, torch_dtype=torch.float32, skip_sharding=True
        ),
        load_config=load_diffusers_config(transformer_dir),
        height=32,
        width=48,
        num_frames=5,
        text_seq_len=7,
        text_seq_len_2=3,
        image_seq_len=4,
    )


def _hv15_modules():
    from difflet.backends.trainium.hunyuan_video import teacache_probe15 as probe_mod
    from difflet.backends.trainium.hunyuan_video.backbone15 import (
        NeuronHunyuanVideo15BackboneApplication,
        _HunyuanVideo15TraceModule,
    )

    return probe_mod, NeuronHunyuanVideo15BackboneApplication, _HunyuanVideo15TraceModule


def _hv15_build(probe_mod, trace_cls, config):
    backbone = trace_cls(config).eval()
    probe = probe_mod.HunyuanVideo15TeacacheProbeFusedModel(
        config,
        seq_len=probe_mod._probe_seq_len(config),
        inner_dim=int(config.inner_dim),
        batch_size=1,
    ).eval()
    return backbone, probe


def test_hv15_probe_model_is_the_backbone_trace_module():
    probe_mod, _, trace_cls = _hv15_modules()
    assert issubclass(probe_mod.HunyuanVideo15TeacacheProbeFusedModel, trace_cls)
    assert probe_mod.NeuronHunyuanVideo15TeacacheProbeFusedApplication.state_tensor_names == {
        "prev_mod"
    }


def test_hv15_parameter_names_equal_backbone_names_plus_declared_state(tmp_path):
    probe_mod, _, trace_cls = _hv15_modules()
    config = _hv15_config(tmp_path)
    backbone, probe = _hv15_build(probe_mod, trace_cls, config)
    state = probe_mod.NeuronHunyuanVideo15TeacacheProbeFusedApplication.state_tensor_names

    backbone_names = set(backbone.state_dict())
    assert set(probe.state_dict()) - state == backbone_names
    assert set(probe.state_dict()) - backbone_names == state
    assert all(k.startswith("transformer.") for k in backbone_names)
    assert not any(k.startswith("trace_module.") for k in probe.state_dict())


def test_hv15_converter_is_the_backbone_converter_and_serves_every_weight(tmp_path):
    probe_mod, backbone_app, trace_cls = _hv15_modules()
    probe_app = probe_mod.NeuronHunyuanVideo15TeacacheProbeFusedApplication
    assert probe_app.convert_hf_to_neuron_state_dict is backbone_app.convert_hf_to_neuron_state_dict

    config = _hv15_config(tmp_path)
    backbone, probe = _hv15_build(probe_mod, trace_cls, config)
    prefix = "transformer."
    hf = {k[len(prefix):]: v for k, v in backbone.state_dict().items()}
    converted = backbone_app.convert_hf_to_neuron_state_dict(dict(hf), config)
    assert set(probe_app.convert_hf_to_neuron_state_dict(dict(hf), config)) == set(converted)

    assert checkpoint_missing_weights(probe, converted, probe_app.state_tensor_names) == set()
    assert checkpoint_missing_weights(probe, converted) == {"prev_mod"}
    nested = {f"trace_module.{k}": v for k, v in converted.items()}
    assert "transformer.x_embedder.proj.weight" in checkpoint_missing_weights(
        probe, nested, probe_app.state_tensor_names
    )


def test_hv15_aliased_tensors_and_forward_parity(tmp_path):
    probe_mod, _, trace_cls = _hv15_modules()
    config = _hv15_config(tmp_path)
    wrapper = probe_mod.ModelWrapperHunyuanVideo15TeacacheProbeFused(
        config, probe_mod.HunyuanVideo15TeacacheProbeFusedModel, tag="probe",
        compiler_args="", priority_model_idx=0,
    )
    module, aliases, aliased = _aliased_names(wrapper)
    assert isinstance(module, trace_cls)
    assert aliased == probe_mod.NeuronHunyuanVideo15TeacacheProbeFusedApplication.state_tensor_names
    assert set(aliases.values()) == {1}

    backbone, probe = _hv15_build(probe_mod, trace_cls, config)
    missing, unexpected = probe.load_state_dict(backbone.state_dict(), strict=False)
    assert set(missing) == {"prev_mod"} and unexpected == []
    x = torch.randn(
        1, config.in_channels, config.latent_frames, config.latent_height, config.latent_width
    )
    t = torch.ones(1)
    with torch.no_grad():
        rel_l1, mod_input = probe(x, t, t)
        tt = backbone.transformer
        temb = tt.time_embed(t, timestep_r=None)
        expected, *_ = tt.transformer_blocks[0].norm1(tt.x_embedder(x), emb=temb)
    assert torch.equal(mod_input, expected)
    assert tuple(mod_input.shape) == tuple(probe.prev_mod.shape)
    assert torch.isfinite(rel_l1) and rel_l1.item() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("guidance_embeds", [True, False])
def test_hv1_prefix_fused_l2_matches_reference_across_steps(tmp_path, dtype, guidance_embeds):
    probe_app, probe_model, _, transformer_cls = _hv1_modules()
    config = _hv1_config(tmp_path)
    config.guidance_embeds = guidance_embeds
    reference = transformer_cls(config).to(dtype).eval()
    probe = (
        probe_model.HunyuanVideoTeacacheProbeFusedModel(
            config, seq_len=probe_app._probe_seq_len(config), inner_dim=config.inner_dim
        )
        .to(dtype)
        .eval()
    )
    subset = (
        probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication.convert_hf_to_neuron_state_dict(
            reference.state_dict(), config
        )
    )
    missing, unexpected = probe.load_state_dict(subset, strict=False)
    assert missing == ["prev_mod"] and unexpected == []
    inputs = list(
        probe_app.ModelWrapperHunyuanVideoTeacacheProbeFused(config, type(probe)).input_generator()[
            0
        ]
    )
    inputs = [x.to(dtype) if x.is_floating_point() else x for x in inputs]
    previous = torch.zeros_like(probe.prev_mod)
    with torch.no_grad():
        for step in range(3):
            inputs[0] = inputs[0] + 0.1
            inputs[1] = torch.full_like(inputs[1], step + 1)
            expected = reference.teacache_mod_input(*inputs)
            delta, current = probe(*inputs)
            torch.testing.assert_close(current, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                delta,
                torch.linalg.vector_norm((expected - previous).reshape(-1)),
                rtol=0,
                atol=0,
            )
            # CPU emulates the wrapper's verified device alias update.
            probe.prev_mod.copy_(current)
            previous = expected


def test_hv1_selective_checkpoint_load_and_cache_identity(tmp_path):
    from safetensors.torch import save_file
    from difflet.pipeline.difflet_pipeline import _cache_application_kwargs

    probe_app, _, _, transformer_cls = _hv1_modules()
    config = _hv1_config(tmp_path)
    reference = transformer_cls(config)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = _hv1_hf_state_dict(reference, config)
    save_file(state, str(checkpoint / "model.safetensors"))
    cls = probe_app.NeuronHunyuanVideoTeacacheProbeFusedApplication
    loaded = cls.get_state_dict(str(checkpoint), config)
    assert set(loaded) == {k for k in state if k.startswith(cls.weight_prefixes)}
    assert set(loaded) < set(state)
    assert "prev_mod" not in loaded
    cache = _cache_application_kwargs({"teacache_fused": True}, model_name="hunyuan_video")
    assert cache["teacache_probe_layout"] == cls.shared_weights_layout
    assert "teacache_probe_enabled" not in _cache_application_kwargs(
        {}, model_name="hunyuan_video"
    )


def test_hv1_mask_fix_invalidates_backbone_cache_without_probe():
    from dataclasses import replace

    from difflet.pipeline.compile_cache import CacheSpec, cache_key
    from difflet.pipeline.difflet_pipeline import _cache_application_kwargs
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    old = CacheSpec(
        model_id="hunyuanvideo-community/HunyuanVideo",
        model_path="/unused/source",
        model_name="hunyuan_video",
        parallel=DiffletParallelConfig(tp_degree=4),
        dtype=torch.bfloat16,
        height=320,
        width=512,
        num_frames=121,
    )
    updated = replace(old, application_kwargs=_cache_application_kwargs(
        None, model_name="hunyuan_video"
    ))
    assert cache_key(updated) != cache_key(old)
    assert cache_key(updated) == cache_key(replace(old, application_kwargs=(
        _cache_application_kwargs({}, model_name="hunyuan_video")
    )))
