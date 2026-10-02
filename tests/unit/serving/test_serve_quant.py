"""--quant through `difflet serve`: options, profile, Wan adapter identity/kwargs."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.quant.spec import QuantSpec
from difflet.registry import resolve_model
from difflet.serving.errors import DiffletServingError
from difflet.serving.models import wan
from difflet.serving.options import build_serving_profile
from difflet.serving.types import ResolvedModelSource, ServingProfile

WAN = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def _build(model_id: str, model_type: str, **overrides):
    entry = resolve_model(model_id)
    kwargs = dict(
        model_id=model_id,
        model_type=model_type,
        entry=entry,
        output_modality="video" if model_type == "wan" else "image",
        output_mime_type="video/mp4" if model_type == "wan" else "image/png",
        default_fps=16 if model_type == "wan" else None,
        default_host_vae=True,
        revision=None,
        cache_dir="/c",
        tp_degree=4,
        cp_degree=1,
        cp_mode=None,
        cfg_parallel=None,
        sp_enabled=None,
        height=None,
        width=None,
        num_frames=None,
        host_vae=False,
        clip_placement=None,
        teacache_cadence=None,
        teacache_online_delta=None,
        teacache_speedup=None,
        teacache_calibration=None,
    )
    kwargs.update(overrides)
    return build_serving_profile(**kwargs)


def _ns(**overrides) -> argparse.Namespace:
    base = dict(
        model_id=WAN, revision=None, host="0.0.0.0", port=8091, api_key=None,
        tp_degree=4, cp_degree=None, cp_mode=None, cfg_parallel=None, sp_enabled=None,
        height=None, width=None, num_frames=None, shapes=None, cache_dir=None,
        quant=None, quant_granularity=None, quant_act=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_options_from_args_carries_quant_fields():
    from difflet.cli.serve import options_from_args

    options = options_from_args(_ns(quant="fp8", quant_granularity="channel", quant_act="none"))
    assert (options.quant, options.quant_granularity, options.quant_act) == ("fp8", "channel", "none")
    plain = options_from_args(_ns())
    assert plain.quant is None and plain.quant_granularity == "tensor" and plain.quant_act == "dynamic"


def test_profile_carries_the_spec_for_wan_and_rejects_other_models():
    profile = _build(WAN, "wan", quant="fp8", quant_granularity="channel", quant_act="none")
    assert profile.quant == QuantSpec(weight_granularity="channel", activation="none")
    assert _build(WAN, "wan").quant is None

    with pytest.raises(DiffletServingError, match="does not support --quant"):
        _build("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", "hunyuan_video_15",
               output_modality="video", quant="fp8")
    with pytest.raises(DiffletServingError, match="mutually exclusive"):
        _build(WAN, "wan", quant="fp8", teacache_speedup=1.5, teacache_calibration="c.json")
    with pytest.raises(DiffletServingError, match="invalid --quant"):
        _build(WAN, "wan", quant="fp8", quant_granularity="token")


def test_model_registry_threads_options_into_the_profile():
    from difflet.serving.model_registry import resolve_serving_model
    from difflet.serving.options import ServeOptions

    resolved = resolve_serving_model(ServeOptions(model_id=WAN, quant="fp8", quant_act="none"))
    assert resolved.profile.quant == QuantSpec(activation="none")
    assert resolve_serving_model(ServeOptions(model_id=WAN)).profile.quant is None


def _profile(tmp_path: Path, quant: QuantSpec | None) -> ServingProfile:
    return ServingProfile(
        model_id=WAN,
        model_type="wan",
        height=64,
        width=96,
        num_frames=5,
        parallel=DiffletParallelConfig(tp_degree=4),
        cache_dir=str(tmp_path / "cache"),
        dtype="bfloat16",
        output_modality="video",
        output_mime_type="video/mp4",
        output_fps=16,
        host_vae=True,
        quant=quant,
    )


def _source(tmp_path: Path) -> ResolvedModelSource:
    return ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id=WAN,
        requested_revision=None,
        pinned_model_path=str(tmp_path / "snapshots" / ("a" * 40)),
        resolved_source_id="a" * 40,
    )


def test_wan_application_kwargs_add_quant_only_when_set(tmp_path):
    bf16 = wan._application_kwargs(_profile(tmp_path, None))
    assert bf16 == wan._application_kwargs()
    assert "quant" not in bf16
    fp8 = wan._application_kwargs(_profile(tmp_path, QuantSpec()))
    assert fp8["quant"] == QuantSpec().to_dict()
    assert fp8["quant_cache_dir"] == str(tmp_path / "cache")
    assert {k: v for k, v in fp8.items() if k not in ("quant", "quant_cache_dir")} == bf16


def test_generation_identity_hashes_the_spec_but_not_the_cache_path(monkeypatch, tmp_path):
    monkeypatch.setattr(wan, "_torch_bfloat16", lambda: "bfloat16")
    source = _source(tmp_path)
    bf16 = wan._compile_spec(source, _profile(tmp_path, None))
    fp8 = wan._compile_spec(source, _profile(tmp_path, QuantSpec()))
    fp8_wo = wan._compile_spec(source, _profile(tmp_path, QuantSpec(activation="none")))
    assert len({bf16.identity.digest, fp8.identity.digest, fp8_wo.identity.digest}) == 3
    app_kwargs = json.loads(fp8.identity.canonical_cache_inputs_json)["cache_inputs"]["application_kwargs"]
    assert app_kwargs["quant"] == QuantSpec().to_dict()
    assert "quant_cache_dir" not in app_kwargs
    assert str(tmp_path) not in fp8.identity.canonical_cache_inputs_json.decode()
    bf16_kwargs = json.loads(bf16.identity.canonical_cache_inputs_json)["cache_inputs"]["application_kwargs"]
    assert "quant" not in bf16_kwargs and "quant_cache_dir" not in bf16_kwargs


def test_build_application_passes_quant_kwargs_and_rejects_tpu(monkeypatch, tmp_path):
    captured = {}

    class FakeApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("difflet.models.wan.application")
    fake.NeuronWanApplication = FakeApp
    monkeypatch.setitem(sys.modules, "difflet.models.wan.application", fake)
    monkeypatch.setattr(wan, "_torch_bfloat16", lambda: "bfloat16")
    monkeypatch.setattr(wan, "_backend_is_tpu", lambda: False)

    wan._build_application(_source(tmp_path), _profile(tmp_path, QuantSpec()))
    assert captured["quant"] == QuantSpec().to_dict()
    assert captured["quant_cache_dir"] == str(tmp_path / "cache")

    monkeypatch.setattr(wan, "_backend_is_tpu", lambda: True)
    with pytest.raises(ValueError, match="Trainium-only"):
        wan._build_application(_source(tmp_path), _profile(tmp_path, QuantSpec()))


def test_profile_carries_model_targets_for_every_wired_model():
    """The serving profile's spec uses the model's own target set (Review Focus 2/5):
    a FLUX profile must never carry Wan's ffn.* targets."""
    for model_id, model_type, modality in (
        ("black-forest-labs/FLUX.1-dev", "flux", "image"),
        ("Qwen/Qwen-Image", "qwen_image", "image"),
        ("hunyuanvideo-community/HunyuanVideo", "hunyuan_video", "video"),
        ("Lightricks/LTX-2", "ltx_2", "video"),
        (WAN, "wan", "video"),
    ):
        extra = {"output_modality": modality, "default_fps": 16 if modality == "video" else None}
        profile = _build(model_id, model_type, quant="fp8", quant_act="none", **extra)
        assert profile.quant is not None, model_id
        assert profile.quant.targets == QuantSpec.for_model(model_type).targets, model_id
        assert profile.quant.activation == "none"
        plain = _build(model_id, model_type, **extra)
        assert plain.quant is None



def test_flux_serving_application_kwargs_add_quant_only_when_set(tmp_path):
    from difflet.common.orchestrators.flux import quant_application_kwargs

    assert quant_application_kwargs(_profile(tmp_path, None)) is None
    kwargs = quant_application_kwargs(_profile(tmp_path, QuantSpec.for_model("flux", activation="none")))
    assert kwargs == {"quant": QuantSpec.for_model("flux", activation="none").to_dict(),
                      "quant_cache_dir": str(tmp_path / "cache")}



def test_qwen_serving_quant_kwargs_add_quant_only_when_set(tmp_path):
    from difflet.serving.orchestrators.qwen_image import _quant_kwargs

    assert _quant_kwargs(_profile(tmp_path, None)) == {}
    spec = QuantSpec.for_model("qwen_image", activation="none")
    assert _quant_kwargs(_profile(tmp_path, spec)) == {"quant": spec.to_dict(), "quant_cache_dir": str(tmp_path / "cache")}
