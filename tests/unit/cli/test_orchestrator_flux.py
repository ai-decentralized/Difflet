from __future__ import annotations

import argparse
import types

from difflet.cli.orchestrators.flux import FluxOrchestrator


def _flux_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="black-forest-labs/FLUX.1-dev", tp_degree=4, cp_degree=1,
        cp_mode="gather_kv", height=1024, width=1024, num_frames=None,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.png",
        steps=28, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False, cfg_parallel=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
        cache_profile_file=None, cache_profile_qualification_file=None,
        taef1=False, taef1_path=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _FakeImage:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path


def test_generate_success_saves_image(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/path",
    )
    monkeypatch.setattr(
        "difflet.pipeline.compile_cache.has_valid_manifest",
        lambda *a, **kw: True,
    )
    monkeypatch.setattr(
        "difflet.pipeline.compile_cache.cache_path",
        lambda *a, **kw: "/fake/compiled",
    )

    fake_image = _FakeImage()
    captured = {}

    class FakePipe:
        def __call__(self, **kw):
            captured.update(kw)
            return types.SimpleNamespace(images=[fake_image])

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
        classmethod(lambda cls, *a, **kw: FakePipe()),
    )

    out = tmp_path / "sub" / "img.png"
    FluxOrchestrator(_flux_args(output=str(out), steps=None, guidance_scale=None,
                                height=None, width=None)).generate()
    assert fake_image.saved_to == str(out)
    # Defaults applied when steps/guidance/shape are None.
    assert captured["num_inference_steps"] == 28
    assert captured["guidance_scale"] == 3.5
    assert captured["height"] == 1024
    assert captured["width"] == 1024


def test_teacache_kwargs_speedup_and_app_kwargs():
    orch = FluxOrchestrator(_flux_args(
        teacache_speedup=1.5, teacache_calibration="/cal.json",
        teacache_cadence=3, teacache_online_delta=0.4,
    ))
    kw = orch._model_kwargs()
    assert kw["teacache_speedup"] == 1.5
    assert kw["teacache_calibration_path"] == "/cal.json"
    assert kw["application_kwargs"]["teacache_cadence"] == 3
    assert kw["application_kwargs"]["teacache_online_delta_alpha"] == 0.4


def test_teacache_kwargs_empty_when_unset():
    assert FluxOrchestrator(_flux_args())._model_kwargs() == {}


def test_model_kwargs_tolerate_compile_namespace_without_teacache_fields():
    args = _flux_args()
    del args.teacache_speedup
    del args.teacache_calibration

    assert FluxOrchestrator(args)._model_kwargs() == {}


def test_taef1_kwargs_reach_compile_and_application_cache_identity():
    kwargs = FluxOrchestrator(
        _flux_args(taef1=True, taef1_path="madebyollin/taef1")
    )._model_kwargs()
    assert kwargs == {
        "application_kwargs": {
            "taef1": True,
            "taef1_path": "madebyollin/taef1",
        }
    }


def test_qualified_cache_profile_is_forwarded_to_application():
    kwargs = FluxOrchestrator(
        _flux_args(
            cache_profile_file="/profile.json",
            cache_profile_qualification_file="/qualification.json",
        )
    )._model_kwargs()
    assert kwargs["application_kwargs"] == {
        "cache_profile_file": "/profile.json",
        "cache_profile_qualification_file": "/qualification.json",
    }


def test_cli_profile_preflight_uses_executable_identity_without_policy_paths(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *args, **kwargs: "/fake/model",
    )
    monkeypatch.setattr(
        "difflet.cli.prewarm.prewarm_neuron_runtime",
        lambda *args, **kwargs: None,
    )

    def cache_path(cache_dir, spec):
        del cache_dir
        captured["cache_inputs"] = spec.cache_inputs()
        return "/fake/compiled"

    monkeypatch.setattr("difflet.pipeline.compile_cache.cache_path", cache_path)
    monkeypatch.setattr(
        "difflet.pipeline.compile_cache.has_valid_manifest",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
        classmethod(lambda cls, *args, **kwargs: object()),
    )

    FluxOrchestrator(
        _flux_args(
            cache_profile_file="/profile.json",
            cache_profile_qualification_file="/qualification.json",
        )
    )._load_pipeline()

    assert captured["cache_inputs"]["application_kwargs"] == {}


def test_parallel_uses_explicit_tp_and_cp_mode():
    parallel = FluxOrchestrator(_flux_args(tp_degree=8, cp_degree=1,
                                           cp_mode="gather_kv"))._parallel()
    assert parallel.tp_degree == 8
    assert parallel.cp_degree == 1


def test_parallel_falls_back_to_registry_default():
    # tp_degree None -> registry default is used (non-None positive int).
    parallel = FluxOrchestrator(_flux_args(tp_degree=None))._parallel()
    assert parallel.tp_degree >= 1
