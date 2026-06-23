from __future__ import annotations
import argparse
import pytest


def _flux_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="black-forest-labs/FLUX.1-dev", tp_degree=4, cp_degree=1,
        height=1024, width=1024, num_frames=None,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.png",
        steps=28, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_generate_exits_when_weights_missing(monkeypatch, capsys):
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    def fake_resolve(model_id, *, revision=None, local_files_only=False, allow_patterns=None):
        if local_files_only:
            raise OSError("not cached")
        return "/fake/path"

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", fake_resolve)

    with pytest.raises(SystemExit) as exc:
        FluxOrchestrator(_flux_args()).generate()
    assert exc.value.code == 1
    assert "difflet download --model-id black-forest-labs/FLUX.1-dev" in capsys.readouterr().err


def test_generate_exits_when_no_compiled_cache(monkeypatch, capsys):
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/path",
    )
    monkeypatch.setattr(
        "difflet.pipeline.compile_cache.has_valid_manifest",
        lambda *a, **kw: False,
    )

    with pytest.raises(SystemExit) as exc:
        FluxOrchestrator(_flux_args()).generate()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "difflet compile --model-id black-forest-labs/FLUX.1-dev" in err


def test_compile_calls_precompile(monkeypatch):
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/path",
    )
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.precompile",
        classmethod(lambda cls, model_id, **kw: calls.append((model_id, kw))),
    )

    FluxOrchestrator(_flux_args()).compile()
    assert len(calls) == 1
    model_id, kw = calls[0]
    assert model_id == "black-forest-labs/FLUX.1-dev"
    assert kw["model_type"] == "flux"


def test_download_calls_resolve_model_path_with_remote(monkeypatch):
    from difflet.cli.orchestrators.flux import FluxOrchestrator

    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, local_files_only, **kw: calls.append(local_files_only),
    )
    FluxOrchestrator(_flux_args()).download()
    assert calls == [False]
