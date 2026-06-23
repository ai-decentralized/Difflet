from __future__ import annotations
import argparse
import pytest


def _ltx2_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Lightricks/LTX-2", tp_degree=4, cp_degree=1,
        height=512, width=768, num_frames=121,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.mp4",
        steps=40, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_generate_exits_when_no_compiled_cache(monkeypatch, capsys):
    from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator

    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/path",
    )
    monkeypatch.setattr(
        "difflet.pipeline.compile_cache.has_valid_manifest",
        lambda *a, **kw: False,
    )
    with pytest.raises(SystemExit) as exc:
        LTX2Orchestrator(_ltx2_args()).generate()
    assert exc.value.code == 1
    assert "difflet compile --model-id Lightricks/LTX-2" in capsys.readouterr().err


def test_compile_calls_precompile(monkeypatch):
    from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator

    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/path",
    )
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.precompile",
        classmethod(lambda cls, model_id, **kw: calls.append(model_id)),
    )
    LTX2Orchestrator(_ltx2_args()).compile()
    assert calls == ["Lightricks/LTX-2"]
