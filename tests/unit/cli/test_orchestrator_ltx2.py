from __future__ import annotations

import argparse
import types

import pytest

from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator


def _ltx2_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Lightricks/LTX-2", tp_degree=4, cp_degree=1,
        cp_mode="gather_kv", height=512, width=768, num_frames=121,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.mp4",
        steps=40, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False, cfg_parallel=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_download_requests_remote(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, revision=None, local_files_only=False, allow_patterns=None:
            calls.append((model_id, local_files_only)),
    )
    LTX2Orchestrator(_ltx2_args()).download()
    assert calls == [("Lightricks/LTX-2", False)]


def test_generate_exits_when_weights_missing(monkeypatch, capsys):
    def fake_resolve(model_id, *, revision=None, local_files_only=False, allow_patterns=None):
        if local_files_only:
            raise OSError("not cached")
        return "/fake/path"

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", fake_resolve)
    with pytest.raises(SystemExit) as exc:
        LTX2Orchestrator(_ltx2_args()).generate()
    assert exc.value.code == 1
    assert "difflet download --model-id Lightricks/LTX-2" in capsys.readouterr().err


def test_generate_success_saves_pt(monkeypatch, tmp_path):
    import torch

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

    captured = {}

    class FakePipe:
        def __call__(self, **kw):
            captured.update(kw)
            return types.SimpleNamespace(frames=torch.zeros(1, 3, 2, 4, 4))

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
        classmethod(lambda cls, *a, **kw: FakePipe()),
    )

    out = tmp_path / "vid.mp4"
    LTX2Orchestrator(_ltx2_args(output=str(out), steps=None,
                                guidance_scale=None)).generate()
    # frames saved as .pt sibling.
    assert (tmp_path / "vid.pt").exists()
    assert captured["num_inference_steps"] == 40
    assert captured["guidance_scale"] == 3.5


def test_parallel_threads_cfg_flag():
    parallel = LTX2Orchestrator(_ltx2_args(cfg_parallel=True, tp_degree=4,
                                           cp_degree=1))._parallel()
    assert parallel.cfg_parallel_enabled is True
    assert parallel.tp_degree == 4
