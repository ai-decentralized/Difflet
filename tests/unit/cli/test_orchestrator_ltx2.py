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


def test_generate_success_uses_default_steps_and_guidance(monkeypatch, tmp_path):
    import torch

    from difflet.cli.orchestrators import ltx_2 as ltx_mod

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
            return types.SimpleNamespace(frames=torch.zeros(1, 2, 3, 4, 4))

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
        classmethod(lambda cls, *a, **kw: FakePipe()),
    )
    monkeypatch.setattr(ltx_mod, "_save_video", lambda frames, path: True)

    out = tmp_path / "vid.mp4"
    LTX2Orchestrator(_ltx2_args(output=str(out), steps=None,
                                guidance_scale=None)).generate()
    assert captured["num_inference_steps"] == 40
    assert captured["guidance_scale"] == 3.5


def test_parallel_threads_cfg_flag():
    parallel = LTX2Orchestrator(_ltx2_args(cfg_parallel=True, tp_degree=4,
                                           cp_degree=1))._parallel()
    assert parallel.cfg_parallel_enabled is True
    assert parallel.tp_degree == 4


# ----------------------------------------------------------------- _save_video

def test_save_video_converts_bfchw_unit_range_frames(monkeypatch, tmp_path):
    # LTX-2 frames arrive as (B, F, C, H, W) in [0, 1] (diffusers
    # VideoProcessor.postprocess_video) — no (x+1)/2 denorm, fps 24.
    import numpy as np
    import torch

    from difflet.cli.orchestrators import ltx_2 as ltx_mod

    exported = {}

    def fake_export(frames, path, fps):
        exported.update(frames=frames, path=path, fps=fps)

    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)
    frames = torch.full((1, 2, 3, 4, 4), 0.5)
    ok = ltx_mod._save_video(frames, str(tmp_path / "v.mp4"))
    assert ok is True
    assert exported["fps"] == 24
    assert len(exported["frames"]) == 2
    f0 = exported["frames"][0]
    # export_to_video multiplies ndarray frames by 255 itself — passing uint8
    # wraps pixels to 256-v (color inversion). The contract is float32 [0, 1].
    assert f0.shape == (4, 4, 3) and f0.dtype == np.float32
    assert abs(float(f0[0, 0, 0]) - 0.5) < 1e-6


def test_save_video_export_failure_returns_false(monkeypatch, tmp_path):
    import torch

    from difflet.cli.orchestrators import ltx_2 as ltx_mod

    def boom(*a, **kw):
        raise RuntimeError("no codec")

    monkeypatch.setattr("diffusers.utils.export_to_video", boom)
    ok = ltx_mod._save_video(torch.zeros(1, 2, 3, 4, 4), str(tmp_path / "v.mp4"))
    assert ok is False


def _generate_with_fake_pipe(monkeypatch, tmp_path, output_name):
    import torch

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **kw: "/fake/path")
    monkeypatch.setattr("difflet.pipeline.compile_cache.has_valid_manifest",
                        lambda *a, **kw: True)
    monkeypatch.setattr("difflet.pipeline.compile_cache.cache_path",
                        lambda *a, **kw: "/fake/compiled")

    class FakePipe:
        def __call__(self, **kw):
            return types.SimpleNamespace(frames=torch.zeros(1, 2, 3, 4, 4))

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
        classmethod(lambda cls, *a, **kw: FakePipe()),
    )
    out = tmp_path / output_name
    LTX2Orchestrator(_ltx2_args(output=str(out))).generate()
    return out


def test_generate_mp4_branch_skips_pt_fallback(monkeypatch, tmp_path):
    from difflet.cli.orchestrators import ltx_2 as ltx_mod
    monkeypatch.setattr(ltx_mod, "_save_video", lambda frames, path: True)
    out = _generate_with_fake_pipe(monkeypatch, tmp_path, "vid.mp4")
    assert not (tmp_path / "vid.pt").exists()


def test_generate_pt_fallback_when_export_fails(monkeypatch, tmp_path):
    from difflet.cli.orchestrators import ltx_2 as ltx_mod
    monkeypatch.setattr(ltx_mod, "_save_video", lambda frames, path: False)
    out = _generate_with_fake_pipe(monkeypatch, tmp_path, "vid.mp4")
    assert (tmp_path / "vid.pt").exists()
