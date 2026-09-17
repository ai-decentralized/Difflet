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


# ------------------------------------------------------ probe-free TeaCache

def _load_pipeline_kwargs(monkeypatch, **arg_overrides):
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **kw: "/fake/path")
    monkeypatch.setattr("difflet.pipeline.compile_cache.has_valid_manifest",
                        lambda *a, **kw: True)
    monkeypatch.setattr("difflet.pipeline.compile_cache.cache_path",
                        lambda *a, **kw: "/fake/compiled")
    captured = {}
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained",
        classmethod(lambda cls, *a, **kw: captured.update(kw) or object()),
    )
    LTX2Orchestrator(_ltx2_args(**arg_overrides))._load_pipeline()
    return captured


def test_load_pipeline_threads_probe_free_teacache_kwargs(monkeypatch):
    # --teacache-cadence / --teacache-online-delta were never forwarded (only
    # the host-pipeline switches were); they must reach the application as
    # runtime-only kwargs next to the existing ones.
    kw = _load_pipeline_kwargs(monkeypatch, teacache_cadence=2)
    assert kw["application_kwargs"] == {
        "enable_host_pipeline": True,
        "enable_decode_components": True,
        "teacache_cadence": 2,
    }
    kw = _load_pipeline_kwargs(monkeypatch, teacache_online_delta=0.6)
    assert kw["application_kwargs"]["teacache_online_delta_alpha"] == 0.6
    assert "teacache_cadence" not in kw["application_kwargs"]
    kw = _load_pipeline_kwargs(monkeypatch)
    assert set(kw["application_kwargs"]) == {"enable_host_pipeline", "enable_decode_components"}


def test_probe_free_teacache_kwargs_do_not_change_ltx2_cache_key():
    # The warm tp4 artifact must hit with --teacache-cadence: the kwargs are in
    # _RUNTIME_ONLY_APP_KWARGS, so the CacheSpec key is byte-identical.
    from difflet.pipeline.compile_cache import CacheSpec, cache_key
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    def spec(app_kwargs):
        return CacheSpec(
            model_id="Lightricks/LTX-2", model_path="/fake/path", model_name="ltx_2",
            parallel=DiffletParallelConfig(tp_degree=4), dtype="bfloat16",
            height=256, width=384, num_frames=121, application_kwargs=app_kwargs,
        )

    plain = spec({"enable_host_pipeline": True, "enable_decode_components": True})
    cadence = spec({"enable_host_pipeline": True, "enable_decode_components": True,
                    "teacache_cadence": 2, "teacache_online_delta_alpha": 0.6})
    assert cache_key(plain) == cache_key(cadence) == cache_key(spec(None))
    # the calibrated-adaptive calibration path is runtime-only too (no probe
    # NEFF for LTX-2): the warm tp4 artifact must hit with --teacache-speedup
    adaptive = spec({"enable_host_pipeline": True, "enable_decode_components": True,
                     "teacache_calibration_path": "/c.json"})
    assert cache_key(plain) == cache_key(adaptive)


def _write_ltx2_calibration(path, *, target=1.5, shape_label="512x768x121"):
    import json

    from difflet.pipeline.teacache import CALIBRATION_SCHEMA
    path.write_text(json.dumps({
        "schema": CALIBRATION_SCHEMA, "model": "ltx_2", "shape_label": shape_label,
        "num_steps": 40, "poly_coef": [0.0, 1.0], "threshold": 0.1,
        "target_speedup": target,
    }))
    return str(path)


def test_load_pipeline_threads_adaptive_calibration_as_runtime_only(monkeypatch, tmp_path):
    # Calibrated-adaptive TeaCache on LTX-2 (host CPU block-0 signal, no probe
    # NEFF): only the calibration path reaches the application. teacache_speedup
    # must NOT -- it would flip the cache key to a probe identity that never
    # exists for LTX-2 and miss the warm tp4 artifact.
    calib = _write_ltx2_calibration(tmp_path / "ltx2.json")
    kw = _load_pipeline_kwargs(monkeypatch, teacache_speedup=1.5, teacache_calibration=calib)
    assert kw["application_kwargs"] == {
        "enable_host_pipeline": True,
        "enable_decode_components": True,
        "teacache_calibration_path": calib,
    }
    assert "teacache_speedup" not in kw and "teacache_calibration_path" not in kw


@pytest.mark.parametrize("bad", [
    dict(target=1.5, speedup=2.0, shape_label="512x768x121", match="lower than requested"),
    dict(target=1.5, speedup=1.5, shape_label="256x384x121", match="shape mismatch"),
])
def test_load_pipeline_validates_the_calibration_like_the_probe_pipelines(
    monkeypatch, tmp_path, bad,
):
    calib = _write_ltx2_calibration(tmp_path / "ltx2.json", target=bad["target"],
                                    shape_label=bad["shape_label"])
    with pytest.raises(ValueError, match=bad["match"]):
        _load_pipeline_kwargs(monkeypatch, teacache_speedup=bad["speedup"],
                              teacache_calibration=calib)
