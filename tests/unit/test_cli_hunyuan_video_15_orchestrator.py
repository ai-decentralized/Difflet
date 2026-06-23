from __future__ import annotations
import argparse
import pytest


def _hunyuan_video_15_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        tp_degree=4, cp_degree=1,
        height=480, width=848, num_frames=121,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.mp4",
        steps=40, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_compile_raises_not_implemented_with_model_id_flag(monkeypatch):
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator

    with pytest.raises(NotImplementedError) as exc_info:
        HunyuanVideo15Orchestrator(_hunyuan_video_15_args()).compile()

    error_msg = str(exc_info.value)
    assert "--model-id" in error_msg
    assert "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v" in error_msg


def test_generate_raises_not_implemented_with_model_id_flag(monkeypatch):
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator

    with pytest.raises(NotImplementedError) as exc_info:
        HunyuanVideo15Orchestrator(_hunyuan_video_15_args()).generate()

    error_msg = str(exc_info.value)
    assert "--model-id" in error_msg
    assert "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v" in error_msg


def test_download_calls_resolve_model_path(monkeypatch):
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator

    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, local_files_only, **kw: calls.append((model_id, local_files_only)),
    )
    HunyuanVideo15Orchestrator(_hunyuan_video_15_args()).download()
    assert len(calls) == 1
    model_id, local_files_only = calls[0]
    assert model_id == "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    assert local_files_only is False
