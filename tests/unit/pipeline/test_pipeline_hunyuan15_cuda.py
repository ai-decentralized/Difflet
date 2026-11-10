"""CUDA 1.5 public factory policy and shared entrypoint registration."""

import os
import subprocess
import sys

import pytest
import torch

MODEL = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"


def test_hv15_gpu_spec_preserves_legacy_backend_contract():
    from difflet.models.spec import gpu_model_spec, gpu_spec_for_cli_model_id
    from difflet.registry import resolve_model

    spec = gpu_model_spec("hunyuan_video_15")
    assert spec is not None and gpu_spec_for_cli_model_id(MODEL) is spec
    assert (spec.guidance, spec.accepts_negative_prompt, spec.accepts_cfg_parallel) == (
        "cfg",
        True,
        False,
    )
    assert (
        spec.default_height,
        spec.default_width,
        spec.default_num_frames,
        spec.default_steps,
        spec.default_guidance_scale,
        spec.default_fps,
    ) == (480, 848, 121, 50, 6, 24)
    entry = resolve_model(MODEL)
    assert entry.backends == ("trainium", "neuron", "cuda")
    # The registry keeps the Trainium layout (#139: TP2 x CFG2, standard CFG); the GPU
    # spec above does not change it.
    assert entry.default_parallel.tp_degree == 2 and entry.default_parallel.cfg_parallel_enabled
    assert not entry.require_capabilities().is_distilled
    assert spec.served_checkpoint_ids == (MODEL,) and not spec.prompt_budgets


@pytest.mark.parametrize("load_in_factory", [True, False])
def test_hv15_public_factory_loads_real_transformer(tmp_path, monkeypatch, load_in_factory):
    from diffusers import HunyuanVideo15Transformer3DModel
    from difflet.backends.cuda.hunyuan_video_15.application import CudaHunyuanVideo15Application
    from difflet.pipeline import difflet_pipeline as factory
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.platforms.cuda import CudaPlatform
    from tests.unit.backends.test_cuda_hunyuan15_transformer import TINY, _inputs

    ref = HunyuanVideo15Transformer3DModel(**TINY).eval()
    ref.save_pretrained(tmp_path / "transformer")
    monkeypatch.setattr(CudaPlatform, "prepare_runtime", lambda *a: None)
    monkeypatch.setattr(factory, "resolve_model_path", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr(CudaHunyuanVideo15Application, "load_companions", lambda *a: None)
    pipe = factory.DiffletPipeline.from_pretrained(
        MODEL,
        backend="cuda",
        # The registry's default layout is Trainium's (TP2 x CFG2); GPU callers pass TP=1.
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype=torch.float32,
        height=32,
        width=32,
        num_frames=9,
        load=load_in_factory,
        skip_compile=not load_in_factory,
        skip_warmup=True,
        compile_cache_dir=str(tmp_path / "cache"),
        application_kwargs={"device": "cpu"},
    )
    if not load_in_factory:
        pipe.compile()
        pipe.load(skip_warmup=True)
    assert pipe.parallel.tp_degree == 1 and pipe.app.transformer.is_loaded
    with torch.no_grad():
        torch.testing.assert_close(
            pipe.app.transformer(**_inputs(), return_dict=False)[0],
            ref(**_inputs(), return_dict=False)[0],
            atol=1e-4,
            rtol=1e-4,
        )


def test_hv15_cli_cfg_refusal_names_the_missing_feature():
    from difflet.models.spec import gpu_model_spec
    from tests.unit.cli.test_gpu_cli import _orch

    with pytest.raises(SystemExit, match="not implemented") as refused:
        _orch(gpu_model_spec("hunyuan_video_15"), cfg_parallel=True).refuse_unsupported()
    assert "no conditional/unconditional pair" not in str(refused.value)


def test_hv15_spec_and_cli_import_without_torch():
    code = """
import sys
from difflet.models.spec import gpu_model_spec
from difflet.backends.gpu.cli import GpuOrchestrator
spec=gpu_model_spec("hunyuan_video_15")
assert spec.guidance=="cfg"
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
