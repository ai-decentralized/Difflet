"""Explicit Hunyuan GPU factory topology preserves registry defaults."""

from difflet.pipeline.parallel_config import DiffletParallelConfig

MODEL = "hunyuanvideo-community/HunyuanVideo"


def test_public_cuda_factory_accepts_explicit_single_gpu_without_changing_registry_defaults(
    monkeypatch, tmp_path
):
    from difflet.pipeline import difflet_pipeline as factory
    from difflet.platforms.cuda import CudaPlatform
    from difflet.registry import resolve_model

    prepared = []
    monkeypatch.setattr(CudaPlatform, "prepare_runtime", lambda self, cfg: prepared.append(cfg))
    monkeypatch.setattr(factory, "resolve_model_path", lambda *a, **kw: str(tmp_path))
    pipe = factory.DiffletPipeline.from_pretrained(
        MODEL,
        backend="cuda",
        parallel=DiffletParallelConfig(tp_degree=1),
        load=False,
        skip_compile=True,
        compile_cache_dir=str(tmp_path / "cache"),
        application_kwargs={"device": "cpu"},
    )
    assert pipe.parallel.tp_degree == prepared[0].tp_degree == 1
    assert resolve_model(MODEL).default_parallel.tp_degree == 4
    assert pipe.app.spec.model_type == "hunyuan_video"
