"""Public GPU factory defaults and refusals before checkpoint I/O."""

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig

MODEL = "hunyuanvideo-community/HunyuanVideo"


def test_public_cuda_factory_defaults_to_one_gpu_without_changing_trainium(monkeypatch, tmp_path):
    from difflet.pipeline import difflet_pipeline as factory
    from difflet.platforms.cuda import CudaPlatform
    from difflet.registry import resolve_model

    prepared = []
    monkeypatch.setattr(CudaPlatform, "prepare_runtime", lambda self, cfg: prepared.append(cfg))
    monkeypatch.setattr(factory, "resolve_model_path", lambda *a, **kw: str(tmp_path))
    pipe = factory.DiffletPipeline.from_pretrained(
        MODEL,
        backend="cuda",
        load=False,
        skip_compile=True,
        compile_cache_dir=str(tmp_path / "cache"),
        application_kwargs={"device": "cpu"},
    )
    assert pipe.parallel.tp_degree == prepared[0].tp_degree == 1
    assert resolve_model(MODEL).default_parallel.tp_degree == 4
    assert pipe.app.spec.model_type == "hunyuan_video"


@pytest.mark.parametrize("load_in_factory", [True, False])
def test_public_cuda_factory_prepares_manifest_before_loading_real_transformer(
    monkeypatch, tmp_path, load_in_factory
):
    import torch
    from diffusers import HunyuanVideoTransformer3DModel

    from difflet.backends.cuda.hunyuan_video.application import CudaHunyuanVideoApplication
    from difflet.pipeline import difflet_pipeline as factory
    from difflet.platforms.cuda import CudaPlatform
    from tests.unit.backends.test_cuda_hunyuan_transformer import TINY, _inputs

    torch.manual_seed(21)
    reference = HunyuanVideoTransformer3DModel(**TINY).eval()
    reference.save_pretrained(tmp_path / "transformer")
    monkeypatch.setattr(CudaPlatform, "prepare_runtime", lambda self, cfg: None)
    monkeypatch.setattr(factory, "resolve_model_path", lambda *a, **kw: str(tmp_path))
    # Companion checkpoints are outside this transformer-loading regression.
    monkeypatch.setattr(CudaHunyuanVideoApplication, "load_companions", lambda self: None)

    pipe = factory.DiffletPipeline.from_pretrained(
        MODEL,
        backend="cuda",
        dtype=torch.float32,
        height=32,
        width=32,
        num_frames=9,
        load=load_in_factory,
        skip_compile=not load_in_factory,
        skip_warmup=True,
        compile_cache_dir=str(tmp_path / "cache"),
        application_kwargs={"device": "cpu", "text_seq_len": 5},
    )
    if not load_in_factory:
        pipe.compile()
        pipe.load(skip_warmup=True)

    assert pipe.app.transformer.is_loaded
    assert pipe.app.has_compiled_artifacts(str(pipe.compiled_path))
    inputs = _inputs()
    with torch.no_grad():
        actual = pipe.app.transformer(*inputs, return_dict=False)[0]
        expected = reference(*inputs, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"height": 321}, ValueError),
        ({"width": 0}, ValueError),
        ({"num_frames": 60}, ValueError),
        ({"application_kwargs": {"host_vae": True}}, NotImplementedError),
        ({"application_kwargs": {"teacache_speedup": 2}}, NotImplementedError),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, NotImplementedError),
    ],
)
def test_invalid_public_gpu_request_is_refused_before_runtime_or_checkpoint_resolution(
    monkeypatch, options, error
):
    from difflet.pipeline import difflet_pipeline as factory
    from difflet.platforms.cuda import CudaPlatform

    def forbidden(*args, **kwargs):
        pytest.fail("invalid GPU request reached runtime or checkpoint resolution")

    monkeypatch.setattr(CudaPlatform, "prepare_runtime", forbidden)
    monkeypatch.setattr(factory, "resolve_model_path", forbidden)
    with pytest.raises(error):
        factory.DiffletPipeline.from_pretrained(MODEL, backend="cuda", load=False, **options)
