"""Wan 2.2 serving on CUDA: preparer, validator and resident adapter, CPU-only.

DiffletPipeline.from_pretrained is replaced by a recorder whose app returns a small frames
tensor; the MP4 is really encoded (imageio-ffmpeg) and checked with PyAV. No weights, no GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import pathlib
import tempfile
import types

import pytest
import torch

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.engines.stage_pipeline import StagePipelineEngine
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators import wan_cuda as mod
from difflet.serving.types import (
    ArtifactSet,
    CancellationSignal,
    DiffletGenerateRequest,
    FileOutputTarget,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    ServingProfile,
    VideoGenerateOptions,
    WorkerRequestContext,
)

_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_SOURCE = ResolvedModelSource("hf_snapshot", _MODEL, None, "/snap/wan", "e" * 40)
_FROM_PRETRAINED = "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained"
H, W, F, FPS = 32, 48, 9, 16  # a tiny profile: the MP4 is really encoded


def _profile(**over) -> ServingProfile:
    d = dict(
        model_id=_MODEL,
        model_type="wan",
        height=H,
        width=W,
        num_frames=F,
        output_fps=FPS,
        parallel=DiffletParallelConfig(tp_degree=1),
        cache_dir="/c",
    )
    d.update(over)
    return ServingProfile(**d)


@pytest.fixture
def source(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mod, "resolve_hf_model_source", lambda *a, **k: calls.append((a, k)) or _SOURCE
    )
    return calls


def _prepare(profile, policy=CompilePolicy.AUTO):
    return mod.CudaWanServingArtifactPreparer().prepare_runtime(
        profile, download_policy=DownloadPolicy.AUTO, compile_policy=policy
    )


# ------------------------------------------------------------------ preparer


def test_prepare_runtime_resolves_the_snapshot_and_builds_an_eager_bundle(source):
    bundle = _prepare(_profile())
    assert source[0][0] == (_MODEL,)
    assert bundle.compile_specs == () and bundle.artifacts.bindings == ()
    assert [s.stage_id for s in bundle.pipeline_definition.stages] == ["pipeline"]
    (stage,) = bundle.runtime_plan.stages
    assert stage.placement == "cuda" and stage.artifact_id is None
    assert bundle.runtime_plan.allocations[0].allocation_id == "wan-cuda"
    # The clip length is part of the resident profile's identity.
    other = _prepare(_profile(num_frames=17)).runtime_plan.profile_identity
    assert other != bundle.runtime_plan.profile_identity


@pytest.mark.parametrize(
    "over,named",
    [
        ({"shapes": ((H, W, F), (H, W, 17))}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "--tp-degree > 1"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, sp_enabled=True)}, "--sp (sequence"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cp_degree=2)}, "not supported"),
        (
            {"parallel": DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True)},
            "not supported",
        ),
        ({"teacache_speedup": 1.5}, "not supported"),
        ({"host_vae": True}, "not supported"),
    ],
)
def test_unsupported_profiles_are_refused_before_the_download(source, over, named):
    with pytest.raises(DiffletServingError) as exc:
        _prepare(_profile(**over))
    assert exc.value.code == "feature_not_supported" and named in exc.value.message
    assert "[wan/cuda serve]" in exc.value.message and source == []


# ----------------------------------------------------------------- validator


class _Tok:
    def __init__(self, n):
        self.n = n

    def __call__(self, text, **kw):
        return types.SimpleNamespace(input_ids=[list(range(self.n))])


def _validator(bundle, n_tokens=40):
    v = mod.CudaWanServingRequestValidator(bundle)
    v._tokenizer_for_runtime = lambda: _Tok(n_tokens)
    return v


def _target() -> FileOutputTarget:
    root = pathlib.Path(tempfile.mkdtemp(prefix="wan-cuda-test-"))
    path = root / "out.part.mp4"
    path.touch(mode=0o600)
    return FileOutputTarget(staging_path=str(path))


def _request(video=None, **over):
    v = dict(num_frames=F, fps=FPS, output_target=None)
    v.update(video or {})
    d = dict(
        request_id="r1",
        model=_MODEL,
        prompt="a cat",
        height=H,
        width=W,
        num_inference_steps=40,
        guidance_scale=4.0,
        seed=7,
        output_format="mp4",
        video=VideoGenerateOptions(**v),
    )
    d.update(over)
    return DiffletGenerateRequest(**d)


def test_default_request_is_accepted(source):
    _validator(_prepare(_profile())).validate(_request())


def test_both_experts_guidance_and_a_negative_prompt_are_accepted(source):
    v = _validator(_prepare(_profile()))
    v.validate(_request({"guidance_scale_2": 3.0, "negative_prompt": "blurry"}))
    v.validate(_request({"guidance_scale_2": 0.0}, guidance_scale=20.0))
    for bad in ({"guidance_scale_2": 20.5}, {"guidance_scale_2": -1.0}):
        with pytest.raises(DiffletServingError) as exc:
            v.validate(_request(bad))
        assert exc.value.code == "invalid_extra_body"
    with pytest.raises(DiffletServingError):
        v.validate(_request(guidance_scale=21.0))


@pytest.mark.parametrize(
    "field,value", [("boundary_ratio", 0.9), ("flow_shift", 5.0), ("true_cfg_scale", 4.0)]
)
def test_fields_the_cuda_app_does_not_take_are_refused(source, field, value):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(_profile())).validate(_request({field: value}))
    assert exc.value.code == "feature_not_supported" and field in exc.value.message


@pytest.mark.parametrize(
    "request_over,video_over",
    [
        ({"height": 64}, {}),
        ({"width": 64}, {}),
        ({}, {"num_frames": 17}),
        ({}, {"fps": 24}),
    ],
)
def test_request_shape_and_fps_must_equal_the_profile(source, request_over, video_over):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(_profile())).validate(_request(video_over, **request_over))
    assert exc.value.code == "profile_mismatch"


def test_only_mp4_with_video_options_is_served(source):
    v = _validator(_prepare(_profile()))
    with pytest.raises(DiffletServingError) as exc:
        v.validate(_request(output_format="png"))
    assert exc.value.code == "invalid_extra_body"
    with pytest.raises(DiffletServingError):
        v.validate(dataclasses.replace(_request(), video=None))


def test_prompts_longer_than_umt5_s_512_tokens_are_refused(source):
    bundle = _prepare(_profile())
    _validator(bundle, n_tokens=512).validate(_request({"negative_prompt": "blurry"}))
    with pytest.raises(DiffletServingError) as exc:
        _validator(bundle, n_tokens=513).validate(_request())
    assert exc.value.code == "prompt_too_long" and "prompt" in exc.value.message


def test_preload_loads_the_umt5_tokenizer_from_the_pinned_snapshot(source, monkeypatch):
    seen = {}
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda p, **k: seen.update(path=str(p)) or _Tok(5)),
    )
    mod.CudaWanServingRequestValidator(_prepare(_profile())).preload()
    assert seen["path"] == "/snap/wan/tokenizer"


# ------------------------------------------------------------------- adapter


class _App:
    """CudaWanApplication's generate: (B, 3, F, H, W) frames in [-1, 1]."""

    def __init__(self, frames, height, width):
        self.calls, self.shape = [], (1, 3, frames, height, width)

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        g = torch.Generator().manual_seed(int(kw.get("seed") or 0))
        return torch.rand(self.shape, generator=g) * 2 - 1


class _Pipe:
    def __init__(self, frames, height, width):
        self.app = _App(frames, height, width)


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe(kw["num_frames"], kw["height"], kw["width"])

    monkeypatch.setattr(_FROM_PRETRAINED, staticmethod(fake))
    return calls


def _engine(bundle):
    adapter = mod.CudaWanServingStageAdapter()
    return adapter, StagePipelineEngine(runtime=bundle, adapter=adapter)


def _cli(**over):
    from difflet.cli.orchestrators.wan_cuda import WanCudaOrchestrator

    d = dict(
        model_id=_MODEL,
        tp_degree=1,
        height=H,
        width=W,
        num_frames=F,
        cache_dir="/c",
        revision="r1",
        force=True,
        cfg_parallel=False,
        torch_compile=False,
        steps=None,
        guidance_scale=None,
        seed=0,
    )
    d.update(over)
    return WanCudaOrchestrator(argparse.Namespace(**d))


def test_startup_loads_the_pipeline_exactly_as_the_cuda_cli_would(source, recorder):
    bundle = _prepare(_profile(revision="r1"), CompilePolicy.FORCE)
    adapter, engine = _engine(bundle)
    asyncio.run(engine.start())
    model_id, kw = recorder[0]
    cli_kw = _cli().pipeline_kwargs(load=True)
    assert model_id == _MODEL
    assert {k: v for k, v in kw.items() if k != "revision"} == {
        k: v for k, v in cli_kw.items() if k != "revision"
    }
    assert (kw["height"], kw["width"], kw["num_frames"]) == (H, W, F)
    assert kw["revision"] == _SOURCE.resolved_source_id and kw["force_compile"] is True


def test_startup_smoke_writes_a_valid_mp4_at_the_profile_shape(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())  # validate_smoke_output decodes the MP4 with PyAV
    prompt, kw = adapter.pipe.app.calls[0]
    assert kw == {
        "num_inference_steps": mod.SMOKE_STEPS,
        "guidance_scale": 4.0,
        "seed": 0,
        "num_frames": F,
        "output_type": "pt",
    }
    assert mod.SMOKE_STEPS == 4 and adapter._smoke is None  # the smoke's temp dir is gone


def test_generate_passes_the_cli_s_arguments_and_writes_the_mp4(source, recorder):
    from difflet.serving.video_media import validate_mp4

    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    target = _target()
    req = _request({"output_target": target}, num_inference_steps=7, guidance_scale=5.0, seed=3)
    out = asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r1", 60.0)))
    prompt, kw = adapter.pipe.app.calls[-1]
    cli_kw = _cli(steps=7, guidance_scale=5.0, seed=3).generate_kwargs()
    assert prompt == "a cat" and kw == cli_kw
    assert out.path == target.staging_path and out.mime_type == "video/mp4"
    assert (out.width, out.height, out.num_frames, out.fps) == (W, H, F, float(FPS))
    meta = validate_mp4(
        pathlib.Path(out.path),
        expected_width=W,
        expected_height=H,
        expected_num_frames=F,
        expected_fps=FPS,
    )
    assert meta.size_bytes == out.size_bytes > 0


def test_guidance_scale_2_and_negative_prompt_reach_generate_only_when_given(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    req = _request(
        {"output_target": _target(), "guidance_scale_2": 2.5, "negative_prompt": "blurry"}
    )
    asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r1", 60.0)))
    _, kw = adapter.pipe.app.calls[-1]
    assert kw["guidance_scale_2"] == 2.5 and kw["negative_prompt"] == "blurry"
    asyncio.run(
        engine.generate(
            _request({"output_target": _target()}), WorkerRequestContext.with_timeout("r2", 60.0)
        )
    )
    _, kw = adapter.pipe.app.calls[-1]
    assert "guidance_scale_2" not in kw and "negative_prompt" not in kw


def test_smoke_output_of_the_wrong_shape_fails_startup(source, monkeypatch):
    monkeypatch.setattr(_FROM_PRETRAINED, staticmethod(lambda m, **kw: _Pipe(F, H * 2, W)))
    adapter, engine = _engine(_prepare(_profile()))
    with pytest.raises(RuntimeError):
        asyncio.run(engine.start())
    assert adapter.pipe is None


def test_cancelled_request_never_reaches_generate(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    n = len(adapter.pipe.app.calls)
    signal = CancellationSignal()
    signal.cancel()
    context = WorkerRequestContext.with_timeout("r", 30.0, cancellation=signal)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.generate(_request({"output_target": _target()}), context))
    assert len(adapter.pipe.app.calls) == n


def test_unsupported_profile_is_refused_before_any_load(source, recorder):
    profile = _profile(parallel=DiffletParallelConfig(tp_degree=2))
    bundle = ResolvedRuntimeBundle(
        profile=profile,
        source=_SOURCE,
        pipeline_definition=_prepare(_profile()).pipeline_definition,
        runtime_plan=mod.cuda_runtime_plan(profile, _SOURCE),
        compile_specs=(),
        artifacts=ArtifactSet(()),
        adapter_config={"compile_policy": CompilePolicy.AUTO},
    )
    with pytest.raises(DiffletServingError) as exc:
        asyncio.run(mod.CudaWanServingStageAdapter().create_loaded_runners(bundle))
    assert "not supported" in exc.value.message and recorder == []


def test_shutdown_releases_the_resident_pipeline(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    asyncio.run(engine.shutdown())
    assert adapter.pipe is None and adapter.active_runtime is None


def test_cuda_serving_metadata_is_one_opaque_stage_with_the_cli_defaults():
    from difflet.common.registry.wan import cuda_serving_metadata

    meta = cuda_serving_metadata()
    assert meta.model_type == "wan" and meta.checkpoint_ids == (_MODEL,)
    assert (meta.default_steps, meta.default_guidance_scale, meta.default_fps) == (40, 4.0, 16)
    assert meta.output_modality == "video" and meta.chat_content_type == "video_url"
    assert meta.output_mime_type == "video/mp4" and meta.default_host_vae is False
    (stage,) = meta.pipeline_definition.stages
    assert stage.kind == "opaque_pipeline" and stage.stage_id == "pipeline"
    assert meta.orchestrator_factory.endswith("wan_cuda:CudaWanServingStageAdapter")


# ------------------------------------------------------------- over HTTP


class _InProcessEngine:
    """The resident engine's surface over an in-process StagePipelineEngine (no worker)."""

    def __init__(self, stage_engine):
        self.stage_engine = stage_engine

    async def start(self) -> None:
        await self.stage_engine.start()

    async def shutdown(self) -> None:
        await self.stage_engine.shutdown()

    async def generate(self, request):
        context = WorkerRequestContext.with_timeout(request.request_id, 60.0)
        return await self.stage_engine.generate(request, context)

    async def wait_for_recovery(self, *, timeout=None) -> None:
        return None

    def health(self):
        return {"status": "ok"}


def test_videos_sync_serves_a_wan_clip_through_the_cuda_adapter(
    source, recorder, monkeypatch, tmp_path
):
    """POST /v1/videos/sync → the CUDA validator → the resident adapter → a real MP4."""
    from fastapi.testclient import TestClient

    from difflet.serving.model_registry import resolve_serving_model
    from difflet.serving.openai.api_server import create_app
    from difflet.serving.options import ServeOptions
    from difflet.serving.video_jobs import InMemoryVideoJobRepository
    from difflet.serving.video_media import validate_mp4
    from difflet.serving.video_service import VideoGenerationService
    from difflet.serving.video_storage import LocalVideoArtifactStore

    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    options = ServeOptions(model_id=_MODEL, cache_dir=str(tmp_path), height=H, width=W)
    options = dataclasses.replace(options, num_frames=F)
    resolved = resolve_serving_model(options)
    assert resolved.metadata.orchestrator_factory.endswith("CudaWanServingStageAdapter")
    bundle = _prepare(resolved.profile)
    adapter = mod.CudaWanServingStageAdapter()
    engine = _InProcessEngine(StagePipelineEngine(runtime=bundle, adapter=adapter))
    validator = _validator(bundle)
    service = VideoGenerationService(
        engine=engine,
        jobs=InMemoryVideoJobRepository(),
        artifacts=LocalVideoArtifactStore(tmp_path / "media"),
        max_queued_requests=0,
        queue_timeout_s=60.0,
        request_timeout_s=60.0,
        recovery_timeout_s=1.0,
    )
    app = create_app(
        options=options,
        resolved_model=resolved,
        engine=engine,
        request_validator=validator,
        video_service=service,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos/sync",
            files=[
                (name, (None, value))
                for name, value in (
                    ("model", _MODEL),
                    ("prompt", "a cat in a garden"),
                    ("size", f"{W}x{H}"),
                    ("num_frames", str(F)),
                    ("guidance_scale_2", "3.0"),
                    ("seed", "5"),
                )
            ],
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "video/mp4"
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(response.content)
        meta = validate_mp4(
            clip, expected_width=W, expected_height=H, expected_num_frames=F, expected_fps=FPS
        )
        assert meta.size_bytes == len(response.content)
        prompt, kw = adapter.pipe.app.calls[-1]
        assert prompt == "a cat in a garden" and kw["seed"] == 5
        assert kw["guidance_scale"] == 4.0 and kw["guidance_scale_2"] == 3.0
        assert kw["num_inference_steps"] == 40 and kw["num_frames"] == F
        # A field the CUDA app does not take is refused before the worker.
        n = len(adapter.pipe.app.calls)
        refused = client.post(
            "/v1/videos/sync",
            files=[
                (name, (None, value)) for name, value in (("prompt", "x"), ("flow_shift", "5.0"))
            ],
        )
        assert refused.status_code == 400, refused.text
        # The CUDA validator's message, not the form parser's: the validator runs in the server.
        assert "flow_shift is not taken by Wan 2.2 on the CUDA backend" in refused.text
        assert len(adapter.pipe.app.calls) == n
