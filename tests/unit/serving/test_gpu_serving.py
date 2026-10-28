"""``difflet serve`` on the generic GPU flow: preparer, validator and resident adapter, CPU-only.

Every test runs once per model spec unless it names the specs it covers. DiffletPipeline.
from_pretrained is replaced by a recorder whose app returns PIL images or a small frames tensor;
the MP4 is really encoded (imageio-ffmpeg) and checked with PyAV. No weights, no GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import pathlib
import types

import pytest
import torch
from PIL import Image

from difflet.backends.gpu import serving as mod
from difflet.models.spec import gpu_model_specs
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.engines.stage_pipeline import StagePipelineEngine
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import CompilePolicy, DownloadPolicy
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

SPECS = gpu_model_specs()
H, W, F = 32, 48, 9  # a tiny profile: the MP4 is really encoded
_FROM_PRETRAINED = "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained"


def _ids(spec):
    return spec.model_type


def _specs(predicate):
    """Run a test on the specs ``predicate`` selects instead of all of them."""
    return pytest.mark.parametrize("spec", [s for s in SPECS if predicate(s)], ids=_ids)


@pytest.fixture(params=SPECS, ids=_ids)
def spec(request):
    return request.param


def _model(spec):
    return spec.served_checkpoint_ids[0]


def _source(spec):
    return ResolvedModelSource(
        "hf_snapshot", _model(spec), None, f"/snap/{spec.model_type}", "a" * 40
    )


def _profile(spec, **over) -> ServingProfile:
    d = dict(
        model_id=_model(spec),
        model_type=spec.model_type,
        height=H,
        width=W,
        num_frames=F if spec.output == "video" else None,
        output_fps=spec.default_fps,
        parallel=DiffletParallelConfig(tp_degree=1),
        cache_dir="/c",
    )
    d.update(over)
    return ServingProfile(**d)


@pytest.fixture
def source(monkeypatch, spec):
    calls = []
    monkeypatch.setattr(
        mod, "resolve_hf_model_source", lambda *a, **k: calls.append((a, k)) or _source(spec)
    )
    return calls


def _prepare(spec, profile, policy=CompilePolicy.AUTO):
    return mod.GpuServingArtifactPreparer(_model(spec)).prepare_runtime(
        profile, download_policy=DownloadPolicy.AUTO, compile_policy=policy
    )


# ------------------------------------------------------------------ preparer


def test_prepare_runtime_resolves_the_snapshot_and_builds_an_eager_bundle(spec, source):
    bundle = _prepare(spec, _profile(spec))
    assert source == [((_model(spec),), {"revision": None, "download_policy": DownloadPolicy.AUTO})]
    assert bundle.compile_specs == () and bundle.artifacts.bindings == ()
    assert [s.stage_id for s in bundle.pipeline_definition.stages] == ["pipeline"]
    (stage,) = bundle.runtime_plan.stages
    assert stage.placement == "cuda" and stage.artifact_id is None
    assert bundle.runtime_plan.allocations[0].allocation_id == f"{spec.short_name}-cuda"
    assert bundle.adapter_config["compile_policy"] is CompilePolicy.AUTO


@pytest.mark.parametrize(
    "over,named",
    [
        ({"shapes": ((H, W, None), (16, 16, None))}, "--shapes"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "--tp-degree > 1"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, sp_enabled=True)}, "--sp (sequence"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cp_degree=2)}, "--cp-degree > 1"),
        # A `difflet run` feature for Wan; Qwen-Image and Flux have no second branch to place.
        (
            {"parallel": DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True)},
            "--cfg-parallel",
        ),
    ],
)
def test_unsupported_profiles_are_refused_before_the_download(spec, source, over, named):
    with pytest.raises(DiffletServingError) as exc:
        _prepare(spec, _profile(spec, **over))
    assert exc.value.code == "feature_not_supported" and named in exc.value.message
    assert f"[{spec.model_type}/cuda serve]" in exc.value.message and source == []


def test_the_model_s_refused_serving_options_are_refused_before_the_download(spec, source):
    for attribute, label in spec.refused_serving_options:
        with pytest.raises(DiffletServingError) as exc:
            _prepare(spec, _profile(spec, **{attribute: True}))
        assert exc.value.code == "feature_not_supported" and label in exc.value.message
    assert source == []


def test_profile_identity_tracks_the_shape(spec, source):
    def identity(**over):
        return _prepare(spec, _profile(spec, **over)).runtime_plan.profile_identity

    assert identity() == identity() != identity(height=16)


# ----------------------------------------------------------------- validator


class _Tok:
    def __init__(self, n):
        self.n = n

    def __call__(self, text, **kw):
        return types.SimpleNamespace(input_ids=[list(range(self.n))])


def _validator(bundle, n_tokens=40):
    v = mod.GpuServingRequestValidator(bundle)
    v._tokenizer = lambda subdir: _Tok(n_tokens)
    return v


def _target(tmp_path, name="out") -> FileOutputTarget:
    path = tmp_path / f"{name}.part.mp4"
    path.touch(mode=0o600)
    return FileOutputTarget(staging_path=str(path))


def _request(spec, target=None, video=None, **over):
    d = dict(
        request_id="r1",
        model=_model(spec),
        prompt="a cat",
        height=H,
        width=W,
        num_inference_steps=spec.default_steps,
        guidance_scale=mod.gpu_serving_metadata(spec).default_guidance_scale,
        seed=7,
    )
    if spec.output == "video":
        options = dict(num_frames=F, fps=spec.default_fps, output_target=target)
        options.update(video or {})
        d.update(output_format="mp4", video=VideoGenerateOptions(**options))
    d.update(over)
    return DiffletGenerateRequest(**d)


def test_default_request_is_accepted(spec, source):
    _validator(_prepare(spec, _profile(spec))).validate(_request(spec))


def test_request_shape_must_equal_the_profile_shape(spec, source):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(spec, _profile(spec))).validate(_request(spec, height=64))
    assert exc.value.code == "profile_mismatch"


def test_prompt_over_the_text_encoder_budget_is_refused(spec, source):
    (budget,) = spec.prompt_budgets
    limit = budget.drop_prefix_tokens + budget.max_tokens
    bundle = _prepare(spec, _profile(spec))
    _validator(bundle, n_tokens=limit).validate(_request(spec))
    with pytest.raises(DiffletServingError) as exc:
        _validator(bundle, n_tokens=limit + 1).validate(_request(spec))
    assert exc.value.code == "prompt_too_long"


def test_preload_loads_the_tokenizers_from_the_pinned_snapshot(spec, source, monkeypatch):
    seen = []
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda p, **k: seen.append(str(p)) or _Tok(5)),
    )
    mod.GpuServingRequestValidator(_prepare(spec, _profile(spec))).preload()
    path = _source(spec).pinned_model_path
    assert seen == [f"{path}/{budget.tokenizer_subdir}" for budget in spec.prompt_budgets]


@_specs(lambda s: s.guidance != "none")
def test_guidance_scale_is_honoured_within_the_range(spec, source):
    v = _validator(_prepare(spec, _profile(spec)))
    v.validate(_request(spec, guidance_scale=0.0))
    v.validate(_request(spec, guidance_scale=spec.max_guidance_scale))
    with pytest.raises(DiffletServingError) as exc:
        v.validate(_request(spec, guidance_scale=spec.max_guidance_scale + 0.5))
    assert exc.value.code == "invalid_extra_body"


@_specs(lambda s: s.guidance == "none")
def test_guidance_scale_other_than_the_noop_value_is_refused(spec, source):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(spec, _profile(spec))).validate(_request(spec, guidance_scale=1.0))
    assert exc.value.code == "feature_not_supported"
    assert spec.no_guidance_reason in exc.value.message


# ------------------------------------------------------------------- adapter


class _App:
    """``app.generate``'s result: PIL images, or (B, 3, F, H, W) frames in [-1, 1]."""

    def __init__(self, height, width, num_frames):
        self.calls, self.shape = [], (height, width, num_frames)

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        height, width, num_frames = self.shape
        if num_frames is None:
            return [Image.new("RGB", (width, height), "red")]
        g = torch.Generator().manual_seed(kw["seed"])
        return torch.rand((1, 3, num_frames, height, width), generator=g) * 2 - 1


class _Pipe:
    def __init__(self, height, width, num_frames=None):
        self.app = _App(height, width, num_frames)


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe(kw["height"], kw["width"], kw.get("num_frames"))

    monkeypatch.setattr(_FROM_PRETRAINED, staticmethod(fake))
    return calls


def _engine(spec, bundle):
    adapter = mod.GpuServingStageAdapter(_model(spec))
    return adapter, StagePipelineEngine(runtime=bundle, adapter=adapter)


def _cli(spec, **over):
    from difflet.backends.gpu.cli import GpuOrchestrator

    d = dict(
        model_id=_model(spec),
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
    return GpuOrchestrator(argparse.Namespace(**d), spec)


def test_startup_loads_the_pipeline_exactly_as_the_cli_would(spec, source, recorder):
    bundle = _prepare(spec, _profile(spec, revision="r1"), CompilePolicy.FORCE)
    adapter, engine = _engine(spec, bundle)
    asyncio.run(engine.start())
    model_id, kw = recorder[0]
    cli_kw = _cli(spec).pipeline_kwargs(load=True)
    # The CLI's kwargs except the revision: the server loads the commit the parent pinned, so
    # a worker restart after a newer snapshot landed still loads what the bundle describes.
    assert model_id == _model(spec)
    assert kw == {**cli_kw, "revision": _source(spec).resolved_source_id}
    assert kw["backend"] == "cuda" and kw["load"] is True and kw["force_compile"] is True


def test_auto_compile_policy_does_not_force(spec, source, recorder):
    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    asyncio.run(engine.start())
    assert recorder[0][1]["force_compile"] is False


def test_startup_smoke_runs_the_spec_s_steps_at_the_profile_shape(spec, source, recorder):
    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    asyncio.run(engine.start())  # validate_smoke_output checks the PNG size or decodes the MP4
    prompt, kw = adapter.pipe.app.calls[0]
    guidance = mod.gpu_serving_metadata(spec).default_guidance_scale
    assert kw == _cli(spec, steps=spec.smoke_steps, guidance_scale=guidance).generate_kwargs()
    assert prompt == spec.smoke_prompt and adapter._smoke is None  # the smoke's temp dir is gone


def test_generate_passes_the_cli_s_arguments(spec, source, recorder, tmp_path):
    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    asyncio.run(engine.start())
    req = _request(spec, _target(tmp_path), num_inference_steps=7, guidance_scale=2.5, seed=3)
    out = asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r1", 60.0)))
    prompt, kw = adapter.pipe.app.calls[-1]
    assert prompt == "a cat"
    assert kw == _cli(spec, steps=7, guidance_scale=2.5, seed=3).generate_kwargs()
    assert out.mime_type == mod.gpu_serving_metadata(spec).output_mime_type


def test_smoke_output_of_the_wrong_shape_fails_startup(spec, source, monkeypatch):
    monkeypatch.setattr(
        _FROM_PRETRAINED, staticmethod(lambda m, **kw: _Pipe(H * 2, W, kw.get("num_frames")))
    )
    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    with pytest.raises(RuntimeError):
        asyncio.run(engine.start())
    assert adapter.pipe is None


def test_cancelled_request_never_reaches_generate(spec, source, recorder, tmp_path):
    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    asyncio.run(engine.start())
    n = len(adapter.pipe.app.calls)
    signal = CancellationSignal()
    signal.cancel()
    context = WorkerRequestContext.with_timeout("r", 30.0, cancellation=signal)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.generate(_request(spec, _target(tmp_path)), context))
    assert len(adapter.pipe.app.calls) == n


def test_unsupported_profile_is_refused_before_any_load(spec, source, recorder):
    # The preparer refuses tp=2, so build the bundle by hand to prove the worker side too.
    profile = _profile(spec, parallel=DiffletParallelConfig(tp_degree=2))
    bundle = ResolvedRuntimeBundle(
        profile=profile,
        source=_source(spec),
        pipeline_definition=_prepare(spec, _profile(spec)).pipeline_definition,
        runtime_plan=mod.resident_runtime_plan(profile, _source(spec), spec),
        compile_specs=(),
        artifacts=ArtifactSet(()),
        adapter_config={"compile_policy": CompilePolicy.AUTO},
    )
    with pytest.raises(DiffletServingError) as exc:
        asyncio.run(mod.GpuServingStageAdapter(_model(spec)).create_loaded_runners(bundle))
    assert "not supported" in exc.value.message and recorder == []


def test_shutdown_releases_the_resident_pipeline(spec, source, recorder):
    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    asyncio.run(engine.start())
    asyncio.run(engine.shutdown())
    assert adapter.pipe is None and adapter.active_runtime is None


def test_gpu_serving_metadata_is_one_opaque_stage_with_the_cli_defaults(spec):
    meta = mod.gpu_serving_metadata(spec)
    assert meta.checkpoint_ids == spec.served_checkpoint_ids
    assert meta.default_steps == spec.default_steps and meta.output_modality == spec.output
    (stage,) = meta.pipeline_definition.stages
    assert stage.kind == "opaque_pipeline" and stage.stage_id == "pipeline"
    assert meta.orchestrator_factory == "difflet.backends.gpu.serving:GpuServingStageAdapter"


# --------------------------------------------------------------------- video

_video_only = _specs(lambda s: s.output == "video")


@_video_only
def test_the_clip_length_is_part_of_the_profile_identity(spec, source):
    a = _prepare(spec, _profile(spec)).runtime_plan.profile_identity
    assert a != _prepare(spec, _profile(spec, num_frames=17)).runtime_plan.profile_identity


@_video_only
def test_both_experts_guidance_and_a_negative_prompt_are_accepted(spec, source):
    v = _validator(_prepare(spec, _profile(spec)))
    v.validate(_request(spec, video={"guidance_scale_2": 3.0, "negative_prompt": "blurry"}))
    v.validate(_request(spec, video={"guidance_scale_2": 0.0}, guidance_scale=20.0))
    for bad in ({"guidance_scale_2": 20.5}, {"guidance_scale_2": -1.0}):
        with pytest.raises(DiffletServingError) as exc:
            v.validate(_request(spec, video=bad))
        assert exc.value.code == "invalid_extra_body"


@_video_only
def test_fields_the_application_does_not_take_are_refused(spec, source):
    v = _validator(_prepare(spec, _profile(spec)))
    for field, _ in spec.refused_video_fields:
        with pytest.raises(DiffletServingError) as exc:
            v.validate(_request(spec, video={field: 1.0}))
        assert exc.value.code == "feature_not_supported" and field in exc.value.message


@_video_only
@pytest.mark.parametrize("video", [{"num_frames": 17}, {"fps": 24}])
def test_request_clip_length_and_fps_must_equal_the_profile(spec, source, video):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(spec, _profile(spec))).validate(_request(spec, video=video))
    assert exc.value.code == "profile_mismatch"


@_video_only
def test_only_mp4_with_video_options_is_served(spec, source):
    v = _validator(_prepare(spec, _profile(spec)))
    with pytest.raises(DiffletServingError) as exc:
        v.validate(_request(spec, output_format="png"))
    assert exc.value.code == "invalid_extra_body"
    with pytest.raises(DiffletServingError):
        v.validate(dataclasses.replace(_request(spec), video=None))


@_video_only
def test_generate_writes_a_valid_mp4_at_the_profile_shape(spec, source, recorder, tmp_path):
    from difflet.serving.video_media import validate_mp4

    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    asyncio.run(engine.start())
    target = _target(tmp_path)
    out = asyncio.run(
        engine.generate(_request(spec, target), WorkerRequestContext.with_timeout("r1", 60.0))
    )
    fps = spec.default_fps
    assert out.path == target.staging_path
    assert (out.width, out.height, out.num_frames, out.fps) == (W, H, F, float(fps))
    meta = validate_mp4(
        pathlib.Path(out.path),
        expected_width=W,
        expected_height=H,
        expected_num_frames=F,
        expected_fps=fps,
    )
    assert meta.size_bytes == out.size_bytes > 0


@_video_only
def test_guidance_scale_2_and_negative_prompt_reach_generate_only_when_given(
    spec, source, recorder, tmp_path
):
    adapter, engine = _engine(spec, _prepare(spec, _profile(spec)))
    asyncio.run(engine.start())
    video = {"guidance_scale_2": 2.5, "negative_prompt": "blurry"}
    req = _request(spec, _target(tmp_path, "a"), video=video)
    asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r1", 60.0)))
    _, kw = adapter.pipe.app.calls[-1]
    assert kw["guidance_scale_2"] == 2.5 and kw["negative_prompt"] == "blurry"
    req = _request(spec, _target(tmp_path, "b"))
    asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r2", 60.0)))
    _, kw = adapter.pipe.app.calls[-1]
    assert "guidance_scale_2" not in kw and "negative_prompt" not in kw


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


@_video_only
def test_videos_sync_serves_a_clip_through_the_gpu_adapter(
    spec, source, recorder, monkeypatch, tmp_path
):
    """POST /v1/videos/sync → the GPU validator → the resident adapter → a real MP4."""
    from fastapi.testclient import TestClient

    from difflet.serving.model_registry import resolve_serving_model
    from difflet.serving.openai.api_server import create_app
    from difflet.serving.options import ServeOptions
    from difflet.serving.video_jobs import InMemoryVideoJobRepository
    from difflet.serving.video_media import validate_mp4
    from difflet.serving.video_service import VideoGenerationService
    from difflet.serving.video_storage import LocalVideoArtifactStore

    model = _model(spec)
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    options = ServeOptions(model_id=model, cache_dir=str(tmp_path), height=H, width=W)
    options = dataclasses.replace(options, num_frames=F)
    resolved = resolve_serving_model(options)
    assert resolved.metadata.orchestrator_factory.endswith(":GpuServingStageAdapter")
    bundle = _prepare(spec, resolved.profile)
    adapter = mod.GpuServingStageAdapter(model)
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
                    ("model", model),
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
            clip,
            expected_width=W,
            expected_height=H,
            expected_num_frames=F,
            expected_fps=spec.default_fps,
        )
        assert meta.size_bytes == len(response.content)
        prompt, kw = adapter.pipe.app.calls[-1]
        assert prompt == "a cat in a garden" and kw["seed"] == 5
        assert kw["guidance_scale"] == spec.default_guidance_scale and kw["guidance_scale_2"] == 3.0
        assert kw["num_inference_steps"] == spec.default_steps and kw["num_frames"] == F
        # A field the application does not take is refused before the worker.
        n = len(adapter.pipe.app.calls)
        refused = client.post(
            "/v1/videos/sync",
            files=[
                (name, (None, value)) for name, value in (("prompt", "x"), ("flow_shift", "5.0"))
            ],
        )
        assert refused.status_code == 400, refused.text
        # The GPU validator's message, not the form parser's: the validator runs in the server.
        assert "[wan/cuda serve] flow_shift is not taken by Wan" in refused.text
        assert len(adapter.pipe.app.calls) == n
