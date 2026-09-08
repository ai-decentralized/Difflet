from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError
from difflet.serving.models import ltx_2
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.types import (
    ArtifactBinding,
    ArtifactSet,
    CompileArtifactIdentity,
    DiffletCompileSpec,
    DiffletGenerateRequest,
    FileBackedGenerateOutput,
    FileOutputTarget,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    ServingProfile,
    StageInvocation,
    VideoGenerateOptions,
    WorkerRequestContext,
)
from difflet.serving.video_media import VideoMediaEncodingError, VideoMediaMetadata


def _profile(
    tmp_path: Path,
    *,
    parallel: DiffletParallelConfig | None = None,
    revision: str | None = None,
    num_frames: int = 17,
) -> ServingProfile:
    return ServingProfile(
        model_id="Lightricks/LTX-2",
        model_type="ltx_2",
        height=64,
        width=96,
        num_frames=num_frames,
        parallel=parallel or DiffletParallelConfig(tp_degree=4),
        cache_dir=str(tmp_path / "cache"),
        revision=revision,
        dtype="bfloat16",
        output_modality="video",
        output_mime_type="video/mp4",
        output_fps=24,
        host_vae=True,
    )


def _source(tmp_path: Path, *, commit: str = "a" * 40) -> ResolvedModelSource:
    return ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id="Lightricks/LTX-2",
        requested_revision=None,
        pinned_model_path=str(tmp_path / "snapshots" / commit),
        resolved_source_id=commit,
    )


def _runtime(tmp_path: Path, profile: ServingProfile | None = None) -> ResolvedRuntimeBundle:
    profile = profile or _profile(tmp_path)
    source = _source(tmp_path)
    pipeline = ltx_2._pipeline_definition()
    identity = CompileArtifactIdentity.from_cache_inputs(
        {
            "model": profile.model_id,
            "shape": profile.shape_dict(),
            "revision": source.resolved_source_id,
        }
    )
    spec = DiffletCompileSpec("pipeline", "pipeline", identity)
    plan = ltx_2.build_runtime_plan(profile, pipeline, (spec,))
    artifact_path = tmp_path / "cache" / "serving" / "ltx_2" / identity.digest / "g1"
    binding = ArtifactBinding(
        artifact_id="pipeline",
        path=artifact_path,
        manifest_path=artifact_path / "difflet_generation_manifest.json",
        identity=identity,
        generation_id="g0000000000000001",
        content_digest="0" * 64,
    )
    return ResolvedRuntimeBundle(
        profile=profile,
        source=source,
        pipeline_definition=pipeline,
        runtime_plan=plan,
        compile_specs=(spec,),
        artifacts=ArtifactSet((binding,)),
    )


def _request(
    profile: ServingProfile,
    *,
    target: FileOutputTarget | None = None,
    **overrides,
) -> DiffletGenerateRequest:
    request = DiffletGenerateRequest(
        request_id="request-1",
        model=profile.model_id,
        prompt="a paper boat on water",
        height=profile.height,
        width=profile.width,
        num_inference_steps=4,
        guidance_scale=3.5,
        seed=42,
        output_format="mp4",
        video=VideoGenerateOptions(
            num_frames=int(profile.num_frames),
            fps=int(profile.output_fps),
            output_target=target,
            negative_prompt="blurred",
        ),
    )
    return replace(request, **overrides)


def _invocation(
    runtime: ResolvedRuntimeBundle,
    request: DiffletGenerateRequest,
) -> StageInvocation[ltx_2.LTX2InitialPayload]:
    return StageInvocation(
        request=request,
        stage=runtime.pipeline_definition.stages[0],
        input=ltx_2.LTX2InitialPayload(),
        context=WorkerRequestContext.with_timeout(request.request_id, 10.0),
    )


def test_compile_plan_binds_commit_and_fixed_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(ltx_2, "_torch_bfloat16", lambda: "bfloat16")
    profile = _profile(tmp_path)
    first = ltx_2.build_compile_plan(_source(tmp_path, commit="a" * 40), profile)[0]
    second = ltx_2.build_compile_plan(_source(tmp_path, commit="b" * 40), profile)[0]
    other_shape = ltx_2.build_compile_plan(
        _source(tmp_path, commit="a" * 40),
        replace(profile, num_frames=25),
    )[0]

    assert first.artifact_id == first.component_id == "pipeline"
    assert first.identity != second.identity
    assert first.identity != other_shape.identity
    cache_inputs = json.loads(first.identity.canonical_cache_inputs_json)
    assert cache_inputs["cache_inputs"]["revision"] == "a" * 40
    # Schema v5: single "shape" dict replaced by the canonical "shapes" list.
    assert cache_inputs["cache_inputs"]["shapes"] == [[64, 96, 17]]
    assert str(tmp_path) not in first.identity.canonical_cache_inputs_json.decode()


def test_build_pipeline_uses_bound_lower_pipeline_and_host_components(monkeypatch, tmp_path):
    calls = []

    def fake_from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return object()

    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    monkeypatch.setattr(
        DiffletPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    monkeypatch.setattr(ltx_2, "_torch_bfloat16", lambda: "bfloat16")
    profile = _profile(tmp_path, revision="release")
    source = replace(_source(tmp_path), requested_revision="release")
    artifact = tmp_path / "compiled"

    result = ltx_2.build_pipeline(
        source,
        profile,
        load=True,
        skip_compile=True,
        compiled_path_override=artifact,
        host_components=True,
    )

    assert result is not None
    assert calls[0][0] == profile.model_id
    kwargs = calls[0][1]
    assert kwargs["model_path_override"] == source.pinned_model_path
    assert kwargs["resolved_source_id"] == source.resolved_source_id
    assert kwargs["compiled_path_override"] == str(artifact)
    assert kwargs["skip_compile"] is True
    assert kwargs["load"] is True
    assert kwargs["application_kwargs"] == {
        "enable_host_pipeline": True,
        "enable_decode_components": True,
        "host_device": "cpu",
    }


def test_runtime_plan_is_one_tp4_hybrid_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ltx_2,
        "resolve_available_neuron_core_ids",
        lambda *, required_num_cores: (4, 5, 6, 7),
    )
    profile = _profile(tmp_path)
    identity = CompileArtifactIdentity.from_cache_inputs({"profile": "ltx2"})
    spec = DiffletCompileSpec("pipeline", "pipeline", identity)

    plan = ltx_2.build_runtime_plan(
        profile,
        ltx_2._pipeline_definition(),
        (spec,),
    )

    assert plan.profile_identity == identity.digest
    assert plan.environment.available_core_ids == (4, 5, 6, 7)
    assert len(plan.allocations) == 1
    assert plan.allocations[0].effective_num_cores == 4
    assert len(plan.stages) == 1
    assert plan.stages[0].stage_id == "pipeline"
    assert plan.stages[0].placement == "hybrid"
    assert plan.stages[0].topology.tp_degree == 4
    assert plan.stages[0].topology.cp_degree == 1
    assert plan.stages[0].topology.world_size == 4


@pytest.mark.parametrize(
    "parallel",
    [
        DiffletParallelConfig(tp_degree=2),
        DiffletParallelConfig(tp_degree=2, cp_degree=2),
        DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True),
        DiffletParallelConfig(tp_degree=4, sp_enabled=True),
    ],
)
def test_runtime_rejects_non_tp4_profile(parallel, tmp_path):
    profile = _profile(tmp_path, parallel=parallel)
    identity = CompileArtifactIdentity.from_cache_inputs({"profile": "bad"})
    spec = DiffletCompileSpec("pipeline", "pipeline", identity)

    with pytest.raises(ValueError, match="TP4, CP1, DP1, CFG off, and SP off"):
        ltx_2.build_runtime_plan(profile, ltx_2._pipeline_definition(), (spec,))


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"width": 80}, "divisible by 32"),
        ({"num_frames": 18}, "8n\\+1"),
        ({"dtype": "float32"}, "bfloat16"),
        ({"output_modality": "image"}, "video/mp4"),
        ({"output_mime_type": "image/png"}, "video/mp4"),
        ({"host_vae": False}, "host decode"),
        # No TeaCache in LTX-2 serving: adaptive OR probe-free.
        ({"teacache_speedup": 1.5}, "TeaCache"),
        ({"teacache_cadence": 2}, "TeaCache"),
        ({"teacache_online_delta": 0.6}, "TeaCache"),
    ],
)
def test_profile_rejects_shapes_and_runtime_contract_mismatches(changed, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        ltx_2._validate_profile(replace(_profile(tmp_path), **changed))


def test_artifact_preparer_uses_commit_source_and_immutable_manager(monkeypatch, tmp_path):
    profile = _profile(tmp_path, revision="release")
    source = replace(_source(tmp_path, commit="c" * 40), requested_revision="release")
    calls = {}

    monkeypatch.setattr(ltx_2, "_torch_bfloat16", lambda: "bfloat16")
    monkeypatch.setattr(
        ltx_2,
        "resolve_model",
        lambda model_id, model_type: SimpleNamespace(download_patterns=("*.json",)),
    )

    def fake_resolve(model_id, *, revision, download_policy, allow_patterns):
        calls["resolve"] = (model_id, revision, download_policy, allow_patterns)
        return source

    monkeypatch.setattr(ltx_2, "resolve_hf_model_source", fake_resolve)

    class FakeManager:
        def __init__(self, root):
            calls["root"] = Path(root)

        def prepare(self, **kwargs):
            calls["prepare"] = kwargs
            path = tmp_path / "published" / "g0000000000000001"
            return ArtifactBinding(
                artifact_id=kwargs["artifact_id"],
                path=path,
                manifest_path=path / "difflet_generation_manifest.json",
                identity=kwargs["identity"],
                generation_id=path.name,
                content_digest="1" * 64,
            )

    monkeypatch.setattr(ltx_2, "ImmutableArtifactManager", FakeManager)
    preparer = ltx_2.LTX2ServingArtifactPreparer(revision="release")

    runtime = preparer.prepare_runtime(
        profile,
        download_policy=DownloadPolicy.AUTO,
        compile_policy=CompilePolicy.AUTO,
    )

    assert runtime.source is source
    assert calls["resolve"] == (
        profile.model_id,
        "release",
        DownloadPolicy.AUTO,
        ("*.json",),
    )
    assert calls["root"] == Path(profile.cache_dir)
    assert calls["prepare"]["model_type"] == "ltx_2"
    assert calls["prepare"]["artifact_id"] == "pipeline"
    assert callable(calls["prepare"]["compile_artifact"])
    assert callable(calls["prepare"]["validate_payload"])
    identity_inputs = json.loads(runtime.compile_specs[0].identity.canonical_cache_inputs_json)
    assert identity_inputs["cache_inputs"]["revision"] == "c" * 40


def test_request_validator_accepts_fixed_ltx2_profile(tmp_path):
    runtime = _runtime(tmp_path)
    validator = ltx_2.LTX2ServingRequestValidator(runtime)
    validator._tokenizer = lambda *args, **kwargs: SimpleNamespace(
        input_ids=SimpleNamespace(shape=(1, 10))
    )

    validator.validate(_request(runtime.profile))


def test_request_validator_rejects_prompt_outside_text_bucket(tmp_path):
    runtime = _runtime(tmp_path)
    validator = ltx_2.LTX2ServingRequestValidator(runtime)
    validator._tokenizer = lambda *args, **kwargs: SimpleNamespace(
        input_ids=SimpleNamespace(shape=(1, 1025))
    )

    with pytest.raises(DiffletServingError) as exc:
        validator.validate(_request(runtime.profile))

    assert exc.value.code == "prompt_too_long"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda request: replace(request, num_inference_steps=1), "invalid_extra_body"),
        (lambda request: replace(request, num_inference_steps=201), "invalid_extra_body"),
        (lambda request: replace(request, guidance_scale=float("inf")), "invalid_extra_body"),
        (lambda request: replace(request, width=request.width + 2), "profile_mismatch"),
        (
            lambda request: replace(
                request,
                video=replace(request.video, num_frames=request.video.num_frames + 8),
            ),
            "profile_mismatch",
        ),
        (
            lambda request: replace(
                request,
                video=replace(request.video, guidance_scale_2=2.0),
            ),
            "invalid_extra_body",
        ),
        (
            lambda request: replace(
                request,
                video=replace(request.video, flow_shift=5.0),
            ),
            "invalid_extra_body",
        ),
    ],
)
def test_request_validator_rejects_unsupported_or_mismatched_requests(
    mutate,
    code,
    tmp_path,
):
    runtime = _runtime(tmp_path)
    validator = ltx_2.LTX2ServingRequestValidator(runtime)
    validator._tokenizer = lambda *args, **kwargs: SimpleNamespace(
        input_ids=SimpleNamespace(shape=(1, 10))
    )

    with pytest.raises(DiffletServingError) as exc:
        validator.validate(mutate(_request(runtime.profile)))

    assert exc.value.code == code


def test_runner_encodes_bfchw_frames_to_exact_parent_target(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path)
    target_path = (tmp_path / "job.part.mp4").resolve()
    target_path.touch()
    target = FileOutputTarget(staging_path=str(target_path))
    encoded_frames = object()
    generator = object()
    calls = {}

    class Frames:
        def float(self):
            calls["float"] = True
            return self

        def clamp(self, minimum, maximum):
            calls["clamp"] = (minimum, maximum)
            return encoded_frames

    frames = Frames()

    class FakePipe:
        def __call__(self, **kwargs):
            calls["pipeline"] = kwargs
            return SimpleNamespace(frames=frames)

    def fake_encode(tensor, path, **kwargs):
        calls["encode"] = (tensor, path, kwargs)
        target_path.write_bytes(b"mp4")
        return VideoMediaMetadata(
            width=runtime.profile.width,
            height=runtime.profile.height,
            num_frames=int(runtime.profile.num_frames),
            fps=float(runtime.profile.output_fps),
            duration_s=runtime.profile.num_frames / runtime.profile.output_fps,
            codec_name="h264",
            pixel_format="yuv420p",
            size_bytes=3,
        )

    monkeypatch.setattr(ltx_2, "_seeded_generator", lambda seed: generator)
    monkeypatch.setattr(ltx_2, "encode_tensor_to_mp4", fake_encode)
    runner = ltx_2.LTX2PipelineRunner(FakePipe(), runtime.profile)
    request = _request(runtime.profile, target=target)

    result = asyncio.run(runner.execute(_invocation(runtime, request)))

    assert calls["pipeline"] == {
        "prompt": request.prompt,
        "negative_prompt": request.video.negative_prompt,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "generator": generator,
        "output_type": "pt",
    }
    assert calls["encode"] == (
        encoded_frames,
        str(target_path),
        {"fps": 24, "layout": "BFCHW", "value_range": "zero_to_one"},
    )
    assert calls["float"] is True
    assert calls["clamp"] == (0.0, 1.0)
    assert result.output.output == FileBackedGenerateOutput(
        path=str(target_path),
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=3,
        width=96,
        height=64,
        num_frames=17,
        fps=24.0,
        duration_s=17 / 24,
    )


@pytest.mark.parametrize(
    "target_factory",
    [
        lambda tmp_path: None,
        lambda tmp_path: FileOutputTarget(staging_path=str((tmp_path / "job.mp4.part").resolve())),
        lambda tmp_path: FileOutputTarget(
            staging_path=str((tmp_path / "missing.part.mp4").resolve())
        ),
        lambda tmp_path: FileOutputTarget(
            staging_path=str((tmp_path / "job.part.mp4").resolve()),
            mime_type="application/octet-stream",
        ),
    ],
)
def test_runner_rejects_missing_or_invalid_parent_target(target_factory, tmp_path):
    runtime = _runtime(tmp_path)
    target = target_factory(tmp_path)
    if target is not None and Path(target.staging_path).name == "job.mp4.part":
        Path(target.staging_path).touch()
    if target is not None and Path(target.staging_path).name == "job.part.mp4":
        Path(target.staging_path).touch()

    class UnexpectedPipe:
        def __call__(self, **kwargs):
            raise AssertionError("pipeline must not run for an invalid target")

    runner = ltx_2.LTX2PipelineRunner(UnexpectedPipe(), runtime.profile)

    with pytest.raises(ValueError):
        asyncio.run(runner.execute(_invocation(runtime, _request(runtime.profile, target=target))))


def test_runner_propagates_encoding_failure_without_fallback(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path)
    target_path = (tmp_path / "failed.part.mp4").resolve()
    target_path.touch()
    target = FileOutputTarget(staging_path=str(target_path))

    class FakePipe:
        def __call__(self, **kwargs):
            frames = SimpleNamespace(
                float=lambda: SimpleNamespace(clamp=lambda minimum, maximum: object())
            )
            return SimpleNamespace(frames=frames)

    def fail_encode(*args, **kwargs):
        raise VideoMediaEncodingError("codec failed")

    monkeypatch.setattr(ltx_2, "_seeded_generator", lambda seed: object())
    monkeypatch.setattr(ltx_2, "encode_tensor_to_mp4", fail_encode)
    runner = ltx_2.LTX2PipelineRunner(FakePipe(), runtime.profile)

    with pytest.raises(VideoMediaEncodingError, match="codec failed"):
        asyncio.run(runner.execute(_invocation(runtime, _request(runtime.profile, target=target))))


def test_adapter_loads_pinned_runtime_binding_with_host_components(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path)
    calls = {}

    class FakeManager:
        def __init__(self, root):
            calls["root"] = Path(root)

        def validate_binding(self, binding, *, validate_payload):
            calls["binding"] = binding
            validate_payload(binding.path)

    def fake_validate(source, profile, spec, path):
        calls["validated"] = (source, profile, spec, path)

    pipe = object()

    def fake_build(source, profile, **kwargs):
        calls["build"] = (source, profile, kwargs)
        return pipe

    monkeypatch.setattr(ltx_2, "ImmutableArtifactManager", FakeManager)
    monkeypatch.setattr(ltx_2, "validate_compiled_artifact", fake_validate)
    monkeypatch.setattr(ltx_2, "build_pipeline", fake_build)
    adapter = ltx_2.LTX2ServingStageAdapter()

    runners = asyncio.run(adapter.create_loaded_runners(runtime))

    assert tuple(runners) == ("pipeline",)
    assert calls["binding"] == runtime.artifacts.require("pipeline")
    assert calls["validated"][0] is runtime.source
    assert calls["validated"][3] == runtime.artifacts.require("pipeline").path
    assert calls["build"] == (
        runtime.source,
        runtime.profile,
        {
            "load": True,
            "skip_compile": True,
            "compiled_path_override": runtime.artifacts.require("pipeline").path,
            "host_components": True,
        },
    )
    assert runners["pipeline"].inner.pipe is pipe


def test_startup_smoke_uses_temporary_target_and_cleans_it(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path)
    adapter = ltx_2.LTX2ServingStageAdapter()
    adapter.active_runtime = runtime
    adapter.active_profile = runtime.profile
    request = adapter.smoke_request()
    target = Path(request.video.output_target.staging_path)
    parent = target.parent
    assert target.is_file()
    assert target.name.endswith(".part.mp4")
    target.write_bytes(b"mp4")
    output = FileBackedGenerateOutput(
        path=str(target),
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=3,
        width=runtime.profile.width,
        height=runtime.profile.height,
        num_frames=int(runtime.profile.num_frames),
        fps=float(runtime.profile.output_fps),
        duration_s=runtime.profile.num_frames / runtime.profile.output_fps,
    )
    calls = []

    def fake_validate(path, **kwargs):
        calls.append((Path(path), kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(ltx_2, "validate_mp4", fake_validate)

    adapter.validate_smoke_output(output)

    assert calls == [
        (
            target,
            {
                "expected_width": 96,
                "expected_height": 64,
                "expected_num_frames": 17,
                "expected_fps": 24,
                "require_silent": True,
            },
        )
    ]
    assert not parent.exists()
    assert adapter._smoke_target is None
