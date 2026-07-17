from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import difflet.serving.models._common as video_common
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError
from difflet.serving.models import wan
from difflet.serving.types import (
    ArtifactBinding,
    ArtifactPublishTarget,
    ArtifactSet,
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
from difflet.serving.video_media import VideoMediaMetadata


def _profile(
    tmp_path: Path,
    *,
    parallel: DiffletParallelConfig | None = None,
    host_vae: bool = True,
) -> ServingProfile:
    return ServingProfile(
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        model_type="wan",
        height=64,
        width=96,
        num_frames=5,
        parallel=parallel or DiffletParallelConfig(tp_degree=4),
        cache_dir=str(tmp_path / "cache"),
        dtype="bfloat16",
        output_modality="video",
        output_mime_type="video/mp4",
        output_fps=16,
        host_vae=host_vae,
    )


def _source(tmp_path: Path, *, commit: str = "a" * 40) -> ResolvedModelSource:
    return ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        requested_revision=None,
        pinned_model_path=str(tmp_path / "snapshots" / commit),
        resolved_source_id=commit,
    )


def _runtime(
    tmp_path: Path,
    monkeypatch,
    *,
    profile: ServingProfile | None = None,
) -> ResolvedRuntimeBundle:
    profile = profile or _profile(tmp_path)
    source = _source(tmp_path)
    monkeypatch.setattr(wan, "_torch_bfloat16", lambda: "bfloat16")
    monkeypatch.setattr(
        video_common,
        "resolve_available_neuron_core_ids",
        lambda *, required_num_cores: tuple(range(4, 4 + required_num_cores)),
    )
    spec = wan._compile_spec(source, profile)
    pipeline = wan._pipeline_definition()
    plan = wan._runtime_plan(profile, pipeline, spec)
    artifact_path = tmp_path / "published" / "generation"
    binding = ArtifactBinding(
        artifact_id="generation",
        path=artifact_path,
        manifest_path=artifact_path / "difflet_generation_manifest.json",
        identity=spec.identity,
        generation_id="g0000000000000001",
        content_digest="1" * 64,
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
    **video_overrides,
) -> DiffletGenerateRequest:
    video = VideoGenerateOptions(
        num_frames=int(profile.num_frames),
        fps=int(profile.output_fps),
        output_target=target,
        negative_prompt="blurred",
        **video_overrides,
    )
    return DiffletGenerateRequest(
        request_id="request-1",
        model=profile.model_id,
        prompt="a paper boat on water",
        height=profile.height,
        width=profile.width,
        num_inference_steps=3,
        guidance_scale=4.0,
        seed=42,
        output_format="mp4",
        video=video,
    )


def _invocation(runtime, request, stage_index, payload):
    return StageInvocation(
        request=request,
        stage=runtime.pipeline_definition.stages[stage_index],
        input=payload,
        context=WorkerRequestContext.with_timeout(request.request_id, 10.0),
    )


class _SymbolicTensor:
    def __init__(self, name: str, calls: list | None = None) -> None:
        self.name = name
        self.calls = calls if calls is not None else []

    def detach(self):
        self.calls.append((self.name, "detach"))
        return self

    def to(self, *args, **kwargs):
        self.calls.append((self.name, "to", args, kwargs))
        return _SymbolicTensor(f"to({self.name})", self.calls)

    def view(self, *shape):
        self.calls.append((self.name, "view", shape))
        return _SymbolicTensor(f"view({self.name})", self.calls)

    def clamp(self, minimum, maximum):
        self.calls.append((self.name, "clamp", minimum, maximum))
        return _SymbolicTensor(f"clamp({self.name})", self.calls)

    def __rtruediv__(self, other):
        return _SymbolicTensor(f"({other}/{self.name})", self.calls)

    def __truediv__(self, other):
        return _SymbolicTensor(f"({self.name}/{other.name})", self.calls)

    def __add__(self, other):
        return _SymbolicTensor(f"({self.name}+{other.name})", self.calls)


def _fake_torch(monkeypatch, calls: dict) -> ModuleType:
    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"
    torch.int32 = "int32"
    torch.int64 = "int64"

    class Generator:
        def manual_seed(self, seed):
            calls["seed"] = seed
            return self

    torch.Generator = Generator
    torch.tensor = lambda value, **kwargs: _SymbolicTensor(f"tensor({value})")
    torch.no_grad = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_registry_default_shape_matches_cli_and_lower_application_defaults():
    entry = wan.resolve_model(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        model_type="wan",
    )

    assert entry.default_shape == {"height": 480, "width": 832, "num_frames": 9}
    assert entry.default_parallel == DiffletParallelConfig(tp_degree=4)


def test_compile_identity_binds_commit_shape_topology_and_component(monkeypatch, tmp_path):
    monkeypatch.setattr(wan, "_torch_bfloat16", lambda: "bfloat16")
    profile = _profile(tmp_path)
    first = wan._compile_spec(_source(tmp_path, commit="a" * 40), profile)
    other_commit = wan._compile_spec(_source(tmp_path, commit="b" * 40), profile)
    other_shape = wan._compile_spec(
        _source(tmp_path, commit="a" * 40),
        replace(profile, num_frames=9),
    )

    assert first.artifact_id == first.component_id == "generation"
    assert first.identity != other_commit.identity
    assert first.identity != other_shape.identity
    identity = json.loads(first.identity.canonical_cache_inputs_json)
    assert identity["component_id"] == "generation"
    assert identity["cache_inputs"]["revision"] == "a" * 40
    assert identity["cache_inputs"]["shape"] == {
        "height": 64,
        "num_frames": 5,
        "width": 96,
    }
    assert identity["cache_inputs"]["application_kwargs"] == {
        "batch_size": 1,
        "enable_text_encoder": True,
        "enable_transformer": True,
        "enable_transformer_2": False,
        "enable_vae_decoder": False,
        "text_seq_len": 512,
    }
    assert str(tmp_path) not in first.identity.canonical_cache_inputs_json.decode()


def test_compile_artifact_checks_target_environment_and_payload(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    spec = runtime.compile_specs[0]
    target = ArtifactPublishTarget(
        artifact_id=spec.artifact_id,
        identity=spec.identity,
        identity_root=tmp_path / "identity",
        staging_path=tmp_path / "staging",
    )
    calls = {}

    class FakeApplication:
        def compile(self, path):
            calls["compile"] = path
            component = Path(path) / "transformer"
            component.mkdir(parents=True)
            (component / "model.pt").write_bytes(b"model")
            (component / "neuron_config.json").write_text("{}")

        def has_compiled_artifacts(self, path):
            calls["validated"] = path
            return True

        def components(self):
            return [SimpleNamespace(name="transformer", artifact_name=None)]

    @contextmanager
    def fake_environment(world_size, **kwargs):
        calls["environment"] = (world_size, kwargs)
        yield

    monkeypatch.setattr(wan, "_build_application", lambda source, profile: FakeApplication())
    monkeypatch.setattr(wan, "serving_compile_environment", fake_environment)

    wan._compile_artifact(runtime.source, runtime.profile, spec, target)

    assert calls == {
        "environment": (4, {}),
        "compile": str(target.staging_path),
        "validated": str(target.staging_path),
    }
    with pytest.raises(ValueError, match="does not match"):
        wan._compile_artifact(
            runtime.source,
            runtime.profile,
            spec,
            replace(target, artifact_id="other"),
        )


def test_artifact_validation_rejects_unknown_or_incomplete_component(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    spec = runtime.compile_specs[0]

    monkeypatch.setattr(
        wan,
        "_build_application",
        lambda source, profile: SimpleNamespace(has_compiled_artifacts=lambda path: False),
    )
    with pytest.raises(ValueError, match="incomplete"):
        wan._validate_artifact(runtime.source, runtime.profile, spec, tmp_path / "artifact")
    with pytest.raises(ValueError, match="unknown.*component"):
        wan._validate_artifact(
            runtime.source,
            runtime.profile,
            replace(spec, component_id="other"),
            tmp_path / "artifact",
        )


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"width": 88}, "divisible by 16"),
        ({"num_frames": 6}, "4n\\+1"),
        ({"dtype": "float32"}, "bfloat16"),
        ({"output_modality": "image"}, "video/mp4"),
        ({"output_mime_type": "image/png"}, "video/mp4"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "tp_degree=4"),
        (
            {"parallel": DiffletParallelConfig(tp_degree=2, cp_degree=2)},
            "tp_degree=4",
        ),
        ({"parallel": DiffletParallelConfig(tp_degree=4, dp_degree=2)}, "dp_degree=1"),
    ],
)
def test_profile_rejects_shapes_and_runtime_contract_mismatches(changed, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        wan._validate_profile(replace(_profile(tmp_path), **changed))


def test_profile_allows_lower_layer_sequence_parallelism(tmp_path):
    wan._validate_profile(
        _profile(tmp_path, parallel=DiffletParallelConfig(tp_degree=4, sp_enabled=True))
    )


def test_artifact_validation_rejects_empty_model_payload(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    spec = runtime.compile_specs[0]
    root = tmp_path / "artifact"
    component = root / "transformer"
    component.mkdir(parents=True)
    (component / "model.pt").touch()
    (component / "neuron_config.json").write_text("{}")
    application = SimpleNamespace(
        has_compiled_artifacts=lambda path: True,
        components=lambda: [SimpleNamespace(name="transformer", artifact_name=None)],
    )
    monkeypatch.setattr(wan, "_build_application", lambda source, profile: application)

    with pytest.raises(ValueError, match="incomplete"):
        wan._validate_artifact(runtime.source, runtime.profile, spec, root)

    (component / "model.pt").write_bytes(b"model")
    wan._validate_artifact(runtime.source, runtime.profile, spec, root)


def test_runtime_plan_places_prompt_and_denoiser_on_neuron_and_decoder_on_host(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    plan = runtime.runtime_plan

    assert plan.mode == "resident"
    assert plan.environment.available_core_ids == (4, 5, 6, 7)
    assert len(plan.allocations) == 1
    assert plan.allocations[0].effective_num_cores == 4
    assert [stage.stage_id for stage in plan.stages] == [
        "prompt_encoder",
        "denoiser",
        "decoder",
    ]
    assert [stage.placement for stage in plan.stages] == ["neuron", "neuron", "host"]
    assert [stage.artifact_id for stage in plan.stages] == [
        "generation",
        "generation",
        None,
    ]
    assert plan.stages[0].topology.world_size == 4


def test_request_validator_checks_profile_features_and_prompt_bucket(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    validator = wan.WanServingRequestValidator(runtime)
    token_lengths = iter((10, 20, 513))
    validator._tokenizer = lambda *args, **kwargs: SimpleNamespace(
        input_ids=SimpleNamespace(shape=(1, next(token_lengths)))
    )
    request = _request(runtime.profile)

    validator.validate(request)
    with pytest.raises(DiffletServingError) as prompt_error:
        validator.validate(replace(request, video=replace(request.video, negative_prompt=None)))
    assert prompt_error.value.code == "prompt_too_long"

    for changed, code in (
        (replace(request, model="other/model"), "profile_mismatch"),
        (replace(request, output_format="png"), "invalid_extra_body"),
        (replace(request, width=request.width + 2), "profile_mismatch"),
        (
            replace(request, video=replace(request.video, guidance_scale_2=3.0)),
            "invalid_extra_body",
        ),
        (
            replace(request, video=replace(request.video, boundary_ratio=0.5)),
            "invalid_extra_body",
        ),
        (
            replace(request, video=replace(request.video, flow_shift=4.0)),
            "invalid_extra_body",
        ),
        (
            replace(request, video=replace(request.video, true_cfg_scale=2.0)),
            "invalid_extra_body",
        ),
    ):
        fresh = wan.WanServingRequestValidator(runtime)
        fresh._tokenizer = lambda *args, **kwargs: SimpleNamespace(
            input_ids=SimpleNamespace(shape=(1, 10))
        )
        with pytest.raises(DiffletServingError) as exc:
            fresh.validate(changed)
        assert exc.value.code == code


def test_prompt_and_denoiser_stages_preserve_cfg_embeddings_seed_and_shape(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    target_path = (tmp_path / "job.part.mp4").resolve()
    target_path.touch()
    request = _request(
        runtime.profile,
        target=FileOutputTarget(staging_path=str(target_path)),
    )
    calls = {"encoded": []}
    latent = object()

    class FakePipeline:
        def encode_prompt(self, *, prompt):
            calls["encoded"].append(prompt)
            return f"embeds:{prompt}"

        def __call__(self, **kwargs):
            calls["denoiser"] = kwargs
            return SimpleNamespace(latents=latent)

    adapter = wan.WanServingStageAdapter()
    adapter.application = SimpleNamespace(pipeline=FakePipeline())
    _fake_torch(monkeypatch, calls)
    prompt_runner = wan.WanPromptEncoderStageRunner(adapter)
    denoiser_runner = wan.WanDenoiserStageRunner(adapter)

    prompt_result = asyncio.run(
        prompt_runner.execute(_invocation(runtime, request, 0, wan.WanInitialPayload()))
    )
    latent_result = asyncio.run(
        denoiser_runner.execute(_invocation(runtime, request, 1, prompt_result.output))
    )

    assert calls["encoded"] == [request.prompt, request.video.negative_prompt]
    assert calls["seed"] == 42
    denoiser = calls["denoiser"]
    assert denoiser["prompt_embeds"] == f"embeds:{request.prompt}"
    assert denoiser["negative_prompt_embeds"] == "embeds:blurred"
    assert (denoiser["height"], denoiser["width"], denoiser["num_frames"]) == (
        64,
        96,
        5,
    )
    assert denoiser["num_inference_steps"] == 3
    assert denoiser["guidance_scale"] == 4.0
    assert denoiser["output_type"] == "latent"
    assert latent_result.output.latents is latent


def test_host_decoder_normalizes_latents_on_cpu_and_encodes_exact_target(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    target_path = (tmp_path / "job.part.mp4").resolve()
    target_path.touch()
    request = _request(
        runtime.profile,
        target=FileOutputTarget(staging_path=str(target_path)),
    )
    calls = {}
    torch = _fake_torch(monkeypatch, calls)
    tensor_calls = []
    latents = _SymbolicTensor("latents", tensor_calls)
    frames = _SymbolicTensor("frames", tensor_calls)

    class FakeVAE:
        config = SimpleNamespace(latents_mean=(1.0, 2.0), latents_std=(0.5, 0.25), z_dim=2)

        def decode(self, value, *, return_dict):
            calls["decode"] = (value, return_dict)
            return (frames,)

    expected_output = FileBackedGenerateOutput(
        path=str(target_path),
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=3,
        width=96,
        height=64,
        num_frames=5,
        fps=16.0,
        duration_s=5 / 16,
    )

    def fake_encode(tensor, encoded_request, **kwargs):
        calls["encode"] = (tensor, encoded_request, kwargs)
        return expected_output

    monkeypatch.setattr(wan, "encode_video_tensor", fake_encode)
    adapter = wan.WanServingStageAdapter()
    adapter.vae = FakeVAE()
    runner = wan.WanHostDecoderStageRunner(adapter)

    result = asyncio.run(
        runner.execute(_invocation(runtime, request, 2, wan.WanLatentPayload(latents)))
    )

    latent_to = next(call for call in tensor_calls if call[:2] == ("latents", "to"))
    assert latent_to[2:] == ((), {"device": "cpu", "dtype": torch.float32})
    decoded, return_dict = calls["decode"]
    assert "latents" in decoded.name and "tensor" in decoded.name
    assert return_dict is False
    encoded_frames, encoded_request, encode_kwargs = calls["encode"]
    assert encoded_frames.name.startswith("clamp(")
    assert encoded_request is request
    assert encode_kwargs == {"layout": "BCTHW", "value_range": "minus_one_to_one"}
    assert result.output.output == expected_output


def test_initial_payload_requires_precreated_parent_owned_mp4_target(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    adapter = wan.WanServingStageAdapter()

    with pytest.raises(ValueError, match="output target"):
        adapter.initial_payload(_request(runtime.profile, target=None))
    missing = FileOutputTarget(staging_path=str((tmp_path / "missing.part.mp4").resolve()))
    with pytest.raises(ValueError, match="pre-created"):
        adapter.initial_payload(_request(runtime.profile, target=missing))

    path = (tmp_path / "valid.part.mp4").resolve()
    path.touch()
    assert isinstance(
        adapter.initial_payload(
            _request(runtime.profile, target=FileOutputTarget(staging_path=str(path)))
        ),
        wan.WanInitialPayload,
    )


def test_adapter_validates_binding_before_loading_and_shutdown_releases_state(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = {}

    class FakeManager:
        def __init__(self, root):
            calls["root"] = Path(root)

        def validate_binding(self, binding, *, validate_payload):
            calls["binding"] = binding
            validate_payload(binding.path)

    class FakeApplication:
        def load(self, path, **kwargs):
            calls["load"] = (path, kwargs)

    application = FakeApplication()
    vae = object()
    monkeypatch.setattr(wan, "ImmutableArtifactManager", FakeManager)
    monkeypatch.setattr(
        wan,
        "_validate_artifact",
        lambda source, profile, spec, path: calls.setdefault(
            "validated", (source, profile, spec, path)
        ),
    )
    monkeypatch.setattr(wan, "_build_application", lambda source, profile: application)
    monkeypatch.setattr(wan, "_load_host_vae", lambda path: vae)
    adapter = wan.WanServingStageAdapter()

    runners = asyncio.run(adapter.create_loaded_runners(runtime))

    binding = runtime.artifacts.require("generation")
    assert tuple(runners) == ("prompt_encoder", "denoiser", "decoder")
    assert calls["binding"] == binding
    assert calls["validated"][3] == binding.path
    assert calls["load"] == (
        str(binding.path),
        {"start_rank_id": 0, "local_ranks_size": 4, "skip_warmup": True},
    )
    assert adapter.application is application
    assert adapter.vae is vae

    asyncio.run(adapter.shutdown())
    assert adapter.application is None
    assert adapter.vae is None
    assert adapter.runtime is None
    assert adapter.profile is None


def test_startup_smoke_is_target_bound_reentrant_and_always_cleans(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    adapter = wan.WanServingStageAdapter()
    adapter.profile = runtime.profile

    first_request = adapter.smoke_request()
    first_target = Path(first_request.video.output_target.staging_path)
    first_parent = first_target.parent
    second_request = adapter.smoke_request()
    second_target = Path(second_request.video.output_target.staging_path)
    second_parent = second_target.parent
    assert not first_parent.exists()

    wrong = tmp_path / "wrong.mp4"
    wrong.write_bytes(b"mp4")
    output = FileBackedGenerateOutput(
        path=str(wrong),
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=3,
        width=96,
        height=64,
        num_frames=5,
        fps=16.0,
        duration_s=5 / 16,
    )
    monkeypatch.setattr(
        "difflet.serving.video_media.validate_mp4",
        lambda path, **kwargs: VideoMediaMetadata(
            width=96,
            height=64,
            num_frames=5,
            fps=16.0,
            duration_s=5 / 16,
            codec_name="h264",
            pixel_format="yuv420p",
            size_bytes=3,
        ),
    )

    with pytest.raises(RuntimeError, match="temporary target"):
        adapter.validate_smoke_output(output)
    assert not second_parent.exists()
    assert adapter._smoke is None

    success_request = adapter.smoke_request()
    success_target = Path(success_request.video.output_target.staging_path)
    success_target.write_bytes(b"mp4")
    success_output = replace(output, path=str(success_target))

    adapter.validate_smoke_output(success_output)

    assert not success_target.parent.exists()
    assert adapter._smoke is None


def test_host_decoder_propagates_encoding_failure(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    target = (tmp_path / "job.part.mp4").resolve()
    target.touch()
    request = _request(
        runtime.profile,
        target=FileOutputTarget(staging_path=str(target)),
    )
    _fake_torch(monkeypatch, {})
    adapter = wan.WanServingStageAdapter()
    adapter.vae = SimpleNamespace(
        config=SimpleNamespace(latents_mean=(0.0,), latents_std=(1.0,), z_dim=1),
        decode=lambda value, return_dict: (_SymbolicTensor("frames"),),
    )
    monkeypatch.setattr(
        wan,
        "encode_video_tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("codec failed")),
    )

    with pytest.raises(RuntimeError, match="codec failed"):
        asyncio.run(
            wan.WanHostDecoderStageRunner(adapter).execute(
                _invocation(
                    runtime,
                    request,
                    2,
                    wan.WanLatentPayload(_SymbolicTensor("latents")),
                )
            )
        )
