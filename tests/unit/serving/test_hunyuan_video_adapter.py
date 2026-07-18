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
from difflet.registry import resolve_model
from difflet.serving.errors import DiffletServingError
from difflet.serving.models import hunyuan_video
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
    clip_placement: str | None = None,
) -> ServingProfile:
    return ServingProfile(
        model_id="hunyuanvideo-community/HunyuanVideo",
        model_type="hunyuan_video",
        height=64,
        width=96,
        num_frames=5,
        parallel=parallel or DiffletParallelConfig(tp_degree=4),
        cache_dir=str(tmp_path / "cache"),
        dtype="bfloat16",
        output_modality="video",
        output_mime_type="video/mp4",
        output_fps=24,
        host_vae=host_vae,
        clip_placement=clip_placement,
    )


def _source(tmp_path: Path, *, commit: str = "a" * 40) -> ResolvedModelSource:
    return ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id="hunyuanvideo-community/HunyuanVideo",
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
    monkeypatch.setattr(
        hunyuan_video,
        "toolchain_versions",
        lambda: {"python": "3.10", "neuronx-cc": "test"},
    )
    monkeypatch.setattr(
        video_common,
        "resolve_available_neuron_core_ids",
        lambda *, required_num_cores: tuple(range(8, 8 + required_num_cores)),
    )
    specs = hunyuan_video._compile_specs(source, profile)
    pipeline = hunyuan_video._pipeline_definition(profile)
    plan = hunyuan_video._runtime_plan(profile, specs)
    bindings = tuple(
        ArtifactBinding(
            artifact_id=spec.artifact_id,
            path=tmp_path / "published" / spec.artifact_id,
            manifest_path=(
                tmp_path / "published" / spec.artifact_id / "difflet_generation_manifest.json"
            ),
            identity=spec.identity,
            generation_id=f"g-{spec.artifact_id}",
            content_digest="2" * 64,
        )
        for spec in specs
    )
    return ResolvedRuntimeBundle(
        profile=profile,
        source=source,
        pipeline_definition=pipeline,
        runtime_plan=plan,
        compile_specs=specs,
        artifacts=ArtifactSet(bindings),
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
        **video_overrides,
    )
    return DiffletGenerateRequest(
        request_id="request-1",
        model=profile.model_id,
        prompt="a paper boat on water",
        height=profile.height,
        width=profile.width,
        num_inference_steps=4,
        guidance_scale=6.0,
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


class _TraceTensor:
    def __init__(
        self,
        name: str,
        *,
        shape: tuple[int, ...] = (),
        calls: list | None = None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.calls = calls if calls is not None else []

    def detach(self):
        self.calls.append((self.name, "detach"))
        return self

    def to(self, *args, **kwargs):
        self.calls.append((self.name, "to", args, kwargs))
        return _TraceTensor(f"to({self.name})", shape=self.shape, calls=self.calls)

    def cpu(self):
        self.calls.append((self.name, "cpu"))
        return _TraceTensor(f"cpu({self.name})", shape=self.shape, calls=self.calls)

    def reshape(self, *shape):
        self.calls.append((self.name, "reshape", shape))
        return _TraceTensor(f"reshape({self.name})", shape=shape, calls=self.calls)

    def unsqueeze(self, dimension):
        self.calls.append((self.name, "unsqueeze", dimension))
        return _TraceTensor(f"unsqueeze({self.name})", calls=self.calls)

    def clamp(self, minimum, maximum):
        self.calls.append((self.name, "clamp", minimum, maximum))
        return _TraceTensor(f"clamp({self.name})", shape=self.shape, calls=self.calls)

    def __getitem__(self, item):
        self.calls.append((self.name, "getitem", item))
        return _TraceTensor(f"slice({self.name})", calls=self.calls)

    def __truediv__(self, value):
        return _TraceTensor(f"({self.name}/{value})", shape=self.shape, calls=self.calls)


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

    def randn(*shape, **kwargs):
        calls["randn"] = (shape, kwargs)
        return _TraceTensor("latents", shape=shape)

    def zeros(*shape, **kwargs):
        calls.setdefault("zeros", []).append((shape, kwargs))
        return _TraceTensor("zeros", shape=tuple(shape))

    def full(shape, value, **kwargs):
        calls["full"] = (shape, value, kwargs)
        return _TraceTensor("guidance", shape=tuple(shape))

    torch.Generator = Generator
    torch.randn = randn
    torch.zeros = zeros
    torch.full = full
    torch.arange = lambda *args, **kwargs: _TraceTensor("position_ids")
    torch.tensor = lambda value, **kwargs: _TraceTensor("sampling_params")
    torch.no_grad = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_registry_default_shape_matches_cli_and_lower_application_defaults():
    entry = resolve_model("hunyuanvideo-community/HunyuanVideo")

    assert entry.default_shape == {"height": 320, "width": 512, "num_frames": 61}
    assert entry.default_parallel == DiffletParallelConfig(tp_degree=4)


def test_compile_identities_bind_commit_toolchain_components_and_denoiser_shape(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        hunyuan_video,
        "toolchain_versions",
        lambda: {"python": "3.10", "neuronx-cc": "test"},
    )
    profile = _profile(tmp_path)
    first = hunyuan_video._compile_specs(_source(tmp_path, commit="a" * 40), profile)
    other_commit = hunyuan_video._compile_specs(_source(tmp_path, commit="b" * 40), profile)
    other_shape = hunyuan_video._compile_specs(
        _source(tmp_path, commit="a" * 40),
        replace(profile, num_frames=9),
    )

    assert [(spec.artifact_id, spec.component_id) for spec in first] == [
        ("llama", "llama"),
        ("denoiser", "denoiser"),
    ]
    assert first[0].identity != other_commit[0].identity
    assert first[1].identity != other_commit[1].identity
    assert first[0].identity == other_shape[0].identity
    assert first[1].identity != other_shape[1].identity
    llama = json.loads(first[0].identity.canonical_cache_inputs_json)
    denoiser = json.loads(first[1].identity.canonical_cache_inputs_json)
    assert llama["resolved_source_id"] == "a" * 40
    assert llama["sequence_length"] == 351
    assert llama["tensor_capture"] == "layers.29"
    assert llama["virtual_core_size"] == 2
    assert denoiser["height"] == 64
    assert denoiser["width"] == 96
    assert denoiser["num_frames"] == 5
    assert denoiser["text_seq_len"] == 256
    assert str(tmp_path) not in first[0].identity.canonical_cache_inputs_json.decode()


@pytest.mark.parametrize(
    ("clip_placement", "host_vae", "expected"),
    [
        ("host", True, ["llama", "denoiser"]),
        ("neuron", True, ["clip", "llama", "denoiser"]),
        ("host", False, ["llama", "denoiser", "decoder"]),
        ("neuron", False, ["clip", "llama", "denoiser", "decoder"]),
    ],
)
def test_compile_specs_include_only_selected_neuron_placement_artifacts(
    monkeypatch,
    tmp_path,
    clip_placement,
    host_vae,
    expected,
):
    monkeypatch.setattr(
        hunyuan_video,
        "toolchain_versions",
        lambda: {"python": "3.10", "neuronx-cc": "test"},
    )
    profile = _profile(
        tmp_path,
        clip_placement=clip_placement,
        host_vae=host_vae,
    )

    specs = hunyuan_video._compile_specs(_source(tmp_path), profile)

    assert [spec.artifact_id for spec in specs] == expected


def test_placement_profiles_keep_non_selected_artifact_identities_identical(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        hunyuan_video,
        "toolchain_versions",
        lambda: {"python": "3.10", "neuronx-cc": "test"},
    )
    source = _source(tmp_path)
    profiles = (
        _profile(tmp_path, clip_placement="host", host_vae=True),
        _profile(tmp_path, clip_placement="neuron", host_vae=True),
        _profile(tmp_path, clip_placement="host", host_vae=False),
        _profile(tmp_path, clip_placement="neuron", host_vae=False),
    )

    identities = [
        {spec.artifact_id: spec.identity for spec in hunyuan_video._compile_specs(source, profile)}
        for profile in profiles
    ]

    assert len({item["llama"] for item in identities}) == 1
    assert len({item["denoiser"] for item in identities}) == 1
    assert identities[1]["clip"] == identities[3]["clip"]
    assert identities[2]["decoder"] == identities[3]["decoder"]


def test_runtime_plan_places_clip_and_decoder_on_host_with_one_shared_neuron_allocation(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    plan = runtime.runtime_plan

    assert plan.environment.available_core_ids == (8, 9, 10, 11)
    assert plan.environment.virtual_core_size_override == 2
    assert len(plan.allocations) == 1
    allocation = plan.allocations[0]
    assert allocation.effective_num_cores == 4
    assert allocation.effective_virtual_core_size == 2
    assert [stage.stage_id for stage in plan.stages] == [
        "clip",
        "llama",
        "denoiser",
        "decoder",
    ]
    assert [stage.placement for stage in plan.stages] == [
        "host",
        "neuron",
        "neuron",
        "host",
    ]
    assert [stage.artifact_id for stage in plan.stages] == [
        None,
        "llama",
        "denoiser",
        None,
    ]


@pytest.mark.parametrize(
    ("clip_placement", "host_vae", "placements", "artifacts"),
    [
        (
            "neuron",
            True,
            ["neuron", "neuron", "neuron", "host"],
            ["clip", "llama", "denoiser", None],
        ),
        (
            "host",
            False,
            ["host", "neuron", "neuron", "neuron"],
            [None, "llama", "denoiser", "decoder"],
        ),
        (
            "neuron",
            False,
            ["neuron", "neuron", "neuron", "neuron"],
            ["clip", "llama", "denoiser", "decoder"],
        ),
    ],
)
def test_runtime_plan_binds_selected_placement_to_one_resident_allocation(
    monkeypatch,
    tmp_path,
    clip_placement,
    host_vae,
    placements,
    artifacts,
):
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        profile=_profile(
            tmp_path,
            clip_placement=clip_placement,
            host_vae=host_vae,
        ),
    )

    assert [stage.placement for stage in runtime.runtime_plan.stages] == placements
    assert [stage.artifact_id for stage in runtime.runtime_plan.stages] == artifacts
    for stage in runtime.runtime_plan.stages:
        if stage.placement == "neuron":
            assert stage.allocation_id == "hunyuan-video-resident"
    if clip_placement == "neuron":
        assert runtime.runtime_plan.stages[0].topology.tp_degree == 1
        assert runtime.runtime_plan.stages[0].topology.world_size == 4
    if not host_vae:
        assert runtime.runtime_plan.stages[-1].topology.tp_degree == 1
        assert runtime.runtime_plan.stages[-1].topology.world_size == 4


def test_pipeline_definition_switches_only_selected_stage_runner_bindings(tmp_path):
    baseline = hunyuan_video._pipeline_definition(
        _profile(tmp_path, clip_placement="host", host_vae=True)
    )
    neuron = hunyuan_video._pipeline_definition(
        _profile(tmp_path, clip_placement="neuron", host_vae=False)
    )

    assert baseline.stages[1:3] == neuron.stages[1:3]
    assert baseline.stages[0].runner_factory.endswith("HunyuanVideoHostClipStageRunner")
    assert neuron.stages[0].runner_factory.endswith("HunyuanVideoNeuronClipStageRunner")
    assert baseline.stages[-1].runner_factory.endswith("HunyuanVideoHostDecoderStageRunner")
    assert neuron.stages[-1].runner_factory.endswith("HunyuanVideoNeuronDecoderStageRunner")


@pytest.mark.parametrize(
    "parallel,match",
    [
        (DiffletParallelConfig(tp_degree=2, cp_degree=2), "cp_degree=1"),
        (DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True), "CFG"),
    ],
)
def test_profile_rejects_unsupported_parallel_axes(parallel, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        hunyuan_video._validate_profile(_profile(tmp_path, parallel=parallel))


@pytest.mark.parametrize(
    ("changed", "match"),
    [
        ({"width": 88}, "divisible by 16"),
        ({"num_frames": 6}, "4n\\+1"),
        ({"dtype": "float32"}, "bfloat16"),
        ({"output_modality": "image"}, "video/mp4"),
        ({"output_mime_type": "image/png"}, "video/mp4"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "tp_degree=4"),
        ({"parallel": DiffletParallelConfig(tp_degree=4, dp_degree=2)}, "dp_degree=1"),
    ],
)
def test_profile_rejects_shapes_and_runtime_contract_mismatches(changed, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        hunyuan_video._validate_profile(replace(_profile(tmp_path), **changed))


def test_profile_allows_lower_layer_sequence_parallelism(tmp_path):
    hunyuan_video._validate_profile(
        _profile(tmp_path, parallel=DiffletParallelConfig(tp_degree=4, sp_enabled=True))
    )


def test_compile_artifact_routes_llama_and_denoiser_under_vcore2_environment(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = []

    class FakeApplication:
        def __init__(self, component):
            self.component = component

        def compile(self, path):
            calls.append((self.component, "compile", path))

    @contextmanager
    def fake_environment(world_size, **kwargs):
        calls.append(("environment", world_size, kwargs))
        yield

    monkeypatch.setattr(hunyuan_video, "serving_compile_environment", fake_environment)
    monkeypatch.setattr(
        hunyuan_video,
        "_build_llama_app",
        lambda source, profile, **kwargs: calls.append(("llama_build", kwargs))
        or FakeApplication("llama"),
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_build_denoiser",
        lambda source, profile: FakeApplication("denoiser"),
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_validate_artifact",
        lambda source, profile, spec, path: calls.append((spec.component_id, "validate", path)),
    )

    for spec in runtime.compile_specs:
        target = ArtifactPublishTarget(
            artifact_id=spec.artifact_id,
            identity=spec.identity,
            identity_root=tmp_path / spec.artifact_id,
            staging_path=tmp_path / f"staging-{spec.artifact_id}",
        )
        hunyuan_video._compile_artifact(
            runtime.source,
            runtime.profile,
            spec,
            target,
        )

    assert calls.count(("environment", 4, {"virtual_core_size": 2})) == 2
    assert (
        "llama_build",
        {"compiled_path": tmp_path / "staging-llama", "for_compile": True},
    ) in calls
    assert ("llama", "compile", str(tmp_path / "staging-llama")) in calls
    assert ("denoiser", "compile", str(tmp_path / "staging-denoiser")) in calls
    assert ("llama", "validate", tmp_path / "staging-llama") in calls
    assert ("denoiser", "validate", tmp_path / "staging-denoiser") in calls

    llama_spec = runtime.require_compile_spec("llama")
    with pytest.raises(ValueError, match="does not match"):
        hunyuan_video._compile_artifact(
            runtime.source,
            runtime.profile,
            llama_spec,
            ArtifactPublishTarget(
                artifact_id="other",
                identity=llama_spec.identity,
                identity_root=tmp_path / "bad",
                staging_path=tmp_path / "bad-staging",
            ),
        )


def test_compile_artifact_routes_experimental_clip_and_decoder(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        profile=_profile(tmp_path, clip_placement="neuron", host_vae=False),
    )
    calls = []

    class FakeApplication:
        def __init__(self, component):
            self.component = component

        def compile(self, path):
            calls.append((self.component, "compile", path))

    @contextmanager
    def fake_environment(world_size, **kwargs):
        calls.append(("environment", world_size, kwargs))
        yield

    monkeypatch.setattr(hunyuan_video, "serving_compile_environment", fake_environment)
    monkeypatch.setattr(
        hunyuan_video,
        "_build_clip_app",
        lambda source, profile: FakeApplication("clip"),
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_build_llama_app",
        lambda source, profile, **kwargs: FakeApplication("llama"),
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_build_denoiser",
        lambda source, profile: FakeApplication("denoiser"),
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_build_vae_decoder",
        lambda source, profile: FakeApplication("decoder"),
    )
    monkeypatch.setattr(hunyuan_video, "_validate_artifact", lambda *args: None)

    for spec in runtime.compile_specs:
        target = ArtifactPublishTarget(
            artifact_id=spec.artifact_id,
            identity=spec.identity,
            identity_root=tmp_path / spec.artifact_id,
            staging_path=tmp_path / f"staging-{spec.artifact_id}",
        )
        hunyuan_video._compile_artifact(runtime.source, runtime.profile, spec, target)

    assert calls.count(("environment", 4, {"virtual_core_size": 2})) == 4
    assert ("clip", "compile", str(tmp_path / "staging-clip")) in calls
    assert ("decoder", "compile", str(tmp_path / "staging-decoder")) in calls


def test_artifact_validation_checks_nxdi_files_and_denoiser_components(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    llama_spec = runtime.require_compile_spec("llama")
    denoiser_spec = runtime.require_compile_spec("denoiser")
    llama_path = tmp_path / "llama"
    llama_path.mkdir()

    with pytest.raises(ValueError, match="Llama artifact is incomplete"):
        hunyuan_video._validate_artifact(runtime.source, runtime.profile, llama_spec, llama_path)
    (llama_path / "model.pt").write_bytes(b"model")
    (llama_path / "neuron_config.json").write_text("{}")
    hunyuan_video._validate_artifact(runtime.source, runtime.profile, llama_spec, llama_path)

    monkeypatch.setattr(
        hunyuan_video,
        "_build_denoiser",
        lambda source, profile: SimpleNamespace(has_compiled_artifacts=lambda path: False),
    )
    with pytest.raises(ValueError, match="denoiser artifact is incomplete"):
        hunyuan_video._validate_artifact(
            runtime.source,
            runtime.profile,
            denoiser_spec,
            tmp_path / "denoiser",
        )


def test_artifact_validation_checks_experimental_clip_and_decoder_components(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        profile=_profile(tmp_path, clip_placement="neuron", host_vae=False),
    )
    clip_spec = runtime.require_compile_spec("clip")
    decoder_spec = runtime.require_compile_spec("decoder")
    clip_path = tmp_path / "clip"
    clip_path.mkdir()

    with pytest.raises(ValueError, match="CLIP artifact is incomplete"):
        hunyuan_video._validate_artifact(
            runtime.source,
            runtime.profile,
            clip_spec,
            clip_path,
        )
    (clip_path / "model.pt").write_bytes(b"model")
    (clip_path / "neuron_config.json").write_text("{}")
    hunyuan_video._validate_artifact(
        runtime.source,
        runtime.profile,
        clip_spec,
        clip_path,
    )

    monkeypatch.setattr(
        hunyuan_video,
        "_build_vae_decoder",
        lambda source, profile: SimpleNamespace(has_compiled_artifacts=lambda path: False),
    )
    with pytest.raises(ValueError, match="decoder artifact is incomplete"):
        hunyuan_video._validate_artifact(
            runtime.source,
            runtime.profile,
            decoder_spec,
            tmp_path / "decoder",
        )


def test_llama_builder_preserves_cli_sharded_checkpoint_load_contract(
    monkeypatch,
    tmp_path,
):
    profile = _profile(tmp_path)
    source = _source(tmp_path)
    calls = {}
    torch = _fake_torch(monkeypatch, calls)

    config_module = ModuleType("neuronx_distributed_inference.models.config")

    class NeuronConfig:
        def __init__(self, **kwargs):
            calls["neuron_config"] = kwargs
            self.kwargs = kwargs

    class TensorCaptureConfig:
        def __init__(self, **kwargs):
            calls["capture"] = kwargs

    config_module.NeuronConfig = NeuronConfig
    config_module.TensorCaptureConfig = TensorCaptureConfig

    llama_module = ModuleType("neuronx_distributed_inference.models.llama.modeling_llama")

    class FakeConfigClass:
        def __init__(self, neuron_config, *, load_config):
            calls["model_config"] = (neuron_config, load_config)

    class NeuronLlamaForCausalLM:
        @classmethod
        def get_config_cls(cls):
            return FakeConfigClass

        def __init__(self, path, config):
            calls["app"] = (path, config)

    llama_module.NeuronLlamaForCausalLM = NeuronLlamaForCausalLM
    adapter_module = ModuleType("neuronx_distributed_inference.utils.hf_adapter")
    adapter_module.load_pretrained_config = lambda *, hf_config: (
        "loaded",
        hf_config,
    )
    transformers = ModuleType("transformers")
    hf_config = SimpleNamespace(pad_token_id=None, tie_word_embeddings=False)
    transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda path: hf_config)
    for name, module in (
        ("neuronx_distributed_inference.models.config", config_module),
        (
            "neuronx_distributed_inference.models.llama.modeling_llama",
            llama_module,
        ),
        ("neuronx_distributed_inference.utils.hf_adapter", adapter_module),
        ("transformers", transformers),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    compiled = tmp_path / "compiled"
    app = hunyuan_video._build_llama_app(
        source,
        profile,
        compiled_path=compiled,
        for_compile=False,
    )

    assert isinstance(app, NeuronLlamaForCausalLM)
    assert calls["neuron_config"]["tp_degree"] == 4
    assert calls["neuron_config"]["seq_len"] == 351
    assert calls["neuron_config"]["torch_dtype"] == torch.bfloat16
    assert calls["neuron_config"]["save_sharded_checkpoint"] is False
    assert calls["capture"] == {"modules_to_capture": ["layers.29"]}
    assert hf_config.pad_token_id == 0
    assert hf_config.tie_word_embeddings is True

    weights = compiled / "weights"
    weights.mkdir(parents=True)
    for rank in range(4):
        (weights / f"tp{rank}_sharded_checkpoint.safetensors").touch()
    hunyuan_video._build_llama_app(
        source,
        profile,
        compiled_path=compiled,
        for_compile=False,
    )
    assert calls["neuron_config"]["save_sharded_checkpoint"] is True

    for path in weights.iterdir():
        path.unlink()
    hunyuan_video._build_llama_app(
        source,
        profile,
        compiled_path=compiled,
        for_compile=True,
    )
    assert calls["neuron_config"]["save_sharded_checkpoint"] is True


def test_clip_builder_uses_serving_specific_replicated_world4_contract(
    monkeypatch,
    tmp_path,
):
    profile = _profile(tmp_path, clip_placement="neuron")
    source = _source(tmp_path)
    calls = {}
    torch = _fake_torch(monkeypatch, calls)

    config_module = ModuleType("difflet.backends.trainium.core.config")

    class NeuronConfig:
        def __init__(self, **kwargs):
            calls["neuron_config"] = kwargs

    config_module.NeuronConfig = NeuronConfig
    clip_module = ModuleType("difflet.models.flux.clip.modeling_clip")

    class CLIPInferenceConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            calls["clip_config"] = self

    class NeuronClipApplication:
        def __init__(self, **kwargs):
            calls["app"] = kwargs

    clip_module.CLIPInferenceConfig = CLIPInferenceConfig
    clip_module.NeuronClipApplication = NeuronClipApplication
    adapter_module = ModuleType("difflet.utils.diffusers_adapter")
    adapter_module.load_diffusers_config = lambda path: ("loaded", path)
    monkeypatch.setitem(sys.modules, "difflet.backends.trainium.core.config", config_module)
    monkeypatch.setitem(sys.modules, "difflet.models.flux.clip.modeling_clip", clip_module)
    monkeypatch.setitem(sys.modules, "difflet.utils.diffusers_adapter", adapter_module)

    app = hunyuan_video._build_clip_app(source, profile)

    assert isinstance(app, NeuronClipApplication)
    assert calls["neuron_config"] == {
        "tp_degree": 1,
        "world_size": 4,
        "torch_dtype": torch.bfloat16,
    }
    config = calls["clip_config"]
    assert config.output_attentions is False
    assert config.output_hidden_states is False
    assert config.use_return_dict is True
    assert calls["app"]["model_path"].endswith("/text_encoder_2")


def test_vae_builder_reuses_lower_layer_decoder_only_world4_application(
    monkeypatch,
    tmp_path,
):
    calls = {}
    _fake_torch(monkeypatch, calls)
    application = ModuleType("difflet.models.hunyuan_video.application")

    class NeuronHunyuanVideoApplication:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    application.NeuronHunyuanVideoApplication = NeuronHunyuanVideoApplication
    monkeypatch.setitem(sys.modules, "difflet.models.hunyuan_video.application", application)
    profile = _profile(tmp_path, host_vae=False)

    app = hunyuan_video._build_vae_decoder(_source(tmp_path), profile)

    assert isinstance(app, NeuronHunyuanVideoApplication)
    assert calls["parallel"] is profile.parallel
    assert calls["shape"] == {"height": 64, "width": 96, "num_frames": 5}
    assert calls["enable_transformer"] is False
    assert calls["enable_vae_decoder"] is True


def test_request_validator_checks_llama_template_bucket_and_unsupported_fields(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    validator = hunyuan_video.HunyuanVideoServingRequestValidator(runtime)
    lengths = iter((351, 352))
    seen_prompts = []

    def tokenizer(prompt, **kwargs):
        seen_prompts.append(prompt)
        return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, next(lengths))))

    validator._tokenizer = tokenizer
    request = _request(runtime.profile)
    validator.validate(request)
    assert request.prompt in seen_prompts[0]
    assert seen_prompts[0].startswith("<|start_header_id|>system")

    with pytest.raises(DiffletServingError) as too_long:
        validator.validate(request)
    assert too_long.value.code == "prompt_too_long"

    for changed, code in (
        (replace(request, model="other/model"), "profile_mismatch"),
        (replace(request, output_format="png"), "invalid_extra_body"),
        (replace(request, height=request.height + 2), "profile_mismatch"),
        (
            replace(request, video=replace(request.video, negative_prompt="bad")),
            "invalid_extra_body",
        ),
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
        fresh = hunyuan_video.HunyuanVideoServingRequestValidator(runtime)
        fresh._tokenizer = lambda *args, **kwargs: SimpleNamespace(
            input_ids=SimpleNamespace(shape=(1, 10))
        )
        with pytest.raises(DiffletServingError) as exc:
            fresh.validate(changed)
        assert exc.value.code == code


def test_host_clip_and_nxdi_llama_stages_preserve_conditioning_contract(
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
    clip_ids = _TraceTensor("clip_ids")
    clip_mask = _TraceTensor("clip_mask")
    pooled = _TraceTensor("pooled")

    def clip_tokenizer(prompt, **kwargs):
        calls["clip_tokenizer"] = (prompt, kwargs)
        return SimpleNamespace(input_ids=clip_ids, attention_mask=clip_mask)

    def clip_model(**kwargs):
        calls["clip_model"] = kwargs
        return SimpleNamespace(pooler_output=pooled)

    llama_ids = _TraceTensor("llama_ids")
    llama_mask = _TraceTensor("llama_mask")
    captured = _TraceTensor("captured")

    def llama_tokenizer(prompt, **kwargs):
        calls["llama_tokenizer"] = (prompt, kwargs)
        return SimpleNamespace(input_ids=llama_ids, attention_mask=llama_mask)

    def llama_app(**kwargs):
        calls["llama_app"] = kwargs
        return SimpleNamespace(captured_tensors=(captured,))

    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()
    adapter.clip_tokenizer = clip_tokenizer
    adapter.clip_model = clip_model
    adapter.llama_tokenizer = llama_tokenizer
    adapter.llama_app = llama_app

    clip_result = asyncio.run(
        hunyuan_video.HunyuanVideoHostClipStageRunner(adapter).execute(
            _invocation(
                runtime,
                request,
                0,
                hunyuan_video.HunyuanVideoInitialPayload(),
            )
        )
    )
    llama_result = asyncio.run(
        hunyuan_video.HunyuanVideoLlamaStageRunner(adapter).execute(
            _invocation(runtime, request, 1, clip_result.output)
        )
    )

    assert calls["clip_tokenizer"] == (
        request.prompt,
        {
            "padding": "max_length",
            "max_length": 77,
            "truncation": True,
            "return_tensors": "pt",
        },
    )
    assert calls["clip_model"] == {
        "input_ids": clip_ids,
        "attention_mask": clip_mask,
    }
    assert ("pooled", "to", (torch.bfloat16,), {}) in pooled.calls
    assert calls["llama_tokenizer"][1] == {
        "max_length": 351,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }
    assert calls["llama_app"]["input_ids"].name == "to(llama_ids)"
    assert calls["llama_app"]["attention_mask"].name == "to(llama_mask)"
    assert llama_result.output.pooled_projections is clip_result.output.pooled_projections
    assert llama_result.output.encoder_hidden_states.name.startswith("cpu(to(slice(")
    assert llama_result.output.encoder_attention_mask.name.startswith("to(slice(")


def test_neuron_clip_stage_preserves_cli_token_and_pooling_contract(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        profile=_profile(tmp_path, clip_placement="neuron"),
    )
    request = _request(runtime.profile)
    calls = {}
    torch = _fake_torch(monkeypatch, calls)
    input_ids = _TraceTensor("clip_ids")
    pooled = _TraceTensor("pooled")

    def tokenizer(prompt, **kwargs):
        calls["tokenizer"] = (prompt, kwargs)
        return SimpleNamespace(input_ids=input_ids)

    def app(value):
        calls["app"] = value
        return SimpleNamespace(pooler_output=pooled)

    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()
    adapter.clip_tokenizer = tokenizer
    adapter.clip_app = app

    result = asyncio.run(
        hunyuan_video.HunyuanVideoNeuronClipStageRunner(adapter).execute(
            _invocation(
                runtime,
                request,
                0,
                hunyuan_video.HunyuanVideoInitialPayload(),
            )
        )
    )

    assert calls["tokenizer"] == (
        request.prompt,
        {
            "padding": "max_length",
            "max_length": 77,
            "truncation": True,
            "return_tensors": "pt",
        },
    )
    assert calls["app"].name == "to(clip_ids)"
    assert ("clip_ids", "to", (torch.int64,), {}) in input_ids.calls
    assert result.output.pooled_projections.name == "reshape(cpu(to(pooled)))"


def test_denoiser_builds_seeded_latents_bundle_and_uses_lower_scheduler_loop(
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
    _fake_torch(monkeypatch, calls)

    class FakeBundle:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            calls["bundle"] = kwargs

    application = ModuleType("difflet.models.hunyuan_video.application")
    application.HunyuanVideoDiTInputBundle = FakeBundle
    monkeypatch.setitem(
        sys.modules,
        "difflet.models.hunyuan_video.application",
        application,
    )
    final_latents = object()

    class FakeDenoiser:
        def pipeline(self, **kwargs):
            calls["pipeline"] = kwargs
            return SimpleNamespace(latents=final_latents)

    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()
    adapter.denoiser = FakeDenoiser()
    conditioning = hunyuan_video.HunyuanVideoConditioningPayload(
        pooled_projections=_TraceTensor("pooled"),
        encoder_hidden_states=_TraceTensor("hidden"),
        encoder_attention_mask=_TraceTensor("mask"),
    )

    result = asyncio.run(
        hunyuan_video.HunyuanVideoDenoiserStageRunner(adapter).execute(
            _invocation(runtime, request, 2, conditioning)
        )
    )

    assert calls["seed"] == 42
    assert calls["randn"][0] == (1, 16, 2, 8, 12)
    assert calls["randn"][1]["dtype"] == "bfloat16"
    assert calls["bundle"]["hidden_states"].shape == (1, 16, 2, 8, 12)
    assert calls["bundle"]["encoder_hidden_states"] is conditioning.encoder_hidden_states
    assert calls["bundle"]["encoder_attention_mask"] is conditioning.encoder_attention_mask
    assert calls["bundle"]["pooled_projections"] is conditioning.pooled_projections
    assert calls["full"] == ([1], 6000.0, {"dtype": "bfloat16"})
    assert calls["pipeline"] == {
        "bundle": calls["pipeline"]["bundle"],
        "num_inference_steps": 4,
        "output_type": "latent",
        "return_trajectory": False,
    }
    assert calls["pipeline"]["bundle"].__dict__ == calls["bundle"]
    assert result.output.latents is final_latents


def test_host_decoder_scales_cpu_latents_and_encodes_bcthw_mp4(monkeypatch, tmp_path):
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
    latents = _TraceTensor("latents", calls=tensor_calls)
    frames = _TraceTensor("frames", calls=tensor_calls)

    class FakeVAE:
        config = SimpleNamespace(scaling_factor=0.476986)

        def decode(self, value, *, return_dict):
            calls["decode"] = (value, return_dict)
            return (frames,)

    expected = FileBackedGenerateOutput(
        path=str(target_path),
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=3,
        width=96,
        height=64,
        num_frames=5,
        fps=24.0,
        duration_s=5 / 24,
    )

    def fake_encode(tensor, encoded_request, **kwargs):
        calls["encode"] = (tensor, encoded_request, kwargs)
        return expected

    monkeypatch.setattr(hunyuan_video, "encode_video_tensor", fake_encode)
    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()
    adapter.vae = FakeVAE()

    result = asyncio.run(
        hunyuan_video.HunyuanVideoHostDecoderStageRunner(adapter).execute(
            _invocation(
                runtime,
                request,
                3,
                hunyuan_video.HunyuanVideoLatentPayload(latents),
            )
        )
    )

    latent_to = next(call for call in tensor_calls if call[:2] == ("latents", "to"))
    assert latent_to[2:] == ((), {"device": "cpu", "dtype": torch.float32})
    decoded, return_dict = calls["decode"]
    assert decoded.name == "(to(latents)/0.476986)"
    assert return_dict is False
    assert calls["encode"][1] is request
    assert calls["encode"][2] == {
        "layout": "BCTHW",
        "value_range": "minus_one_to_one",
    }
    assert result.output.output == expected


def test_neuron_decoder_uses_lower_pipeline_scaling_and_encodes_bcthw_mp4(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        profile=_profile(tmp_path, host_vae=False),
    )
    target_path = (tmp_path / "job.part.mp4").resolve()
    target_path.touch()
    request = _request(
        runtime.profile,
        target=FileOutputTarget(staging_path=str(target_path)),
    )
    calls = {}
    torch = _fake_torch(monkeypatch, calls)
    latents = _TraceTensor("latents")
    frames = _TraceTensor("frames")

    class Pipeline:
        def _decode_latents(self, value):
            calls["decode"] = value
            return frames

    expected = FileBackedGenerateOutput(
        path=str(target_path),
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=3,
        width=96,
        height=64,
        num_frames=5,
        fps=24.0,
        duration_s=5 / 24,
    )
    monkeypatch.setattr(
        hunyuan_video,
        "encode_video_tensor",
        lambda tensor, encoded_request, **kwargs: calls.update(
            encode=(tensor, encoded_request, kwargs)
        )
        or expected,
    )
    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()
    adapter.vae_app = SimpleNamespace(pipeline=Pipeline())

    result = asyncio.run(
        hunyuan_video.HunyuanVideoNeuronDecoderStageRunner(adapter).execute(
            _invocation(
                runtime,
                request,
                3,
                hunyuan_video.HunyuanVideoLatentPayload(latents),
            )
        )
    )

    assert calls["decode"].name == "to(latents)"
    assert ("latents", "to", (), {"dtype": torch.bfloat16}) in latents.calls
    assert calls["encode"][0].name == "clamp(to(frames))"
    assert calls["encode"][1] is request
    assert calls["encode"][2] == {
        "layout": "BCTHW",
        "value_range": "minus_one_to_one",
    }
    assert result.output.output == expected


def test_initial_payload_requires_precreated_parent_target(monkeypatch, tmp_path):
    runtime = _runtime(tmp_path, monkeypatch)
    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()

    with pytest.raises(ValueError, match="output target"):
        adapter.initial_payload(_request(runtime.profile))
    missing = FileOutputTarget(staging_path=str((tmp_path / "missing.part.mp4").resolve()))
    with pytest.raises(ValueError, match="pre-created"):
        adapter.initial_payload(_request(runtime.profile, target=missing))

    path = (tmp_path / "valid.part.mp4").resolve()
    path.touch()
    payload = adapter.initial_payload(
        _request(
            runtime.profile,
            target=FileOutputTarget(staging_path=str(path)),
        )
    )
    assert isinstance(payload, hunyuan_video.HunyuanVideoInitialPayload)


def test_adapter_validates_both_bindings_and_loads_host_and_neuron_components(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = []

    class FakeManager:
        def __init__(self, root):
            calls.append(("manager", Path(root)))

        def validate_binding(self, binding, *, validate_payload):
            calls.append(("binding", binding.artifact_id))
            validate_payload(binding.path)

    class FakeLlama:
        def load(self, path):
            calls.append(("llama_load", path))

    class FakeDenoiser:
        def load(self, path, **kwargs):
            calls.append(("denoiser_load", path, kwargs))

    clip = (object(), object())
    llama = FakeLlama()
    llama_tokenizer = object()
    denoiser = FakeDenoiser()
    vae = object()
    monkeypatch.setattr(hunyuan_video, "ImmutableArtifactManager", FakeManager)
    monkeypatch.setattr(
        hunyuan_video,
        "_validate_artifact",
        lambda source, profile, spec, path: calls.append(("validated", spec.artifact_id, path)),
    )
    monkeypatch.setattr(hunyuan_video, "_load_host_clip", lambda path: clip)
    monkeypatch.setattr(
        hunyuan_video,
        "_build_llama_app",
        lambda source, profile, **kwargs: calls.append(("llama_build", kwargs)) or llama,
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_load_llama_tokenizer",
        lambda path: llama_tokenizer,
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_build_denoiser",
        lambda source, profile: denoiser,
    )
    monkeypatch.setattr(hunyuan_video, "_load_host_vae", lambda path: vae)
    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()

    runners = asyncio.run(adapter.create_loaded_runners(runtime))

    llama_binding = runtime.artifacts.require("llama")
    denoiser_binding = runtime.artifacts.require("denoiser")
    assert tuple(runners) == ("clip", "llama", "denoiser", "decoder")
    assert ("binding", "llama") in calls and ("binding", "denoiser") in calls
    assert ("validated", "llama", llama_binding.path) in calls
    assert ("validated", "denoiser", denoiser_binding.path) in calls
    assert (
        "llama_build",
        {"compiled_path": llama_binding.path, "for_compile": False},
    ) in calls
    assert ("llama_load", str(llama_binding.path)) in calls
    assert (
        "denoiser_load",
        str(denoiser_binding.path),
        {"start_rank_id": 0, "local_ranks_size": 4, "skip_warmup": True},
    ) in calls
    assert (adapter.clip_tokenizer, adapter.clip_model) == clip
    assert adapter.llama_app is llama
    assert adapter.llama_tokenizer is llama_tokenizer
    assert adapter.denoiser is denoiser
    assert adapter.vae is vae

    asyncio.run(adapter.shutdown())
    assert adapter.clip_tokenizer is None
    assert adapter.clip_model is None
    assert adapter.llama_app is None
    assert adapter.llama_tokenizer is None
    assert adapter.denoiser is None
    assert adapter.vae is None
    assert adapter.runtime is None
    assert adapter.profile is None


def test_adapter_loads_neuron_clip_and_vae_without_host_models(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        profile=_profile(tmp_path, clip_placement="neuron", host_vae=False),
    )
    calls = []

    class FakeManager:
        def __init__(self, root):
            calls.append(("manager", Path(root)))

        def validate_binding(self, binding, *, validate_payload):
            calls.append(("binding", binding.artifact_id))
            validate_payload(binding.path)

    class FakeApplication:
        def __init__(self, name):
            self.name = name

        def load(self, path, **kwargs):
            calls.append((f"{self.name}_load", path, kwargs))

    class FakeLlama(FakeApplication):
        def load(self, path, **kwargs):
            calls.append(("llama_load", path, kwargs))

    clip_app = FakeApplication("clip")
    llama_app = FakeLlama("llama")
    denoiser = FakeApplication("denoiser")
    vae_app = FakeApplication("decoder")
    clip_tokenizer = object()
    llama_tokenizer = object()
    monkeypatch.setattr(hunyuan_video, "ImmutableArtifactManager", FakeManager)
    monkeypatch.setattr(
        hunyuan_video,
        "_validate_artifact",
        lambda source, profile, spec, path: calls.append(("validated", spec.artifact_id)),
    )
    monkeypatch.setattr(hunyuan_video, "_build_clip_app", lambda *args: clip_app)
    monkeypatch.setattr(
        hunyuan_video,
        "_load_clip_tokenizer",
        lambda path: clip_tokenizer,
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_build_llama_app",
        lambda source, profile, **kwargs: llama_app,
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_load_llama_tokenizer",
        lambda path: llama_tokenizer,
    )
    monkeypatch.setattr(hunyuan_video, "_build_denoiser", lambda *args: denoiser)
    monkeypatch.setattr(hunyuan_video, "_build_vae_decoder", lambda *args: vae_app)
    monkeypatch.setattr(
        hunyuan_video,
        "_load_host_clip",
        lambda path: pytest.fail("host CLIP must not load"),
    )
    monkeypatch.setattr(
        hunyuan_video,
        "_load_host_vae",
        lambda path: pytest.fail("host VAE must not load"),
    )
    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()

    runners = asyncio.run(adapter.create_loaded_runners(runtime))

    assert tuple(runners) == ("clip", "llama", "denoiser", "decoder")
    assert {item[1] for item in calls if item[0] == "binding"} == {
        "clip",
        "llama",
        "denoiser",
        "decoder",
    }
    load_kwargs = {
        "start_rank_id": 0,
        "local_ranks_size": 4,
        "skip_warmup": True,
    }
    assert ("clip_load", str(runtime.artifacts.require("clip").path), load_kwargs) in calls
    assert (
        "decoder_load",
        str(runtime.artifacts.require("decoder").path),
        load_kwargs,
    ) in calls
    load_events = [item[0] for item in calls if item[0].endswith("_load")]
    assert load_events == ["llama_load", "denoiser_load", "clip_load", "decoder_load"]
    assert adapter.clip_app is clip_app
    assert adapter.clip_model is None
    assert adapter.vae_app is vae_app
    assert adapter.vae is None

    asyncio.run(adapter.shutdown())
    assert adapter.clip_app is None
    assert adapter.vae_app is None


def test_startup_smoke_is_target_bound_reentrant_and_cleans_on_error(
    monkeypatch,
    tmp_path,
):
    runtime = _runtime(tmp_path, monkeypatch)
    adapter = hunyuan_video.HunyuanVideoServingStageAdapter()
    adapter.profile = runtime.profile
    first = adapter.smoke_request()
    first_parent = Path(first.video.output_target.staging_path).parent
    second = adapter.smoke_request()
    second_target = Path(second.video.output_target.staging_path)
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
        fps=24.0,
        duration_s=5 / 24,
    )
    monkeypatch.setattr(
        "difflet.serving.video_media.validate_mp4",
        lambda path, **kwargs: VideoMediaMetadata(
            width=96,
            height=64,
            num_frames=5,
            fps=24.0,
            duration_s=5 / 24,
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
