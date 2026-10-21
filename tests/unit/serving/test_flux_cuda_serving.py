"""FLUX.1-dev serving on CUDA: preparer, validator and resident adapter, CPU-only.

DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import types

import pytest
from PIL import Image

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.engines.resident_worker import _apply_worker_runtime_environment
from difflet.serving.engines.stage_pipeline import StagePipelineEngine
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.orchestrators import flux_cuda as mod
from difflet.serving.types import (
    ArtifactSet,
    CancellationSignal,
    DiffletGenerateRequest,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    ServingProfile,
    WorkerRequestContext,
)

_MODEL = "black-forest-labs/FLUX.1-dev"
_SOURCE = ResolvedModelSource("hf_snapshot", _MODEL, None, "/snap/flux", "f" * 40)
_FROM_PRETRAINED = "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained"


def _profile(**over) -> ServingProfile:
    d = dict(
        model_id=_MODEL,
        model_type="flux",
        height=1024,
        width=1024,
        num_frames=None,
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
    return mod.CudaFluxServingArtifactPreparer().prepare_runtime(
        profile, download_policy=DownloadPolicy.AUTO, compile_policy=policy
    )


# ------------------------------------------------------------------ preparer


def test_prepare_runtime_resolves_the_snapshot_and_builds_an_eager_bundle(source):
    bundle = _prepare(_profile())
    assert source[0][0] == (_MODEL,)
    assert source[0][1] == {"revision": None, "download_policy": DownloadPolicy.AUTO}
    assert bundle.compile_specs == () and bundle.artifacts.bindings == ()
    assert [s.stage_id for s in bundle.pipeline_definition.stages] == ["pipeline"]
    (stage,) = bundle.runtime_plan.stages
    assert stage.placement == "cuda" and stage.artifact_id is None
    assert bundle.runtime_plan.allocations[0].allocation_id == "flux-cuda"
    assert bundle.adapter_config["compile_policy"] is CompilePolicy.AUTO


@pytest.mark.parametrize(
    "over,named",
    [
        ({"shapes": ((1024, 1024, None), (768, 768, None))}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "--tp-degree > 1"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, sp_enabled=True)}, "--sp (sequence"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cp_degree=2)}, "not supported"),
        # FLUX.1-dev is guidance-distilled: one pass per step, no second branch to place.
        (
            {"parallel": DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True)},
            "guidance-distilled",
        ),
        ({"teacache_speedup": 1.5}, "not supported"),
    ],
)
def test_unsupported_profiles_are_refused_before_the_download(source, over, named):
    with pytest.raises(DiffletServingError) as exc:
        _prepare(_profile(**over))
    assert exc.value.code == "feature_not_supported" and named in exc.value.message
    assert "[flux/cuda serve]" in exc.value.message and source == []


def test_runtime_plan_is_accepted_by_the_resident_worker_environment(source, monkeypatch):
    # setenv records each variable, so monkeypatch restores what the worker setup overwrites.
    for name in mod_worker_environment_names():
        monkeypatch.setenv(name, "sentinel")
    _apply_worker_runtime_environment(_prepare(_profile()))  # raises on a plan it rejects


def mod_worker_environment_names():
    return (
        "NEURON_RT_VISIBLE_CORES",
        "NEURON_RT_NUM_CORES",
        "NEURON_RT_VIRTUAL_CORE_SIZE",
        "NEURON_LOGICAL_NC_CONFIG",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
    )


# ----------------------------------------------------------------- validator


class _Tok:
    def __init__(self, n):
        self.n = n

    def __call__(self, text, **kw):
        return types.SimpleNamespace(input_ids=[list(range(self.n))])


def _validator(bundle, n_tokens=40):
    v = mod.CudaFluxServingRequestValidator(bundle)
    v._tokenizer_for_runtime = lambda: _Tok(n_tokens)
    return v


def _request(**over):
    d = dict(
        request_id="r1",
        model=_MODEL,
        prompt="a cat",
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=7,
    )
    d.update(over)
    return DiffletGenerateRequest(**d)


def test_default_request_is_accepted(source):
    _validator(_prepare(_profile())).validate(_request())


def test_guidance_scale_is_honoured_within_the_flux_range(source):
    v = _validator(_prepare(_profile()))
    v.validate(_request(guidance_scale=0.0))
    v.validate(_request(guidance_scale=20.0))
    with pytest.raises(DiffletServingError) as exc:
        v.validate(_request(guidance_scale=20.5))
    assert exc.value.code == "invalid_extra_body"


def test_request_shape_must_equal_the_profile_shape(source):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(_profile())).validate(_request(height=768, width=768))
    assert exc.value.code == "profile_mismatch"


def test_prompt_longer_than_t5s_512_tokens_is_refused(source):
    bundle = _prepare(_profile())
    _validator(bundle, n_tokens=512).validate(_request())
    with pytest.raises(DiffletServingError) as exc:
        _validator(bundle, n_tokens=513).validate(_request())
    assert exc.value.code == "prompt_too_long"


def test_preload_loads_the_t5_tokenizer_from_the_pinned_snapshot(source, monkeypatch):
    seen = {}
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda p, **k: seen.update(path=str(p)) or _Tok(5)),
    )
    mod.CudaFluxServingRequestValidator(_prepare(_profile())).preload()
    assert seen["path"] == "/snap/flux/tokenizer_2"


# ------------------------------------------------------------------- adapter


class _App:
    def __init__(self, size):
        self.calls, self.size = [], size

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return [Image.new("RGB", self.size, "red")]


class _Pipe:
    def __init__(self, size):
        self.app = _App(size)


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe((kw["width"], kw["height"]))

    monkeypatch.setattr(_FROM_PRETRAINED, staticmethod(fake))
    return calls


def _engine(bundle):
    adapter = mod.CudaFluxServingStageAdapter()
    return adapter, StagePipelineEngine(runtime=bundle, adapter=adapter)


def _cli(**over):
    from difflet.cli.orchestrators.flux_cuda import FluxCudaOrchestrator

    d = dict(
        model_id=_MODEL,
        tp_degree=1,
        height=1024,
        width=1024,
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
    return FluxCudaOrchestrator(argparse.Namespace(**d))


def test_startup_loads_the_pipeline_exactly_as_the_cuda_cli_would(source, recorder):
    bundle = _prepare(_profile(revision="r1"), CompilePolicy.FORCE)
    adapter, engine = _engine(bundle)
    asyncio.run(engine.start())
    model_id, kw = recorder[0]
    cli_kw = _cli().pipeline_kwargs(load=True)
    # Same kwargs as the CLI except the revision: the server loads the commit the parent pinned.
    assert model_id == _MODEL
    assert {k: v for k, v in kw.items() if k != "revision"} == {
        k: v for k, v in cli_kw.items() if k != "revision"
    }
    assert kw["revision"] == _SOURCE.resolved_source_id
    assert kw["backend"] == "cuda" and kw["load"] is True and kw["force_compile"] is True


def test_auto_compile_policy_does_not_force(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    assert recorder[0][1]["force_compile"] is False


def test_startup_smoke_runs_four_steps_at_the_profile_shape(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    prompt, kw = adapter.pipe.app.calls[0]
    assert kw == {
        "num_inference_steps": mod.SMOKE_STEPS,
        "guidance_scale": 3.5,
        "seed": 0,
        "output_type": "pil",
    }
    assert mod.SMOKE_STEPS == 4


def test_generate_passes_the_cli_s_arguments_and_returns_png(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    req = _request(prompt="a cat", num_inference_steps=7, guidance_scale=2.5, seed=3)
    out = asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r1", 30.0)))
    prompt, kw = adapter.pipe.app.calls[-1]
    cli_kw = _cli(steps=7, guidance_scale=2.5, seed=3).generate_kwargs()
    assert prompt == "a cat" and kw == cli_kw
    assert kw == {"num_inference_steps": 7, "guidance_scale": 2.5, "seed": 3, "output_type": "pil"}
    assert out.mime_type == "image/png" and out.output_format == "png"
    assert Image.open(io.BytesIO(out.data)).size == (1024, 1024)


def test_smoke_output_of_the_wrong_size_fails_startup(source, monkeypatch):
    monkeypatch.setattr(_FROM_PRETRAINED, staticmethod(lambda m, **kw: _Pipe((8, 8))))
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
        asyncio.run(engine.generate(_request(), context))
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
        asyncio.run(mod.CudaFluxServingStageAdapter().create_loaded_runners(bundle))
    assert "not supported" in exc.value.message and recorder == []


def test_shutdown_releases_the_resident_pipeline(source, recorder):
    adapter, engine = _engine(_prepare(_profile()))
    asyncio.run(engine.start())
    asyncio.run(engine.shutdown())
    assert adapter.pipe is None and adapter.active_runtime is None


def test_cuda_serving_metadata_is_one_opaque_stage_with_the_cli_defaults():
    from difflet.common.registry.flux import cuda_serving_metadata

    meta = cuda_serving_metadata()
    assert meta.model_type == "flux" and meta.checkpoint_ids == (_MODEL,)
    assert (meta.default_steps, meta.default_guidance_scale) == (28, 3.5)
    assert meta.output_modality == "image" and meta.chat_content_type == "image_url"
    (stage,) = meta.pipeline_definition.stages
    assert stage.kind == "opaque_pipeline" and stage.stage_id == "pipeline"
    assert meta.orchestrator_factory.endswith("flux_cuda:CudaFluxServingStageAdapter")
