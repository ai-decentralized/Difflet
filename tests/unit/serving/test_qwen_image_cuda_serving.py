"""Qwen-Image serving on CUDA: preparer, validator and resident adapter, CPU-only.

DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""

from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.engines.resident_worker import _apply_worker_runtime_environment
from difflet.serving.errors import DiffletServingError
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.backends.gpu import serving as mod
from difflet.models.qwen_image.spec import SPEC
from difflet.serving.types import ResolvedModelSource, ServingProfile

import os as _os

_AMBIENT_VISIBLE_CORES = _os.environ.get("NEURON_RT_VISIBLE_CORES")
_SOURCE = ResolvedModelSource("hf_snapshot", "Qwen/Qwen-Image", None, "/snap/qwen", "a" * 40)


def _profile(**over) -> ServingProfile:
    d = dict(
        model_id="Qwen/Qwen-Image",
        model_type="qwen_image",
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


def _prepare(profile, source_calls, policy=CompilePolicy.AUTO):
    return mod.GpuServingArtifactPreparer("Qwen/Qwen-Image").prepare_runtime(
        profile, download_policy=DownloadPolicy.AUTO, compile_policy=policy
    )


# ------------------------------------------------------------------ preparer


def test_prepare_runtime_resolves_the_snapshot_and_builds_an_eager_bundle(source):
    bundle = _prepare(_profile(), source)
    assert source and source[0][0] == ("Qwen/Qwen-Image",)
    assert source[0][1] == {"revision": None, "download_policy": DownloadPolicy.AUTO}
    assert bundle.compile_specs == () and bundle.artifacts.bindings == ()
    assert [s.stage_id for s in bundle.pipeline_definition.stages] == ["pipeline"]
    stage = bundle.runtime_plan.stages[0]
    assert stage.placement == "cuda" and stage.artifact_id is None
    assert bundle.runtime_plan.profile_identity
    assert bundle.adapter_config["compile_policy"] is CompilePolicy.AUTO
    assert bundle.source is _SOURCE and bundle.profile is not None


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"shapes": ((1024, 1024, None), (768, 768, None))}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, sp_enabled=True)}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cp_degree=2)}, "not supported"),
        # Qwen-Image runs single-pass (no CFG): there is no second branch to place.
        (
            {"parallel": DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True)},
            "does not apply",
        ),
        ({"teacache_speedup": 1.5}, "not supported"),
    ],
)
def test_unsupported_profiles_are_refused_before_the_download(source, over, expected):
    with pytest.raises(DiffletServingError) as exc:
        _prepare(_profile(**over), source)
    assert exc.value.code == "feature_not_supported" and expected in exc.value.message
    assert source == []


_WORKER_ENVIRONMENT_NAMES = (
    "NEURON_RT_VISIBLE_CORES",
    "NEURON_RT_NUM_CORES",
    "NEURON_RT_VIRTUAL_CORE_SIZE",
    "NEURON_LOGICAL_NC_CONFIG",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
)


@pytest.fixture
def preserve_worker_environment():
    import os

    original = {name: os.environ.get(name) for name in _WORKER_ENVIRONMENT_NAMES}
    yield
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_runtime_plan_is_accepted_by_the_resident_worker_environment(
    source, preserve_worker_environment
):
    bundle = _prepare(_profile(), source)
    _apply_worker_runtime_environment(bundle)  # raises ValueError on a plan the worker rejects


def test_worker_environment_does_not_leak_out_of_the_previous_test():
    # delenv(raising=False) on an unset variable records nothing to undo, so a worker's
    # NEURON_RT_VISIBLE_CORES=0 would stay set for every later test in the session.
    import os

    assert os.environ.get("NEURON_RT_VISIBLE_CORES") == _AMBIENT_VISIBLE_CORES


def test_profile_identity_tracks_shape_and_source(source):
    a = _prepare(_profile(), source).runtime_plan.profile_identity
    b = _prepare(_profile(height=768, width=768), source).runtime_plan.profile_identity
    assert a != b and a == _prepare(_profile(), source).runtime_plan.profile_identity


# ----------------------------------------------------------------- validator


class _Tok:
    def __init__(self, n):
        self.n = n

    def __call__(self, text, **kw):
        import types

        return types.SimpleNamespace(input_ids=[list(range(self.n))])


def _validator(bundle, n_tokens=40):
    v = mod.GpuServingRequestValidator(bundle)
    v._tokenizer = lambda subdir: _Tok(n_tokens)
    return v


def _request(**over):
    from difflet.serving.types import DiffletGenerateRequest

    d = dict(
        request_id="r1",
        model="Qwen/Qwen-Image",
        prompt="a cat",
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=SPEC.serving_noop_guidance_scale,
        seed=7,
    )
    d.update(over)
    return DiffletGenerateRequest(**d)


def test_default_request_is_accepted(source):
    _validator(_prepare(_profile(), source)).validate(_request())


def test_guidance_scale_other_than_the_noop_value_is_refused(source):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(_profile(), source)).validate(_request(guidance_scale=1.0))
    assert exc.value.code == "feature_not_supported" and "guidance_embeds" in exc.value.message


def test_request_shape_must_equal_the_profile_shape(source):
    with pytest.raises(DiffletServingError) as exc:
        _validator(_prepare(_profile(), source)).validate(_request(height=768, width=768))
    assert exc.value.code == "profile_mismatch"


def test_prompt_longer_than_the_dit_text_length_is_refused(source):
    # diffusers truncates at tokenizer_max_length (1024) + the 34-token template prefix.
    bundle = _prepare(_profile(), source)
    _validator(bundle, n_tokens=34 + 1024).validate(_request())
    with pytest.raises(DiffletServingError) as exc:
        _validator(bundle, n_tokens=34 + 1025).validate(_request())
    assert exc.value.code == "prompt_too_long"


def test_preload_loads_the_tokenizer_from_the_pinned_snapshot(source, monkeypatch):
    seen = {}
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda p, **k: seen.update(path=str(p)) or _Tok(5)),
    )
    mod.GpuServingRequestValidator(_prepare(_profile(), source)).preload()
    assert seen["path"] == "/snap/qwen/tokenizer"


# ------------------------------------------------------------------- adapter

import argparse
import asyncio
import io

from PIL import Image

from difflet.serving.engines.stage_pipeline import StagePipelineEngine
from difflet.serving.types import (
    ArtifactSet,
    CancellationSignal,
    ResolvedRuntimeBundle,
    WorkerRequestContext,
)

_FROM_PRETRAINED = "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained"


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
    adapter = mod.GpuServingStageAdapter("Qwen/Qwen-Image")
    return adapter, StagePipelineEngine(runtime=bundle, adapter=adapter)


def test_startup_loads_the_pipeline_exactly_as_the_cuda_cli_would(source, recorder):
    from difflet.backends.gpu.cli import GpuOrchestrator

    bundle = _prepare(_profile(cache_dir="/c", revision="r1"), source, CompilePolicy.FORCE)
    adapter, engine = _engine(bundle)
    asyncio.run(engine.start())
    model_id, kw = recorder[0]
    cli_args = argparse.Namespace(
        tp_degree=1, height=1024, width=1024, cache_dir="/c", revision="r1", force=True
    )
    cli_kw = GpuOrchestrator(cli_args, SPEC).pipeline_kwargs(load=True)
    # Same kwargs as the CLI except the revision: the server loads the commit the parent pinned
    # (test below), where the CLI resolves the user's ref itself.
    assert model_id == "Qwen/Qwen-Image"
    assert {k: v for k, v in kw.items() if k != "revision"} == {
        k: v for k, v in cli_kw.items() if k != "revision"
    }
    assert kw["backend"] == "cuda" and kw["load"] is True and kw["force_compile"] is True


def test_worker_loads_the_commit_the_parent_pinned(source, recorder):
    # revision=profile.revision would re-resolve refs/main in the worker, so a worker
    # restart after a newer snapshot landed would quietly load a commit the bundle, the
    # profile_identity and the validator's tokenizer do not describe.
    adapter, engine = _engine(_prepare(_profile(revision=None), source))
    asyncio.run(engine.start())
    assert recorder[0][1]["revision"] == _SOURCE.resolved_source_id == "a" * 40


def test_auto_compile_policy_does_not_force(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    assert recorder[0][1]["force_compile"] is False


def test_startup_smoke_runs_four_steps_at_the_profile_shape(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    prompt, kw = adapter.pipe.app.calls[0]
    assert kw == {"num_inference_steps": SPEC.smoke_steps, "seed": 0, "output_type": "pil"}
    assert SPEC.smoke_steps == 4 and "guidance_scale" not in kw


def test_generate_passes_prompt_steps_and_seed_and_returns_png(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    req = _request(prompt="a cat", num_inference_steps=7, seed=3)
    out = asyncio.run(engine.generate(req, WorkerRequestContext.with_timeout("r1", 30.0)))
    prompt, kw = adapter.pipe.app.calls[-1]
    assert prompt == "a cat" and kw == {"num_inference_steps": 7, "seed": 3, "output_type": "pil"}
    assert out.mime_type == "image/png" and out.output_format == "png"
    assert Image.open(io.BytesIO(out.data)).size == (1024, 1024)


def test_smoke_output_of_the_wrong_size_fails_startup(source, monkeypatch):
    monkeypatch.setattr(_FROM_PRETRAINED, staticmethod(lambda m, **kw: _Pipe((8, 8))))
    adapter, engine = _engine(_prepare(_profile(), source))
    with pytest.raises(RuntimeError):
        asyncio.run(engine.start())
    assert adapter.pipe is None


def test_cancelled_request_never_reaches_generate(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    n = len(adapter.pipe.app.calls)
    signal = CancellationSignal()
    signal.cancel()
    context = WorkerRequestContext.with_timeout("r", 30.0, cancellation=signal)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(engine.generate(_request(), context))
    assert len(adapter.pipe.app.calls) == n


def test_unsupported_profile_is_refused_before_any_load(source, recorder):
    # The preparer refuses tp=2, so build the bundle by hand to prove the worker side too.
    profile = _profile(parallel=DiffletParallelConfig(tp_degree=2))
    bundle = ResolvedRuntimeBundle(
        profile=profile,
        source=_SOURCE,
        pipeline_definition=_prepare(_profile(), source).pipeline_definition,
        runtime_plan=mod.resident_runtime_plan(profile, _SOURCE, SPEC),
        compile_specs=(),
        artifacts=ArtifactSet(()),
        adapter_config={"compile_policy": CompilePolicy.AUTO},
    )
    with pytest.raises(DiffletServingError) as exc:
        asyncio.run(mod.GpuServingStageAdapter("Qwen/Qwen-Image").create_loaded_runners(bundle))
    assert "not supported" in exc.value.message and recorder == []


def test_shutdown_releases_the_resident_pipeline(source, recorder):
    adapter, engine = _engine(_prepare(_profile(), source))
    asyncio.run(engine.start())
    asyncio.run(engine.shutdown())
    assert adapter.pipe is None and adapter.active_runtime is None
