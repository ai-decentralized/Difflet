"""Flux on the GPU flow through the CLI: one process, DiffletPipeline + generate().

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
Mirrors tests/unit/cli/test_orchestrator_qwen_cuda.py.
"""

from __future__ import annotations

import argparse

import pytest
from PIL import Image

from difflet.backends.gpu.cli import GpuOrchestrator
from difflet.models.flux.spec import SPEC


def _orch(args):
    return GpuOrchestrator(args, SPEC)


def _args(**over):
    d = dict(
        model_id="black-forest-labs/FLUX.1-dev",
        tp_degree=None,
        cp_degree=None,
        cp_mode="gather_kv",
        cfg_parallel=False,
        sp_enabled=False,
        dp=None,
        height=None,
        width=None,
        shapes=None,
        cache_dir=None,
        force=False,
        revision=None,
        prompt="a cat",
        output="/tmp/x/flux.png",
        steps=None,
        guidance_scale=None,
        seed=42,
        requests=None,
        requests_dir=None,
        teacache_cadence=None,
        teacache_online_delta=None,
        teacache_speedup=None,
        teacache_calibration=None,
        host_vae=False,
        taef1=False,
        taef1_path=None,
        torch_compile=False,
    )
    d.update(over)
    return argparse.Namespace(**d)


class _App:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return [Image.new("RGB", (8, 8), "white")]


class _Pipe:
    def __init__(self):
        self.app = _App()
        self.compiled_path = "/fake/compiled"


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe()

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained", staticmethod(fake)
    )
    return calls


def test_compile_builds_the_cuda_pipeline_without_loading(recorder):
    _orch(_args()).compile()
    model_id, kw = recorder[0]
    assert model_id == "black-forest-labs/FLUX.1-dev"
    assert kw["backend"] == "cuda" and kw["load"] is False
    assert kw["parallel"].tp_degree == 1 and kw["dtype"] == "bf16"
    assert kw["local_files_only"] is True and kw["height"] == 1024 and kw["width"] == 1024
    assert "application_kwargs" not in kw


def test_generate_without_a_compiled_artifact_compiles_first(recorder, tmp_path):
    out = tmp_path / "flux.png"
    _orch(_args(output=str(out))).generate()
    _, kw = recorder[0]
    assert kw["load"] is True and kw["force_compile"] is False and out.is_file()


def test_generate_creates_the_output_directory(recorder, tmp_path):
    out = tmp_path / "deep" / "er" / "flux.png"
    _orch(_args(output=str(out))).generate()
    assert out.is_file()


def test_steps_default_to_28_and_guidance_to_3_5(recorder, tmp_path):
    orch = _orch(_args(output=str(tmp_path / "f.png")))
    orch.generate()
    prompt, kw = orch._last_pipe.app.calls[0]
    assert prompt == "a cat" and kw["seed"] == 42 and kw["output_type"] == "pil"
    assert kw["num_inference_steps"] == SPEC.default_steps == 28
    assert kw["guidance_scale"] == SPEC.default_guidance_scale == 3.5


def test_steps_seed_and_guidance_are_forwarded(recorder, tmp_path):
    orch = _orch(
        _args(output=str(tmp_path / "f.png"), steps=7, seed=3, guidance_scale=5.0)
    )
    orch.generate()
    _, kw = orch._last_pipe.app.calls[0]
    assert kw["num_inference_steps"] == 7 and kw["seed"] == 3 and kw["guidance_scale"] == 5.0


def test_height_width_cache_dir_and_force_reach_from_pretrained(recorder):
    _orch(_args(cache_dir="/c", force=True, height=768, width=512)).compile()
    _, kw = recorder[0]
    assert kw["compile_cache_dir"] == "/c" and kw["force_compile"] is True
    assert (kw["height"], kw["width"]) == (768, 512)


def test_schnell_is_built_from_its_own_repo(recorder, monkeypatch):
    args = _args(model_id="black-forest-labs/FLUX.1-schnell")
    _orch(args).compile()
    assert recorder[0][0] == "black-forest-labs/FLUX.1-schnell"
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid) or "/snap",
    )
    _orch(args).download()
    assert seen["mid"] == "black-forest-labs/FLUX.1-schnell"


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"tp_degree": 2}, "not supported"),
        ({"sp_enabled": True}, "not supported"),
        ({"cp_degree": 2}, "not supported"),
        ({"cfg_parallel": True}, "does not apply"),
        ({"shapes": "1024x1024,768x768"}, "not supported"),
        ({"teacache_cadence": 2}, "not supported"),
        ({"teacache_speedup": 1.5}, "not supported"),
        ({"teacache_online_delta": 0.1}, "not supported"),
        ({"teacache_calibration": "/x.json"}, "not supported"),
        ({"taef1": True}, "not supported"),
        ({"taef1_path": "madebyollin/taef1"}, "not supported"),
        ({"host_vae": True}, "not supported"),
    ],
)
def test_unsupported_flags_are_refused_before_any_load(recorder, over, expected):
    with pytest.raises(SystemExit) as exc:
        _orch(_args(**over)).generate()
    assert expected in str(exc.value) and "[flux/cuda]" in str(exc.value) and recorder == []


def test_download_resolves_the_hub_snapshot(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid, **kw) or "/snap",
    )
    _orch(_args()).download()
    assert seen["mid"] == "black-forest-labs/FLUX.1-dev" and seen["local_files_only"] is False


def test_run_is_download_compile_generate(recorder, monkeypatch, tmp_path):
    order = []
    orch = _orch(_args(output=str(tmp_path / "f.png")))
    monkeypatch.setattr(orch, "download", lambda: order.append("download"))
    monkeypatch.setattr(orch, "compile", lambda: order.append("compile"))
    monkeypatch.setattr(orch, "generate", lambda: order.append("generate"))
    orch.run()
    assert order == ["download", "compile", "generate"]


def test_run_refuses_unsupported_flags_before_any_download(recorder, monkeypatch):
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: pytest.fail("must refuse before resolving the snapshot"),
    )
    with pytest.raises(SystemExit) as exc:
        _orch(_args(taef1=True)).run()
    assert "not supported" in str(exc.value) and recorder == []


def test_lone_teacache_calibration_is_refused_by_name(recorder):
    """``main._validate_teacache`` lets a lone ``--teacache-calibration`` through and Flux on
    the GPU flow forwards nothing, so without this refusal the calibration file is silently
    dropped."""
    with pytest.raises(SystemExit) as exc:
        _orch(_args(teacache_calibration="/x.json")).generate()
    assert "--teacache-calibration" in str(exc.value) and "not supported" in str(exc.value)
    assert recorder == []


def test_torch_compile_is_forwarded_to_the_flux_application():
    # The measured, opt-in compile option reaches CudaFluxApplication as a kwarg.
    kw = _orch(_args(torch_compile=True)).pipeline_kwargs(load=True)
    assert kw["application_kwargs"] == {"torch_compile": True}
    kw = _orch(_args()).pipeline_kwargs(load=True)
    assert "application_kwargs" not in kw
