"""Qwen-Image on CUDA through the CLI: one process, DiffletPipeline + generate() (CUDA plan F3).

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""

from __future__ import annotations

import argparse

import pytest
from PIL import Image

from difflet.cli.orchestrators import qwen_image_cuda as mod


def _args(**over):
    d = dict(
        model_id="Qwen/Qwen-Image",
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
        output="/tmp/x/qwen.png",
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
    mod.CudaQwenImageOrchestrator(_args()).compile()
    model_id, kw = recorder[0]
    assert model_id == "Qwen/Qwen-Image" and kw["backend"] == "cuda" and kw["load"] is False
    assert kw["parallel"].tp_degree == 1 and kw["dtype"] == "bf16"
    assert kw["local_files_only"] is True and kw["height"] == 1024 and kw["width"] == 1024


def test_generate_without_a_compiled_artifact_compiles_first(recorder, tmp_path):
    # from_pretrained compiles when the manifest is missing; the orchestrator never
    # demands a separate `difflet compile` (Review Focus 3).
    out = tmp_path / "qwen.png"
    mod.CudaQwenImageOrchestrator(_args(output=str(out))).generate()
    _, kw = recorder[0]
    assert kw["load"] is True and kw["force_compile"] is False and out.is_file()


def test_generate_creates_the_output_directory(recorder, tmp_path):
    out = tmp_path / "deep" / "er" / "qwen.png"
    mod.CudaQwenImageOrchestrator(_args(output=str(out))).generate()
    assert out.is_file()


def test_generate_passes_prompt_steps_and_seed(recorder, tmp_path):
    orch = mod.CudaQwenImageOrchestrator(_args(output=str(tmp_path / "q.png"), steps=7, seed=3))
    orch.generate()
    prompt, kw = orch._last_pipe.app.calls[0]
    assert prompt == "a cat" and kw["num_inference_steps"] == 7 and kw["seed"] == 3


def test_steps_default_to_fifty_on_cuda(recorder, tmp_path):
    orch = mod.CudaQwenImageOrchestrator(_args(output=str(tmp_path / "q.png")))
    orch.generate()
    assert orch._last_pipe.app.calls[0][1]["num_inference_steps"] == mod.DEFAULT_STEPS == 50


def test_cache_dir_and_force_reach_from_pretrained(recorder):
    mod.CudaQwenImageOrchestrator(_args(cache_dir="/c", force=True)).compile()
    _, kw = recorder[0]
    assert kw["compile_cache_dir"] == "/c" and kw["force_compile"] is True


@pytest.mark.parametrize(
    "over,follow_up",
    [
        ({"shapes": "1024x1024,768x768"}, "F3"),
        ({"sp_enabled": True}, "F6"),
        ({"teacache_speedup": 1.5}, "F4b"),
        ({"teacache_online_delta": 0.1}, "F4b"),
        ({"cfg_parallel": True}, "distilled model"),
        ({"tp_degree": 2}, "F6"),
        ({"cp_degree": 2}, "F2b"),
    ],
)
def test_unsupported_flags_are_refused_before_any_load(recorder, over, follow_up):
    with pytest.raises(SystemExit) as exc:
        mod.CudaQwenImageOrchestrator(_args(**over)).generate()
    assert follow_up in str(exc.value) and recorder == []


def test_download_resolves_the_hub_snapshot(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid, **kw) or "/snap",
    )
    mod.CudaQwenImageOrchestrator(_args()).download()
    assert seen["mid"] == "Qwen/Qwen-Image" and seen["local_files_only"] is False


def test_run_is_download_compile_generate(recorder, monkeypatch, tmp_path):
    order = []
    orch = mod.CudaQwenImageOrchestrator(_args(output=str(tmp_path / "q.png")))
    monkeypatch.setattr(orch, "download", lambda: order.append("download"))
    monkeypatch.setattr(orch, "compile", lambda: order.append("compile"))
    monkeypatch.setattr(orch, "generate", lambda: order.append("generate"))
    orch.run()
    assert order == ["download", "compile", "generate"]


def test_run_refuses_unsupported_flags_before_any_download(recorder, monkeypatch):
    # Review finding: run() is download → compile → generate, and the refusal lived in
    # compile/generate, so `difflet run --cp-degree 2` pulled the 58 GB snapshot first.
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: pytest.fail("must refuse before resolving the snapshot"),
    )
    with pytest.raises(SystemExit) as exc:
        mod.CudaQwenImageOrchestrator(_args(cp_degree=2)).run()
    assert "F2b" in str(exc.value) and recorder == []


def test_guidance_scale_is_refused_not_dropped(recorder):
    # The application refuses guidance_scale (guidance_embeds=False → silent no-op); the CLI
    # must not hide that by never forwarding the flag.
    with pytest.raises(SystemExit) as exc:
        mod.CudaQwenImageOrchestrator(_args(guidance_scale=3.0)).generate()
    assert "guidance" in str(exc.value) and recorder == []


def test_torch_compile_is_opt_in():
    assert mod.pipeline_kwargs(_args(), load=True).get("application_kwargs") is None
    kw = mod.pipeline_kwargs(_args(torch_compile=True), load=True)
    assert kw["application_kwargs"] == {"torch_compile": True}


def test_teacache_cadence_is_forwarded_as_an_application_kwarg(recorder, tmp_path):
    out = tmp_path / "qwen.png"
    mod.CudaQwenImageOrchestrator(_args(teacache_cadence=2, output=str(out))).generate()
    assert recorder[0][1]["application_kwargs"] == {"teacache_cadence": 2}


def test_cuda_pipeline_kwargs_cfg_parallel_defaults_off():
    """The serving caller passes no ``cfg_parallel`` and keeps the single-branch config."""
    common = dict(
        tp_degree=None,
        height=None,
        width=None,
        cache_dir=None,
        revision=None,
        force_compile=False,
        load=False,
    )
    assert mod.cuda_pipeline_kwargs(**common)["parallel"].cfg_parallel_enabled is False
    on = mod.cuda_pipeline_kwargs(**common, cfg_parallel=True)["parallel"]
    assert on.cfg_parallel_enabled is True and on.tp_degree == 1


def test_qwen_is_one_process_and_refuses_cfg_parallel_as_distilled(recorder):
    assert mod.CudaQwenImageOrchestrator.ACCEPTS_CFG_PARALLEL is False
    assert mod.CudaQwenImageOrchestrator(_args()).world_size == 1
    with pytest.raises(SystemExit) as exc:
        mod.CudaQwenImageOrchestrator(_args(cfg_parallel=True)).compile()
    assert "--cfg-parallel" in str(exc.value) and "[qwen_image/cuda]" in str(exc.value)
    assert recorder == []
