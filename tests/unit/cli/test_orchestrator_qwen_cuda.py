"""Qwen-Image on the GPU flow through the CLI: one process, DiffletPipeline + generate().

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""

from __future__ import annotations

import argparse

import pytest
from PIL import Image

from difflet.backends.gpu.cli import GpuOrchestrator, gpu_pipeline_kwargs
from difflet.models.qwen_image.spec import SPEC


def _orch(args):
    return GpuOrchestrator(args, SPEC)


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
    _orch(_args()).compile()
    model_id, kw = recorder[0]
    assert model_id == "Qwen/Qwen-Image" and kw["backend"] == "cuda" and kw["load"] is False
    assert kw["parallel"].tp_degree == 1 and kw["dtype"] == "bf16"
    assert kw["local_files_only"] is True and kw["height"] == 1024 and kw["width"] == 1024


def test_generate_without_a_compiled_artifact_compiles_first(recorder, tmp_path):
    # from_pretrained compiles when the manifest is missing; the orchestrator never
    # demands a separate `difflet compile`.
    out = tmp_path / "qwen.png"
    _orch(_args(output=str(out))).generate()
    _, kw = recorder[0]
    assert kw["load"] is True and kw["force_compile"] is False and out.is_file()


def test_generate_creates_the_output_directory(recorder, tmp_path):
    out = tmp_path / "deep" / "er" / "qwen.png"
    _orch(_args(output=str(out))).generate()
    assert out.is_file()


def test_generate_passes_prompt_steps_and_seed(recorder, tmp_path):
    orch = _orch(_args(output=str(tmp_path / "q.png"), steps=7, seed=3))
    orch.generate()
    prompt, kw = orch._last_pipe.app.calls[0]
    assert prompt == "a cat" and kw["num_inference_steps"] == 7 and kw["seed"] == 3


def test_steps_default_to_fifty(recorder, tmp_path):
    orch = _orch(_args(output=str(tmp_path / "q.png")))
    orch.generate()
    assert orch._last_pipe.app.calls[0][1]["num_inference_steps"] == SPEC.default_steps == 50


def test_cache_dir_and_force_reach_from_pretrained(recorder):
    _orch(_args(cache_dir="/c", force=True)).compile()
    _, kw = recorder[0]
    assert kw["compile_cache_dir"] == "/c" and kw["force_compile"] is True


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"shapes": "1024x1024,768x768"}, "not supported"),
        ({"sp_enabled": True}, "not supported"),
        ({"teacache_speedup": 1.5}, "not supported"),
        ({"teacache_online_delta": 0.1}, "not supported"),
        ({"cfg_parallel": True}, "does not apply"),
        ({"tp_degree": 2}, "not supported"),
        ({"cp_degree": 2}, "not supported"),
    ],
)
def test_unsupported_flags_are_refused_before_any_load(recorder, over, expected):
    with pytest.raises(SystemExit) as exc:
        _orch(_args(**over)).generate()
    assert expected in str(exc.value) and recorder == []


def test_download_resolves_the_hub_snapshot(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid, **kw) or "/snap",
    )
    _orch(_args()).download()
    assert seen["mid"] == "Qwen/Qwen-Image" and seen["local_files_only"] is False


def test_run_is_download_compile_generate(recorder, monkeypatch, tmp_path):
    order = []
    orch = _orch(_args(output=str(tmp_path / "q.png")))
    monkeypatch.setattr(orch, "download", lambda: order.append("download"))
    monkeypatch.setattr(orch, "compile", lambda: order.append("compile"))
    monkeypatch.setattr(orch, "generate", lambda: order.append("generate"))
    orch.run()
    assert order == ["download", "compile", "generate"]


def test_run_refuses_unsupported_flags_before_any_download(recorder, monkeypatch):
    # run() is download → compile → generate: a refusal that lived only in compile/generate
    # would let `difflet run --cp-degree 2` pull the 58 GB snapshot first.
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: pytest.fail("must refuse before resolving the snapshot"),
    )
    with pytest.raises(SystemExit) as exc:
        _orch(_args(cp_degree=2)).run()
    assert "not supported" in str(exc.value) and recorder == []


def test_guidance_scale_is_refused_not_dropped(recorder):
    # The application refuses guidance_scale (guidance_embeds=False → silent no-op); the CLI
    # must not hide that by never forwarding the flag.
    with pytest.raises(SystemExit) as exc:
        _orch(_args(guidance_scale=3.0)).generate()
    assert "guidance" in str(exc.value) and recorder == []


def test_torch_compile_is_opt_in():
    assert _orch(_args()).pipeline_kwargs(load=True).get("application_kwargs") is None
    kw = _orch(_args(torch_compile=True)).pipeline_kwargs(load=True)
    assert kw["application_kwargs"] == {"torch_compile": True}


def test_teacache_cadence_is_forwarded_as_an_application_kwarg(recorder, tmp_path):
    out = tmp_path / "qwen.png"
    _orch(_args(teacache_cadence=2, output=str(out))).generate()
    assert recorder[0][1]["application_kwargs"] == {"teacache_cadence": 2}


def test_gpu_pipeline_kwargs_cfg_parallel_defaults_off():
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
    assert gpu_pipeline_kwargs(SPEC, **common)["parallel"].cfg_parallel_enabled is False
    on = gpu_pipeline_kwargs(SPEC, **common, cfg_parallel=True)["parallel"]
    assert on.cfg_parallel_enabled is True and on.tp_degree == 1


def test_qwen_is_one_process_and_refuses_cfg_parallel_as_distilled(recorder):
    assert SPEC.accepts_cfg_parallel is False
    assert _orch(_args()).world_size == 1
    with pytest.raises(SystemExit) as exc:
        _orch(_args(cfg_parallel=True)).compile()
    assert "--cfg-parallel" in str(exc.value) and "[qwen_image/cuda]" in str(exc.value)
    assert recorder == []


# ------------------------------------------------- batch mode: a dp worker


def _manifest(tmp_path, specs):
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    requests_dir = tmp_path / "requests"
    reqs = [
        RequestSpec(index=i, output=str(tmp_path / "out" / f"r{i}.png"), **spec)
        for i, spec in enumerate(specs)
    ]
    write_manifest(reqs, requests_dir)
    return requests_dir


def _batch_args(requests_dir, **over):
    batch = dict(
        prompt=None,
        output=None,
        requests_dir=str(requests_dir),
        worker_index=0,
        dp_schedule="least_loaded",
        work_dir=str(requests_dir.parent / "worker_0"),
    )
    return _args(**{**batch, **over})


class _FailingApp(_App):
    def generate(self, prompt, **kw):
        if prompt == "boom":
            self.calls.append((prompt, kw))
            raise RuntimeError("generation exploded")
        return super().generate(prompt, **kw)


def test_batch_mode_loads_once_and_writes_every_request(recorder, tmp_path):
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(
        tmp_path,
        [dict(prompt="p0", seed=0), dict(prompt="p1", seed=1, steps=3), dict(prompt="p2", seed=2)],
    )
    orch = _orch(_batch_args(requests_dir, steps=9))
    orch.generate(argv=[])
    assert [kw["load"] for _, kw in recorder] == [True]  # one load for the whole batch
    calls = orch._last_pipe.app.calls
    assert [(p, kw["seed"], kw["num_inference_steps"]) for p, kw in calls] == [
        ("p0", 0, 9),  # --steps is the worker's default
        ("p1", 1, 3),  # the request's own steps win
        ("p2", 2, 9),
    ]
    assert all((tmp_path / "out" / f"r{i}.png").is_file() for i in range(3))
    s = summarize(requests_dir)
    assert s.done == [0, 1, 2] and not s.failed and not s.unfinished
    # Every request claimed exactly once, by this worker.
    assert [(requests_dir / f"req_{i:04d}.claim").read_text() for i in range(3)] == ["0"] * 3


def test_batch_mode_steps_fall_back_to_the_model_default(recorder, tmp_path):
    requests_dir = _manifest(tmp_path, [dict(prompt="p0")])
    orch = _orch(_batch_args(requests_dir))
    orch.generate(argv=[])
    assert orch._last_pipe.app.calls[0][1]["num_inference_steps"] == SPEC.default_steps


def test_a_failing_request_is_marked_failed_and_the_loop_continues(recorder, monkeypatch, tmp_path):
    from difflet.cli.dp.claims import summarize

    monkeypatch.setattr(_Pipe, "__init__", lambda self: setattr(self, "app", _FailingApp()))
    requests_dir = _manifest(tmp_path, [dict(prompt="p0"), dict(prompt="boom"), dict(prompt="p2")])
    orch = _orch(_batch_args(requests_dir))
    orch.generate(argv=[])
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p0", "boom", "p2"]
    s = summarize(requests_dir)
    assert s.done == [0, 2] and list(s.failed) == [1] and not s.unfinished
    assert "generation exploded" in s.failed[1]
    assert not (tmp_path / "out" / "r1.png").exists()


def test_a_request_guidance_scale_fails_that_request_on_qwen(recorder, tmp_path):
    # Qwen-Image takes no guidance scale: a request carrying one is failed, never dropped.
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(tmp_path, [dict(prompt="p0", guidance_scale=4.0), dict(prompt="p1")])
    orch = _orch(_batch_args(requests_dir))
    orch.generate(argv=[])
    s = summarize(requests_dir)
    assert s.done == [1] and list(s.failed) == [0] and "guidance" in s.failed[0]
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p1"]


def test_a_request_negative_prompt_fails_that_request_on_qwen(recorder, tmp_path):
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(tmp_path, [dict(prompt="p0", negative_prompt="blurry")])
    _orch(_batch_args(requests_dir)).generate(argv=[])
    s = summarize(requests_dir)
    assert list(s.failed) == [0] and "negative" in s.failed[0]


def test_round_robin_claims_only_this_worker_s_requests(recorder, tmp_path):
    import dataclasses

    from difflet.cli.dp.claims import summarize
    from difflet.cli.dp.requests_io import read_manifest, write_manifest

    requests_dir = _manifest(tmp_path, [dict(prompt=f"p{i}") for i in range(4)])
    write_manifest(
        [dataclasses.replace(r, assigned_worker=r.index % 2) for r in read_manifest(requests_dir)],
        requests_dir,
    )
    orch = _orch(
        _batch_args(requests_dir, worker_index=1, dp_schedule="round_robin")
    )
    orch.generate(argv=[])
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p1", "p3"]
    s = summarize(requests_dir)
    assert s.done == [1, 3] and s.unfinished == [0, 2]


def test_generate_kwargs_take_the_request_overrides():
    from difflet.cli.dp.requests_io import RequestSpec

    orch = _orch(_args(steps=9, seed=5))
    assert orch.generate_kwargs()["num_inference_steps"] == 9
    req = RequestSpec(index=0, prompt="p", output="o.png", seed=11, steps=4)
    kw = orch.generate_kwargs(req)
    assert kw["num_inference_steps"] == 4 and kw["seed"] == 11
