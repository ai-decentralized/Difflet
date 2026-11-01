"""The GPU CLI orchestrator (``difflet.backends.gpu.cli``), once per model spec.

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU. What
only one model does is in test_orchestrator_{qwen,wan}_cuda.py, which reuse these helpers.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest
import torch
from PIL import Image

from difflet.backends.gpu.cli import GpuOrchestrator
from difflet.cli.orchestrators import wan as trainium_wan
from difflet.models.spec import gpu_model_specs
from difflet.platforms.base import PARALLEL_MODE_FLAGS

SPECS = gpu_model_specs()


@pytest.fixture(params=SPECS, ids=lambda s: s.model_type)
def spec(request):
    return request.param


def _ext(spec):
    return ".mp4" if spec.output == "video" else ".png"


def _args(spec, **over):
    d = dict(
        model_id=spec.cli_model_ids[0],
        tp_degree=None,
        cp_degree=None,
        cp_mode="gather_kv",
        cfg_parallel=False,
        sp_enabled=False,
        dp=None,
        height=None,
        width=None,
        num_frames=None,
        shapes=None,
        cache_dir=None,
        force=False,
        revision=None,
        prompt="a cat",
        output=f"/tmp/x/out{_ext(spec)}",
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


def _orch(spec, **over):
    return GpuOrchestrator(_args(spec, **over), spec)


class _App:
    def __init__(self, spec):
        self.calls = []
        if spec.output == "video":
            # The application's ``output_type="pt"`` result: (B, 3, F, H, W) in [-1, 1].
            torch.manual_seed(0)
            self.result = torch.rand(1, 3, 9, 16, 16) * 2 - 1
        else:
            self.result = [Image.new("RGB", (8, 8), "white")]

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return self.result


class _Pipe:
    def __init__(self, spec):
        self.app = _App(spec)
        self.compiled_path = "/fake/compiled"


@pytest.fixture
def recorder(monkeypatch, spec):
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe(spec)

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained", staticmethod(fake)
    )
    return calls


@pytest.fixture(autouse=True)
def saved(monkeypatch):
    """Record the MP4 writer's calls; write a marker file so the output exists."""
    calls = []

    def fake(tensor, output_path, *, fps=16):
        calls.append((tensor, output_path))
        Path(output_path).write_bytes(b"mp4")
        return True

    monkeypatch.setattr(trainium_wan, "_save_video", fake)
    return calls


def _exploding_generate(message):
    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        if prompt == "boom":
            raise RuntimeError(message)
        return self.result

    return generate


def test_compile_builds_the_pipeline_from_the_spec_without_loading(spec, recorder):
    _orch(spec).compile()
    ((model_id, kw),) = recorder
    assert model_id == spec.cli_model_ids[0] and kw["backend"] == "cuda" and kw["load"] is False
    assert kw["parallel"].tp_degree == 1 and kw["parallel"].cfg_parallel_enabled is False
    assert kw["dtype"] == "bf16" and kw["local_files_only"] is True
    assert (kw["height"], kw["width"]) == (spec.default_height, spec.default_width)
    assert kw.get("num_frames") == spec.default_num_frames
    assert "application_kwargs" not in kw


def test_generate_without_a_compiled_artifact_compiles_first(spec, recorder, tmp_path):
    # from_pretrained compiles when the manifest is missing; the orchestrator never demands a
    # separate `difflet compile`. The output's directory is created.
    out = tmp_path / "deep" / "er" / f"out{_ext(spec)}"
    _orch(spec, output=str(out)).generate()
    ((_, kw),) = recorder
    assert kw["load"] is True and kw["force_compile"] is False and out.is_file()


def test_generate_defaults_come_from_the_spec(spec, recorder, tmp_path):
    orch = _orch(spec, output=str(tmp_path / f"out{_ext(spec)}"))
    orch.generate()
    ((prompt, kw),) = orch._last_pipe.app.calls
    assert prompt == "a cat" and kw["seed"] == 42
    assert kw["output_type"] == ("pt" if spec.output == "video" else "pil")
    assert kw["num_inference_steps"] == spec.default_steps
    assert kw.get("guidance_scale") == spec.default_guidance_scale
    assert kw.get("num_frames") == spec.default_num_frames
    # No --negative-prompt flag: the application's default applies.
    assert "negative_prompt" not in kw


def test_flags_reach_from_pretrained_and_generate(spec, recorder, tmp_path):
    over = dict(steps=7, seed=3, height=768, width=512, cache_dir="/c", force=True)
    if spec.guidance != "none":
        over["guidance_scale"] = 5.0
    if spec.output == "video":
        over["num_frames"] = 17
    orch = _orch(spec, output=str(tmp_path / f"out{_ext(spec)}"), **over)
    orch.generate()
    ((_, load_kw),) = recorder
    assert load_kw["compile_cache_dir"] == "/c" and load_kw["force_compile"] is True
    assert (load_kw["height"], load_kw["width"]) == (768, 512)
    ((_, kw),) = orch._last_pipe.app.calls
    assert kw["num_inference_steps"] == 7 and kw["seed"] == 3
    assert kw.get("guidance_scale") == over.get("guidance_scale")
    assert kw.get("num_frames") == load_kw.get("num_frames") == over.get("num_frames")


_REFUSED_VALUES = {
    "shapes": "1024x1024,768x768",
    "teacache_cadence": 2,
    "teacache_speedup": 1.5,
    "teacache_online_delta": 0.1,
    "teacache_calibration": "/x.json",
    "taef1": True,
    "taef1_path": "madebyollin/taef1",
    "host_vae": True,
}


def _refusals():
    for spec in SPECS:
        cases = [
            ({"tp_degree": 2}, PARALLEL_MODE_FLAGS["tp"]),
            ({"sp_enabled": True}, PARALLEL_MODE_FLAGS["sp"]),
            ({"cp_degree": 2}, PARALLEL_MODE_FLAGS["cp"]),
            *(({attr: _REFUSED_VALUES[attr]}, label) for attr, label in spec.refused_cli_flags),
        ]
        if not spec.accepts_cfg_parallel:
            cases.append(({"cfg_parallel": True}, "does not apply"))
        for over, expected in cases:
            yield pytest.param(spec, over, expected, id=f"{spec.model_type}-{next(iter(over))}")


@pytest.mark.parametrize("spec,over,expected", list(_refusals()))
def test_unsupported_flags_are_refused_by_name_before_any_load(spec, over, expected, recorder):
    with pytest.raises(SystemExit) as exc:
        _orch(spec, **over).generate()
    assert f"[{spec.model_type}/cuda]" in str(exc.value) and expected in str(exc.value)
    assert recorder == []


def test_run_refuses_unsupported_flags_before_any_download(spec, recorder, monkeypatch):
    # run() is download → compile → generate: a refusal that lived only in compile/generate
    # would let `difflet run --cp-degree 2` pull the snapshot first.
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: pytest.fail("must refuse before resolving the snapshot"),
    )
    with pytest.raises(SystemExit) as exc:
        _orch(spec, cp_degree=2).run()
    assert "not supported" in str(exc.value) and recorder == []


@pytest.mark.parametrize(
    "spec,model_id", [pytest.param(s, m, id=m) for s in SPECS for m in s.cli_model_ids]
)
def test_each_checkpoint_is_downloaded_and_built_from_its_own_repo(
    spec, model_id, recorder, monkeypatch
):
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid, **kw) or "/snap",
    )
    orch = _orch(spec, model_id=model_id)
    orch.download()
    orch.compile()
    assert seen["mid"] == model_id and seen["local_files_only"] is False
    assert recorder[0][0] == model_id


def test_run_is_download_compile_generate(spec, monkeypatch):
    order = []
    orch = _orch(spec)
    monkeypatch.setattr(orch, "download", lambda: order.append("download"))
    monkeypatch.setattr(orch, "compile", lambda: order.append("compile"))
    monkeypatch.setattr(orch, "generate", lambda: order.append("generate"))
    orch.run()
    assert order == ["download", "compile", "generate"]


def test_torch_compile_is_opt_in(spec):
    assert "application_kwargs" not in _orch(spec).pipeline_kwargs(load=True)
    kw = _orch(spec, torch_compile=True).pipeline_kwargs(load=True)
    assert kw["application_kwargs"] == {"torch_compile": True}


# ------------------------------------------------- batch mode: a dp worker


def _manifest(spec, tmp_path, requests):
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    requests_dir = tmp_path / "requests"
    write_manifest(
        [
            RequestSpec(index=i, output=str(tmp_path / "out" / f"r{i}{_ext(spec)}"), **request)
            for i, request in enumerate(requests)
        ],
        requests_dir,
    )
    return requests_dir


def _batch(requests_dir, **over):
    """The flags the dp router gives a worker."""
    batch = dict(
        prompt=None,
        output=None,
        requests_dir=str(requests_dir),
        worker_index=0,
        dp_schedule="least_loaded",
        work_dir=str(requests_dir.parent / "worker_0"),
    )
    return {**batch, **over}


def test_batch_mode_loads_once_and_writes_every_request(spec, recorder, tmp_path):
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(
        spec,
        tmp_path,
        [dict(prompt="p0", seed=0), dict(prompt="p1", seed=1, steps=3), dict(prompt="p2", seed=2)],
    )
    orch = _orch(spec, **_batch(requests_dir, steps=9))
    orch.generate(argv=[])
    assert [kw["load"] for _, kw in recorder] == [True]  # one load for the whole batch
    calls = orch._last_pipe.app.calls
    assert [(p, kw["seed"], kw["num_inference_steps"]) for p, kw in calls] == [
        ("p0", 0, 9),  # --steps is the worker's default
        ("p1", 1, 3),  # the request's own steps win
        ("p2", 2, 9),
    ]
    assert all((tmp_path / "out" / f"r{i}{_ext(spec)}").is_file() for i in range(3))
    s = summarize(requests_dir)
    assert s.done == [0, 1, 2] and not s.failed and not s.unfinished
    # Every request claimed exactly once, by this worker.
    assert [(requests_dir / f"req_{i:04d}.claim").read_text() for i in range(3)] == ["0"] * 3


def test_batch_mode_steps_fall_back_to_the_model_default(spec, recorder, tmp_path):
    requests_dir = _manifest(spec, tmp_path, [dict(prompt="p0")])
    orch = _orch(spec, **_batch(requests_dir))
    orch.generate(argv=[])
    assert orch._last_pipe.app.calls[0][1]["num_inference_steps"] == spec.default_steps


def test_a_failing_request_is_marked_failed_and_the_loop_continues(
    spec, recorder, monkeypatch, tmp_path
):
    from difflet.cli.dp.claims import summarize

    monkeypatch.setattr(_App, "generate", _exploding_generate("generation exploded"))
    requests_dir = _manifest(
        spec, tmp_path, [dict(prompt="p0"), dict(prompt="boom"), dict(prompt="p2")]
    )
    orch = _orch(spec, **_batch(requests_dir))
    orch.generate(argv=[])
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p0", "boom", "p2"]
    s = summarize(requests_dir)
    assert s.done == [0, 2] and list(s.failed) == [1] and not s.unfinished
    assert "generation exploded" in s.failed[1]
    assert not (tmp_path / "out" / f"r1{_ext(spec)}").exists()


def test_round_robin_claims_only_this_worker_s_requests(spec, recorder, tmp_path):
    from difflet.cli.dp.claims import summarize
    from difflet.cli.dp.requests_io import read_manifest, write_manifest

    requests_dir = _manifest(spec, tmp_path, [dict(prompt=f"p{i}") for i in range(4)])
    write_manifest(
        [dataclasses.replace(r, assigned_worker=r.index % 2) for r in read_manifest(requests_dir)],
        requests_dir,
    )
    orch = _orch(spec, **_batch(requests_dir, worker_index=1, dp_schedule="round_robin"))
    orch.generate(argv=[])
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p1", "p3"]
    s = summarize(requests_dir)
    assert s.done == [1, 3] and s.unfinished == [0, 2]
