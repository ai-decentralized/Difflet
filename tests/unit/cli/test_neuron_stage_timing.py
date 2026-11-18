"""The neuron generate stage's step timing (``torchrun._time_scheduler_steps``) on a scheduler
the orchestrator deep-copies after the hook is installed, as ``LTX2Orchestrator`` copies the
video scheduler into its audio scheduler: the copy must drive its own step index, not the
original's (found on the device as an IndexError past the last sigma, one iteration in two),
and the marks must count one video step per denoising iteration. The hook tests are
torch-free; the stage tests drive ``run_generate_stage`` itself (one process, rank 0, no
process group) against a pipeline stand-in whose call steps the scheduler and its deep copy
as LTX-2 does, and pin where the hook is installed: in ``generate`` mode with ``timing``
only, with the marks cleared per request; the LTX-2 driver test runs its stage body against
a recording stand-in for ``run_generate_stage`` (no pipeline is built)."""

import argparse
import copy
from types import SimpleNamespace

from difflet.cli.orchestrators import ltx_2 as ltx2_cli
from difflet.cli.orchestrators import torchrun as tr


class _Scheduler:
    """A scheduler whose class-level ``step`` walks a table of ``n`` entries and raises past
    its end, as diffusers' flow-match scheduler does on ``sigmas[step_index + 1]``."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.i = 0

    def step(self, model_output):
        if self.i >= self.n:
            raise IndexError(f"index {self.i} is out of bounds for dimension 0 with size {self.n}")
        self.i += 1
        return model_output


def _pipe(scheduler) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(pipeline=SimpleNamespace(scheduler=scheduler)))


def test_the_timed_step_survives_a_deep_copied_scheduler():
    n = 4
    scheduler = _Scheduler(n)
    marks: list[float] = []
    tr._time_scheduler_steps(_pipe(scheduler), marks)
    audio = copy.deepcopy(scheduler)

    for step in range(n):
        assert scheduler.step(step) == step
        assert audio.step(step) == step

    assert scheduler.i == n
    assert audio.i == n
    assert len(marks) == n
    assert marks == sorted(marks)


def test_a_pipeline_without_a_scheduler_is_left_alone():
    marks: list[float] = []
    tr._time_scheduler_steps(SimpleNamespace(app=SimpleNamespace()), marks)
    tr._time_scheduler_steps(_pipe(None), marks)
    assert marks == []


def test_a_scheduler_with_an_instance_level_step_is_left_alone():
    """Timing is evidence, never the stage's failure: a scheduler whose ``step`` lives on the
    instance alone (no class-level method to rebind) keeps it, un-timed."""
    calls: list[int] = []
    scheduler = SimpleNamespace(step=lambda model_output: calls.append(model_output))
    marks: list[float] = []
    tr._time_scheduler_steps(_pipe(scheduler), marks)
    scheduler.step(1)
    assert calls == [1]
    assert marks == []


def test_the_timed_step_passes_the_call_through():
    calls: list[tuple] = []

    class _Recording:
        def step(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "latents"

    scheduler = _Recording()
    marks: list[float] = []
    tr._time_scheduler_steps(_pipe(scheduler), marks)
    assert scheduler.step("velocity", 0.5, return_dict=False) == "latents"
    assert calls == [(("velocity", 0.5), {"return_dict": False})]
    assert len(marks) == 1


# ------------------------------------------------------------- the stage body (rank 0)


class _StageApp:
    """What ``run_generate_stage`` reads from the application: ``load``, ``__call__`` and
    ``pipeline.scheduler``. A call resets the scheduler (``set_timesteps``), deep-copies it as
    ``LTX2Orchestrator`` copies the video scheduler into its audio one, and steps both
    ``steps`` times: the hook, installed before the copy, must survive it."""

    def __init__(self, steps: int) -> None:
        self.pipeline = SimpleNamespace(scheduler=_Scheduler(steps))
        self.steps = steps
        self.calls: list[dict] = []

    def load(self, *args, **kwargs) -> None:
        pass

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        video = self.pipeline.scheduler
        video.i = 0
        audio = copy.deepcopy(video)
        for step in range(self.steps):
            video.step(step)
            audio.step(step)
        return {"latents": kwargs["prompt"], "audio_steps": audio.i}


class _StagePipe:
    def __init__(self, steps: int) -> None:
        self.app = _StageApp(steps)
        self.loads = 0

    def load(self, *, skip_warmup: bool) -> None:
        self.loads += 1

    def __call__(self, **kwargs):
        return self.app(**kwargs)


def _stage_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        tp_degree=4, cp_degree=1, cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False,
        cache_dir="/nonexistent-cache", stage_mode="generate", requests_dir=None,
        prompt="a fox", output="/tmp/o.mp4", seed=42, steps=4, guidance_scale=1.0,
        negative_prompt=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_stage(monkeypatch, args, *, timing: bool, steps: int = 4) -> dict:
    """``run_generate_stage`` on one rank against ``_StagePipe``; what it saved and whether it
    finished a compile are recorded."""
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    pipe = _StagePipe(steps)
    monkeypatch.setattr(DiffletPipeline, "from_pretrained",
                        classmethod(lambda cls, model_id, **kwargs: pipe))
    saved: list[tuple] = []
    compiled: list[bool] = []
    tr.run_generate_stage(
        args, "neuron", tag="t", model_id="m", pipeline_kwargs={}, application_kwargs={},
        request_kwargs=lambda req: {"prompt": req.prompt, "seed": req.seed},
        save=lambda req, out: saved.append((req.prompt, out)),
        finish_compile=lambda: compiled.append(True), timing=timing,
    )
    return {"pipe": pipe, "saved": saved, "compiled": compiled}


def _queue(tmp_path, prompts):
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    requests = tmp_path / "requests"
    write_manifest([RequestSpec(index=i, prompt=prompt, output=str(tmp_path / f"o{i}.mp4"),
                                seed=10 + i, assigned_worker=0)
                    for i, prompt in enumerate(prompts)], requests)
    return requests


def test_the_generate_stage_times_every_request_and_the_copied_scheduler_steps_itself(
        monkeypatch, tmp_path, capsys):
    """Two requests of a DP queue: the hook is installed once, before the first request, and
    every request's deep-copied scheduler steps itself (no IndexError past the table); the
    marks are cleared per request, so each line's ``n`` is that request's steps minus one."""
    args = _stage_args(requests_dir=str(_queue(tmp_path, ["a fox", "a cat"])), worker_index=0,
                       dp_schedule="round_robin")
    run = _run_stage(monkeypatch, args, timing=True, steps=4)

    video = run["pipe"].app.pipeline.scheduler
    assert "step" in video.__dict__  # the hook, bound to the instance
    assert [out["audio_steps"] for _, out in run["saved"]] == [4, 4]
    assert [prompt for prompt, _ in run["saved"]] == ["a fox", "a cat"]
    assert run["compiled"] == [] and run["pipe"].loads == 1
    out = capsys.readouterr().out
    assert out.count("[t] start-up: from_pretrained") == 1
    lines = [line for line in out.splitlines() if line.startswith("[t] request:")]
    assert len(lines) == 2
    assert all("step median" in line and "(n=3, max" in line for line in lines)


def test_the_generate_stage_without_timing_leaves_the_scheduler_alone(monkeypatch, capsys):
    run = _run_stage(monkeypatch, _stage_args(), timing=False)
    assert "step" not in run["pipe"].app.pipeline.scheduler.__dict__
    assert [prompt for prompt, _ in run["saved"]] == ["a fox"]
    out = capsys.readouterr().out
    assert "start-up" not in out and "step median" not in out
    assert "[t] neuron rank 0/1:" in out


def test_the_compile_stage_installs_no_hook(monkeypatch, capsys):
    """``compile`` mode is the load and the manifest: no request runs, so nothing is timed
    and the scheduler is not touched, with ``timing`` on."""
    run = _run_stage(monkeypatch, _stage_args(stage_mode="compile"), timing=True)
    assert "step" not in run["pipe"].app.pipeline.scheduler.__dict__
    assert run["compiled"] == [True]
    assert run["pipe"].app.calls == [] and run["saved"] == []
    out = capsys.readouterr().out
    assert out.count("[t] start-up: from_pretrained") == 1 and "[t] request:" not in out


# ------------------------------------------------------------------- the LTX-2 driver


def test_the_ltx2_driver_times_its_generate_stage(monkeypatch):
    """The LTX-2 stage asks ``run_generate_stage`` for the timing lines (the DiT step median
    is the device campaign's per-step number), as the Wan and Qwen-Image stages do."""
    seen: list[dict] = []
    monkeypatch.setattr(ltx2_cli, "run_generate_stage",
                        lambda args, backend, **kwargs: seen.append(kwargs))
    args = argparse.Namespace(
        model_id="Lightricks/LTX-2", tp_degree=4, cp_degree=1, cp_mode="gather_kv",
        cfg_parallel=False, sp_enabled=False, height=480, width=704, num_frames=49, steps=20,
        guidance_scale=1.0, seed=42, prompt="a fox", negative_prompt=None, output="/tmp/o.mp4",
        cache_dir="/tmp/c", work_dir=None, revision="47da56e2", shapes=None, quant=None,
        teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
        teacache_calibration=None, device_text_encoder=False, requests_dir=None,
        stage_mode="generate",
    )
    ltx2_cli.LTX2Orchestrator(args)._stage_generate_neuron(args, "neuron")
    assert len(seen) == 1
    assert seen[0]["tag"] == "ltx_2"
    assert seen[0]["timing"] is True
