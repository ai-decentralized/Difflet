"""The neuron generate stage's step timing (``torchrun._time_scheduler_steps``) on a scheduler
the orchestrator deep-copies after the hook is installed, as ``LTX2Orchestrator`` copies the
video scheduler into its audio scheduler: the copy must drive its own step index, not the
original's (found on the device as an IndexError past the last sigma, one iteration in two),
and the marks must count one video step per denoising iteration. The hook tests are
torch-free; the LTX-2 driver test runs its stage body against a recording stand-in for
``run_generate_stage`` (no pipeline is built)."""

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
