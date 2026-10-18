"""Flux's CLI on the neuron backend: one torchrun generate stage; the stage runs the request(s)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from difflet.cli.orchestrators import flux as flux_cli


def _args(**overrides):
    base = dict(tp_degree=4, cp_degree=None, cfg_parallel=False, sp_enabled=False, taef1=False,
                height=1024, width=1024, prompt="a cat", output="/tmp/o.png", steps=50,
                guidance_scale=3.5, seed=7, cache_dir=None, work_dir=None, requests_dir=None,
                worker_index=None, dp_schedule=None, exec_mode=None, revision=None,
                keep_work_dir=False, teacache_cadence=None, teacache_online_delta=None,
                teacache_speedup=None, teacache_calibration=None, stage_mode="generate", shapes=None)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def launches(monkeypatch):
    calls = []
    import difflet.cli.orchestrators.torchrun as torchrun_mod

    # both the driver's check and the shared refusal helper's own lookup
    monkeypatch.setattr(flux_cli, "torchrun_backend", lambda: "neuron")
    monkeypatch.setattr(torchrun_mod, "torchrun_backend", lambda: "neuron")
    monkeypatch.setattr(flux_cli.runner, "run_stage",
                        lambda *a, **k: calls.append(dict(args=a, **k)))
    return calls


@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_driver_launches_one_torchrun_generate_stage(launches, mode):
    getattr(flux_cli.FluxOrchestrator(_args()), mode)()
    [call] = launches
    assert call["args"][:2] == ("black-forest-labs/FLUX.1-dev", "generate")
    assert call["num_cores"] == 4
    cli = call["cli_args"]
    assert cli[cli.index("--stage-mode") + 1] == mode
    for flag, value in (("--prompt", "a cat"), ("--steps", "50"), ("--guidance-scale", "3.5"),
                        ("--seed", "7"), ("--tp-degree", "4")):
        assert cli[cli.index(flag) + 1] == value


@pytest.mark.parametrize("override, named", [
    (dict(cp_degree=2), "--cp-degree"), (dict(sp_enabled=True), "--sp"),
    (dict(cfg_parallel=True), "--cfg-parallel"), (dict(teacache_cadence=2), "--teacache-cadence"),
])
def test_driver_refuses_what_the_stage_does_not_run(launches, capsys, override, named):
    with pytest.raises(SystemExit) as exc:
        flux_cli.FluxOrchestrator(_args(**override)).generate()
    assert exc.value.code == 2 and launches == []
    assert named in capsys.readouterr().err


def test_stage_runs_the_request_and_saves_the_image(monkeypatch, tmp_path):
    import difflet.pipeline.difflet_pipeline as dp

    saved, calls = [], []

    class FakeImage:
        def save(self, path):
            saved.append(path)

    class FakePipe:
        app = SimpleNamespace(dit=SimpleNamespace(device="cpu"))

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[FakeImage()])

    built = []
    monkeypatch.setattr(dp.DiffletPipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: built.append(k) or FakePipe()))
    out = tmp_path / "img.png"
    flux_cli.FluxOrchestrator(_args(output=str(out)))._run_stage_internal("generate", _args(output=str(out)))
    assert built[0]["model_type"] == "flux" and built[0]["parallel"].tp_degree == 4
    [call] = calls
    assert call["prompt"] == "a cat" and call["num_inference_steps"] == 50 and call["guidance_scale"] == 3.5
    assert saved == [str(out)]


def test_registry_routes_flux_to_the_neuron_application():
    from difflet.models.flux.entry import create_flux_application
    from difflet.models.flux.neuron_application import TorchNeuronFluxApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.registry import resolve_model

    assert "neuron" in resolve_model("black-forest-labs/FLUX.1-dev", model_type="flux").backends
    app = create_flux_application(model_path="/m", parallel=DiffletParallelConfig(tp_degree=4),
                                  dtype=None, shape={"height": 1024, "width": 1024}, backend="neuron")
    assert isinstance(app, TorchNeuronFluxApplication)
