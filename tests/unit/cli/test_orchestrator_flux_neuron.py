"""Flux's CLI on the neuron backend: one torchrun generate stage; the stage runs the request(s)."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import pytest

from difflet.cli.orchestrators import flux as flux_cli


def _args(**overrides):
    base = dict(tp_degree=4, cp_degree=None, cfg_parallel=False, sp_enabled=False, taef1=False,
                height=1024, width=1024, prompt="a cat", output="/tmp/o.png", steps=50,
                guidance_scale=3.5, seed=7, cache_dir=None, work_dir=None, requests_dir=None,
                worker_index=None, dp_schedule=None, exec_mode=None, revision=None,
                keep_work_dir=False, cp_mode=None, teacache_cadence=None, teacache_online_delta=None,
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
@pytest.mark.parametrize("tp, stages", [(4, [("transformer", 4)]),
                                        (2, [("transformer", 2), ("vae", 1)])])
def test_driver_runs_a_vae_stage_only_where_the_decode_does_not_fit(launches, tmp_path, mode, tp, stages):
    getattr(flux_cli.FluxOrchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=tp)), mode)()
    assert [(c["args"][1], c["num_cores"]) for c in launches] == stages
    for call in launches:
        cli = call["cli_args"]
        assert cli[cli.index("--stage-mode") + 1] == mode
        assert cli[cli.index("--work-dir") + 1] == str(tmp_path / "w")
        for flag, value in (("--prompt", "a cat"), ("--steps", "50"), ("--guidance-scale", "3.5"),
                            ("--seed", "7"), ("--tp-degree", str(tp))):
            assert cli[cli.index(flag) + 1] == value
    assert not (tmp_path / "w").exists()  # removed without --keep-work-dir


@pytest.mark.parametrize("override, named", [(dict(cfg_parallel=True), "--cfg-parallel"),
                                             (dict(taef1=True), "--taef1")])
def test_driver_refuses_what_the_stage_does_not_run(launches, capsys, override, named):
    with pytest.raises(SystemExit) as exc:
        flux_cli.FluxOrchestrator(_args(**override)).generate()
    assert exc.value.code == 2 and launches == []
    assert named in capsys.readouterr().err


def test_driver_runs_context_parallel_on_tp_x_cp_cores(launches, tmp_path):
    args = _args(work_dir=str(tmp_path / "w"), tp_degree=2, cp_degree=2, cp_mode="ulysses")
    flux_cli.FluxOrchestrator(args).generate()
    assert [(c["args"][1], c["num_cores"]) for c in launches] == [("transformer", 4), ("vae", 1)]
    cli = launches[0]["cli_args"]
    assert cli[cli.index("--cp-degree") + 1] == "2" and cli[cli.index("--cp-mode") + 1] == "ulysses"


def test_teacache_flags_reach_the_stage_application(launches, monkeypatch, tmp_path):
    args = _args(work_dir=str(tmp_path / "w"), teacache_speedup=1.6, teacache_calibration="/c.json")
    flux_cli.FluxOrchestrator(args).generate()
    cli = launches[0]["cli_args"]
    assert cli[cli.index("--teacache-speedup") + 1] == "1.6"
    assert cli[cli.index("--teacache-calibration") + 1] == "/c.json"

    import difflet.backends.registry as registry
    import difflet.pipeline.difflet_pipeline as dp

    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    built = []
    monkeypatch.setattr(dp.DiffletPipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: built.append(k) or (_ for _ in ()).throw(SystemExit(0))))
    with pytest.raises(SystemExit):
        flux_cli.FluxOrchestrator(args)._run_stage_internal("transformer", args)
    opts = built[0]["application_kwargs"]
    assert opts["teacache_speedup"] == 1.6 and opts["teacache_calibration_path"] == "/c.json"


def test_transformer_stage_saves_the_packed_latents(monkeypatch, tmp_path):
    import torch

    import difflet.backends.registry as registry
    import difflet.pipeline.difflet_pipeline as dp

    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    latents, calls, built = torch.randn(1, 16, 64), [], []

    class FakePipe:
        app = SimpleNamespace(dit=SimpleNamespace(device="cpu"))

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=latents)

    monkeypatch.setattr(dp.DiffletPipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: built.append(k) or FakePipe()))
    args = _args(work_dir=str(tmp_path), tp_degree=2)
    flux_cli.FluxOrchestrator(args)._run_stage_internal("transformer", args)
    assert built[0]["application_kwargs"]["decode_vae"] is False
    assert built[0]["parallel"].tp_degree == 2
    [call] = calls
    assert call["output_type"] == "latent" and call["num_inference_steps"] == 50
    torch.testing.assert_close(torch.load(tmp_path / "latents.pt"), latents)


def test_transformer_stage_at_tp4_decodes_in_process(monkeypatch, tmp_path):
    import difflet.backends.registry as registry
    import difflet.pipeline.difflet_pipeline as dp

    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    saved, calls, built = [], [], []

    class FakeImage:
        def save(self, path):
            saved.append(path)

    class FakePipe:
        app = SimpleNamespace(dit=SimpleNamespace(device="cpu"))

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[FakeImage()])

    monkeypatch.setattr(dp.DiffletPipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: built.append(k) or FakePipe()))
    out = tmp_path / "img.png"
    args = _args(work_dir=str(tmp_path), output=str(out), tp_degree=4)
    flux_cli.FluxOrchestrator(args)._run_stage_internal("transformer", args)
    assert built[0]["application_kwargs"]["decode_vae"] is True
    assert calls[0]["output_type"] == "pil" and saved == [str(out)]
    assert not (tmp_path / "latents.pt").exists()


def test_vae_stage_unpacks_decodes_and_saves(monkeypatch, tmp_path):
    import torch

    import difflet.backends.neuron.flux.vae as vae_mod
    import difflet.backends.registry as registry
    import difflet.pipeline.path_resolver as resolver

    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    monkeypatch.setattr(resolver, "resolve_model_path", lambda *a, **k: str(tmp_path / "model"))
    seen = {}

    class FakeVae:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.vae_config = {"scaling_factor": 0.5, "shift_factor": 0.1}
            self.scale_factor = 8
            self.phase_seconds = {}

        def load(self, skip_warmup=False):
            seen["skip_warmup"] = skip_warmup

        def decode(self, z):
            seen["z"] = z
            return torch.zeros(1, 3, 32, 32)

    monkeypatch.setattr(vae_mod, "NeuronFluxVaeDecoderApplication", FakeVae)
    packed = torch.randn(1, 4, 64)  # 32x32 image: (32 / 16)^2 tokens of 16 channels x 2 x 2
    torch.save(packed, tmp_path / "latents.pt")
    out = tmp_path / "img.png"
    args = _args(work_dir=str(tmp_path), output=str(out), height=32, width=32)
    flux_cli.FluxOrchestrator(args)._run_stage_internal("vae", args)
    assert seen["skip_warmup"] is True and seen["init"]["height"] == 32
    assert tuple(seen["z"].shape) == (1, 16, 4, 4)  # unpacked 2x2 patches
    assert out.exists()


def test_transformer_stage_dumps_the_latents_and_the_frames_it_saves(monkeypatch, tmp_path):
    """With DIFFLET_DUMP_LATENTS_DIR set, the stage writes prepare_latents' packed noise as
    initial.pt, the last step's latents as final.pt and the saved image's uint8 pixels (rule
    round, numpy_to_pil's) as frames.npy, equal to the PNG; unset, nothing is written and
    prepare_latents is not wrapped."""
    import json

    import numpy as np
    import torch
    from PIL import Image

    import difflet.backends.registry as registry
    import difflet.pipeline.difflet_pipeline as dp

    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    monkeypatch.delenv("RANK", raising=False)
    noise = torch.randn(1, 4, 64).to(torch.bfloat16)
    pixels = np.random.default_rng(7).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    fluxes = []

    class FakeFlux:
        num_timesteps = 3

        def prepare_latents(self, *a):
            return noise, "ids"

    class FakePipe:
        def __init__(self):
            fluxes.append(FakeFlux())
            self.app = SimpleNamespace(dit=SimpleNamespace(device="cpu"), pipeline=fluxes[-1])

        def __call__(self, **kwargs):  # FluxPipeline's loop: prepare_latents, a callback per step
            latents, _ = self.app.pipeline.prepare_latents(1, 16, 32, 32)
            for i in range(3):
                latents = latents * 0.5
                kwargs["callback_on_step_end"](self.app.pipeline, i, i, {"latents": latents})
            return SimpleNamespace(images=[Image.fromarray(pixels)])

    monkeypatch.setattr(dp.DiffletPipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: FakePipe()))
    dump = tmp_path / "dump"
    monkeypatch.delenv("DIFFLET_DUMP_LATENTS_DIR", raising=False)
    args = _args(work_dir=str(tmp_path), output=str(tmp_path / "off.png"), tp_degree=4)
    flux_cli.FluxOrchestrator(args)._run_stage_internal("transformer", args)
    assert not dump.exists() and "prepare_latents" not in vars(fluxes[0])

    monkeypatch.setenv("DIFFLET_DUMP_LATENTS_DIR", str(dump))
    out = tmp_path / "on.png"
    args = _args(work_dir=str(tmp_path), output=str(out), tp_degree=4)
    flux_cli.FluxOrchestrator(args)._run_stage_internal("transformer", args)
    assert sorted(p.name for p in dump.iterdir()) == ["final.pt", "frames.json", "frames.npy",
                                                      "initial.pt"]
    torch.testing.assert_close(torch.load(dump / "initial.pt"), noise)
    torch.testing.assert_close(torch.load(dump / "final.pt"), noise * 0.125)
    frames = np.load(dump / "frames.npy")
    assert frames.shape == (1, 32, 32, 3) and np.array_equal(frames[0], np.asarray(Image.open(out)))
    meta = json.loads((dump / "frames.json").read_text())
    assert (meta["rule"], meta["layout"]) == ("round", "FHWC")


def test_vae_stage_dumps_the_uint8_frames_it_saves(monkeypatch, tmp_path):
    """The vae stage's frames.npy is the saved PNG's uint8 pixels (rule round); unset, the dump
    directory is never created."""
    import json

    import numpy as np
    import torch
    from PIL import Image

    import difflet.backends.neuron.flux.vae as vae_mod
    import difflet.backends.registry as registry
    import difflet.pipeline.path_resolver as resolver

    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    monkeypatch.setattr(resolver, "resolve_model_path", lambda *a, **k: str(tmp_path / "model"))
    monkeypatch.delenv("RANK", raising=False)

    class FakeVae:
        def __init__(self, **kwargs):
            self.vae_config = {"scaling_factor": 0.5, "shift_factor": 0.1}
            self.scale_factor = 8
            self.phase_seconds = {}

        def load(self, skip_warmup=False):
            pass

        def decode(self, z):  # past [-1, 1] too: postprocess clamps
            return torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(7)) * 2.4 - 1.2

    monkeypatch.setattr(vae_mod, "NeuronFluxVaeDecoderApplication", FakeVae)
    torch.save(torch.randn(1, 4, 64), tmp_path / "latents.pt")
    dump = tmp_path / "dump"
    monkeypatch.delenv("DIFFLET_DUMP_LATENTS_DIR", raising=False)
    args = _args(work_dir=str(tmp_path), output=str(tmp_path / "off.png"), height=32, width=32)
    flux_cli.FluxOrchestrator(args)._run_stage_internal("vae", args)
    assert not dump.exists()

    monkeypatch.setenv("DIFFLET_DUMP_LATENTS_DIR", str(dump))
    out = tmp_path / "on.png"
    args = _args(work_dir=str(tmp_path), output=str(out), height=32, width=32)
    flux_cli.FluxOrchestrator(args)._run_stage_internal("vae", args)
    frames = np.load(dump / "frames.npy")
    assert frames.dtype == np.uint8 and frames.shape == (1, 32, 32, 3)
    assert np.array_equal(frames[0], np.asarray(Image.open(out)))
    assert np.array_equal(np.asarray(Image.open(out)), np.asarray(Image.open(tmp_path / "off.png")))
    assert json.loads((dump / "frames.json").read_text())["rule"] == "round"


def test_registry_routes_flux_to_the_neuron_application():
    from difflet.models.flux.entry import create_flux_application
    from difflet.models.flux.neuron_application import TorchNeuronFluxApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from difflet.registry import resolve_model

    assert "neuron" in resolve_model("black-forest-labs/FLUX.1-dev", model_type="flux").backends
    app = create_flux_application(model_path="/m", parallel=DiffletParallelConfig(tp_degree=4),
                                  dtype=None, shape={"height": 1024, "width": 1024}, backend="neuron")
    assert isinstance(app, TorchNeuronFluxApplication)


def test_quant_is_forwarded_to_the_transformer_stage(launches, monkeypatch, tmp_path):
    """--quant mxfp8 is forwarded to the transformer stage and turns on the NKI trace cache; the
    driver's quantized-checkpoint step is skipped when the weights are not local (the ranks
    report it)."""
    import difflet.pipeline.path_resolver as resolver

    # setenv first records the original state, so the undo also removes the value the
    # orchestrator sets with os.environ.setdefault.
    monkeypatch.setenv("DIFFLET_NKI_TRACE_CACHE", "")
    monkeypatch.delenv("DIFFLET_NKI_TRACE_CACHE")
    monkeypatch.setattr(resolver, "resolve_model_path",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("not downloaded")))
    args = _args(work_dir=str(tmp_path / "w"), model_id="black-forest-labs/FLUX.1-dev",
                 quant="mxfp8", quant_compute=None)
    flux_cli.FluxOrchestrator(args).generate()
    cli = launches[0]["cli_args"]
    assert cli[cli.index("--quant") + 1] == "mxfp8" and "--quant-compute" not in cli
    assert os.environ.get("DIFFLET_NKI_TRACE_CACHE") == "1"
    plain = _args(work_dir=str(tmp_path / "w2"), model_id="black-forest-labs/FLUX.1-dev")
    flux_cli.FluxOrchestrator(plain).generate()
    assert "--quant" not in launches[1]["cli_args"]


def test_quant_reaches_the_stage_application(monkeypatch, tmp_path):
    """The transformer stage hands the --quant spec to the application (as the TeaCache flags:
    the stage imports torch_neuronx, so on a CPU-only host this is a ledgered Neuron skip)."""
    import difflet.backends.registry as registry
    import difflet.pipeline.difflet_pipeline as dp

    args = _args(work_dir=str(tmp_path / "w"), model_id="black-forest-labs/FLUX.1-dev",
                 quant="mxfp8", quant_compute=None)
    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    built = []
    monkeypatch.setattr(dp.DiffletPipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: built.append(k) or (_ for _ in ()).throw(SystemExit(0))))
    with pytest.raises(SystemExit):
        flux_cli.FluxOrchestrator(args)._run_stage_internal("transformer", args)
    quant = built[0]["application_kwargs"]["quant"]
    assert quant["format"] == "mxfp8" and quant["tp_degree"] == 4
    assert any("single_transformer_blocks" in t for t in quant["targets"])


def test_stage_parses_values_that_start_with_a_dash():
    """A prompt (or cache dir, revision, calibration path) starting with '-' reaches the stage as
    typed: the hand-written argv sent ``--prompt -dark`` and the stage parser exited 2."""
    from difflet.cli.stage import _build_stage_parser

    args = _args(prompt="-dark scene", cache_dir="-c", teacache_calibration="-cal.json")
    argv = flux_cli.FluxOrchestrator(args)._stage_cli_args("generate", work_dir="/w")
    staged = _build_stage_parser().parse_args(
        ["--orchestrator", flux_cli._HF_MODEL_ID, "--stage", "transformer", *argv])
    assert (staged.prompt, staged.cache_dir, staged.teacache_calibration) == \
        ("-dark scene", "-c", "-cal.json")
    assert (staged.tp_degree, staged.height, staged.width, staged.stage_mode, staged.work_dir) == \
        (4, 1024, 1024, "generate", "/w")
