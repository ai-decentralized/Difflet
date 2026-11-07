"""HunyuanVideo 1.5's CLI on the neuron backend (CPU, stages stubbed): the stage chain and its layout,
the refusals, what the encode and transformer stages write, and --quant mxfp8 from the driver's
quantized directory (built from the bf16 copy at the layout's tp) to the spec the ranks build
the pipeline with. A test that needs weights resolves a toy snapshot or none: the real one may
be in the hub cache, and its bf16 copy is 16.7 GB."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import difflet.cli.orchestrators.hunyuan_video_15 as hv15


def _args(**over):
    base = dict(model_id=hv15._HF_MODEL_ID, tp_degree=None, cp_degree=None, cp_mode=None, cfg_parallel=False, height=None,
                width=None, num_frames=None, steps=None, seed=42, prompt="a cat", output="/tmp/o.mp4",
                guidance_scale=None, cache_dir=None, revision=None, exec_mode=None, work_dir=None,
                keep_work_dir=False, sp_enabled=False, requests_dir=None, requests=None, dp=None,
                teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
                teacache_calibration=None, stage_mode="generate", quant=None,
                quant_compute=None)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _no_quant_targets_override(monkeypatch):
    monkeypatch.delenv("DIFFLET_QUANT_TARGETS", raising=False)


class _Calls(list):
    env_seen: list


def _no_local_snapshot(monkeypatch) -> None:
    """compile() makes a bf16 copy of the real fp32 DiT when the snapshot is local (16.7 GB
    under the test's cache, refused by tests/unit/conftest.py): a driver test never needs it."""
    def missing(*args, **kwargs):
        raise FileNotFoundError("no local snapshot (test)")

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", missing)


def _toy_snapshot(monkeypatch, tmp_path) -> Path:
    """The tiny HunyuanVideo-1.5 snapshot (an fp32 DiT, as the hub's) as the local one, with the
    compile cache under ``tmp_path / "cache"``, where the bf16 copy and the quantized
    directory land."""
    from tests.unit.backends._neuron_hv15_toy import write_tiny_hv15_snapshot

    root = write_tiny_hv15_snapshot(tmp_path / "snap")
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *args, **kwargs: str(root))
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "cache"))
    return root


@pytest.fixture
def launches(monkeypatch):
    calls = _Calls()
    import difflet.cli.orchestrators.torchrun as torchrun_helpers

    # The driver and prepare_quant set it: setenv first records the original state (a delenv
    # of an unset variable records nothing to undo, and a refused launch leaves it set).
    monkeypatch.setenv("DIFFLET_NKI_TRACE_CACHE", "")
    monkeypatch.delenv("DIFFLET_NKI_TRACE_CACHE")
    _no_local_snapshot(monkeypatch)
    monkeypatch.setattr(hv15, "torchrun_backend", lambda: "neuron")
    monkeypatch.setattr(torchrun_helpers, "torchrun_backend", lambda: "neuron")  # the refusal checks it too
    def run_stage(model, stage, num_cores, virtual_core_size, cli_args):
        calls.append((stage, num_cores, list(cli_args)))
        env_seen.append(os.environ.get("DIFFLET_NKI_TRACE_CACHE"))

    env_seen = []
    calls.env_seen = env_seen  # what each launch saw
    monkeypatch.setattr(hv15.runner, "run_stage", run_stage)
    return calls


def _flag(cli, name):
    return cli[cli.index(name) + 1] if name in cli else None


def _fake_pipeline(seen: dict, latents=lambda generator: torch.full((1, 4, 2, 2, 3), 7.0)) -> type:
    """The transformer stage's pipeline class without a device: it records in ``seen`` what the
    stage builds it with (``init``), loads it with (``skip_warmup``) and runs it with
    (``steps``, each request's seed under ``seeds``), and returns ``latents(generator)``. One
    fake for every transformer-stage test."""

    class FakePipeline:
        def __init__(self, **kw):
            seen["init"] = kw
            self.dit = SimpleNamespace(phase_seconds={})
            self.guidance_scale = 6.0

        def load(self, skip_warmup=False):
            seen["skip_warmup"] = skip_warmup

        def __call__(self, num_inference_steps, generator, callback, **embeds):
            seen["steps"] = num_inference_steps
            seen.setdefault("seeds", []).append(generator.initial_seed())
            callback(0, None, None)
            return latents(generator)

    return FakePipeline


def test_default_layout_is_tp2_x_cfg2(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"))).generate()
    assert launches.env_seen == ["1", "1", "1"]  # every stage keeps the attention kernel's trace
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ  # restored after the launches
    assert [(s, n) for s, n, _ in launches] == [("encode", 1), ("transformer", 4), ("vae", 1)]
    cli = launches[1][2]
    assert _flag(cli, "--tp-degree") == "2" and "--cfg-parallel" in cli
    assert _flag(cli, "--steps") == "50" and _flag(cli, "--num-frames") == "121"
    # A bf16 command line keeps its words: --quant adds its flags only when given.
    assert cli == ["--model-id", hv15._HF_MODEL_ID, "--stage-mode", "generate", "--work-dir",
                   str(tmp_path / "w"), "--tp-degree", "2", "--cp-degree", "1", "--height", "480",
                   "--width", "848", "--num-frames", "121", "--steps", "50", "--seed", "42",
                   "--cfg-parallel", "--prompt", "a cat", "--output", "/tmp/o.mp4"]
    assert not (tmp_path / "w").exists()


@pytest.mark.parametrize("over, tp, cores", [({}, 2, 4), ({"cfg_parallel": True}, 1, 2),
                                             ({"cp_degree": 2}, 1, 2), ({"tp_degree": 4}, 4, 4)])
def test_quant_reaches_the_transformer_stage_at_the_layout_tp(launches, monkeypatch, tmp_path,
                                                              capsys, over, tp, cores):
    """--cfg-parallel or --cp-degree without --tp-degree runs at tp1 (``_layout``), so the
    driver quantizes for the layout's tp, never a constant: a tp2 directory under a tp1 launch
    would be refused by every rank. compile writes the directory from the bf16 copy it has
    just made, generate resolves the same one, and the transformer stage line rebuilds the
    driver's spec (it carries --tp-degree and --quant)."""
    from difflet.backends.neuron.core.weights_bf16 import valid_copy
    from difflet.cli.quant import quant_spec
    from difflet.cli.stage import _build_stage_parser
    from difflet.quant.checkpoint import check_quantized_dir, read_manifest

    root = _toy_snapshot(monkeypatch, tmp_path)
    args = _args(work_dir=str(tmp_path / "w"), quant="mxfp8", **over)
    hv15.HunyuanVideo15Orchestrator(args).compile()
    orch = hv15.HunyuanVideo15Orchestrator(args)
    orch.generate()
    spec = quant_spec(args, default_tp_degree=tp)
    assert spec.tp_degree == tp and read_manifest(orch._quantized_dit)["spec"]["tp_degree"] == tp
    copy = valid_copy(root / "transformer")
    assert copy is not None
    check_quantized_dir(orch._quantized_dit, copy, spec)  # built from the copy, with this spec
    out = capsys.readouterr().out
    assert out.count(f"[hv15] MXFP8 transformer (nki): {orch._quantized_dit}\n") == 2
    transformer = [(n, cli) for stage, n, cli in launches if stage == "transformer"]
    assert len(transformer) == 2  # compile's, then generate's
    for n, cli in transformer:
        assert n == cores
        assert _flag(cli, "--quant") == "mxfp8" and _flag(cli, "--tp-degree") == str(tp)
        stage_args, _ = _build_stage_parser().parse_known_args(
            ["--orchestrator", hv15._HF_MODEL_ID, "--stage", "transformer", *cli])
        assert quant_spec(stage_args) == spec


def test_quant_generate_before_compile_is_refused_before_any_launch(launches, monkeypatch,
                                                                   tmp_path, capsys):
    """A host where compile never ran has no bf16 copy, so the directory is looked for from the
    fp32 source, which nothing quantized: refused on the driver (exit 2, as its other
    refusals), naming `difflet compile`, before the encode stage, the work dir or any write."""
    _toy_snapshot(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exc:
        hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"),
                                              quant="mxfp8")).generate()
    err = capsys.readouterr().err
    assert exc.value.code == 2 and launches == []
    assert err.startswith("Error: --quant: transformer: ")
    assert "run `difflet compile` with the same --quant flags first" in err
    assert err.endswith("; nothing was launched.\n")
    assert not (tmp_path / "cache" / "quantized").exists()
    assert not (tmp_path / "cache" / "weights-bf16").exists() and not (tmp_path / "w").exists()


@pytest.mark.parametrize("mode", ["compile", "generate"])
def test_quant_without_local_weights_is_refused_naming_download(launches, monkeypatch, tmp_path,
                                                                capsys, mode):
    """--quant needs the weights on the driver (compile quantizes them, generate resolves the
    directory), so a launch before `difflet download` is refused here instead of starting
    ranks that each report the missing weights; the bf16 path leaves them to the ranks."""
    _no_local_snapshot(monkeypatch)
    hf = hv15._HF_MODEL_ID
    with pytest.raises(SystemExit) as exc:
        getattr(hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"),
                                                      quant="mxfp8")), mode)()
    assert exc.value.code == 2 and launches == []
    assert capsys.readouterr().err == (
        f"Error: --quant: the {hf} weights are not local; run `difflet download --model-id "
        f"{hf}` first; nothing was launched.\n")


def test_the_native_path_keeps_no_args_keyed_artifact(launches, monkeypatch, tmp_path):
    """A regression pin, not a proof: the native path never calls the stage-manifest helpers,
    so this passes whether or not the NEFFs key on the HLO (F2's graph-and-store test in
    test_neuron_hv15_fp8.py proves that). It fails once a change gives the native path an
    args-keyed stage artifact, which a --quant launch would then have to key."""
    import difflet.cli.orchestrators.base as base

    def keyed(*args, **kwargs):
        pytest.fail("an args-keyed stage artifact on the native path")

    for name in ("hashed_stage_dir", "write_stage_manifest", "has_valid_stage_manifest"):
        monkeypatch.setattr(base, name, keyed)
    _toy_snapshot(monkeypatch, tmp_path)
    for mode in ("compile", "generate"):
        getattr(hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"),
                                                      quant="mxfp8")), mode)()
    assert [s for s, _, _ in launches] == ["transformer", "vae", "encode", "transformer", "vae"]


def test_a_quant_launch_restores_the_trace_cache_opt_in(launches, monkeypatch, tmp_path):
    """prepare_quant opts in to the NKI trace cache for the MXFP8 kernel; the driver reads the
    user's setting before it, so the launches still see the opt-in and the driver's
    environment is restored after them, as on a bf16 launch."""
    _toy_snapshot(monkeypatch, tmp_path)
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), quant="mxfp8")).compile()
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    launches.env_seen.clear()
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), quant="mxfp8")).generate()
    assert launches.env_seen == ["1", "1", "1"]
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ


def test_the_driver_module_imports_no_torch_and_no_neuron_backend():
    """A regression pin, never skipped: the AoT path's native-free contract needs the driver
    free of torch and of every neuron backend module at import (its own guard is skipped
    without neuronx_distributed), so --quant's torch imports live inside the functions."""
    code = ("import sys\n"
            "import difflet.cli.orchestrators.hunyuan_video_15\n"
            "bad = sorted(m for m in sys.modules if m == 'torch' or "
            "m.startswith('difflet.backends.neuron'))\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env=dict(os.environ, DIFFLET_BACKEND="cpu"))


def test_explicit_tp4_runs_cfg_sequentially(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=4)).generate()
    assert launches[1][:2] == ("transformer", 4) and "--cfg-parallel" not in launches[1][2]


def test_compile_skips_the_encoder(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"))).compile()
    assert [s for s, _, _ in launches] == ["transformer", "vae"]


def test_sp_and_cp_mode_reach_the_transformer_stage(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=2, sp_enabled=True)).generate()
    assert "--sp" in launches[1][2]
    launches.clear()
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=1, cp_degree=4,
                                          cp_mode="ring")).generate()
    assert _flag(launches[1][2], "--cp-mode") == "ring"


def test_encode_stage_uses_diffusers_encode_prompt(monkeypatch, tmp_path):
    import diffusers

    prompts = []

    class FakePipe:
        def encode_prompt(self, prompt, device, dtype):
            prompts.append(prompt)
            return (torch.ones(1, 4, 8), torch.ones(1, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2))

    monkeypatch.setattr(diffusers.HunyuanVideo15Pipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: FakePipe()))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    args = _args(work_dir=str(tmp_path))
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("encode", args)
    saved = torch.load(tmp_path / "embeds.pt")
    assert prompts == ["a cat", ""]
    assert set(saved) == {"prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2",
                          "negative_prompt_embeds", "negative_prompt_embeds_mask", "negative_prompt_embeds_2",
                          "negative_prompt_embeds_mask_2"}


def test_transformer_stage_saves_latents(monkeypatch, tmp_path):
    import difflet.backends.neuron.hunyuan_video15.pipeline as P
    import difflet.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda name: SimpleNamespace(prepare_runtime=lambda p: None))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    seen = {}
    monkeypatch.setattr(P, "TorchNeuronHunyuanVideo15Pipeline", _fake_pipeline(seen))
    torch.save({"prompt_embeds": torch.zeros(1)}, tmp_path / "embeds.pt")
    args = _args(work_dir=str(tmp_path), tp_degree=2, cfg_parallel=True, height=480, width=848,
                 num_frames=121, steps=50, seed=3)
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("transformer", args)
    assert seen["skip_warmup"] is True and seen["steps"] == 50
    assert seen["init"]["parallel"].cfg_parallel_enabled and seen["init"]["parallel"].tp_degree == 2
    assert torch.equal(torch.load(tmp_path / "latents.pt"), torch.full((1, 4, 2, 2, 3), 7.0))


def test_transformer_stage_builds_the_pipeline_with_the_spec(monkeypatch, tmp_path):
    """The native path keys no stage artifact on the args, so a stage that dropped the spec
    would run the bf16 DiT under the driver's MXFP8 line: the ranks build the pipeline with the
    spec the stage flags describe, at the line's tp, and with none without --quant."""
    import difflet.backends.neuron.hunyuan_video15.pipeline as P
    import difflet.backends.registry as registry
    from difflet.quant.targets import default_spec

    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda p: None))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    seen = {}
    monkeypatch.setattr(P, "TorchNeuronHunyuanVideo15Pipeline", _fake_pipeline(seen))
    torch.save({"prompt_embeds": torch.zeros(1)}, tmp_path / "embeds.pt")
    expected = default_spec("hunyuan_video_15", tp_degree=2).to_dict()
    for quant, spec in (("mxfp8", expected), (None, None)):
        args = _args(work_dir=str(tmp_path), model_id=hv15._HF_MODEL_ID, quant=quant,
                     tp_degree=2, cfg_parallel=True, height=480, width=848, num_frames=121,
                     steps=50, seed=3)
        hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("transformer", args)
        assert seen["init"]["quant"] == spec


def test_probe_free_teacache_reaches_the_transformer_stage(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), teacache_cadence=2)).generate()
    assert _flag(launches[1][2], "--teacache-cadence") == "2"


def test_adaptive_teacache_flags_reach_the_transformer_stage(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), teacache_speedup=1.5,
                                          teacache_calibration="/c.json")).generate()
    cli = launches[1][2]
    assert _flag(cli, "--teacache-speedup") == "1.5" and _flag(cli, "--teacache-calibration") == "/c.json"


def _requests_dir(tmp_path):
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    rdir = tmp_path / "requests"
    rdir.mkdir()
    write_manifest([RequestSpec(index=0, prompt="a dog", output=str(tmp_path / "o0.mp4"), seed=5,
                                negative_prompt="blurry", assigned_worker=0),
                    RequestSpec(index=1, prompt="a boat", output=str(tmp_path / "o1.mp4"), seed=9,
                                assigned_worker=0)], rdir)
    return rdir  # a Path, as the stage parser gives


def test_batch_encode_claims_each_request(monkeypatch, tmp_path):
    import diffusers

    prompts = []

    class FakePipe:
        def encode_prompt(self, prompt, device, dtype):
            prompts.append(prompt)
            return (torch.ones(1, 4, 8), torch.ones(1, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2))

    monkeypatch.setattr(diffusers.HunyuanVideo15Pipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: FakePipe()))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    args = _args(work_dir=str(tmp_path), requests_dir=_requests_dir(tmp_path), worker_index=0,
                 dp_schedule="round_robin")
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("encode", args)
    assert prompts == ["a dog", "blurry", "a boat", ""]
    assert (tmp_path / "embeds_req0000.pt").exists() and (tmp_path / "embeds_req0001.pt").exists()


def test_batch_transformer_replays_the_claims_with_each_seed(monkeypatch, tmp_path):
    import difflet.backends.neuron.hunyuan_video15.pipeline as P
    import difflet.backends.registry as registry
    from difflet.cli.dp import claims

    monkeypatch.setattr(registry, "get_backend", lambda name: SimpleNamespace(prepare_runtime=lambda p: None))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    seen = {}
    monkeypatch.setattr(P, "TorchNeuronHunyuanVideo15Pipeline", _fake_pipeline(
        seen, latents=lambda g: torch.full((1, 4, 2, 2, 3), float(g.initial_seed()))))
    rdir = _requests_dir(tmp_path)
    for index in (0, 1):
        claims.claim_next(rdir, 0, "round_robin")
        torch.save({"prompt_embeds": torch.zeros(1)}, tmp_path / f"embeds_req000{index}.pt")
    args = _args(work_dir=str(tmp_path), requests_dir=rdir, worker_index=0, dp_schedule="round_robin",
                 tp_degree=2, height=480, width=848, num_frames=121, steps=50, seed=42)
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("transformer", args)
    assert seen["seeds"] == [5, 9]
    assert torch.load(tmp_path / "latents_req0001.pt").flatten()[0].item() == 9.0
