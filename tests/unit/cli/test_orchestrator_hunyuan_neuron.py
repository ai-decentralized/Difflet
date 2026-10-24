"""The HunyuanVideo CLI on the neuron backend (torchrun MPMD).

With DIFFLET_BACKEND=neuron, ``difflet compile/generate --model-id hunyuanvideo-community/
HunyuanVideo`` is one torchrun launch of the generate stage (the Llama/CLIP encode folded into
it, on rank 0's host) and then the decode: on the host in the driver by default, or a one-core
vae launch with --device-vae, as LTX-2 places it. The ranks build TorchNeuronHunyuanVideoApplication
through DiffletPipeline.from_pretrained, here the CPU fake of tests/unit/cli/_fake_neuron_pipeline.py.
The AoT path (clip, llama, generate) is unchanged. No test touches a NeuronCore.
"""

from __future__ import annotations

import argparse
import importlib
import json
import struct

import pytest

from difflet.cli.orchestrators import hunyuan_video as hv_mod
from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
from tests.unit.cli import _fake_neuron_pipeline as fake_pipeline

_HV = "hunyuanvideo-community/HunyuanVideo"
#: The real DiT's parameter count (12.8 B, 23.9 GiB in bf16).
_HV_DIT_PARAMS = 12_821_012_544
_NEURON_APP_KWARGS = {
    "text_seq_len": 256, "batch_size": 1, "enable_vae_decoder": False, "exec_mode": None,
    "encoder_threads": 12, "text_encoder_device": "host",
}


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        model_id=_HV, tp_degree=2, cp_degree=1, cp_mode="gather_kv", cfg_parallel=False,
        sp_enabled=False, height=320, width=512, num_frames=61, shapes=None, steps=None,
        guidance_scale=None, seed=42, prompt="a fox", output="/tmp/hv.mp4", cache_dir=None,
        work_dir=None, keep_work_dir=False, revision=None, exec_mode=None, host_vae=False,
        device_vae=False, teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None, requests_dir=None,
        stage_mode="generate",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _unset(monkeypatch, *names):
    """Unset ``names`` now and restore them after the test, even when the code under test
    assigns os.environ itself."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _backend(monkeypatch, backend):
    _unset(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", "DIFFLET_COMPILE_CACHE")
    if backend is not None:
        monkeypatch.setenv("DIFFLET_BACKEND", backend)


@pytest.fixture
def launches(monkeypatch):
    """DIFFLET_BACKEND=neuron; run_stage and the host decode recorded; the DiT fit check off
    (pinned on its own below)."""
    _backend(monkeypatch, "neuron")
    stages, decoded = [], []
    monkeypatch.setattr(hv_mod.runner, "run_stage",
                        lambda model, stage, **kw: stages.append((stage, kw)))
    monkeypatch.setattr(hv_mod, "_decode_latents_host", lambda *a, **k: decoded.append((a, k)))
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: "/fake/model")
    monkeypatch.setattr(HunyuanVideoOrchestrator, "_check_dit_fits", lambda self: None)
    return stages, decoded


def _plan(stages):
    return [(stage, kw["num_cores"]) for stage, kw in stages]


# ----------------------------------------------------------------- the stage plan


@pytest.mark.parametrize("tp, cp, cp_mode", [(2, 1, "gather_kv"), (4, 1, "gather_kv"),
                                             (2, 2, "ulysses"), (2, 2, "ring")])
def test_device_vae_runs_generate_on_tp_x_cp_cores_then_vae_on_one(launches, tmp_path, tp, cp,
                                                                   cp_mode):
    stages, decoded = launches
    work = tmp_path / "work"
    HunyuanVideoOrchestrator(_args(tp_degree=tp, cp_degree=cp, cp_mode=cp_mode, device_vae=True,
                                   work_dir=str(work))).generate()
    assert _plan(stages) == [("generate", tp * cp), ("vae", 1)] and decoded == []
    for _, kw in stages:
        cli = kw["cli_args"]
        assert kw["virtual_core_size"] is None
        assert cli[cli.index("--stage-mode") + 1] == "generate"
        assert cli[cli.index("--work-dir") + 1] == str(work)
    assert not work.exists()  # removed without --keep-work-dir
    stages.clear()
    HunyuanVideoOrchestrator(_args(tp_degree=tp, cp_degree=cp, cp_mode=cp_mode,
                                   device_vae=True)).compile()
    assert _plan(stages) == [("generate", tp * cp), ("vae", 1)]
    assert all(kw["cli_args"][kw["cli_args"].index("--stage-mode") + 1] == "compile"
               for _, kw in stages)


def test_host_decode_is_the_default(launches, tmp_path):
    """LTX-2's placement: without --device-vae the driver decodes on the host after the generate
    launch has exited, and compile has nothing to build for it."""
    stages, decoded = launches
    work = tmp_path / "work"
    out = tmp_path / "out.mp4"
    HunyuanVideoOrchestrator(_args(work_dir=str(work), output=str(out), keep_work_dir=True,
                                   height=256, width=448, num_frames=33)).generate()
    assert _plan(stages) == [("generate", 2)]
    assert decoded == [((str(work / "latents.pt"), "/fake/model", str(out)),
                        {"height": 256, "width": 448, "num_frames": 33})]
    assert work.exists()  # --keep-work-dir
    stages.clear()
    HunyuanVideoOrchestrator(_args()).compile()
    assert _plan(stages) == [("generate", 2)]
    assert not hv_mod._decodes_on_device(_args())
    assert not hv_mod._decodes_on_device(_args(host_vae=True))
    assert hv_mod._decodes_on_device(_args(device_vae=True))


def test_generate_preserves_the_work_dir_when_the_decode_fails(launches, tmp_path, capsys,
                                                               monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(hv_mod, "_decode_latents_host", boom)
    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="decode failed"):
        HunyuanVideoOrchestrator(_args(work_dir=str(work))).generate()
    assert work.exists() and "work dir preserved" in capsys.readouterr().err


def test_parallel_exec_mode_and_teacache_flags_reach_the_generate_stage(launches, tmp_path):
    stages, _ = launches
    HunyuanVideoOrchestrator(_args(
        work_dir=str(tmp_path / "w"), tp_degree=2, cp_degree=2, cp_mode="ring", exec_mode="eager",
        teacache_cadence=2, teacache_online_delta=0.6, teacache_speedup=1.6,
        teacache_calibration="/c.json", steps=50, guidance_scale=7.0, seed=3, revision="abc123",
    )).generate()
    cli = stages[0][1]["cli_args"]
    for flag, value in (("--tp-degree", "2"), ("--cp-degree", "2"), ("--cp-mode", "ring"),
                        ("--exec-mode", "eager"), ("--teacache-cadence", "2"),
                        ("--teacache-online-delta", "0.6"), ("--teacache-speedup", "1.6"),
                        ("--teacache-calibration", "/c.json"), ("--steps", "50"),
                        ("--guidance-scale", "7.0"), ("--seed", "3"), ("--prompt", "a fox"),
                        ("--revision", "abc123")):
        assert cli[cli.index(flag) + 1] == value, flag
    stages.clear()
    HunyuanVideoOrchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=4,
                                   sp_enabled=True)).generate()
    cli = stages[0][1]["cli_args"]
    assert "--sp" in cli and "--exec-mode" not in cli and "--revision" not in cli


def test_cache_dir_roots_the_compile_cache(launches, tmp_path, capsys):
    import os

    HunyuanVideoOrchestrator(_args(cache_dir=str(tmp_path / "cache"))).compile()
    assert os.environ["DIFFLET_COMPILE_CACHE"] == str(tmp_path / "cache")
    assert "[hunyuan_video] backend 'neuron' compiles into --cache-dir" in capsys.readouterr().out


# ----------------------------------------------------------------- refusals before any launch


@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_cfg_parallel_is_refused_as_guidance_distilled(launches, capsys, mode):
    stages, _ = launches
    with pytest.raises(SystemExit) as exc:
        getattr(HunyuanVideoOrchestrator(_args(cfg_parallel=True)), mode)()
    assert exc.value.code == 2 and stages == []
    assert "guidance-distilled" in capsys.readouterr().err


def test_main_refuses_cfg_parallel_on_neuron(monkeypatch, tmp_path, capsys):
    cli_main = importlib.import_module("difflet.cli.main")  # the package exports main()

    _backend(monkeypatch, None)
    _unset(monkeypatch, "NEURON_RT_NUM_CORES")
    monkeypatch.setattr(hv_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["generate", "--model-id", _HV, "--backend", "neuron", "--cfg-parallel",
                       "--tp-degree", "2", "--prompt", "a fox", "--output",
                       str(tmp_path / "o.mp4")])
    assert exc.value.code != 0
    assert "guidance-distilled" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_several_shapes_and_both_vae_flags_are_refused(launches, capsys, mode):
    stages, _ = launches
    with pytest.raises(SystemExit) as exc:
        getattr(HunyuanVideoOrchestrator(_args(shapes="320x512x61,320x512x33")), mode)()
    assert exc.value.code == 2 and stages == []
    assert "--shapes with more than one shape" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        getattr(HunyuanVideoOrchestrator(_args(host_vae=True, device_vae=True)), mode)()
    assert exc.value.code == 2 and stages == []
    assert "--host-vae and --device-vae are mutually exclusive" in capsys.readouterr().err


def test_run_refuses_before_the_download(launches, monkeypatch):
    downloads = []
    monkeypatch.setattr(HunyuanVideoOrchestrator, "download", lambda self: downloads.append(1))
    with pytest.raises(SystemExit):
        HunyuanVideoOrchestrator(_args(shapes="320x512x61,320x512x33")).run()
    assert downloads == []


def test_request_shape_outside_the_single_shape_fails_before_the_launch(launches):
    stages, _ = launches
    with pytest.raises(SystemExit, match="not in the compiled bucket set"):
        HunyuanVideoOrchestrator(_args(shapes="320x512x33")).generate()
    assert stages == []


def _header_only_snapshot(root, numel: int) -> str:
    """A snapshot whose transformer is one safetensors header declaring ``numel`` bf16
    elements (no tensor data is read)."""
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    header = json.dumps({"w": {"dtype": "BF16", "shape": [numel], "data_offsets": [0, 0]}})
    (transformer / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header.encode())
    return str(root)


@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_tp1_is_refused_before_any_stage_runs(monkeypatch, tmp_path, capsys, mode):
    """tp1 needs 23.9 GiB of bf16 DiT on a 24 GiB core: refused in the driver, before torchrun
    starts; tp2 (11.9 GiB per core) launches."""
    _backend(monkeypatch, "neuron")
    snapshot = _header_only_snapshot(tmp_path / "snap", _HV_DIT_PARAMS)
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: snapshot)
    stages = []
    monkeypatch.setattr(hv_mod.runner, "run_stage", lambda model, stage, **kw: stages.append(stage))
    monkeypatch.setattr(hv_mod, "_decode_latents_host", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        getattr(HunyuanVideoOrchestrator(_args(tp_degree=1, work_dir=str(tmp_path / "w"))), mode)()
    assert exc.value.code == 2 and stages == []
    err = capsys.readouterr().err
    assert "HunyuanVideo DiT" in err and "--tp-degree 2 or more" in err
    getattr(HunyuanVideoOrchestrator(_args(tp_degree=2, work_dir=str(tmp_path / "w"))), mode)()
    assert stages == ["generate"]


# ----------------------------------------------------------------- the AoT path is unchanged


@pytest.mark.parametrize("backend", [None, "trainium"])
@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_aot_backends_keep_the_three_stages(monkeypatch, tmp_path, backend, mode):
    _backend(monkeypatch, backend)
    stages = []
    monkeypatch.setattr(hv_mod.runner, "run_stage",
                        lambda model, stage, **kw: stages.append((stage, kw)))
    monkeypatch.setattr(HunyuanVideoOrchestrator, "_check_dit_fits",
                        lambda self: pytest.fail("the AoT path has no DiT fit check"))
    getattr(HunyuanVideoOrchestrator(_args(tp_degree=4, cfg_parallel=True,
                                           work_dir=str(tmp_path / "w"))), mode)()
    assert _plan(stages) == [("clip", 1), ("llama", 4), ("generate", 4)]
    assert all(kw["virtual_core_size"] == 2 for _, kw in stages)


@pytest.mark.parametrize("backend", [None, "trainium"])
def test_aot_backends_dispatch_their_own_stages(monkeypatch, backend):
    _backend(monkeypatch, backend)
    orch = HunyuanVideoOrchestrator(_args())
    with pytest.raises(ValueError, match="Unknown HunyuanVideo stage"):
        orch._run_stage_internal("vae", orch.args)


def test_neuron_has_no_clip_or_llama_stage(launches):
    orch = HunyuanVideoOrchestrator(_args())
    for stage in ("clip", "llama"):
        with pytest.raises(ValueError, match="'generate' and 'vae'"):
            orch._run_stage_internal(stage, orch.args)
        with pytest.raises(ValueError):
            orch._stage_cache_inputs(stage, orch.args)


# ----------------------------------------------------------------- stage cache keys


def test_neuron_stage_cache_keys_name_the_backend_and_exec_mode(monkeypatch):
    _backend(monkeypatch, "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "eager")
    orch = HunyuanVideoOrchestrator(_args(cache_dir="/c"))
    generate = orch._stage_cache_inputs("generate", orch.args)
    assert generate["component"] == "hunyuan_video_generate"
    assert (generate["backend"], generate["exec_mode"]) == ("neuron", "eager")
    assert (generate["tp"], generate["cp"], generate["cp_mode"], generate["sp"]) == (
        2, 1, "gather_kv", False)
    assert generate["shapes"] == [[320, 512, 61]] and generate["text_seq_len"] == 256
    vae = orch._stage_cache_inputs("vae", orch.args)
    assert vae["component"] == "hunyuan_video_vae" and "tp" not in vae
    assert (vae["backend"], vae["exec_mode"], vae["dtype"]) == ("neuron", "eager", "float32")
    assert orch._stage_compiled_dir("generate", orch.args).parent.name == "hunyuan_video_generate"

    _backend(monkeypatch, None)
    aot = orch._stage_cache_inputs("generate", orch.args)
    assert aot["component"] == "hunyuan_video_dit" and "backend" not in aot
    with pytest.raises(ValueError):
        orch._stage_cache_inputs("vae", orch.args)


# ----------------------------------------------------------------- the stages themselves


@pytest.fixture
def neuron_stage(monkeypatch):
    """One process, DIFFLET_BACKEND=neuron, no process group (rank 0): from_pretrained is the
    CPU fake."""
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    _backend(monkeypatch, "neuron")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(fake_pipeline, "CALLS", [])
    monkeypatch.setattr(DiffletPipeline, "from_pretrained",
                        classmethod(fake_pipeline.fake_from_pretrained))
    return fake_pipeline


def test_generate_stage_builds_the_application_and_saves_the_latents(neuron_stage, tmp_path,
                                                                     capsys):
    import torch

    cache = tmp_path / "cache"
    args = _args(work_dir=str(tmp_path), cache_dir=str(cache), revision="abc123", steps=50,
                 guidance_scale=6.0, seed=7, tp_degree=2, cp_degree=2, cp_mode="ulysses")
    HunyuanVideoOrchestrator(args)._run_stage_internal("generate", args)

    [call] = neuron_stage.CALLS
    kw = call["kwargs"]
    assert call["model_id"] == _HV
    assert (kw["backend"], kw["model_type"], kw["revision"], kw["local_files_only"]) == (
        "neuron", "hunyuan_video", "abc123", True)
    assert kw["application_kwargs"] == _NEURON_APP_KWARGS
    assert (kw["height"], kw["width"], kw["num_frames"], kw["shapes"], kw["load"]) == (
        320, 512, 61, None, False)
    assert kw["dtype"] is torch.bfloat16 and kw["compile_cache_dir"] == str(cache)
    p = kw["parallel"]
    assert (p.tp_degree, p.cp_degree, p.cp_mode, p.cfg_parallel_enabled, p.sp_enabled) == (
        2, 2, "ulysses", False, False)
    app = call["pipeline"].app
    assert isinstance(app, fake_pipeline.FakeHunyuanVideoApp)
    assert [c["skip_warmup"] for c in app.load_calls] == [False]
    assert app.calls == [{"prompt": "a fox", "num_inference_steps": 50, "guidance_scale": 6.0,
                          "output_type": "latent", "seed": 7}]
    want = torch.randn(fake_pipeline.LATENT_SHAPE, generator=torch.Generator().manual_seed(7))
    assert torch.equal(torch.load(tmp_path / "latents.pt"), want)
    assert "[hunyuan_video] neuron rank 0/1: exec_mode=compile" in capsys.readouterr().out
    assert not cache.exists()  # generate writes no stage manifest


def test_generate_stage_defaults_are_the_trainium_stage_s(neuron_stage, tmp_path):
    args = _args(work_dir=str(tmp_path))
    HunyuanVideoOrchestrator(args)._run_stage_internal("generate", args)
    [call] = neuron_stage.CALLS[-1]["pipeline"].app.calls
    assert (call["num_inference_steps"], call["guidance_scale"]) == (4, 6.0)


def test_generate_stage_hands_the_teacache_flags_to_the_application(neuron_stage, tmp_path):
    args = _args(work_dir=str(tmp_path), teacache_cadence=2, teacache_online_delta=0.6,
                 teacache_speedup=1.6, teacache_calibration="/c.json")
    HunyuanVideoOrchestrator(args)._run_stage_internal("generate", args)
    opts = neuron_stage.CALLS[-1]["kwargs"]["application_kwargs"]
    assert opts == {**_NEURON_APP_KWARGS, "teacache_cadence": 2,
                    "teacache_online_delta_alpha": 0.6, "teacache_speedup": 1.6,
                    "teacache_calibration_path": "/c.json"}


def test_generate_stage_compile_warms_up_and_writes_the_manifest(neuron_stage, tmp_path):
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    args = _args(stage_mode="compile", work_dir=str(tmp_path / "w"),
                 cache_dir=str(tmp_path / "cache"))
    orch = HunyuanVideoOrchestrator(args)
    orch._run_stage_internal("generate", args)
    app = neuron_stage.CALLS[-1]["pipeline"].app
    assert [c["skip_warmup"] for c in app.load_calls] == [False] and app.calls == []
    compiled_dir = orch._stage_compiled_dir("generate", args)
    assert has_valid_stage_manifest(compiled_dir, orch._stage_cache_inputs("generate", args))
    assert not (tmp_path / "w").exists()


@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_generate_stage_writes_files_on_rank0_only(neuron_stage, monkeypatch, tmp_path, mode):
    from difflet.backends.neuron.core import distributed

    monkeypatch.setattr(distributed, "is_rank0", lambda: False)
    work, cache = tmp_path / "work", tmp_path / "cache"
    work.mkdir()
    args = _args(stage_mode=mode, work_dir=str(work), cache_dir=str(cache))
    HunyuanVideoOrchestrator(args)._run_stage_internal("generate", args)
    assert len(neuron_stage.CALLS[-1]["pipeline"].app.calls) == (mode == "generate")
    assert list(work.iterdir()) == [] and not cache.exists()


class _FakeVae:
    """Stands in for NeuronHunyuanVideoVaeDecoderApplication in the vae stage."""

    instances: list = []

    def __init__(self, **kwargs):
        self.init = kwargs
        self.loads, self.decoded = [], []
        _FakeVae.instances.append(self)

    def load(self, skip_warmup=False):
        self.loads.append(skip_warmup)

    def decode(self, latents):
        import torch

        self.decoded.append(latents)
        return torch.zeros(1, 3, 5, 16, 16)


@pytest.fixture
def vae_stage(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import difflet.backends.neuron.hunyuan_video.vae as vae_mod
    import difflet.backends.registry as registry

    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(registry, "get_backend",
                        lambda name: SimpleNamespace(prepare_runtime=lambda parallel: None))
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(tmp_path / "model"))
    monkeypatch.setattr(vae_mod, "NeuronHunyuanVideoVaeDecoderApplication", _FakeVae)
    monkeypatch.setattr(_FakeVae, "instances", [])
    return _FakeVae


def test_vae_stage_decodes_the_saved_latents_and_writes_the_video(vae_stage, monkeypatch,
                                                                  tmp_path):
    import torch

    written = []
    monkeypatch.setattr(hv_mod, "_write_video", lambda video, output: written.append(
        (video, output)))
    latents = torch.randn(1, 16, 2, 2, 2)
    torch.save(latents, tmp_path / "latents.pt")
    args = _args(work_dir=str(tmp_path), output=str(tmp_path / "o.mp4"), height=256, width=448,
                 num_frames=33)
    HunyuanVideoOrchestrator(args)._run_stage_internal("vae", args)
    [vae] = vae_stage.instances
    assert vae.init == {"model_path": str(tmp_path / "model" / "vae"), "height": 256,
                        "width": 448, "num_frames": 33, "device": "neuron"}
    assert vae.loads == [True]  # generate skips the warm-up decode
    [decoded] = vae.decoded
    assert torch.equal(decoded, latents)
    assert [output for _, output in written] == [str(tmp_path / "o.mp4")]


def test_vae_stage_compile_warms_up_and_writes_the_manifest(vae_stage, tmp_path):
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    args = _args(stage_mode="compile", cache_dir=str(tmp_path / "cache"))
    orch = HunyuanVideoOrchestrator(args)
    orch._run_stage_internal("vae", args)
    [vae] = vae_stage.instances
    assert vae.loads == [False] and vae.decoded == []
    assert has_valid_stage_manifest(orch._stage_compiled_dir("vae", args),
                                    orch._stage_cache_inputs("vae", args))


# ----------------------------------------------------------------- the decode itself


@pytest.fixture(scope="module")
def tiny_vae_snapshot(tmp_path_factory):
    import torch

    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLHunyuanVideo

    from tests.unit.backends._neuron_hv1_toy import TINY_HV1_VAE

    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("hv1snap")
    AutoencoderKLHunyuanVideo(**TINY_HV1_VAE).save_pretrained(str(root / "vae"),
                                                              safe_serialization=True)
    return root


def test_host_decode_writes_what_the_diffusers_pipeline_would(tiny_vae_snapshot, monkeypatch,
                                                              tmp_path):
    """The host decode is HunyuanVideoPipeline's: ``vae.decode(latents / scaling_factor)``
    (tiled in space and time) then VideoProcessor's post-processing, an MP4 at 24 fps."""
    import torch
    from diffusers import AutoencoderKLHunyuanVideo
    from diffusers.video_processor import VideoProcessor

    exported = {}
    monkeypatch.setattr("diffusers.utils.export_to_video",
                        lambda frames, path, fps: exported.update(frames=frames, path=path,
                                                                  fps=fps))
    latents = torch.randn(1, 4, 5, 8, 8, generator=torch.Generator().manual_seed(1))
    torch.save(latents, tmp_path / "latents.pt")
    out = tmp_path / "o.mp4"
    hv_mod._decode_latents_host(str(tmp_path / "latents.pt"), str(tiny_vae_snapshot), str(out),
                                height=64, width=64, num_frames=17)

    vae = AutoencoderKLHunyuanVideo.from_pretrained(str(tiny_vae_snapshot / "vae")).eval()
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                      tile_sample_min_num_frames=16, tile_sample_stride_height=192,
                      tile_sample_stride_width=192, tile_sample_stride_num_frames=12)
    with torch.no_grad():
        video = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    want = VideoProcessor(vae_scale_factor=8).postprocess_video(video, output_type="np")[0]
    assert (exported["path"], exported["fps"]) == (str(out), 24)
    assert len(exported["frames"]) == 17
    torch.testing.assert_close(torch.from_numpy(exported["frames"][0]),
                               torch.from_numpy(want[0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.from_numpy(exported["frames"][-1]),
                               torch.from_numpy(want[-1]), atol=1e-5, rtol=1e-5)


def test_write_video_falls_back_to_the_decoded_tensor(monkeypatch, tmp_path):
    import torch

    def boom(*a, **k):
        raise RuntimeError("no codec")

    monkeypatch.setattr("diffusers.utils.export_to_video", boom)
    video = torch.rand(1, 3, 5, 16, 16) * 2 - 1
    hv_mod._write_video(video, str(tmp_path / "v.mp4"))
    assert torch.equal(torch.load(tmp_path / "v.pt"), video)
    hv_mod._write_video(video, str(tmp_path / "w.pt"))
    assert torch.equal(torch.load(tmp_path / "w.pt"), video)

