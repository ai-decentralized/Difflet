from __future__ import annotations

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from difflet.cli.orchestrators import wan as wan_mod
from difflet.cli.orchestrators.wan import WanOrchestrator
from tests.unit.cli import _fake_neuron_pipeline as fake_pipeline


def _wan_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", tp_degree=4, cp_degree=1,
        cp_mode="gather_kv", cfg_parallel=False,
        height=480, width=832, num_frames=9,
        cache_dir=None, force=False, revision=None,
        prompt="a cat walking", output="/tmp/wan.mp4",
        steps=2, guidance_scale=1.0, seed=42, stage_mode="generate",
        work_dir=None, keep_work_dir=False, host_vae=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _inject(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class _FakeWanApp:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.compiled = None
        self.loaded = None
        _FakeWanApp.instances.append(self)

    def compile(self, path):
        self.compiled = path

    def load(self, path, **kw):
        self.loaded = (path, kw)

    def __call__(self, **kw):
        import torch
        self.call_kwargs = kw
        # transformer stage expects .latents; vae stage expects .frames.
        return types.SimpleNamespace(
            latents=torch.zeros(1, 16, 2, 4, 4, dtype=torch.bfloat16),
            frames=torch.zeros(1, 3, 2, 8, 8, dtype=torch.bfloat16),
        )


# ----------------------------------------------------------------- download

_WAN21_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
_WAN21_REV = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"


def test_download_resolves_remote(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, revision=None, local_files_only, allow_patterns=None:
        calls.append((revision, local_files_only)),
    )
    WanOrchestrator(_wan_args()).download()
    assert calls == [(None, False)]


def test_download_passes_revision(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, revision=None, local_files_only, allow_patterns=None:
        calls.append((revision, local_files_only)),
    )
    WanOrchestrator(_wan_args(model_id=_WAN21_ID, revision=_WAN21_REV)).download()
    assert calls == [(_WAN21_REV, False)]


def test_cli_download_revision_reaches_snapshot_download(monkeypatch):
    # `difflet download --model-id <wan> --revision <sha>` end to end: the pinned
    # revision must arrive at huggingface_hub.snapshot_download, not be dropped.
    import importlib

    cli_main = importlib.import_module("difflet.cli.main")  # `difflet.cli.main` is also a function
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda **kw: calls.append(kw) or "/fake/snapshot")
    cli_main.main(["download", "--model-id", _WAN21_ID, "--revision", _WAN21_REV])
    assert [c["revision"] for c in calls] == [_WAN21_REV]
    assert calls[0]["local_files_only"] is False


# ------------------------------------------------------ generate error path

def test_generate_preserves_work_dir_on_failure(monkeypatch, tmp_path, capsys):
    def boom(*a, **kw):
        raise RuntimeError("stage failed")

    monkeypatch.setattr(wan_mod.runner, "run_stage", boom)
    with pytest.raises(RuntimeError):
        WanOrchestrator(_wan_args(work_dir=str(tmp_path))).generate()
    assert "work dir preserved" in capsys.readouterr().err
    # Work dir not removed on failure.
    assert tmp_path.exists()


def test_generate_removes_work_dir_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr(wan_mod.runner, "run_stage", lambda *a, **kw: None)
    wd = tmp_path / "work"
    WanOrchestrator(_wan_args(work_dir=str(wd), keep_work_dir=False)).generate()
    assert not wd.exists()


def test_generate_keeps_work_dir_when_requested(monkeypatch, tmp_path):
    monkeypatch.setattr(wan_mod.runner, "run_stage", lambda *a, **kw: None)
    wd = tmp_path / "work"
    WanOrchestrator(_wan_args(work_dir=str(wd), keep_work_dir=True)).generate()
    assert wd.exists()


def test_compile_doubles_cores_with_cfg_parallel(monkeypatch):
    cores = {}
    monkeypatch.setattr(
        wan_mod.runner, "run_stage",
        lambda orch, stage, num_cores, **kw: cores.update({stage: num_cores}),
    )
    WanOrchestrator(_wan_args(tp_degree=4, cp_degree=1, cfg_parallel=True)).compile()
    assert cores["transformer"] == 8  # 4 * 1 * 2
    assert cores["vae"] == 1


# ------------------------------------------------------ _run_stage_internal

def test_run_stage_internal_dispatch(monkeypatch):
    orch = WanOrchestrator(_wan_args())
    seen = []
    monkeypatch.setattr(orch, "_stage_transformer", lambda args: seen.append("t"))
    monkeypatch.setattr(orch, "_stage_vae", lambda args: seen.append("v"))
    orch._run_stage_internal("transformer", orch.args)
    orch._run_stage_internal("vae", orch.args)
    assert seen == ["t", "v"]


def test_run_stage_internal_unknown_raises():
    with pytest.raises(ValueError):
        WanOrchestrator(_wan_args())._run_stage_internal("bogus", _wan_args())


# ------------------------------------------------------ _stage_compiled_dir

def test_stage_compiled_dir_names():
    # Schema v5: <cache>/<component>/<sha256[:16]>; manifest is authoritative.
    orch = WanOrchestrator(_wan_args(cache_dir="/c", tp_degree=4, cp_degree=2,
                                     cfg_parallel=True))
    t = orch._stage_compiled_dir("transformer", orch.args)
    v = orch._stage_compiled_dir("vae", orch.args)
    assert t.parent == Path("/c/wan_transformer")
    assert v.parent == Path("/c/wan_vae")
    for path in (t, v):
        assert len(path.name) == 16 and int(path.name, 16) >= 0
    # deterministic + shape-sensitive
    assert orch._stage_compiled_dir("transformer", orch.args) == t
    other = _wan_args(cache_dir="/c", tp_degree=4, cp_degree=2, cfg_parallel=True, height=320)
    assert WanOrchestrator(other)._stage_compiled_dir("transformer", other) != t


def test_stage_compiled_dir_unknown_raises():
    orch = WanOrchestrator(_wan_args())
    with pytest.raises(ValueError):
        orch._stage_compiled_dir("nope", orch.args)


def test_stage_compiled_dir_distinguishes_wan_2_1_from_2_2():
    # Both Wan 2.2 and Wan 2.1 route to WanOrchestrator; their compiled
    # artifacts must never share a directory.
    a22 = _wan_args(cache_dir="/c")
    a21 = _wan_args(cache_dir="/c", model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    o22, o21 = WanOrchestrator(a22), WanOrchestrator(a21)
    for stage in ("transformer", "vae"):
        assert o22._stage_compiled_dir(stage, a22) != o21._stage_compiled_dir(stage, a21)


def test_stage_compiled_dir_wan_2_2_keeps_legacy_prefix():
    # The historical model id keeps its bare "wan" component prefix; other
    # Wan versions get a model-derived prefix (schema v5 hash dirs below it).
    args = _wan_args(cache_dir="/c")
    orch = WanOrchestrator(args)
    assert orch._stage_compiled_dir("transformer", args).parent == Path("/c/wan_transformer")
    assert orch._stage_compiled_dir("vae", args).parent == Path("/c/wan_vae")
    a21 = _wan_args(cache_dir="/c", model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    assert WanOrchestrator(a21)._stage_compiled_dir("transformer", a21).parent == Path(
        "/c/wan2_1_t2v_14b_diffusers_transformer"
    )


# ------------------------------------------------------ _shared_cli_args

def test_shared_cli_args_includes_cfg_and_optionals():
    orch = WanOrchestrator(_wan_args(cfg_parallel=True, cache_dir="/c"))
    parts = orch._shared_cli_args("compile", work_dir="/w")
    assert "--cfg-parallel" in parts
    assert parts[parts.index("--prompt") + 1] == "a cat walking"
    assert parts[parts.index("--output") + 1] == "/tmp/wan.mp4"
    assert parts[parts.index("--cache-dir") + 1] == "/c"
    assert parts[parts.index("--work-dir") + 1] == "/w"


def test_shared_cli_args_omits_optionals_when_absent():
    orch = WanOrchestrator(_wan_args(cfg_parallel=False, prompt=None,
                                     output=None, cache_dir=None))
    parts = orch._shared_cli_args("generate")
    assert "--cfg-parallel" not in parts
    assert "--prompt" not in parts
    assert "--output" not in parts
    assert "--cache-dir" not in parts
    assert "--work-dir" not in parts
    assert "--revision" not in parts


def test_shared_cli_args_without_revision_is_byte_identical_to_before():
    # Trainium/TPU stage command lines must not change when --revision is not
    # given (the stage cache identity and every recorded command stay valid).
    parts = WanOrchestrator(_wan_args(cache_dir="/c"))._shared_cli_args(
        "generate", work_dir="/w")
    assert parts == [
        "--model-id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "--tp-degree", "4", "--cp-degree", "1", "--cp-mode", "gather_kv",
        "--height", "480", "--width", "832", "--num-frames", "9",
        "--steps", "2", "--guidance-scale", "1.0", "--seed", "42",
        "--stage-mode", "generate",
        "--prompt", "a cat walking", "--output", "/tmp/wan.mp4",
        "--cache-dir", "/c", "--work-dir", "/w",
    ]


def test_shared_cli_args_forwards_revision_only_when_set():
    orch = WanOrchestrator(_wan_args(model_id=_WAN21_ID, revision=_WAN21_REV))
    parts = orch._shared_cli_args("generate", work_dir="/w")
    assert parts[parts.index("--revision") + 1] == _WAN21_REV
    assert parts.count("--revision") == 1
    # The pair is purely additive: removing it gives the unpinned command line.
    plain = WanOrchestrator(_wan_args(model_id=_WAN21_ID))._shared_cli_args(
        "generate", work_dir="/w")
    without = list(parts)
    i = without.index("--revision")
    del without[i:i + 2]
    assert without == plain


def test_shared_cli_args_revision_round_trips_through_the_stage_parser():
    from difflet.cli import stage

    parts = WanOrchestrator(_wan_args(model_id=_WAN21_ID, revision=_WAN21_REV)
                            )._shared_cli_args("generate")
    ns, _ = stage._build_stage_parser().parse_known_args(
        ["--orchestrator", "wan", "--stage", "transformer", *parts])
    assert ns.revision == _WAN21_REV


@pytest.mark.parametrize("mode", ["compile", "generate"])
def test_orchestrator_hands_revision_to_every_stage(monkeypatch, tmp_path, mode):
    stages = {}
    monkeypatch.setattr(
        wan_mod.runner, "run_stage",
        lambda orch, stage, **kw: stages.update({stage: kw["cli_args"]}),
    )
    args = _wan_args(model_id=_WAN21_ID, revision=_WAN21_REV,
                     work_dir=str(tmp_path / "work"), keep_work_dir=True)
    getattr(WanOrchestrator(args), mode)()
    assert set(stages) == {"transformer", "vae"}
    for cli_args in stages.values():
        assert cli_args[cli_args.index("--revision") + 1] == _WAN21_REV


# ----------------------------------------------------------------- _save_video

def test_save_video_success(monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr("diffusers.utils.export_to_video", lambda *a, **kw: None)
    ok = wan_mod._save_video(torch.zeros(1, 3, 2, 4, 4), str(tmp_path / "v.mp4"))
    assert ok is True


def test_save_video_passes_float_unit_range_frames(monkeypatch, tmp_path):
    # export_to_video multiplies ndarray frames by 255 itself — passing uint8
    # wraps pixels to 256-v (color inversion). Contract: float32 [0, 1].
    import numpy as np
    import torch

    exported = {}
    monkeypatch.setattr("diffusers.utils.export_to_video",
                        lambda frames, path, fps: exported.update(frames=frames, fps=fps))
    ok = wan_mod._save_video(torch.zeros(1, 3, 2, 4, 4), str(tmp_path / "v.mp4"))
    assert ok is True and exported["fps"] == 16
    f0 = exported["frames"][0]
    assert f0.shape == (4, 4, 3) and f0.dtype == np.float32
    assert abs(float(f0[0, 0, 0]) - 0.5) < 1e-6  # [-1,1] zeros -> 0.5


def test_save_video_export_failure_returns_false(monkeypatch, tmp_path):
    import torch

    def boom(*a, **kw):
        raise RuntimeError("no codec")

    monkeypatch.setattr("diffusers.utils.export_to_video", boom)
    ok = wan_mod._save_video(torch.zeros(1, 3, 2, 4, 4), str(tmp_path / "v.mp4"))
    assert ok is False


# ----------------------------------------------------------------- stage internals

def _setup_wan_fakes(monkeypatch):
    _FakeWanApp.instances = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/model",
    )
    _inject(
        monkeypatch, "difflet.models.wan.application",
        NeuronWanApplication=_FakeWanApp,
        _latent_num_frames=lambda f: 2,
    )


def test_stage_transformer_compile(monkeypatch, tmp_path):
    _setup_wan_fakes(monkeypatch)
    orch = WanOrchestrator(_wan_args(stage_mode="compile", cache_dir=str(tmp_path)))
    orch._stage_transformer(orch.args)
    app = _FakeWanApp.instances[-1]
    assert app.compiled is not None
    assert app.loaded is None


def test_stage_transformer_generate_saves_latents(monkeypatch, tmp_path):
    _setup_wan_fakes(monkeypatch)
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path),
                     cache_dir=str(tmp_path))
    orch = WanOrchestrator(args)
    orch._finish_stage_compile("transformer", args, orch._stage_compiled_dir("transformer", args))
    orch._stage_transformer(args)
    assert (tmp_path / "latents.pt").exists()


def test_stage_transformer_lets_pipeline_prepare_latents(monkeypatch, tmp_path):
    # Sampling must start from the pipeline's unit-variance prepare_latents
    # noise (seeded via generator), not orchestrator-injected randn*0.1 smoke
    # latents — those decode to a flat gray video.
    import torch
    _setup_wan_fakes(monkeypatch)
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path),
                     cache_dir=str(tmp_path), seed=1234)
    orch = WanOrchestrator(args)
    orch._finish_stage_compile("transformer", args, orch._stage_compiled_dir("transformer", args))
    orch._stage_transformer(args)
    kw = _FakeWanApp.instances[-1].call_kwargs
    assert "latents" not in kw
    gen = kw.get("generator")
    assert isinstance(gen, torch.Generator)
    assert gen.initial_seed() == 1234


def test_stage_vae_compile(monkeypatch, tmp_path):
    _setup_wan_fakes(monkeypatch)
    orch = WanOrchestrator(_wan_args(stage_mode="compile", cache_dir=str(tmp_path)))
    orch._stage_vae(orch.args)
    assert _FakeWanApp.instances[-1].compiled is not None


@pytest.mark.parametrize("stage", ["transformer", "vae"])
@pytest.mark.parametrize("revision", [None, _WAN21_REV])
def test_stage_resolves_weights_at_the_requested_revision(
    monkeypatch, tmp_path, stage, revision,
):
    # The stage process runs offline (local_files_only=True); without the
    # revision it resolves whatever snapshot the hub cache's refs/main points
    # at, not the one `difflet download --revision` fetched.
    _setup_wan_fakes(monkeypatch)
    resolved = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, **kw: resolved.append(kw) or "/fake/model",
    )
    args = _wan_args(model_id=_WAN21_ID, revision=revision,
                     stage_mode="compile", cache_dir=str(tmp_path))
    orch = WanOrchestrator(args)
    getattr(orch, f"_stage_{stage}")(args)
    assert [kw.get("revision") for kw in resolved] == [revision]
    assert [kw["local_files_only"] for kw in resolved] == [True]


def test_stage_vae_generate_saves_pt_for_non_mp4(monkeypatch, tmp_path):
    import torch
    _setup_wan_fakes(monkeypatch)
    torch.save(torch.zeros(1, 16, 2, 4, 4), tmp_path / "latents.pt")
    out = tmp_path / "out.png"  # non-mp4 -> .pt branch
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path),
                     cache_dir=str(tmp_path), output=str(out))
    orch = WanOrchestrator(args)
    orch._finish_stage_compile("vae", args, orch._stage_compiled_dir("vae", args))
    orch._stage_vae(args)
    assert (tmp_path / "out.pt").exists()


def test_stage_vae_generate_mp4_branch(monkeypatch, tmp_path):
    import torch
    _setup_wan_fakes(monkeypatch)
    monkeypatch.setattr(wan_mod, "_save_video", lambda frames, path: True)
    torch.save(torch.zeros(1, 16, 2, 4, 4), tmp_path / "latents.pt")
    out = tmp_path / "out.mp4"
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path),
                     cache_dir=str(tmp_path), output=str(out))
    orch = WanOrchestrator(args)
    orch._finish_stage_compile("vae", args, orch._stage_compiled_dir("vae", args))
    orch._stage_vae(args)
    # mp4 path taken -> no .pt fallback written.
    assert not (tmp_path / "out.pt").exists()


def test_wan_pipeline_prepare_latents_is_unit_variance_and_seeded():
    # Sampling must start from unit-variance noise (scheduler init sigma 1.0);
    # the old smoke harness injected randn*0.1, which collapses the trajectory
    # to the latent mean and decodes to a flat gray video. The orchestrator now
    # delegates to this pipeline path with a seeded generator.
    import torch
    from difflet.models.wan.pipeline import WanOrchestrator as WanPipelineOrchestrator

    pipe = WanPipelineOrchestrator.__new__(WanPipelineOrchestrator)
    pipe.height, pipe.width, pipe.num_frames = 480, 832, 9
    pipe.dtype = torch.float32
    make = lambda seed: WanPipelineOrchestrator.prepare_latents(
        pipe, batch_size=1, generator=torch.Generator().manual_seed(seed))
    a, b, c = make(42), make(42), make(7)
    assert a.shape == (1, 16, 3, 60, 104)
    assert 0.9 < float(a.float().std()) < 1.1
    assert torch.equal(a, b)          # same seed -> deterministic
    assert not torch.equal(a, c)      # different seed -> different noise


# ----------------------------------------------------------------- --host-vae

def test_compile_host_vae_skips_neuron_vae_stage(monkeypatch):
    # NCC_EVRF007: the single-shot Neuron VAE graph exceeds the compiler
    # instruction limit beyond ~9 frames; --host-vae decodes on CPU instead.
    stages = []
    monkeypatch.setattr(wan_mod.runner, "run_stage",
                        lambda orch, stage, **kw: stages.append(stage))
    WanOrchestrator(_wan_args(host_vae=True, num_frames=81)).compile()
    assert stages == ["transformer"]


def test_generate_host_vae_decodes_on_host(monkeypatch, tmp_path):
    stages = []
    monkeypatch.setattr(wan_mod.runner, "run_stage",
                        lambda orch, stage, **kw: stages.append(stage))
    decoded = {}
    monkeypatch.setattr(wan_mod, "_decode_latents_host",
                        lambda latents_path, model_id, output_path, revision=None:
                        decoded.update(latents=latents_path, out=output_path))
    out = tmp_path / "wan.mp4"
    args = _wan_args(host_vae=True, num_frames=81, work_dir=str(tmp_path / "work"),
                     keep_work_dir=True, output=str(out))
    WanOrchestrator(args).generate()
    assert stages == ["transformer"]
    assert decoded["out"] == str(out)
    assert decoded["latents"].endswith("latents.pt")


# ------------------------------------------------------ probe-free TeaCache

def test_shared_cli_args_forward_probe_free_teacache_flags():
    parts = WanOrchestrator(_wan_args(teacache_cadence=2))._shared_cli_args("generate")
    assert parts[parts.index("--teacache-cadence") + 1] == "2"
    assert "--teacache-online-delta" not in parts
    parts = WanOrchestrator(_wan_args(teacache_online_delta=0.6))._shared_cli_args("generate")
    assert parts[parts.index("--teacache-online-delta") + 1] == "0.6"
    assert "--teacache-cadence" not in parts


def test_stage_transformer_threads_cadence_into_app_without_changing_artifact(
    monkeypatch, tmp_path,
):
    # --teacache-cadence was dropped by the stage (device-confirmed no-op); the
    # transformer stage must hand it to the app, while the compiled-dir identity
    # stays that of a plain generate so the warm cache hits.
    _setup_wan_fakes(monkeypatch)
    plain = _wan_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path))
    cadence = _wan_args(stage_mode="generate", work_dir=str(tmp_path),
                        cache_dir=str(tmp_path), teacache_cadence=2)
    orch = WanOrchestrator(cadence)
    assert orch._stage_compiled_dir("transformer", cadence) == \
        WanOrchestrator(plain)._stage_compiled_dir("transformer", plain)
    orch._finish_stage_compile("transformer", cadence, orch._stage_compiled_dir("transformer", cadence))
    orch._stage_transformer(cadence)
    kw = _FakeWanApp.instances[-1].kwargs
    assert kw["teacache_cadence"] == 2
    assert kw["teacache_online_delta_alpha"] is None


def test_transformer_virtual_core_size_ring_only():
    # Ring CP needs NEURON_RT_VIRTUAL_CORE_SIZE=2: the nkilib ring kernel's
    # per-core send/recv buffers exist only in its LNC2 SPMD-grid variant, and
    # without the env the compile dies with NCC_ILLC059 (found on device,
    # wan ring 512x512x9, 2026-08-30). Every other mode keeps None so existing
    # compile caches stay valid.
    f = wan_mod._transformer_virtual_core_size
    assert f(_wan_args(cp_degree=2, cp_mode="ring")) == 2
    assert f(_wan_args(cp_degree=1, cp_mode="ring")) is None  # ring needs cp>1
    assert f(_wan_args(cp_degree=2, cp_mode="gather_kv")) is None
    assert f(_wan_args(cp_degree=2, cp_mode="ulysses")) is None
    assert f(_wan_args(cp_degree=1, cp_mode="gather_kv")) is None


# ------------------------------------------- calibrated-adaptive TeaCache (CLI wiring)

def _write_calibration(path, *, model, shape_label, target=1.5, num_steps=20):
    import json

    from difflet.pipeline.teacache import CALIBRATION_SCHEMA
    path.write_text(json.dumps({
        "schema": CALIBRATION_SCHEMA, "model": model, "shape_label": shape_label,
        "num_steps": num_steps, "poly_coef": [0.0, 1.0], "threshold": 0.1,
        "target_speedup": target,
    }))
    return str(path)


def test_shared_cli_args_forward_adaptive_teacache_flags():
    parts = WanOrchestrator(_wan_args(
        teacache_speedup=1.5, teacache_calibration="/c.json"))._shared_cli_args("generate")
    assert parts[parts.index("--teacache-speedup") + 1] == "1.5"
    assert parts[parts.index("--teacache-calibration") + 1] == "/c.json"
    plain = WanOrchestrator(_wan_args())._shared_cli_args("generate")
    assert "--teacache-speedup" not in plain and "--teacache-calibration" not in plain


def test_adaptive_teacache_gets_its_own_probe_artifact(monkeypatch, tmp_path):
    # Calibrated-adaptive TeaCache builds the fused block-0 probe NEFF
    # (application.py, requires_teacache_probe), so the transformer artifact
    # identity must carry it. Before the fix the key ignored the probe: an
    # adaptive generate hit the plain artifact and failed on device with
    # "teacache_probe/model.pt does not exist" (trn2.3xlarge, 2026-10-04).
    _setup_wan_fakes(monkeypatch)
    calib = _write_calibration(tmp_path / "wan.json", model="wan", shape_label="480x832x9")
    plain = _wan_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path))
    cadence = _wan_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path),
                        teacache_cadence=2)
    adaptive = _wan_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path),
                         teacache_speedup=1.5, teacache_calibration=calib)
    orch = WanOrchestrator(adaptive)
    plain_inputs = WanOrchestrator(plain)._stage_cache_inputs("transformer", plain)
    # additive: plain and probe-free requests keep the exact pre-fix key
    assert "teacache_probe" not in plain_inputs
    assert WanOrchestrator(cadence)._stage_cache_inputs("transformer", cadence) == plain_inputs
    assert orch._stage_cache_inputs("transformer", adaptive) == {**plain_inputs,
                                                                "teacache_probe": True}
    assert orch._stage_compiled_dir("transformer", adaptive) != \
        WanOrchestrator(plain)._stage_compiled_dir("transformer", plain)
    # the VAE stage is unaffected
    assert orch._stage_cache_inputs("vae", adaptive) == \
        WanOrchestrator(plain)._stage_cache_inputs("vae", plain)
    orch._finish_stage_compile("transformer", adaptive,
                               orch._stage_compiled_dir("transformer", adaptive))
    WanOrchestrator(plain)._finish_stage_compile(
        "transformer", plain, WanOrchestrator(plain)._stage_compiled_dir("transformer", plain))
    orch._stage_transformer(adaptive)
    kw = _FakeWanApp.instances[-1].kwargs
    assert kw["teacache_calibration_path"] == calib
    assert kw["teacache_cadence"] is None and kw["teacache_online_delta_alpha"] is None
    orch._stage_transformer(plain)
    assert _FakeWanApp.instances[-1].kwargs["teacache_calibration_path"] is None


@pytest.mark.parametrize("bad", [
    dict(target=1.5, speedup=2.0, shape_label="480x832x9", match="lower than requested"),
    dict(target=1.5, speedup=1.5, shape_label="512x512x9", match="shape mismatch"),
])
def test_stage_transformer_validates_the_calibration_like_the_probe_pipelines(
    monkeypatch, tmp_path, bad,
):
    _setup_wan_fakes(monkeypatch)
    calib = _write_calibration(tmp_path / "wan.json", model="wan",
                               shape_label=bad["shape_label"], target=bad["target"])
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path),
                     teacache_speedup=bad["speedup"], teacache_calibration=calib)
    orch = WanOrchestrator(args)
    orch._finish_stage_compile("transformer", args, orch._stage_compiled_dir("transformer", args))
    with pytest.raises(ValueError, match=bad["match"]):
        orch._stage_transformer(args)


# ------------------------------------------- neuron backend (torchrun MPMD, task W5)
# With DIFFLET_BACKEND=neuron the transformer stage runs as torchrun ranks; each builds
# TorchNeuronWanApplication through DiffletPipeline.from_pretrained (here the CPU fake of
# tests/unit/cli/_fake_neuron_pipeline.py), the VAE decodes on the host, and only rank 0 writes.

_NEURON_APP_KWARGS = {
    "text_seq_len": 512, "batch_size": 1, "enable_text_encoder": True,
    "enable_transformer": True, "enable_transformer_2": False, "enable_vae_decoder": False,
    "exec_mode": None, "encoder_threads": 12,
}


@pytest.fixture
def neuron_stage(monkeypatch):
    """One process, DIFFLET_BACKEND=neuron, no process group (rank 0): from_pretrained is the
    CPU fake; the Trainium application is a recorder that must stay unused."""
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    _setup_wan_fakes(monkeypatch)
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(fake_pipeline, "CALLS", [])
    monkeypatch.setattr(DiffletPipeline, "from_pretrained",
                        classmethod(fake_pipeline.fake_from_pretrained))
    return fake_pipeline


def _fake_latents(seed: int):
    import torch

    return torch.randn(fake_pipeline.LATENT_SHAPE, generator=torch.Generator().manual_seed(seed))


def test_stage_transformer_neuron_routes_through_pipeline(neuron_stage, monkeypatch, tmp_path,
                                                          capsys):
    import torch

    monkeypatch.setattr(WanOrchestrator, "_require_stage_artifact",
                        lambda *a: pytest.fail("non-AoT: there is no stage artifact to require"))
    cache = tmp_path / "cache"
    args = _wan_args(model_id=_WAN21_ID, revision=_WAN21_REV, stage_mode="generate",
                     work_dir=str(tmp_path), cache_dir=str(cache), prompt="a fox",
                     steps=50, guidance_scale=5.0, seed=42)
    WanOrchestrator(args)._stage_transformer(args)

    assert _FakeWanApp.instances == []  # the Trainium application is never built
    [call] = neuron_stage.CALLS
    kw = call["kwargs"]
    assert call["model_id"] == _WAN21_ID
    assert (kw["backend"], kw["revision"], kw["local_files_only"]) == ("neuron", _WAN21_REV, True)
    assert kw["application_kwargs"] == _NEURON_APP_KWARGS
    assert (kw["height"], kw["width"], kw["num_frames"], kw["shapes"]) == (480, 832, 9, None)
    assert kw["dtype"] is torch.bfloat16 and kw["compile_cache_dir"] == str(cache)
    parallel = kw["parallel"]
    assert (parallel.tp_degree, parallel.cp_degree, parallel.cfg_parallel_enabled,
            parallel.sp_enabled) == (4, 1, False, False)
    app = call["pipeline"].app
    # The warm-up runs in load(), before the step loop (the Trainium stage skips it: AoT NEFFs).
    assert [c["skip_warmup"] for c in app.load_calls] == [False]
    assert app.calls == [{"prompt": "a fox", "height": 480, "width": 832, "num_frames": 9,
                          "num_inference_steps": 50, "guidance_scale": 5.0,
                          "output_type": "latent", "seed": 42}]
    assert torch.equal(torch.load(tmp_path / "latents.pt"), _fake_latents(42))
    out = capsys.readouterr().out
    assert out.count(f"[wan] latents saved to {tmp_path / 'latents.pt'}") == 1
    assert "compiled_blocks=40 unwarmed_shapes=[]" in out
    assert not cache.exists()  # generate writes no stage manifest and no AoT artifact


def test_stage_transformer_neuron_compile_warms_up_and_writes_the_manifest(neuron_stage,
                                                                           tmp_path):
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    args = _wan_args(model_id=_WAN21_ID, stage_mode="compile", work_dir=str(tmp_path / "w"),
                     cache_dir=str(tmp_path / "cache"))
    orch = WanOrchestrator(args)
    orch._stage_transformer(args)

    [call] = neuron_stage.CALLS
    app = call["pipeline"].app
    # `difflet compile --backend neuron` = the warm-up that fills the NEFF cache; no generate.
    assert [c["skip_warmup"] for c in app.load_calls] == [False] and app.calls == []
    compiled_dir = orch._stage_compiled_dir("transformer", args)
    inputs = orch._stage_cache_inputs("transformer", args)
    assert has_valid_stage_manifest(compiled_dir, inputs)
    assert (inputs["backend"], inputs["exec_mode"]) == ("neuron", "compile")
    assert not (tmp_path / "w").exists()


@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_stage_transformer_neuron_writes_files_on_rank0_only(neuron_stage, monkeypatch, tmp_path,
                                                             capsys, mode):
    # work_file() and the compiled dir are the same path on every rank: one writer, rank 0.
    from difflet.backends.neuron.core import distributed

    monkeypatch.setattr(distributed, "is_rank0", lambda: False)
    work, cache = tmp_path / "work", tmp_path / "cache"
    work.mkdir()
    args = _wan_args(model_id=_WAN21_ID, stage_mode=mode, work_dir=str(work),
                     cache_dir=str(cache))
    WanOrchestrator(args)._stage_transformer(args)

    [call] = neuron_stage.CALLS
    assert len(call["pipeline"].app.calls) == (1 if mode == "generate" else 0)  # every rank runs
    assert list(work.iterdir()) == [] and not cache.exists()
    assert "latents saved" not in capsys.readouterr().out


def test_stage_transformer_neuron_rejects_request_batches(neuron_stage, tmp_path):
    # Every rank would claim a different request from the queue and feed its own prompt to
    # the same TP collectives; data parallelism is out of phase 1.
    args = _wan_args(model_id=_WAN21_ID, work_dir=str(tmp_path),
                     requests_dir=str(tmp_path / "requests"), worker_index=0,
                     dp_schedule="round_robin")
    with pytest.raises(NotImplementedError, match="--requests"):
        WanOrchestrator(args)._stage_transformer(args)
    assert neuron_stage.CALLS == []


@pytest.mark.parametrize("overrides, parallel, app_extra", [
    (dict(cp_degree=2), (2, False, False), {}),
    (dict(cfg_parallel=True), (1, True, False), {}),
    (dict(sp_enabled=True), (1, False, True), {}),
    (dict(teacache_cadence=2), (1, False, False), {"teacache_cadence": 2}),
    (dict(teacache_online_delta=0.6), (1, False, False), {"teacache_online_delta_alpha": 0.6}),
])
def test_stage_transformer_neuron_hands_unsupported_options_on(neuron_stage, tmp_path, overrides,
                                                               parallel, app_extra):
    # Not dropped silently: prepare_runtime rejects cp/cfg/sp, the application TeaCache.
    args = _wan_args(model_id=_WAN21_ID, stage_mode="compile", cache_dir=str(tmp_path),
                     **overrides)
    WanOrchestrator(args)._stage_transformer(args)
    kw = neuron_stage.CALLS[0]["kwargs"]
    p = kw["parallel"]
    assert (p.cp_degree, p.cfg_parallel_enabled, p.sp_enabled) == parallel
    assert kw["application_kwargs"] == {**_NEURON_APP_KWARGS, **app_extra}


@pytest.mark.parametrize("backend_name, tracked", [("neuron", True), ("cpu", False)])
def test_stage_transformer_neuron_logs_fallbacks(neuron_stage, monkeypatch, tmp_path, capsys,
                                                 backend_name, tracked):
    # track_fallbacks() synchronizes the device: only a pipeline on the neuron backend uses it.
    from contextlib import contextmanager

    from difflet.backends.neuron import runtime

    entered = []

    @contextmanager
    def fake_track():
        ops: list = []
        entered.append(ops)
        yield ops
        ops.append("aten::fake_cpu_op")

    monkeypatch.setattr(runtime, "track_fallbacks", fake_track)
    monkeypatch.setattr(neuron_stage, "BACKEND_NAME", backend_name)
    args = _wan_args(model_id=_WAN21_ID, work_dir=str(tmp_path), cache_dir=str(tmp_path / "c"))
    WanOrchestrator(args)._stage_transformer(args)

    assert len(entered) == (2 if tracked else 0)  # around load() and around the generate
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if "fallbacks" in ln]
    assert len(line) == 1 and "rank 0/1" in line[0]
    assert line[0].count("aten::fake_cpu_op") == (2 if tracked else 0)


@pytest.mark.parametrize("mode", ["compile", "generate"])
def test_neuron_forces_host_vae(monkeypatch, tmp_path, capsys, mode):
    # The vae stage would run as torchrun ranks building the Trainium VAE application, and the
    # neuron application has no device VAE (enable_vae_decoder=True raises by design).
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    stages, decoded = [], []
    monkeypatch.setattr(wan_mod.runner, "run_stage",
                        lambda orch, stage, **kw: stages.append(stage))
    monkeypatch.setattr(wan_mod, "_decode_latents_host",
                        lambda latents_path, model_id, output_path, revision=None:
                        decoded.append((latents_path, revision)))
    args = _wan_args(model_id=_WAN21_ID, revision=_WAN21_REV, host_vae=False,
                     work_dir=str(tmp_path / "work"), keep_work_dir=True)
    getattr(WanOrchestrator(args), mode)()

    assert stages == ["transformer"] and args.host_vae is True
    expected = [(str(tmp_path / "work" / "latents.pt"), _WAN21_REV)] if mode == "generate" else []
    assert decoded == expected
    assert capsys.readouterr().out.count("--host-vae") == 1


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
def test_aot_backends_keep_the_vae_stage(monkeypatch, tmp_path, capsys, backend):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    if backend is not None:
        monkeypatch.setenv("DIFFLET_BACKEND", backend)
    stages = []
    monkeypatch.setattr(wan_mod.runner, "run_stage",
                        lambda orch, stage, **kw: stages.append(stage))
    args = _wan_args(model_id=_WAN21_ID, host_vae=False, work_dir=str(tmp_path / "work"))
    WanOrchestrator(args).generate()
    assert stages == ["transformer", "vae"] and args.host_vae is False
    assert "--host-vae" not in capsys.readouterr().out


def test_cache_inputs_unchanged_for_trainium_and_distinct_for_neuron(monkeypatch):
    args = _wan_args(model_id=_WAN21_ID, cache_dir="/c")
    orch = WanOrchestrator(args)

    def keyed(backend, exec_mode=None):
        monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
        monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
        if backend is not None:
            monkeypatch.setenv("DIFFLET_BACKEND", backend)
        if exec_mode is not None:
            monkeypatch.setenv("DIFFLET_EXEC_MODE", exec_mode)
        stages = ("transformer", "vae")
        return ({s: orch._stage_cache_inputs(s, args) for s in stages},
                {s: orch._stage_compiled_dir(s, args) for s in stages})

    base, base_dirs = keyed(None)
    assert set(base["transformer"]) == {"component", "model_id", "tp", "cp", "cp_mode",
                                        "cfg_parallel", "sp", "dtype", "text_seq_len", "shapes",
                                        "toolchain"}
    assert set(base["vae"]) == {"component", "model_id", "dtype", "shapes", "toolchain"}
    for backend in ("trainium", "tpu"):  # AoT: byte-identical keys and dirs
        assert keyed(backend) == (base, base_dirs)
    neuron, neuron_dirs = keyed("neuron")
    for stage in ("transformer", "vae"):
        assert neuron[stage] == {**base[stage], "backend": "neuron", "exec_mode": "compile"}
        assert neuron_dirs[stage] != base_dirs[stage]
        assert neuron_dirs[stage].parent == base_dirs[stage].parent
    # resolve_exec_mode: an omitted --exec-mode keys like the default (compile); eager differs.
    assert keyed("neuron", "compile") == (neuron, neuron_dirs)
    eager, eager_dirs = keyed("neuron", "eager")
    assert eager["transformer"] == {**neuron["transformer"], "exec_mode": "eager"}
    assert eager_dirs["transformer"] != neuron_dirs["transformer"]


# CPU torchrun: the real run_stage -> torchrun (2 ranks, gloo) -> difflet.cli.stage ->
# WanOrchestrator, with from_pretrained replaced in every rank (FAKE_PIPELINE_ENV).

_REAL_POPEN = subprocess.Popen
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _bounded_popen(limit: float):
    """subprocess.Popen whose first untimed wait() gives up after ``limit`` s (TimeoutExpired):
    run_stage then sends torchrun SIGTERM and waits for it, so a hung launch fails the test."""

    class BoundedPopen(_REAL_POPEN):
        _bounded = False

        def wait(self, timeout=None):
            if timeout is None and not self._bounded:
                self._bounded = True
                timeout = limit
            return super().wait(timeout=timeout)

    return BoundedPopen


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"  # a zombie has exited


@pytest.fixture
def neuron_torchrun(monkeypatch, tmp_path):
    site, records = tmp_path / "site", tmp_path / "records"
    fake_pipeline.write_sitecustomize(site)
    records.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        filter(None, [str(site), str(_REPO_ROOT), os.environ.get("PYTHONPATH")])))
    monkeypatch.setenv(fake_pipeline.FAKE_PIPELINE_ENV, fake_pipeline.MODULE)
    monkeypatch.setenv(fake_pipeline.FAKE_RECORDS_ENV, str(records))
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_DISABLE_PREWARM", "1")
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.setattr(subprocess, "Popen", _bounded_popen(300))
    return records


def test_generate_neuron_cpu_torchrun_one_writer_and_decode_after_every_rank_exited(
    neuron_torchrun, monkeypatch, tmp_path, capfd,
):
    """Review Focus 5: rank 0 alone writes latents.pt, and the driver decodes only after
    run_stage returned, which is after torchrun exited: rank 1, which lingers at exit after
    rank 0 wrote the file, has exited too."""
    import torch

    seen = {}

    def decode(latents_path, model_id, output_path, revision=None):
        seen["records"] = fake_pipeline.read_rank_records(neuron_torchrun)
        seen["alive"] = [_alive(r["pid"]) for r in seen["records"]]
        seen["latents"] = torch.load(latents_path)

    monkeypatch.setattr(wan_mod, "_decode_latents_host", decode)
    work, cache = tmp_path / "work", tmp_path / "cache"
    args = _wan_args(model_id=_WAN21_ID, revision=_WAN21_REV, tp_degree=2, work_dir=str(work),
                     keep_work_dir=True, cache_dir=str(cache), steps=3, guidance_scale=5.0)
    WanOrchestrator(args).generate()

    records = seen["records"]
    assert [r["rank"] for r in records] == [0, 1] and seen["alive"] == [False, False]
    writes = [w for r in records for w in r["writes"]]
    assert writes == [{"kind": "torch.save", "path": str(work / "latents.pt"), "rank": 0}]
    for record in records:
        [call] = record["from_pretrained"]
        kw = call["kwargs"]
        assert (call["model_id"], kw["backend"], kw["revision"]) == (_WAN21_ID, "neuron",
                                                                      _WAN21_REV)
        assert kw["parallel"]["tp_degree"] == 2 and record["world_size"] == 2
        assert [c["skip_warmup"] for c in record["pipeline_load_calls"]] == [False]
        assert [(c["seed"], c["num_inference_steps"]) for c in record["app_calls"]] == [(42, 3)]
    assert torch.equal(seen["latents"], _fake_latents(42))
    log = "".join(capfd.readouterr())
    assert log.count("[wan] latents saved to") == 1
    assert log.count("compiled_blocks=40 unwarmed_shapes=[]") == 2  # one line per rank
    assert not cache.exists()  # generate: no stage manifest, no AoT artifact


def test_compile_neuron_cpu_torchrun_rank0_writes_the_manifest(neuron_torchrun, tmp_path):
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    args = _wan_args(model_id=_WAN21_ID, tp_degree=2, cache_dir=str(tmp_path / "cache"))
    orch = WanOrchestrator(args)
    orch.compile()

    records = fake_pipeline.read_rank_records(neuron_torchrun)
    assert [r["rank"] for r in records] == [0, 1]
    compiled_dir = orch._stage_compiled_dir("transformer", args)
    writes = [w for r in records for w in r["writes"]]
    assert writes == [{"kind": "manifest", "path": str(compiled_dir), "rank": 0}]
    assert has_valid_stage_manifest(compiled_dir, orch._stage_cache_inputs("transformer", args))
    for record in records:
        assert [c["skip_warmup"] for c in record["pipeline_load_calls"]] == [False]
        assert record["app_calls"] == []
