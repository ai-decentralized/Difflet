from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

from difflet.cli.orchestrators import wan as wan_mod
from difflet.cli.orchestrators.wan import WanOrchestrator


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

def test_download_resolves_remote(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, local_files_only, allow_patterns=None: calls.append(local_files_only),
    )
    WanOrchestrator(_wan_args()).download()
    assert calls == [False]


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


def test_stage_transformer_threads_calibration_into_app_without_changing_artifact(
    monkeypatch, tmp_path,
):
    # Calibrated-adaptive TeaCache: Wan's block-0 signal is a host CPU shadow
    # (no probe NEFF), so the calibration path reaches the app as a runtime-only
    # kwarg and the compiled-dir identity stays that of a plain generate.
    _setup_wan_fakes(monkeypatch)
    calib = _write_calibration(tmp_path / "wan.json", model="wan", shape_label="480x832x9")
    plain = _wan_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path))
    adaptive = _wan_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path),
                         teacache_speedup=1.5, teacache_calibration=calib)
    orch = WanOrchestrator(adaptive)
    assert orch._stage_compiled_dir("transformer", adaptive) == \
        WanOrchestrator(plain)._stage_compiled_dir("transformer", plain)
    orch._finish_stage_compile("transformer", adaptive,
                               orch._stage_compiled_dir("transformer", adaptive))
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
