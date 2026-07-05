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
        work_dir=None, keep_work_dir=False,
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
    orch = WanOrchestrator(_wan_args(cache_dir="/c", tp_degree=4, cp_degree=2,
                                     cfg_parallel=True))
    t = orch._stage_compiled_dir("transformer", orch.args)
    v = orch._stage_compiled_dir("vae", orch.args)
    assert t == Path("/c/wan_transformer_tp4cp2cfg_h480w832f9")
    assert v == Path("/c/wan_vae_h480w832f9")


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


def test_stage_compiled_dir_wan_2_2_keeps_legacy_names():
    # Additive-only: the historical model id keeps its pre-fix dir names so
    # existing compile caches stay valid.
    args = _wan_args(cache_dir="/c")
    orch = WanOrchestrator(args)
    assert orch._stage_compiled_dir("transformer", args) == \
        Path("/c/wan_transformer_tp4cp1_h480w832f9")
    assert orch._stage_compiled_dir("vae", args) == Path("/c/wan_vae_h480w832f9")


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
    WanOrchestrator(args)._stage_transformer(args)
    assert (tmp_path / "latents.pt").exists()


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
    WanOrchestrator(args)._stage_vae(args)
    assert (tmp_path / "out.pt").exists()


def test_stage_vae_generate_mp4_branch(monkeypatch, tmp_path):
    import torch
    _setup_wan_fakes(monkeypatch)
    monkeypatch.setattr(wan_mod, "_save_video", lambda frames, path: True)
    torch.save(torch.zeros(1, 16, 2, 4, 4), tmp_path / "latents.pt")
    out = tmp_path / "out.mp4"
    args = _wan_args(stage_mode="generate", work_dir=str(tmp_path),
                     cache_dir=str(tmp_path), output=str(out))
    WanOrchestrator(args)._stage_vae(args)
    # mp4 path taken -> no .pt fallback written.
    assert not (tmp_path / "out.pt").exists()
