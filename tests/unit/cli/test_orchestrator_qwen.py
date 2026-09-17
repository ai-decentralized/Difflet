from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

from difflet.cli.orchestrators import qwen_image as qwen_mod
from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator


def _qwen_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Qwen/Qwen-Image", tp_degree=4, cp_degree=1,
        cp_mode="gather_kv", height=1024, width=1024, num_frames=None,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/qwen.png",
        steps=4, guidance_scale=4.0, seed=42, stage_mode="generate",
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
        compiled_dir=None, vae_tp_degree=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _inject(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _common_setup(monkeypatch):
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/model",
    )


class _Recorder:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.compiled = None
        self.loaded = None

    def compile(self, path):
        self.compiled = path

    def load(self, path, **kw):
        self.loaded = (path, kw)


# ----------------------------------------------------------------- download / orchestration

def test_download_resolves_remote(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, revision, local_files_only: calls.append(
            (revision, local_files_only)
        ),
    )
    QwenImageOrchestrator(_qwen_args(revision="abc123")).download()
    assert calls == [("abc123", False)]


def test_generate_preserves_work_dir_on_failure(monkeypatch, tmp_path, capsys):
    def boom(*a, **kw):
        raise RuntimeError("fail")

    monkeypatch.setattr(qwen_mod.runner, "run_stage", boom)
    with pytest.raises(RuntimeError):
        QwenImageOrchestrator(_qwen_args(work_dir=str(tmp_path))).generate()
    assert "work dir preserved" in capsys.readouterr().err
    assert tmp_path.exists()


def test_generate_removes_work_dir_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr(qwen_mod.runner, "run_stage", lambda *a, **kw: None)
    wd = tmp_path / "work"
    QwenImageOrchestrator(_qwen_args(work_dir=str(wd))).generate()
    assert not wd.exists()


def test_compile_runs_all_stages(monkeypatch):
    stages = []
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda orch, stage, **kw: stages.append(stage))
    QwenImageOrchestrator(_qwen_args()).compile()
    assert stages == ["text", "generate", "vae"]


def test_run_stage_internal_dispatch(monkeypatch):
    orch = QwenImageOrchestrator(_qwen_args())
    seen = []
    monkeypatch.setattr(orch, "_stage_text", lambda a: seen.append("text"))
    monkeypatch.setattr(orch, "_stage_generate", lambda a: seen.append("generate"))
    monkeypatch.setattr(orch, "_stage_vae", lambda a: seen.append("vae"))
    for s in ("text", "generate", "vae"):
        orch._run_stage_internal(s, orch.args)
    assert seen == ["text", "generate", "vae"]


def test_run_stage_internal_unknown_raises():
    with pytest.raises(ValueError):
        QwenImageOrchestrator(_qwen_args())._run_stage_internal("x", _qwen_args())


def test_stage_compiled_dir_names():
    # Schema v5: dirs are <cache>/<component>/<sha256[:16]>.
    orch = QwenImageOrchestrator(_qwen_args(cache_dir="/c", tp_degree=4, cp_degree=2))
    for stage, component in (
        ("text", "qwen_image_enc"),
        ("generate", "qwen_image_dit"),
        ("vae", "qwen_image_vae"),
    ):
        path = orch._stage_compiled_dir(stage, orch.args)
        assert path.parent == Path("/c") / component
        assert len(path.name) == 16 and int(path.name, 16) >= 0
        assert orch._stage_compiled_dir(stage, orch.args) == path
    other = _qwen_args(cache_dir="/c", tp_degree=4, cp_degree=2, height=512)
    assert QwenImageOrchestrator(other)._stage_compiled_dir("generate", other) != \
        orch._stage_compiled_dir("generate", orch.args)


def test_stage_compiled_dir_unknown_raises():
    orch = QwenImageOrchestrator(_qwen_args())
    with pytest.raises(ValueError):
        orch._stage_compiled_dir("nope", orch.args)


def test_shared_cli_args_optionals():
    orch = QwenImageOrchestrator(_qwen_args(cache_dir="/c"))
    parts = orch._shared_cli_args("generate", work_dir="/w")
    assert parts[parts.index("--prompt") + 1] == "a cat"
    assert parts[parts.index("--cache-dir") + 1] == "/c"
    assert parts[parts.index("--work-dir") + 1] == "/w"
    bare = QwenImageOrchestrator(
        _qwen_args(prompt=None, output=None, cache_dir=None)
    )._shared_cli_args("compile")
    assert "--prompt" not in bare and "--output" not in bare


def test_shared_cli_args_forwards_revision():
    parts = QwenImageOrchestrator(
        _qwen_args(revision="refs/pr/1")
    )._shared_cli_args("compile")

    assert parts[parts.index("--revision") + 1] == "refs/pr/1"


# ----------------------------------------------------------------- stage: text

def _setup_text(monkeypatch, app_cls):
    _common_setup(monkeypatch)
    _inject(monkeypatch, "neuronx_distributed_inference.models.config",
            NeuronConfig=lambda **kw: object(),
            TensorCaptureConfig=lambda **kw: object())
    _inject(monkeypatch,
            "neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text",
            NeuronQwen2VLTextForCausalLM=app_cls)
    _inject(monkeypatch, "neuronx_distributed_inference.utils.hf_adapter",
            load_pretrained_config=lambda **kw: {})

    def _tok(*a, **kw):
        import torch
        return types.SimpleNamespace(
            input_ids=torch.ones(1, 256, dtype=torch.int64),
            attention_mask=torch.ones(1, 256, dtype=torch.int64),
        )

    _inject(monkeypatch, "transformers",
            AutoConfig=types.SimpleNamespace(
                from_pretrained=lambda p: types.SimpleNamespace(
                    text_config=types.SimpleNamespace(pad_token_id=None))),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda p: _tok))


def test_stage_text_compile(monkeypatch, tmp_path):
    class FakeText(_Recorder):
        @classmethod
        def get_config_cls(cls):
            return lambda *a, **kw: object()

    _setup_text(monkeypatch, FakeText)
    args = _qwen_args(stage_mode="compile", cache_dir=str(tmp_path))
    created = []
    orig = FakeText.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(FakeText, "__init__", spy)
    QwenImageOrchestrator(args)._stage_text(args)
    assert created[-1].compiled is not None


def test_stage_text_generate_saves_text_pt(monkeypatch, tmp_path):
    import torch

    class FakeText(_Recorder):
        @classmethod
        def get_config_cls(cls):
            return lambda *a, **kw: object()

        def __call__(self, **kw):
            return types.SimpleNamespace(captured_tensors=[torch.zeros(1, 256, 16)])

    _setup_text(monkeypatch, FakeText)
    args = _qwen_args(stage_mode="generate", work_dir=str(tmp_path),
                      cache_dir=str(tmp_path))
    orch = QwenImageOrchestrator(args)
    orch._finish_stage_compile("text", args, orch._stage_compiled_dir("text", args))
    orch._stage_text(args)
    assert (tmp_path / "text.pt").exists()


# ----------------------------------------------------------------- stage: generate

def _setup_generate(monkeypatch, app_cls):
    _common_setup(monkeypatch)
    _inject(monkeypatch, "difflet.models.qwen_image.application",
            NeuronQwenImageApplication=app_cls)


def test_stage_generate_compile(monkeypatch, tmp_path):
    _setup_generate(monkeypatch, _Recorder)
    args = _qwen_args(stage_mode="compile", work_dir=str(tmp_path),
                      cache_dir=str(tmp_path))
    created = []
    orig = _Recorder.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(_Recorder, "__init__", spy)
    QwenImageOrchestrator(args)._stage_generate(args)
    assert created[-1].compiled is not None


def test_stage_generate_runs_and_saves_latents(monkeypatch, tmp_path):
    import torch

    sched = types.SimpleNamespace(
        config=types.SimpleNamespace(
            max_shift=1.15, base_shift=0.5,
            max_image_seq_len=4096, base_image_seq_len=256),
        timesteps=torch.zeros(4),
        set_timesteps=lambda **kw: None,
    )

    class FakeGen(_Recorder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.pipeline = types.SimpleNamespace(
                scheduler=sched,
                __call__=None,
            )
            # pipeline must be callable
            self.pipeline = _FakePipeline(sched)

    _setup_generate(monkeypatch, FakeGen)
    torch.save({"encoder_hidden_states": torch.zeros(1, 1024, 16),
                "encoder_hidden_states_mask": torch.ones(1, 1024, dtype=torch.bool)},
               tmp_path / "text.pt")
    args = _qwen_args(stage_mode="generate", work_dir=str(tmp_path),
                      cache_dir=str(tmp_path))
    orch = QwenImageOrchestrator(args)
    orch._finish_stage_compile("generate", args, orch._stage_compiled_dir("generate", args))
    orch._stage_generate(args)
    assert (tmp_path / "latents.pt").exists()


class _FakePipeline:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, **kw):
        import torch
        return types.SimpleNamespace(latents=torch.zeros(1, 4, 64))


# ----------------------------------------------------------------- stage: vae

def _setup_vae(monkeypatch, app_cls):
    _common_setup(monkeypatch)
    _inject(monkeypatch, "difflet.backends.trainium.core.config",
            NeuronConfig=lambda **kw: object())
    _inject(monkeypatch, "difflet.backends.trainium.wan.vae",
            NeuronWanVAEDecoderApplication=app_cls,
            WanVAEDecoderInferenceConfig=lambda **kw: types.SimpleNamespace(
                latents_mean=[0.0] * 16, latents_std=[1.0] * 16, **kw))
    _inject(monkeypatch, "difflet.utils.diffusers_adapter",
            load_diffusers_config=lambda p: {})


def test_stage_vae_compile(monkeypatch, tmp_path):
    _setup_vae(monkeypatch, _Recorder)
    args = _qwen_args(stage_mode="compile", work_dir=str(tmp_path),
                      cache_dir=str(tmp_path))
    created = []
    orig = _Recorder.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(_Recorder, "__init__", spy)
    QwenImageOrchestrator(args)._stage_vae(args)
    assert created[-1].compiled is not None


def test_stage_vae_compile_supports_serving_tp_and_artifact_override(monkeypatch, tmp_path):
    configs = []
    _common_setup(monkeypatch)
    _inject(
        monkeypatch,
        "difflet.backends.trainium.core.config",
        NeuronConfig=lambda **kw: configs.append(kw) or object(),
    )
    _inject(
        monkeypatch,
        "difflet.backends.trainium.wan.vae",
        NeuronWanVAEDecoderApplication=_Recorder,
        WanVAEDecoderInferenceConfig=lambda **kw: types.SimpleNamespace(
            latents_mean=[0.0] * 16, latents_std=[1.0] * 16, **kw
        ),
    )
    _inject(
        monkeypatch,
        "difflet.utils.diffusers_adapter",
        load_diffusers_config=lambda p: {},
    )
    compiled_dir = tmp_path / "serving-vae"
    args = _qwen_args(
        stage_mode="compile",
        cache_dir=str(tmp_path),
        compiled_dir=str(compiled_dir),
        vae_tp_degree=4,
    )

    app = QwenImageOrchestrator(args)
    app._stage_vae(args)

    assert configs[-1]["tp_degree"] == 4
    assert configs[-1]["world_size"] == 4


def test_stage_vae_generate_saves_output(monkeypatch, tmp_path):
    import torch

    class FakeVae(_Recorder):
        def __call__(self, z):
            return torch.zeros(1, 3, 1, 8, 8)

    _setup_vae(monkeypatch, FakeVae)
    torch.save(torch.zeros(1, 4, 64), tmp_path / "latents.pt")
    out = tmp_path / "out.png"
    args = _qwen_args(stage_mode="generate", work_dir=str(tmp_path),
                      cache_dir=str(tmp_path), output=str(out),
                      height=32, width=32)
    orch = QwenImageOrchestrator(args)
    orch._finish_stage_compile("vae", args, orch._stage_compiled_dir("vae", args))
    orch._stage_vae(args)
    # Either the png (via torchvision) or a .pt fallback exists.
    assert out.exists() or (tmp_path / "out.pt").exists()


def test_stage_vae_generate_pt_fallback(monkeypatch, tmp_path):
    import torch

    class FakeVae(_Recorder):
        def __call__(self, z):
            return torch.zeros(1, 3, 1, 8, 8)

    _setup_vae(monkeypatch, FakeVae)
    # Force the save_image fallback by injecting a torchvision.utils that raises.
    _inject(monkeypatch, "torchvision.utils",
            save_image=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no tv")))
    torch.save(torch.zeros(1, 4, 64), tmp_path / "latents.pt")
    out = tmp_path / "out.png"
    args = _qwen_args(stage_mode="generate", work_dir=str(tmp_path),
                      cache_dir=str(tmp_path), output=str(out),
                      height=32, width=32)
    orch = QwenImageOrchestrator(args)
    orch._finish_stage_compile("vae", args, orch._stage_compiled_dir("vae", args))
    orch._stage_vae(args)
    assert (tmp_path / "out.pt").exists()


def test_stage_text_compile_skips_when_manifest_matches(monkeypatch, tmp_path):
    """A repeated `difflet compile` must not rebuild a stage whose manifest
    already matches (single-component stages recompiled unconditionally)."""
    class FakeText(_Recorder):
        @classmethod
        def get_config_cls(cls):
            return lambda *a, **kw: object()

    _setup_text(monkeypatch, FakeText)
    args = _qwen_args(stage_mode="compile", cache_dir=str(tmp_path))
    created = []
    orig = FakeText.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(FakeText, "__init__", spy)
    orch = QwenImageOrchestrator(args)
    orch._stage_text(args)
    assert created[-1].compiled is not None          # first compile builds
    orch._stage_text(args)
    assert created[-1].compiled is None              # manifest matches -> skipped
    # a changed configuration invalidates the manifest and compiles again
    args2 = _qwen_args(stage_mode="compile", cache_dir=str(tmp_path), tp_degree=2)
    QwenImageOrchestrator(args2)._stage_text(args2)
    assert created[-1].compiled is not None


# ------------------------------------------- calibrated-adaptive TeaCache (additive probe)

def test_stage_generate_compiles_the_probe_additively_for_adaptive_teacache(monkeypatch, tmp_path):
    """--teacache-speedup against a DiT stage compiled without the probe: the
    fused probe is built into the SAME stage dir (the hunyuan_video rule --
    identity unchanged, DiT NEFF reused, select=["teacache_probe"]) and the
    load proceeds; a plain generate never touches the probe."""
    import json

    import torch

    from difflet.pipeline.teacache import CALIBRATION_SCHEMA

    sched = types.SimpleNamespace(
        config=types.SimpleNamespace(
            max_shift=1.15, base_shift=0.5,
            max_image_seq_len=4096, base_image_seq_len=256),
        timesteps=torch.zeros(4),
        set_timesteps=lambda **kw: None,
    )

    class FakeGen(_Recorder):
        instances = []

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.pipeline = _FakePipeline(sched)
            self.compiled_select = None
            self.probe_checked = None
            FakeGen.instances.append(self)

        def has_compiled_artifacts(self, path, select=None):
            self.probe_checked = list(select or [])
            return False

        def compile(self, path, select=None):
            self.compiled = path
            self.compiled_select = list(select) if select else None

    _setup_generate(monkeypatch, FakeGen)
    torch.save({"encoder_hidden_states": torch.zeros(1, 1024, 16),
                "encoder_hidden_states_mask": torch.ones(1, 1024, dtype=torch.bool)},
               tmp_path / "text.pt")
    calib = tmp_path / "qwen.json"
    calib.write_text(json.dumps({
        "schema": CALIBRATION_SCHEMA, "model": "qwen_image", "shape_label": "1024x1024",
        "num_steps": 20, "poly_coef": [0.0, 1.0], "threshold": 0.1, "target_speedup": 1.5,
    }))
    plain = _qwen_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path))
    adaptive = _qwen_args(stage_mode="generate", work_dir=str(tmp_path), cache_dir=str(tmp_path),
                          teacache_speedup=1.5, teacache_calibration=str(calib))
    orch = QwenImageOrchestrator(adaptive)
    compiled_dir = orch._stage_compiled_dir("generate", adaptive)
    assert compiled_dir == QwenImageOrchestrator(plain)._stage_compiled_dir("generate", plain)
    orch._finish_stage_compile("generate", adaptive, compiled_dir)
    orch._stage_generate(adaptive)
    app = FakeGen.instances[-1]
    assert app.kwargs["teacache_fused"] is True
    assert app.kwargs["teacache_speedup"] == 1.5
    assert app.kwargs["teacache_calibration_path"] == str(calib)
    assert app.probe_checked == ["teacache_probe"]
    assert app.compiled == str(compiled_dir) and app.compiled_select == ["teacache_probe"]
    assert app.loaded[0] == str(compiled_dir)

    QwenImageOrchestrator(plain)._stage_generate(plain)
    app = FakeGen.instances[-1]
    assert app.kwargs["teacache_fused"] is False
    assert app.probe_checked is None and app.compiled is None and app.loaded is not None
