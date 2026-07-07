from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest

from difflet.cli.orchestrators import hunyuan_video as hv_mod
from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator


def _hv_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="hunyuanvideo-community/HunyuanVideo", tp_degree=4, cp_degree=1,
        cp_mode="gather_kv", height=320, width=512, num_frames=61,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/hv.mp4",
        steps=4, guidance_scale=6.0, seed=42, stage_mode="generate",
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


class _Recorder:
    """Generic fake application: records compile/load and returns canned output."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.compiled = None
        self.loaded = None

    def compile(self, path):
        self.compiled = path

    def load(self, path, **kw):
        self.loaded = (path, kw)


class _FakeTokenizer:
    def __init__(self, n):
        import torch
        self._n = n

    @classmethod
    def make(cls, n):
        return lambda *a, **kw: cls(n)

    def __call__(self, *a, **kw):
        import torch
        return types.SimpleNamespace(
            input_ids=torch.ones(1, self._n, dtype=torch.int64),
            attention_mask=torch.ones(1, self._n, dtype=torch.int64),
        )


# ----------------------------------------------------------------- download

def test_download_resolves_remote(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, local_files_only: calls.append(local_files_only),
    )
    HunyuanVideoOrchestrator(_hv_args()).download()
    assert calls == [False]


# ----------------------------------------------------------------- generate orchestration

def test_generate_preserves_work_dir_on_failure(monkeypatch, tmp_path, capsys):
    def boom(*a, **kw):
        raise RuntimeError("stage failed")

    monkeypatch.setattr(hv_mod.runner, "run_stage", boom)
    with pytest.raises(RuntimeError):
        HunyuanVideoOrchestrator(_hv_args(work_dir=str(tmp_path))).generate()
    assert "work dir preserved" in capsys.readouterr().err
    assert tmp_path.exists()


def test_generate_removes_work_dir_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr(hv_mod.runner, "run_stage", lambda *a, **kw: None)
    wd = tmp_path / "work"
    HunyuanVideoOrchestrator(_hv_args(work_dir=str(wd))).generate()
    assert not wd.exists()


def test_compile_runs_all_stages(monkeypatch):
    stages = []
    monkeypatch.setattr(
        hv_mod.runner, "run_stage",
        lambda orch, stage, **kw: stages.append(stage),
    )
    HunyuanVideoOrchestrator(_hv_args()).compile()
    assert stages == ["clip", "llama", "generate"]


# ----------------------------------------------------------------- dispatch

def test_run_stage_internal_dispatch(monkeypatch):
    orch = HunyuanVideoOrchestrator(_hv_args())
    seen = []
    monkeypatch.setattr(orch, "_stage_clip", lambda a: seen.append("clip"))
    monkeypatch.setattr(orch, "_stage_llama", lambda a: seen.append("llama"))
    monkeypatch.setattr(orch, "_stage_generate", lambda a: seen.append("generate"))
    for s in ("clip", "llama", "generate"):
        orch._run_stage_internal(s, orch.args)
    assert seen == ["clip", "llama", "generate"]


def test_run_stage_internal_unknown_raises():
    with pytest.raises(ValueError):
        HunyuanVideoOrchestrator(_hv_args())._run_stage_internal("x", _hv_args())


# ----------------------------------------------------------------- helpers

def test_stage_compiled_dir_names():
    orch = HunyuanVideoOrchestrator(_hv_args(cache_dir="/c", tp_degree=4, cp_degree=2))
    assert orch._stage_compiled_dir("clip", orch.args) == Path("/c/hunyuan_video_clip")
    assert orch._stage_compiled_dir("llama", orch.args) == Path("/c/hunyuan_video_llama_seq351")
    assert orch._stage_compiled_dir("generate", orch.args) == \
        Path("/c/hunyuan_video_dit_tp4cp2_h320w512f61")


def test_stage_compiled_dir_unknown_raises():
    orch = HunyuanVideoOrchestrator(_hv_args())
    with pytest.raises(ValueError):
        orch._stage_compiled_dir("nope", orch.args)


def test_shared_cli_args_optionals():
    orch = HunyuanVideoOrchestrator(_hv_args(cache_dir="/c"))
    parts = orch._shared_cli_args("generate", work_dir="/w")
    assert parts[parts.index("--prompt") + 1] == "a cat"
    assert parts[parts.index("--output") + 1] == "/tmp/hv.mp4"
    assert parts[parts.index("--cache-dir") + 1] == "/c"
    assert parts[parts.index("--work-dir") + 1] == "/w"

    bare = HunyuanVideoOrchestrator(
        _hv_args(prompt=None, output=None, cache_dir=None)
    )._shared_cli_args("compile")
    assert "--prompt" not in bare and "--cache-dir" not in bare


def test_save_video_passes_float_unit_range_frames(monkeypatch, tmp_path):
    # export_to_video multiplies ndarray frames by 255 itself — passing uint8
    # wraps pixels to 256-v (color inversion). Contract: float32 [0, 1].
    import numpy as np
    import torch

    exported = {}
    monkeypatch.setattr("diffusers.utils.export_to_video",
                        lambda frames, path, fps: exported.update(frames=frames, fps=fps))
    ok = hv_mod._save_video(torch.zeros(1, 3, 2, 4, 4), str(tmp_path / "v.mp4"))
    assert ok is True and exported["fps"] == 24
    f0 = exported["frames"][0]
    assert f0.shape == (4, 4, 3) and f0.dtype == np.float32
    assert abs(float(f0[0, 0, 0]) - 0.5) < 1e-6  # [-1,1] zeros -> 0.5


def test_save_video_success_and_failure(monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr("diffusers.utils.export_to_video", lambda *a, **kw: None)
    assert hv_mod._save_video(torch.zeros(1, 3, 2, 4, 4), str(tmp_path / "v.mp4")) is True

    def boom(*a, **kw):
        raise RuntimeError("x")

    monkeypatch.setattr("diffusers.utils.export_to_video", boom)
    assert hv_mod._save_video(torch.zeros(1, 3, 2, 4, 4), str(tmp_path / "v.mp4")) is False


# ----------------------------------------------------------------- stage: clip

def _common_setup(monkeypatch):
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: "/fake/model",
    )


def test_stage_clip_compile(monkeypatch, tmp_path):
    _common_setup(monkeypatch)
    _inject(monkeypatch, "difflet.backends.trainium.core.config",
            NeuronConfig=lambda **kw: object())
    _inject(monkeypatch, "difflet.models.flux.clip.modeling_clip",
            CLIPInferenceConfig=lambda **kw: types.SimpleNamespace(),
            NeuronClipApplication=_Recorder)
    _inject(monkeypatch, "difflet.utils.diffusers_adapter",
            load_diffusers_config=lambda p: {})
    _inject(monkeypatch, "transformers", CLIPTokenizer=object)

    args = _hv_args(stage_mode="compile", cache_dir=str(tmp_path))
    # capture the constructed app
    created = []
    orig = _Recorder.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(_Recorder, "__init__", spy)
    HunyuanVideoOrchestrator(args)._stage_clip(args)
    assert created[-1].compiled is not None


def test_stage_clip_generate_saves_clip_pt(monkeypatch, tmp_path):
    import torch
    _common_setup(monkeypatch)

    class FakeClipApp(_Recorder):
        def __call__(self, ids):
            return types.SimpleNamespace(pooler_output=torch.zeros(1, 768))

    _inject(monkeypatch, "difflet.backends.trainium.core.config",
            NeuronConfig=lambda **kw: object())
    _inject(monkeypatch, "difflet.models.flux.clip.modeling_clip",
            CLIPInferenceConfig=lambda **kw: types.SimpleNamespace(),
            NeuronClipApplication=FakeClipApp)
    _inject(monkeypatch, "difflet.utils.diffusers_adapter",
            load_diffusers_config=lambda p: {})
    _inject(monkeypatch, "transformers",
            CLIPTokenizer=types.SimpleNamespace(from_pretrained=_FakeTokenizer.make(77)))

    args = _hv_args(stage_mode="generate", work_dir=str(tmp_path),
                    cache_dir=str(tmp_path))
    HunyuanVideoOrchestrator(args)._stage_clip(args)
    assert (tmp_path / "clip.pt").exists()


# ----------------------------------------------------------------- stage: llama

def _setup_llama(monkeypatch, app_cls):
    _common_setup(monkeypatch)
    _inject(monkeypatch, "neuronx_distributed_inference.models.config",
            NeuronConfig=lambda **kw: object(),
            TensorCaptureConfig=lambda **kw: object())
    _inject(monkeypatch, "neuronx_distributed_inference.models.llama.modeling_llama",
            NeuronLlamaForCausalLM=app_cls)
    _inject(monkeypatch, "neuronx_distributed_inference.utils.hf_adapter",
            load_pretrained_config=lambda **kw: {})
    _inject(monkeypatch, "transformers",
            AutoConfig=types.SimpleNamespace(
                from_pretrained=lambda p: types.SimpleNamespace(
                    pad_token_id=None, tie_word_embeddings=False)),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=_FakeTokenizer.make(351)))


def test_stage_llama_compile(monkeypatch, tmp_path):
    class FakeLlama(_Recorder):
        @classmethod
        def get_config_cls(cls):
            return lambda *a, **kw: object()

    _setup_llama(monkeypatch, FakeLlama)
    args = _hv_args(stage_mode="compile", cache_dir=str(tmp_path))
    created = []
    orig = FakeLlama.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(FakeLlama, "__init__", spy)
    HunyuanVideoOrchestrator(args)._stage_llama(args)
    assert created[-1].compiled is not None


def test_stage_llama_generate_saves_llama_pt(monkeypatch, tmp_path):
    import torch

    class FakeLlama(_Recorder):
        @classmethod
        def get_config_cls(cls):
            return lambda *a, **kw: object()

        def __call__(self, **kw):
            return types.SimpleNamespace(captured_tensors=[torch.zeros(1, 351, 16)])

    _setup_llama(monkeypatch, FakeLlama)
    args = _hv_args(stage_mode="generate", work_dir=str(tmp_path),
                    cache_dir=str(tmp_path))
    HunyuanVideoOrchestrator(args)._stage_llama(args)
    assert (tmp_path / "llama.pt").exists()


# ----------------------------------------------------------------- stage: generate

def _setup_generate(monkeypatch, app_cls):
    _common_setup(monkeypatch)
    _inject(monkeypatch, "difflet.models.hunyuan_video.application",
            HunyuanVideoDiTInputBundle=lambda **kw: types.SimpleNamespace(**kw),
            NeuronHunyuanVideoApplication=app_cls)
    _inject(monkeypatch, "difflet.models.hunyuan_video.pipeline",
            _retrieve_timesteps=lambda sched, steps, dev, sigmas=None: (
                __import__("torch").zeros(steps), None))


def test_stage_generate_compile(monkeypatch, tmp_path):
    _setup_generate(monkeypatch, _Recorder)
    args = _hv_args(stage_mode="compile", cache_dir=str(tmp_path),
                    work_dir=str(tmp_path))
    created = []
    orig = _Recorder.__init__

    def spy(self, *a, **kw):
        orig(self, *a, **kw)
        created.append(self)

    monkeypatch.setattr(_Recorder, "__init__", spy)
    HunyuanVideoOrchestrator(args)._stage_generate(args)
    assert created[-1].compiled is not None


def test_stage_generate_runs_and_saves_pt(monkeypatch, tmp_path):
    import torch

    class FakeGen(_Recorder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.pipeline = types.SimpleNamespace(scheduler=object())

        def __call__(self, **kw):
            return types.SimpleNamespace(frames=torch.zeros(1, 3, 2, 8, 8))

    _setup_generate(monkeypatch, FakeGen)
    # inter-stage inputs
    torch.save({"encoder_hidden_states": torch.zeros(1, 256, 16),
                "encoder_attention_mask": torch.ones(1, 256, dtype=torch.int64)},
               tmp_path / "llama.pt")
    torch.save({"pooled_projections": torch.zeros(1, 768)}, tmp_path / "clip.pt")

    out = tmp_path / "out.png"  # non-mp4 -> .pt branch
    args = _hv_args(stage_mode="generate", work_dir=str(tmp_path),
                    cache_dir=str(tmp_path), output=str(out))
    HunyuanVideoOrchestrator(args)._stage_generate(args)
    assert (tmp_path / "out.pt").exists()


def test_stage_generate_mp4_branch(monkeypatch, tmp_path):
    import torch

    class FakeGen(_Recorder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.pipeline = types.SimpleNamespace(scheduler=object())

        def __call__(self, **kw):
            return types.SimpleNamespace(frames=torch.zeros(1, 3, 2, 8, 8))

    _setup_generate(monkeypatch, FakeGen)
    monkeypatch.setattr(hv_mod, "_save_video", lambda frames, path: True)
    torch.save({"encoder_hidden_states": torch.zeros(1, 256, 16),
                "encoder_attention_mask": torch.ones(1, 256, dtype=torch.int64)},
               tmp_path / "llama.pt")
    torch.save({"pooled_projections": torch.zeros(1, 768)}, tmp_path / "clip.pt")
    out = tmp_path / "out.mp4"
    args = _hv_args(stage_mode="generate", work_dir=str(tmp_path),
                    cache_dir=str(tmp_path), output=str(out))
    HunyuanVideoOrchestrator(args)._stage_generate(args)
    assert not (tmp_path / "out.pt").exists()
