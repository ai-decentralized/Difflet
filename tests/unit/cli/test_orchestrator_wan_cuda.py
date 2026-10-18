"""Wan 2.2 on CUDA through the CLI: one process, DiffletPipeline + generate() (CUDA plan F1).

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
Mirrors tests/unit/cli/test_orchestrator_flux_cuda.py; the differences are the video output
(``OUTPUT_TYPE = "pt"``, MP4 through the Trainium orchestrator's ``_save_video``, ``.pt``
fallback), ``--num-frames`` and the 480x832x9 default shape.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from difflet.cli.orchestrators import wan as trainium_wan
from difflet.cli.orchestrators import wan_cuda as mod
from difflet.cli.orchestrators.qwen_image_cuda import CudaOneProcessOrchestrator

WAN22 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN21 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def _args(**over):
    d = dict(
        model_id=WAN22,
        tp_degree=None,
        cp_degree=None,
        cp_mode="gather_kv",
        cfg_parallel=False,
        sp_enabled=False,
        dp=None,
        height=None,
        width=None,
        num_frames=None,
        shapes=None,
        cache_dir=None,
        force=False,
        revision=None,
        prompt="a cat",
        output="/tmp/x/wan.mp4",
        steps=None,
        guidance_scale=None,
        seed=42,
        requests=None,
        requests_dir=None,
        teacache_cadence=None,
        teacache_online_delta=None,
        teacache_speedup=None,
        teacache_calibration=None,
        host_vae=False,
        taef1=False,
        taef1_path=None,
        torch_compile=False,
    )
    d.update(over)
    return argparse.Namespace(**d)


def _video(frames: int = 9) -> torch.Tensor:
    """The application's ``output_type="pt"`` result: (B, 3, F, H, W) in [-1, 1]."""
    torch.manual_seed(0)
    return torch.rand(1, 3, frames, 16, 16) * 2 - 1


class _App:
    def __init__(self):
        self.calls = []
        self.video = _video()

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return self.video


class _Pipe:
    def __init__(self):
        self.app = _App()
        self.compiled_path = "/fake/compiled"


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw))
        return _Pipe()

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained", staticmethod(fake)
    )
    return calls


@pytest.fixture
def saved(monkeypatch):
    """Record _save_video calls; write a marker file so the output exists."""
    calls = []

    def fake(tensor, output_path):
        calls.append((tensor, output_path))
        Path(output_path).write_bytes(b"mp4")
        return True

    monkeypatch.setattr(mod, "_save_video", fake)
    return calls


def test_wan_orchestrator_is_the_shared_one_process_orchestrator():
    assert issubclass(mod.WanCudaOrchestrator, CudaOneProcessOrchestrator)
    assert mod.WanCudaOrchestrator.HF_MODEL_ID == WAN22
    assert set(mod.WanCudaOrchestrator.MODEL_IDS) == {WAN22, WAN21}
    assert mod.WanCudaOrchestrator.DEFAULT_STEPS == 40
    assert mod.WanCudaOrchestrator.ACCEPTS_GUIDANCE_SCALE is True
    assert mod.WanCudaOrchestrator.DEFAULT_GUIDANCE_SCALE == 4.0
    assert mod.WanCudaOrchestrator.OUTPUT_TYPE == "pt"
    # The image models keep PIL output.
    assert CudaOneProcessOrchestrator.OUTPUT_TYPE == "pil"


def test_the_mp4_writer_is_the_trainium_orchestrator_s_helper():
    # Reused by import, not copied.
    assert mod._save_video is trainium_wan._save_video


def test_compile_builds_the_cuda_pipeline_at_480x832x9_without_loading(recorder):
    mod.WanCudaOrchestrator(_args()).compile()
    model_id, kw = recorder[0]
    assert model_id == WAN22
    assert kw["backend"] == "cuda" and kw["load"] is False
    assert kw["parallel"].tp_degree == 1 and kw["dtype"] == "bf16"
    assert kw["local_files_only"] is True
    assert (kw["height"], kw["width"], kw["num_frames"]) == (480, 832, 9)
    assert "application_kwargs" not in kw


def test_height_width_num_frames_cache_dir_and_force_reach_from_pretrained(recorder):
    args = _args(cache_dir="/c", force=True, height=320, width=512, num_frames=17)
    mod.WanCudaOrchestrator(args).compile()
    _, kw = recorder[0]
    assert kw["compile_cache_dir"] == "/c" and kw["force_compile"] is True
    assert (kw["height"], kw["width"], kw["num_frames"]) == (320, 512, 17)


def test_generate_writes_the_mp4_through_save_video(recorder, saved, tmp_path):
    out = tmp_path / "deep" / "wan.mp4"
    orch = mod.WanCudaOrchestrator(_args(output=str(out)))
    orch.generate()
    _, kw = recorder[0]
    assert kw["load"] is True and kw["force_compile"] is False
    # The whole (B, 3, F, H, W) tensor goes to the helper, which takes frames[0] itself.
    ((tensor, path),) = saved
    assert tensor is orch._last_pipe.app.video and path == str(out)
    assert out.is_file() and not out.with_suffix(".pt").exists()


def test_a_failed_mp4_export_falls_back_to_a_pt_tensor(recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_save_video", lambda tensor, path: False)
    out = tmp_path / "wan.mp4"
    orch = mod.WanCudaOrchestrator(_args(output=str(out)))
    orch.generate()
    pt = out.with_suffix(".pt")
    assert pt.is_file() and not out.exists()
    assert torch.equal(torch.load(pt), orch._last_pipe.app.video)


def test_a_non_mp4_output_is_saved_as_the_tensor(recorder, saved, tmp_path):
    out = tmp_path / "wan.pt"
    orch = mod.WanCudaOrchestrator(_args(output=str(out)))
    orch.generate()
    assert saved == [] and torch.equal(torch.load(out), orch._last_pipe.app.video)


def test_generate_writes_a_real_mp4(recorder, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "wan.mp4"
    mod.WanCudaOrchestrator(_args(output=str(out))).generate()
    assert out.is_file() and out.stat().st_size > 0 and not out.with_suffix(".pt").exists()


def test_steps_default_to_40_guidance_to_4_and_frames_to_9(recorder, saved, tmp_path):
    orch = mod.WanCudaOrchestrator(_args(output=str(tmp_path / "w.mp4")))
    orch.generate()
    prompt, kw = orch._last_pipe.app.calls[0]
    assert prompt == "a cat" and kw["seed"] == 42 and kw["output_type"] == "pt"
    assert kw["num_inference_steps"] == mod.DEFAULT_STEPS == 40
    assert kw["guidance_scale"] == mod.DEFAULT_GUIDANCE_SCALE == 4.0
    assert kw["num_frames"] == 9
    # No --negative-prompt flag: the application's default (None → "") applies.
    assert "negative_prompt" not in kw


def test_steps_seed_guidance_and_frames_are_forwarded(recorder, saved, tmp_path):
    orch = mod.WanCudaOrchestrator(
        _args(output=str(tmp_path / "w.mp4"), steps=7, seed=3, guidance_scale=5.0, num_frames=17)
    )
    orch.generate()
    _, kw = orch._last_pipe.app.calls[0]
    assert kw["num_inference_steps"] == 7 and kw["seed"] == 3 and kw["guidance_scale"] == 5.0
    assert kw["num_frames"] == 17
    assert recorder[0][1]["num_frames"] == 17


def test_wan21_is_built_from_its_own_repo(recorder, monkeypatch):
    args = _args(model_id=WAN21)
    mod.WanCudaOrchestrator(args).compile()
    assert recorder[0][0] == WAN21
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid) or "/snap",
    )
    mod.WanCudaOrchestrator(args).download()
    assert seen["mid"] == WAN21


@pytest.mark.parametrize(
    "over,follow_up",
    [
        ({"tp_degree": 2}, "F6"),
        ({"sp_enabled": True}, "F6"),
        ({"cp_degree": 2}, "F2"),
        ({"cfg_parallel": True}, "F2"),
        ({"shapes": "480x832x9,480x832x17"}, "F1b"),
        ({"teacache_cadence": 2}, "F1b"),
        ({"teacache_speedup": 1.5}, "F1b"),
        ({"teacache_online_delta": 0.1}, "F1b"),
        ({"teacache_calibration": "/x.json"}, "F1b"),
        ({"taef1": True}, "F1b"),
        ({"taef1_path": "madebyollin/taef1"}, "F1b"),
        ({"host_vae": True}, "F1b"),
        ({"torch_compile": True}, "F1b"),
    ],
)
def test_unsupported_flags_are_refused_before_any_load(recorder, over, follow_up):
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(**over)).generate()
    assert follow_up in str(exc.value) and "[wan/cuda]" in str(exc.value) and recorder == []


@pytest.mark.parametrize(
    "over,flag",
    [
        ({"shapes": "480x832x9"}, "--shapes"),
        ({"teacache_cadence": 2}, "--teacache-cadence"),
        ({"teacache_calibration": "/x.json"}, "--teacache-calibration"),
        ({"taef1_path": "madebyollin/taef1"}, "--taef1"),
        ({"host_vae": True}, "--host-vae"),
        ({"torch_compile": True}, "--torch-compile"),
    ],
)
def test_refusals_name_the_flag(recorder, over, flag):
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(**over)).compile()
    assert flag in str(exc.value) and recorder == []


def test_download_resolves_the_hub_snapshot(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda mid, **kw: seen.update(mid=mid, **kw) or "/snap",
    )
    mod.WanCudaOrchestrator(_args()).download()
    assert seen["mid"] == WAN22 and seen["local_files_only"] is False


def test_run_is_download_compile_generate(recorder, monkeypatch, tmp_path):
    order = []
    orch = mod.WanCudaOrchestrator(_args(output=str(tmp_path / "w.mp4")))
    monkeypatch.setattr(orch, "download", lambda: order.append("download"))
    monkeypatch.setattr(orch, "compile", lambda: order.append("compile"))
    monkeypatch.setattr(orch, "generate", lambda: order.append("generate"))
    orch.run()
    assert order == ["download", "compile", "generate"]


def test_run_refuses_unsupported_flags_before_any_download(recorder, monkeypatch):
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *a, **kw: pytest.fail("must refuse before resolving the snapshot"),
    )
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(host_vae=True)).run()
    assert "F1b" in str(exc.value) and recorder == []


def test_lone_teacache_calibration_is_refused_by_name(recorder):
    """``main._validate_teacache`` lets a lone ``--teacache-calibration`` through and Wan on
    CUDA forwards nothing, so without this refusal the calibration file is silently dropped."""
    with pytest.raises(SystemExit) as exc:
        mod.WanCudaOrchestrator(_args(teacache_calibration="/x.json")).generate()
    assert "--teacache-calibration" in str(exc.value) and "F1b" in str(exc.value)
    assert recorder == []


_FORBIDDEN = (
    "neuronx_distributed",
    "torch_neuronx",
    "torch_xla",
    "difflet.backends.trainium",
    "difflet.backends.tpu",
)


def test_wan_cuda_imports_nothing_from_neuron_or_the_sibling_backends():
    tree = ast.parse(Path(mod.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    bad = [m for m in imported if any(m == p or m.startswith(p + ".") for p in _FORBIDDEN)]
    assert bad == []


def test_importing_the_cuda_and_trainium_wan_orchestrators_pulls_no_neuron_module():
    # _save_video lives in the Trainium orchestrator: importing it must stay Neuron-free.
    code = textwrap.dedent("""
        import importlib.abc, sys
        BLOCKED = {"neuronx_distributed", "torch_neuronx", "neuronxcc", "nki", "torch_xla",
                   "libneuronxla"}
        BLOCKED_PREFIXES = ("difflet.backends.trainium", "difflet.backends.tpu")
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED or name.startswith(BLOCKED_PREFIXES):
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        import difflet.cli.orchestrators.wan_cuda
        import difflet.cli.orchestrators.wan
        print("OK")
        """)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
