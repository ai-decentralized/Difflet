"""main() picks the CUDA orchestrator by backend name and routes DP/batch on CUDA (F2)."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import textwrap

import pytest


def _cli():
    return importlib.import_module("difflet.cli.main")


def test_cuda_backend_routes_qwen_to_the_cuda_orchestrator(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id="Qwen/Qwen-Image"))
    assert type(orch).__name__ == "CudaQwenImageOrchestrator"


def test_cpu_and_trainium_keep_the_staged_orchestrator(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id="Qwen/Qwen-Image"))
    assert type(orch).__name__ == "QwenImageOrchestrator"


@pytest.mark.parametrize(
    "model_id", ["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"]
)
def test_cuda_backend_routes_flux_to_the_flux_orchestrator(monkeypatch, model_id):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id=model_id))
    assert type(orch).__name__ == "FluxCudaOrchestrator"


@pytest.mark.parametrize(
    "model_id", ["Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan-AI/Wan2.1-T2V-14B-Diffusers"]
)
def test_cuda_backend_routes_wan_to_the_wan_orchestrator(monkeypatch, model_id):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id=model_id))
    assert type(orch).__name__ == "WanCudaOrchestrator" and orch.model_id == model_id


def test_other_models_on_cuda_point_at_the_trainium_path(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    with pytest.raises(SystemExit) as exc:
        _cli()._get_orchestrator(argparse.Namespace(model_id="hunyuanvideo-community/HunyuanVideo"))
    message = str(exc.value)
    assert (
        "DIFFLET_BACKEND=trainium" in message and "hunyuanvideo-community/HunyuanVideo" in message
    )
    # F1 (Wan) has landed: the fallback no longer names it.
    assert "F1" not in message and "Wan" not in message


def test_requests_and_dp_dispatch_to_the_router_on_cuda(monkeypatch, tmp_path):
    # F2 task 4: the Release 1 refusal is gone; batch and dp>1 go through the dp router
    # (download + compile once in the parent, then one single-GPU worker per replica).
    import torch

    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    cli = _cli()
    parent = []

    class _Orch:
        def download(self):
            parent.append("download")

        def compile(self):
            parent.append("compile")

        def generate(self):
            pytest.fail("the parent never generates in batch/dp mode")

    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: _Orch())
    routed = []

    def fake_router(args, requests, *, replica_cores, worker_argv_prefix=None):
        routed.append((args.dp, replica_cores, [(r.prompt, r.output) for r in requests]))
        return 0

    monkeypatch.setattr("difflet.cli.dp.router.run_router", fake_router)
    req = tmp_path / "r.jsonl"
    req.write_text('{"prompt": "a", "output": "a.png"}\n{"prompt": "b", "output": "b.png"}\n')
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--model-id", "Qwen/Qwen-Image", "--requests", str(req)])
    assert exc.value.code == 0
    assert parent == ["download", "compile"]
    assert routed == [(None, 1, [("a", "a.png"), ("b", "b.png")])]
    parent.clear()
    routed.clear()
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "run",
                "--model-id",
                "Qwen/Qwen-Image",
                "--prompt",
                "a",
                "--output",
                "a.png",
                "--dp",
                "2",
            ]
        )
    assert exc.value.code == 0
    assert parent == ["download", "compile"] and routed == [(2, 1, [("a", "a.png")])]


def test_dp_beyond_the_gpu_count_is_refused_before_any_work(monkeypatch, capsys):
    import torch

    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: pytest.fail("must not reach"))
    monkeypatch.setattr(
        "difflet.cli.dp.router.run_router", lambda *a, **k: pytest.fail("must not route")
    )
    argv = ["run", "--model-id", "Qwen/Qwen-Image", "--prompt", "a", "--output", "a.png"]
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--dp", "3"])
    assert exc.value.code == 1 and "needs 3 GPUs" in capsys.readouterr().err


def test_cli_and_serving_import_without_neuron():
    # F3's import clause: no module-level Neuron import on the CLI/serving path.
    code = textwrap.dedent("""
        import importlib.abc, sys
        BLOCKED = {"neuronx_distributed", "torch_neuronx", "neuronxcc", "nki", "torch_xla", "libneuronxla"}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        for m in ("difflet.cli.main", "difflet.cli.orchestrators.qwen_image",
                  "difflet.cli.orchestrators.qwen_image_cuda", "difflet.cli.orchestrators.flux_cuda",
                  "difflet.cli.orchestrators.wan_cuda", "difflet.cli.orchestrators.wan",
                  "difflet.cli.runner",
                  "difflet.cli.stage", "difflet.serving.orchestrators.qwen_image",
                  "difflet.serving.models._common"):
            __import__(m)
        print("OK")
        """)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]


def test_torch_compile_is_refused_off_cuda(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    with pytest.raises(SystemExit) as exc:
        _cli()._get_orchestrator(argparse.Namespace(model_id="Qwen/Qwen-Image", torch_compile=True))
    assert "CUDA" in str(exc.value)
