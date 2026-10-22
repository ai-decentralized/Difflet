"""main() picks the GPU orchestrator on a GPU platform and routes DP/batch there."""

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


@pytest.mark.parametrize(
    "model_id,model_type",
    [
        ("Qwen/Qwen-Image", "qwen_image"),
        ("black-forest-labs/FLUX.1-dev", "flux"),
        ("black-forest-labs/FLUX.1-schnell", "flux"),
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan"),
        ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "wan"),
    ],
)
def test_cuda_backend_routes_to_the_gpu_orchestrator(monkeypatch, model_id, model_type):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id=model_id))
    assert type(orch).__name__ == "GpuOrchestrator" and orch.spec.model_type == model_type
    assert orch.model_id == model_id


def test_cpu_and_trainium_keep_the_staged_orchestrator(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    orch = _cli()._get_orchestrator(argparse.Namespace(model_id="Qwen/Qwen-Image"))
    assert type(orch).__name__ == "QwenImageOrchestrator"


def test_other_models_on_cuda_point_at_the_trainium_path(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    with pytest.raises(SystemExit) as exc:
        _cli()._get_orchestrator(argparse.Namespace(model_id="hunyuanvideo-community/HunyuanVideo"))
    message = str(exc.value)
    assert (
        "DIFFLET_BACKEND=trainium" in message and "hunyuanvideo-community/HunyuanVideo" in message
    )
    # Wan runs on the GPU flow: the fallback does not name it.
    assert "Wan" not in message


def test_requests_and_dp_dispatch_to_the_router_on_cuda(monkeypatch, tmp_path):
    # Batch and dp>1 go through the dp router (download + compile once in the parent, then
    # one single-GPU worker per replica).
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
    # No module-level Neuron, Trainium-backend or TPU-backend import on the CLI/serving path.
    code = textwrap.dedent("""
        import importlib.abc, sys
        BLOCKED = {"neuronx_distributed", "torch_neuronx", "neuronxcc", "nki", "torch_xla", "libneuronxla"}
        BLOCKED_PREFIXES = ("difflet.backends.trainium", "difflet.backends.tpu")
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED or name.startswith(BLOCKED_PREFIXES):
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        for m in ("difflet.cli.main", "difflet.cli.orchestrators.qwen_image",
                  "difflet.backends.gpu.cli", "difflet.backends.gpu.serving",
                  "difflet.cli.orchestrators.wan",
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


def test_torch_compile_is_refused_off_gpu(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    with pytest.raises(SystemExit) as exc:
        _cli()._get_orchestrator(argparse.Namespace(model_id="Qwen/Qwen-Image", torch_compile=True))
    assert "GPU-backend option" in str(exc.value) and "cpu" in str(exc.value)
