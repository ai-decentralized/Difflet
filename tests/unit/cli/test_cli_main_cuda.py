"""main() picks the CUDA orchestrator by backend name and refuses DP/batch on CUDA."""

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


def test_other_models_on_cuda_are_refused_by_name(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    with pytest.raises(SystemExit) as exc:
        _cli()._get_orchestrator(argparse.Namespace(model_id="black-forest-labs/FLUX.1-dev"))
    assert "not supported" in str(exc.value) and "not supported" in str(exc.value)


def test_requests_and_dp_are_refused_on_cuda(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    cli = _cli()
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: pytest.fail("must not reach"))
    req = tmp_path / "r.jsonl"
    req.write_text('{"prompt": "a", "output": "a.png"}\n')
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--model-id", "Qwen/Qwen-Image", "--requests", str(req)])
    assert exc.value.code == 1 and "not supported" in capsys.readouterr().err
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
    assert exc.value.code == 1 and "not supported" in capsys.readouterr().err


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
                  "difflet.cli.orchestrators.qwen_image_cuda", "difflet.cli.runner",
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
