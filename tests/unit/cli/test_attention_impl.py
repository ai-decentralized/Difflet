"""Attention selection must survive CLI dispatch and produce distinct artifacts."""

import argparse
import importlib

import pytest

from difflet.cli.main import _build_parser
from difflet.cli.stage import _build_stage_parser
from difflet.ops.attention_config import attention_implementation, get_attention_impl
from difflet.pipeline.compile_cache import CacheSpec, cache_key, has_valid_manifest, write_manifest
from difflet.pipeline.parallel_config import DiffletParallelConfig

MODEL = "black-forest-labs/FLUX.1-dev"


def _args(model=MODEL, impl="sdpa"):
    return _build_parser().parse_args([
        "compile", "--model-id", model, "--attention-impl", impl,
    ])


def test_cli_selection_reaches_compile_and_restores_environment(monkeypatch):
    cli = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
    monkeypatch.delenv("DIFFLET_ATTENTION_IMPL", raising=False)
    monkeypatch.setattr(cli, "_validate_capacity", lambda args: None)
    seen = []
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: argparse.Namespace(
        compile=lambda: seen.append(get_attention_impl())))
    cli.main(["compile", "--model-id", MODEL, "--attention-impl", "sdpa"])
    cli.main(["compile", "--model-id", MODEL])
    assert seen == ["sdpa", "megakernel"]
    assert get_attention_impl() == "megakernel"


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_cli_accepts_both_implementations(command):
    for impl in ["megakernel", "sdpa"]:
        args = _build_parser().parse_args([command, "--model-id", MODEL,
                                          "--attention-impl", impl])
        assert args.attention_impl == impl


def test_sdpa_rejects_ring_before_dispatch(monkeypatch):
    cli = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
    monkeypatch.setattr(cli, "_get_orchestrator", lambda args: pytest.fail("must not dispatch"))
    with pytest.raises(SystemExit, match="ring"):
        cli.main(["compile", "--model-id", MODEL, "--cp-mode", "ring", "--cp-degree", "2",
                  "--attention-impl", "sdpa"])


def test_sdpa_rejects_other_backends(monkeypatch):
    cli = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    with pytest.raises(SystemExit, match="Trainium"):
        cli.main(["compile", "--model-id", MODEL, "--attention-impl", "sdpa"])


@pytest.mark.parametrize("name,cls,model,stage", [
    ("wan", "WanOrchestrator", "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "transformer"),
    ("hunyuan_video", "HunyuanVideoOrchestrator", "hunyuanvideo-community/HunyuanVideo", "generate"),
    ("qwen_image", "QwenImageOrchestrator", "Qwen/Qwen-Image", "generate"),
])
def test_stages_forward_selection_and_separate_artifacts(name, cls, model, stage):
    orchestrator = getattr(importlib.import_module(f"difflet.cli.orchestrators.{name}"), cls)
    paths = []
    for impl in ["megakernel", "sdpa"]:
        args = _args(model, impl)
        orch = orchestrator(args)
        forwarded, unknown = _build_stage_parser().parse_known_args([
            "--orchestrator", model, "--stage", stage, *orch._shared_cli_args("compile"),
        ])
        assert not unknown
        assert forwarded.attention_impl == impl
        assert orch._stage_cache_inputs(stage, args) == orch._stage_cache_inputs(stage, forwarded)
        paths.append(orch._stage_compiled_dir(stage, args))
    assert paths[0] != paths[1]


def test_stage_dispatch_uses_selection(monkeypatch):
    stage = importlib.import_module("difflet.cli.stage")
    seen = []
    monkeypatch.setattr(stage, "_load_orchestrator_class", lambda name: lambda args:
                        argparse.Namespace(_run_stage_internal=lambda *args:
                                           seen.append(get_attention_impl())))
    stage.main(["--orchestrator", MODEL, "--stage", "transformer", "--stage-mode", "compile",
                "--attention-impl", "sdpa"])
    assert seen == ["sdpa"]


def test_dp_workers_forward_selection():
    from difflet.cli.dp.router import worker_cli_args

    args = _args()
    args.steps, args.guidance_scale, args.seed = None, None, 42
    forwarded = worker_cli_args(args)
    assert forwarded[forwarded.index("--attention-impl") + 1] == "sdpa"


def test_cache_snapshot_separates_neffs_and_keeps_default_identity(tmp_path):
    def spec():
        return CacheSpec(model_id=MODEL, model_path="/weights", model_name="flux",
                         parallel=DiffletParallelConfig(), dtype="bf16")

    with attention_implementation("megakernel"):
        optimized = spec()
    with attention_implementation("sdpa"):
        sdpa = spec()
    assert "attention_impl" not in optimized.cache_inputs()
    assert sdpa.cache_inputs()["attention_impl"] == "sdpa"
    assert cache_key(optimized) != cache_key(sdpa)
    write_manifest(tmp_path, optimized)
    assert not has_valid_manifest(tmp_path, sdpa)
    assert sdpa.manifest_metadata()["attention_impl"] == "sdpa"
