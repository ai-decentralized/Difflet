"""Public CLI parsing and dispatch must preserve or refuse LTX-2 generation flags."""

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from difflet.cli.dp.requests_io import load_requests_jsonl, write_manifest


@pytest.fixture(autouse=True)
def cuda_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")


def _orchestrator(*flags, model="Lightricks/LTX-2", command="generate"):
    cli = importlib.import_module("difflet.cli.main")
    args = cli._build_parser().parse_args([command, "--model-id", model, *flags])
    return cli._get_orchestrator(args)


@pytest.fixture
def generation(monkeypatch):
    calls = []

    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return torch.zeros(1, 3, 9, 32, 32)

    def pipeline(*args, **kwargs):
        return SimpleNamespace(app=SimpleNamespace(generate=generate), compiled_path="compiled")

    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.from_pretrained", pipeline
    )
    return calls


def test_single_prompt_negative_prompt_reaches_generation(tmp_path, generation):
    orch = _orchestrator(
        "--prompt",
        "a cat",
        "--output",
        str(tmp_path / "single.pt"),
        "--negative-prompt",
        "blurry",
    )
    orch.generate()
    assert generation[0][1]["negative_prompt"] == "blurry"


def test_batch_negative_prompt_uses_global_fallback_and_per_line_overrides(tmp_path, generation):
    path = tmp_path / "requests.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                dict(prompt="fallback", output=str(tmp_path / "fallback.pt")),
                dict(
                    prompt="override",
                    output=str(tmp_path / "override.pt"),
                    negative_prompt="distorted",
                ),
                dict(prompt="empty", output=str(tmp_path / "empty.pt"), negative_prompt=""),
            ]
        )
    )
    queue = tmp_path / "queue"
    write_manifest(load_requests_jsonl(path), queue)
    orch = _orchestrator(
        "--requests",
        str(path),
        "--requests-dir",
        str(queue),
        "--worker-index",
        "0",
        "--negative-prompt",
        "blurry",
        "--dp-schedule",
        "least_loaded",
    )
    orch.generate(argv=[])
    assert [(prompt, kwargs["negative_prompt"]) for prompt, kwargs in generation] == [
        ("fallback", "blurry"),
        ("override", "distorted"),
        ("empty", ""),
    ]


UNSUPPORTED = [
    ("--guidance-scale-2", "7"),
    ("--true-cfg-scale", "7"),
    ("--dual-expert",),
    ("--device-vae",),
    ("--vae-in-process",),
    ("--vae-dtype", "fp32"),
    ("--vae-exec-mode", "compile"),
    ("--host-text-encoder",),
    ("--text-encoder-dtype", "fp32"),
    ("--device-text-encoder",),
    ("--exec-mode", "compile"),
    ("--tp-reduce-dtype", "fp32"),
]


@pytest.mark.parametrize("flags", UNSUPPORTED, ids=lambda flags: flags[0])
def test_unsupported_ltx2_flags_refuse_before_snapshot_resolution(monkeypatch, flags):
    downloads = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda *args, **kwargs: downloads.append((args, kwargs)) or "/snapshot",
    )
    orch = _orchestrator(*flags, command="run")
    with pytest.raises(SystemExit, match=flags[0]):
        orch.download()
    assert downloads == []


@pytest.mark.parametrize("command", ["generate", "run"])
@pytest.mark.parametrize("flags", UNSUPPORTED, ids=lambda flags: flags[0])
def test_batch_dispatch_refuses_flags_before_request_or_model_io(monkeypatch, command, flags):
    cli = importlib.import_module("difflet.cli.main")

    def unexpected_io(*args, **kwargs):
        pytest.fail("invalid GPU batch reached request or checkpoint I/O")

    monkeypatch.setattr("difflet.cli.dp.requests_io.load_requests_jsonl", unexpected_io)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", unexpected_io)
    with pytest.raises(SystemExit, match=flags[0]):
        cli.main(
            [command, "--model-id", "Lightricks/LTX-2", "--requests", "requests.jsonl", *flags]
        )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--height", "0"),
        ("--height", "33"),
        ("--width", "31"),
        ("--num-frames", "8"),
        ("--steps", "0"),
        ("--tp-degree", "2"),
    ],
)
def test_invalid_ltx2_request_refuses_before_download(monkeypatch, flag, value):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid LTX-2 request reached a checkpoint download")

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", unexpected)
    orch = _orchestrator(flag, value, command="run")
    with pytest.raises(SystemExit):
        orch.download()


def test_ltx2_download_uses_registered_checkpoint_subset(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *a, **kw: calls.append((a, kw))
    )
    orch = _orchestrator(command="run")
    orch.download()
    from difflet.registry import resolve_model

    assert calls[0][1]["allow_patterns"] == resolve_model("Lightricks/LTX-2").download_patterns
