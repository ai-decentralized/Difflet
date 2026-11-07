"""The CLI's MXFP8 flags (``difflet/cli/quant.py``): parsing on compile/generate/run, the
driver's refusals, the spec and its forwarding to the stage parser and the DP workers, and the
Qwen-Image DiT stage key (additive: a bf16 command line's key is unchanged)."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import pytest

from difflet.cli import quant as Q
from difflet.cli.stage import _build_stage_parser

QWEN = "Qwen/Qwen-Image"
FLUX = "black-forest-labs/FLUX.1-dev"


def _parse(*flags):
    cli_main = importlib.import_module("difflet.cli.main")
    return cli_main._build_parser().parse_args(list(flags))


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_flags_parse_on_every_command(command):
    args = _parse(command, "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--quant-compute", "dequant")
    assert (args.quant, args.quant_compute) == ("mxfp8", "dequant")
    args = _parse(command, "--model-id", QWEN)
    assert (args.quant, args.quant_compute) == (None, None)
    with pytest.raises(SystemExit):  # the per-tensor static recipe is gone
        _parse(command, "--model-id", QWEN, "--quant", "fp8")


def _refused(args, match, capsys):
    with pytest.raises(SystemExit):
        Q.validate_quant(args)
    assert match in capsys.readouterr().err


def test_refusals(capsys, monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv("UNSAFE_FP8FNCAST", raising=False)
    base = dict(model_id=QWEN, backend="neuron", quant="mxfp8", quant_compute=None,
                exec_mode=None, teacache_speedup=None)
    Q.validate_quant(argparse.Namespace(**base))  # the happy path
    Q.validate_quant(argparse.Namespace(**{**base, "model_id": FLUX}))
    _refused(argparse.Namespace(**{**base, "quant": None, "quant_compute": "nki"}),
             "--quant-compute needs --quant", capsys)
    _refused(argparse.Namespace(**{**base, "backend": "trainium"}), "neuron backend", capsys)
    _refused(argparse.Namespace(**{**base, "backend": None}), "neuron backend", capsys)
    _refused(argparse.Namespace(**{**base, "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"}),
             "not available for", capsys)
    _refused(argparse.Namespace(**{**base, "exec_mode": "eager"}), "compile", capsys)
    _refused(argparse.Namespace(**{**base, "teacache_speedup": 1.5}), "TeaCache", capsys)
    monkeypatch.setenv("UNSAFE_FP8FNCAST", "1")
    _refused(argparse.Namespace(**base), "UNSAFE_FP8FNCAST", capsys)


def test_the_backend_env_counts_as_the_flag(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    Q.validate_quant(argparse.Namespace(model_id=QWEN, backend=None, quant="mxfp8",
                                        quant_compute=None))


def test_spec_from_args():
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--tp-degree", "2")
    spec = Q.quant_spec(args)
    assert spec.format == "mxfp8" and spec.compute == "nki"
    assert spec.tp_degree == 2 and spec.k_shards("transformer_blocks.0.img_mlp.net.2") == 2
    args.quant_compute = "dequant"
    assert Q.quant_spec(args).compute == "dequant"
    assert Q.quant_spec(_parse("generate", "--model-id", QWEN)) is None
    assert Q.quant_spec(_parse("generate", "--model-id", QWEN, "--quant", "mxfp8")).tp_degree == 4


def test_quant_spec_takes_the_drivers_tp():
    """The row-parallel weights are quantized per rank K range, so the spec's tp is the launch's:
    the args' --tp-degree or 4, unless the driver passes its own default."""
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    assert Q.quant_spec(args).tp_degree == 4 and Q.quant_spec(args, tp_degree=2).tp_degree == 2
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--tp-degree", "8")
    assert Q.quant_spec(args).tp_degree == 8 and Q.quant_spec(args, tp_degree=2).tp_degree == 2


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """The tiny diffusers-layout Qwen-Image DiT's transformer directory (4 heads x 32, a
    512-wide MLP: every target's K per rank is 128 at the driver's default tp4). torch and
    diffusers load here only; the rest of this module is as torch-free as the driver."""
    pytest.importorskip("diffusers")
    from tests.unit.backends._neuron_qwen_toy import write_tiny_qwen_model

    root = tmp_path_factory.mktemp("tiny_qwen_quant")
    write_tiny_qwen_model(root)
    return str(root / "transformer")


@pytest.fixture
def _trace_cache_restored(monkeypatch):
    """Undo the DIFFLET_NKI_TRACE_CACHE the preparer sets with os.environ.setdefault: setenv
    first records the variable's original state (delenv of an unset variable records
    nothing, and the value leaked into later tests' compile-cache layout)."""
    monkeypatch.setenv("DIFFLET_NKI_TRACE_CACHE", "")
    monkeypatch.delenv("DIFFLET_NKI_TRACE_CACHE")


def test_prepare_quant_writes_on_create_and_refuses_when_missing(
        tmp_path, monkeypatch, capsys, tiny_dir, _trace_cache_restored):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    with pytest.raises(SystemExit):  # generate's resolve-only call before any compile wrote it
        Q.prepare_quant(args, {"transformer": tiny_dir}, create=False, tag="t")
    assert "Error: --quant: " in capsys.readouterr().err
    out = Q.prepare_quant(args, {"transformer": tiny_dir}, create=True, tag="t")
    assert (Path(out["transformer"]) / "difflet_quant.json").is_file()
    assert Path(out["transformer"]).is_relative_to(tmp_path)
    assert capsys.readouterr().out.startswith(f"[t] MXFP8 transformer (nki): {out['transformer']}")
    assert os.environ["DIFFLET_NKI_TRACE_CACHE"] == "1"
    # Resolve-only now finds it, and a second component is prepared under its own name.
    both = Q.prepare_quant(args, {"transformer": tiny_dir, "twin": tiny_dir}, create=False,
                           tag="t")
    assert both == {"transformer": out["transformer"], "twin": out["transformer"]}


def test_prepare_quant_is_off_without_the_flag_or_the_weights(tmp_path, capsys,
                                                              _trace_cache_restored):
    """None and no line when --quant is off, or when the weights are not downloaded yet (the
    ranks report that); the trace-cache choice is still made for the ranks in the second case,
    and dequant compute leaves it alone."""
    missing = {"transformer": str(tmp_path / "not-downloaded" / "transformer")}
    assert Q.prepare_quant(_parse("compile", "--model-id", QWEN), missing, create=True,
                           tag="t") is None
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--quant-compute", "dequant")
    assert Q.prepare_quant(args, missing, create=True, tag="t") is None
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    assert Q.prepare_quant(args, missing, create=True, tag="t") is None
    assert Q.prepare_quant(args, {}, create=True, tag="t") is None
    assert os.environ["DIFFLET_NKI_TRACE_CACHE"] == "1"
    assert capsys.readouterr() == ("", "")


def test_flags_are_forwarded_only_when_given_and_consumed_by_the_stage_parser():
    assert Q.quant_cli_args(argparse.Namespace()) == []
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--quant-compute", "dequant")
    forwarded = Q.quant_cli_args(args)
    assert forwarded == ["--quant", "mxfp8", "--quant-compute", "dequant"]
    parsed, extra = _build_stage_parser().parse_known_args(
        ["--orchestrator", QWEN, "--stage", "generate", *forwarded])
    assert extra == [] and parsed.quant == "mxfp8" and parsed.quant_compute == "dequant"
    assert Q.quant_cli_args(_parse("generate", "--model-id", QWEN, "--quant", "mxfp8")) == \
        ["--quant", "mxfp8"]

    from difflet.cli.dp.router import worker_cli_args

    args.dp = 2
    worker = worker_cli_args(args)
    assert "--quant" in worker and "--quant-compute" in worker


def test_qwen_orchestrator_forwards_and_keys_quant(monkeypatch):
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")

    def key(*flags):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", QWEN, "--cache-dir", "/c", "--tp-degree", "2", *flags])
        orch = cli_main._get_orchestrator(args)
        return (orch._stage_cache_inputs("generate", args),
                orch._stage_compiled_dir("generate", args),
                orch._shared_cli_args("generate", work_dir="/w"))

    plain, plain_dir, plain_cli = key()
    assert not any(k.startswith("quant") for k in plain)
    assert "--quant" not in plain_cli
    mx, mx_dir, mx_cli = key("--quant", "mxfp8")
    added = sorted(k for k in mx if k not in plain)
    assert added == ["quant", "quant_compute", "quant_schema", "quant_targets_sha256",
                     "quant_tp_degree"]
    assert mx["quant"] == "mxfp8" and mx["quant_tp_degree"] == 2
    assert {k: v for k, v in mx.items() if k not in added} == plain
    assert mx_dir != plain_dir and mx_dir.parent == plain_dir.parent
    assert mx_cli[mx_cli.index("--quant") + 1] == "mxfp8"
    dequant, dequant_dir, _ = key("--quant", "mxfp8", "--quant-compute", "dequant")
    assert sorted(k for k in mx if mx[k] != dequant.get(k)) == ["quant_compute"]
    assert dequant_dir != mx_dir


def test_application_kwargs_carry_the_spec(monkeypatch):
    from difflet.cli.orchestrators.qwen_image import _torchrun_application_kwargs

    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    kwargs = _torchrun_application_kwargs(args)
    assert kwargs["quant"]["format"] == "mxfp8" and kwargs["quant"]["compute"] == "nki"
    assert kwargs["quant"]["tp_degree"] == 4 and "quant_calibration" not in kwargs
    assert "quant" not in _torchrun_application_kwargs(_parse("generate", "--model-id", QWEN))


def test_the_kernel_path_turns_on_the_nki_trace_cache(monkeypatch, tmp_path):
    """--quant mxfp8 (kernel compute) sets DIFFLET_NKI_TRACE_CACHE=1 for the ranks before any
    launch unless the user chose; dequant compute and no --quant leave it alone."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")
    # setenv first records the variable's original state, so the undo restores it even
    # though the orchestrator sets it with os.environ.setdefault (delenv of an unset variable
    # records nothing, and the value leaked into later tests' compile-cache layout).
    monkeypatch.setenv("DIFFLET_NKI_TRACE_CACHE", "")
    monkeypatch.delenv("DIFFLET_NKI_TRACE_CACHE")

    def prepare(*flags):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", QWEN, "--cache-dir", str(tmp_path), "--tp-degree", "4",
             *flags])
        orch = cli_main._get_orchestrator(args)
        # the model is not downloaded: _prepare_quant returns after the environment choice
        orch._model_dir = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
        orch._prepare_quant(create=False)

    prepare()
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    prepare("--quant", "mxfp8", "--quant-compute", "dequant")
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    prepare("--quant", "mxfp8")
    assert os.environ["DIFFLET_NKI_TRACE_CACHE"] == "1"
    monkeypatch.setenv("DIFFLET_NKI_TRACE_CACHE", "0")
    prepare("--quant", "mxfp8")
    assert os.environ["DIFFLET_NKI_TRACE_CACHE"] == "0"


def test_quant_targets_env_replaces_the_layer_set_and_keys_the_spec(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv(Q.QUANT_TARGETS_ENV, raising=False)
    args = _parse("generate", "--model-id", QWEN, "--quant", "mxfp8")
    default = Q.quant_spec(args)
    monkeypatch.setenv(Q.QUANT_TARGETS_ENV,
                       r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.net\.0\.proj;;"
                       r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.net\.2")
    narrowed = Q.quant_spec(args)
    assert len(narrowed.targets) == 2 and narrowed.targets != default.targets
    assert narrowed.matches("transformer_blocks.3.img_mlp.net.2")
    assert not narrowed.matches("transformer_blocks.3.attn.to_q")
    assert default.matches("transformer_blocks.3.attn.to_q")
    assert narrowed.cache_fields()["quant_targets_sha256"] != default.cache_fields()["quant_targets_sha256"]
    assert narrowed.weights_identity() != default.weights_identity()
    monkeypatch.setenv(Q.QUANT_TARGETS_ENV, "")
    assert Q.quant_spec(args).targets == default.targets
