"""The CLI's FP8 flags (``difflet/cli/quant.py``): parsing on compile/generate/run, the driver's
refusals, the spec and its forwarding to the stage parser and the DP workers, and the Qwen-Image
DiT stage key (additive: a bf16 command line's key is unchanged)."""

from __future__ import annotations

import argparse
import importlib
import json

import pytest

from difflet.cli import quant as Q
from difflet.cli.stage import _build_stage_parser

QWEN = "Qwen/Qwen-Image"
FLUX = "black-forest-labs/FLUX.1-dev"


@pytest.fixture
def calibration(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"schema": 1, "amax": {"transformer_blocks.0.attn.to_q": 1.0}}))
    return str(path)


def _parse(*flags):
    cli_main = importlib.import_module("difflet.cli.main")
    return cli_main._build_parser().parse_args(list(flags))


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_flags_parse_on_every_command(command, calibration):
    args = _parse(command, "--model-id", QWEN, "--backend", "neuron", "--quant", "fp8",
                  "--quant-calibration", calibration, "--quant-compute", "dequant")
    assert (args.quant, args.quant_calibration, args.quant_compute) == ("fp8", calibration, "dequant")
    args = _parse(command, "--model-id", QWEN)
    assert (args.quant, args.quant_calibration, args.quant_compute) == (None, None, None)


def _refused(args, match, capsys):
    with pytest.raises(SystemExit):
        Q.validate_quant(args)
    assert match in capsys.readouterr().err


def test_refusals(calibration, capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv("UNSAFE_FP8FNCAST", raising=False)
    base = dict(model_id=QWEN, backend="neuron", quant="fp8", quant_calibration=calibration,
                quant_compute=None, exec_mode=None, teacache_speedup=None)
    Q.validate_quant(argparse.Namespace(**base))  # the happy path
    _refused(argparse.Namespace(**{**base, "quant": None, "quant_calibration": None,
                                   "quant_compute": "nki"}),
             "--quant-compute needs --quant", capsys)
    _refused(argparse.Namespace(**{**base, "quant": None, "quant_calibration": calibration}),
             "--quant-calibration needs --quant", capsys)
    _refused(argparse.Namespace(**{**base, "backend": "trainium"}), "neuron backend", capsys)
    _refused(argparse.Namespace(**{**base, "backend": None}), "neuron backend", capsys)
    _refused(argparse.Namespace(**{**base, "model_id": FLUX}), "not available for", capsys)
    _refused(argparse.Namespace(**{**base, "exec_mode": "eager"}), "compile", capsys)
    _refused(argparse.Namespace(**{**base, "quant_calibration": None}), "--quant-calibration PATH",
             capsys)
    _refused(argparse.Namespace(**{**base, "quant_calibration": str(tmp_path / "no.json")}),
             "no such file", capsys)
    _refused(argparse.Namespace(**{**base, "teacache_speedup": 1.5}), "TeaCache", capsys)
    monkeypatch.setenv("UNSAFE_FP8FNCAST", "1")
    _refused(argparse.Namespace(**base), "UNSAFE_FP8FNCAST", capsys)


def test_the_backend_env_counts_as_the_flag(calibration, monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    Q.validate_quant(argparse.Namespace(model_id=QWEN, backend=None, quant="fp8",
                                        quant_calibration=calibration, quant_compute=None))


def test_spec_from_args(calibration):
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "fp8",
                  "--quant-calibration", calibration)
    spec = Q.quant_spec(args)
    assert spec.format == "fp8" and spec.compute == "nki" and len(spec.calibration_sha256) == 64
    args.quant_compute = "dequant"
    assert Q.quant_spec(args).compute == "dequant"
    assert Q.quant_spec(_parse("generate", "--model-id", QWEN)) is None


def test_flags_are_forwarded_only_when_given_and_consumed_by_the_stage_parser(calibration):
    assert Q.quant_cli_args(argparse.Namespace()) == []
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "fp8",
                  "--quant-calibration", calibration, "--quant-compute", "dequant")
    forwarded = Q.quant_cli_args(args)
    assert forwarded == ["--quant", "fp8", "--quant-calibration", calibration,
                         "--quant-compute", "dequant"]
    parsed, extra = _build_stage_parser().parse_known_args(
        ["--orchestrator", QWEN, "--stage", "generate", *forwarded])
    assert extra == [] and parsed.quant == "fp8" and parsed.quant_compute == "dequant"

    from difflet.cli.dp.router import worker_cli_args

    args.dp = 2
    worker = worker_cli_args(args)
    assert "--quant" in worker and "--quant-calibration" in worker and "--quant-compute" in worker


def test_qwen_orchestrator_forwards_and_keys_quant(calibration, monkeypatch):
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
    fp8, fp8_dir, fp8_cli = key("--quant", "fp8", "--quant-calibration", calibration)
    added = sorted(k for k in fp8 if k not in plain)
    assert added == ["quant", "quant_activation", "quant_activation_margin",
                     "quant_calibration_sha256", "quant_compute", "quant_fp8_max",
                     "quant_schema", "quant_targets_sha256", "quant_weight_granularity"]
    assert {k: v for k, v in fp8.items() if k not in added} == plain
    assert fp8_dir != plain_dir and fp8_dir.parent == plain_dir.parent
    assert fp8_cli[fp8_cli.index("--quant") + 1] == "fp8"
    dequant, dequant_dir, _ = key("--quant", "fp8", "--quant-calibration", calibration,
                                  "--quant-compute", "dequant")
    assert sorted(k for k in fp8 if fp8[k] != dequant.get(k)) == ["quant_compute"]
    assert dequant_dir != fp8_dir


def test_application_kwargs_carry_the_spec(calibration, monkeypatch):
    from difflet.cli.orchestrators.qwen_image import _torchrun_application_kwargs

    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "fp8",
                  "--quant-calibration", calibration)
    kwargs = _torchrun_application_kwargs(args)
    assert kwargs["quant"]["format"] == "fp8" and kwargs["quant"]["compute"] == "nki"
    assert kwargs["quant_calibration"] == calibration
    assert "quant" not in _torchrun_application_kwargs(_parse("generate", "--model-id", QWEN))
