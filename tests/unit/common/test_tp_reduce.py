from __future__ import annotations

import argparse
import importlib

import pytest
import torch

from difflet.cli.orchestrators.minimax_h3 import MiniMaxH3Orchestrator
from difflet.common.tp_reduce import parse_tp_reduce_dtype, tp_reduce_cache_fields

# difflet.cli re-exports the ``main`` function, which shadows the submodule.
cli_main = importlib.import_module("difflet.cli.main")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, torch.bfloat16), ("bf16", torch.bfloat16), ("fp32", torch.float32)],
)
def test_parse_tp_reduce_dtype(value, expected):
    assert parse_tp_reduce_dtype(value) is expected


def test_parse_tp_reduce_dtype_rejects_other_values():
    with pytest.raises(ValueError, match="bf16, fp32"):
        parse_tp_reduce_dtype("fp16")


def test_cache_fields_empty_for_legacy_dtype():
    assert tp_reduce_cache_fields(torch.float32, legacy=torch.float32) == {}
    assert tp_reduce_cache_fields(torch.bfloat16, legacy=torch.bfloat16) == {}


def test_cache_fields_name_a_changed_dtype():
    assert tp_reduce_cache_fields(torch.bfloat16, legacy=torch.float32) == {
        "tp_reduce_dtype": "bfloat16"
    }
    assert tp_reduce_cache_fields(torch.float32, legacy=torch.bfloat16) == {
        "tp_reduce_dtype": "float32"
    }


def test_cache_fields_always_present_for_mixed_legacy():
    assert tp_reduce_cache_fields(torch.bfloat16, legacy=None) == {
        "tp_reduce_dtype": "bfloat16"
    }


@pytest.mark.parametrize("command", ["compile", "generate"])
def test_cli_flag_parses(command):
    parser = cli_main._build_parser()
    args = parser.parse_args(
        [command, "--model-id", "Qwen/Qwen-Image", "--tp-reduce-dtype", "fp32"]
    )
    assert args.tp_reduce_dtype == "fp32"
    args = parser.parse_args([command, "--model-id", "Qwen/Qwen-Image"])
    assert args.tp_reduce_dtype is None


def test_cli_flag_rejects_other_values():
    parser = cli_main._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["generate", "--model-id", "Qwen/Qwen-Image", "--tp-reduce-dtype", "fp16"]
        )


def test_stage_parser_carries_flag():
    from difflet.cli.stage import _build_stage_parser

    args, _ = _build_stage_parser().parse_known_args(
        ["--orchestrator", "x", "--stage", "generate", "--tp-reduce-dtype", "fp32"]
    )
    assert args.tp_reduce_dtype == "fp32"


def _minimax(tp_reduce_dtype):
    orch = MiniMaxH3Orchestrator.__new__(MiniMaxH3Orchestrator)
    orch.args = argparse.Namespace(
        model_id="MiniMaxAI/MiniMax-H3",
        tp_degree=4,
        cp_degree=1,
        cfg_parallel=False,
        sp_enabled=False,
        tp_reduce_dtype=tp_reduce_dtype,
    )
    return orch


@pytest.mark.parametrize("value", ["bf16", "fp32"])
def test_minimax_rejects_flag(value):
    with pytest.raises(
        NotImplementedError, match="--tp-reduce-dtype is not supported for MiniMax-H3"
    ):
        _minimax(value)._validate_topology()


def test_minimax_accepts_absent_flag():
    _minimax(None)._validate_topology()
