"""Activation calibration: collection on a toy model, the file, and the static scale."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from difflet.quant.calibration import (
    ActivationCalibrator,
    read_calibration,
    write_calibration,
)
from difflet.quant.spec import QuantSpec


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.mod = nn.Linear(8, 8)

    def forward(self, x):
        return self.to_q(x) + self.mod(x)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x


SPEC = QuantSpec(targets=(r"transformer_blocks\.\d+\.to_q",))


def test_calibrator_records_running_amax_per_target_only():
    torch.manual_seed(0)
    model = _Toy()
    with ActivationCalibrator(model, SPEC) as cal:
        x1 = torch.randn(3, 8)
        model(x1)
        model(x1 * 3)
    amax = cal.amax()
    assert sorted(amax) == ["transformer_blocks.0.to_q", "transformer_blocks.1.to_q"]
    assert amax["transformer_blocks.0.to_q"] == pytest.approx((x1 * 3).abs().amax().item())
    assert cal.calls()["transformer_blocks.1.to_q"] == 2
    # Hooks are gone after the context: a forward records nothing new.
    model(x1 * 100)
    assert cal.amax() == amax


def test_calibrator_strips_the_trace_prefix():
    wrapper = nn.Module()
    wrapper.transformer = _Toy()
    with ActivationCalibrator(wrapper, SPEC) as cal:
        wrapper.transformer(torch.ones(2, 8))
    assert "transformer_blocks.0.to_q" in cal.amax()


def test_calibrator_refuses_a_spec_matching_nothing():
    with pytest.raises(ValueError, match="match no nn.Linear"):
        ActivationCalibrator(_Toy(), QuantSpec(targets=(r"nothing",)))


def test_calibrator_reports_targets_that_never_ran():
    model = _Toy()
    with ActivationCalibrator(model, SPEC) as cal:
        model.transformer_blocks[0](torch.ones(2, 8))
    with pytest.raises(ValueError, match="saw no forward"):
        cal.amax()


def test_write_and_read_calibration(tmp_path):
    path = write_calibration(
        tmp_path / "cal.json", amax={"transformer_blocks.0.to_q": 12.5}, model="qwen_image",
        checkpoint={"model_id": "Qwen/Qwen-Image", "snapshot": "75e0b4be"},
        prompts=["a cat"], steps=20, shape={"height": 1024, "width": 1024},
        calls={"transformer_blocks.0.to_q": 20},
    )
    cal = read_calibration(path)
    assert cal.amax == {"transformer_blocks.0.to_q": 12.5}
    assert cal.provenance["checkpoint"]["snapshot"] == "75e0b4be"
    assert cal.provenance["steps"] == 20 and "amax" not in cal.provenance
    assert len(cal.sha256) == 64
    scale = cal.input_scale("transformer.transformer_blocks.0.to_q", SPEC)
    assert scale.dtype == torch.float32 and scale.shape == (1,)
    assert scale.item() == pytest.approx(12.5 * 1.25 / 448.0)
    assert cal.covers(["transformer_blocks.0.to_q", "transformer_blocks.1.to_q"]) == [
        "transformer_blocks.1.to_q"]
    with pytest.raises(KeyError, match="no activation amax"):
        cal.input_scale("transformer_blocks.1.to_q", SPEC)


def test_read_calibration_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": 1, "amax": {"a": -1.0}}))
    with pytest.raises(ValueError, match="negative"):
        read_calibration(bad)
    bad.write_text(json.dumps({"schema": 99, "amax": {"a": 1.0}}))
    with pytest.raises(ValueError, match="schema"):
        read_calibration(bad)
    bad.write_text(json.dumps({"schema": 1}))
    with pytest.raises(ValueError, match="amax"):
        read_calibration(bad)
