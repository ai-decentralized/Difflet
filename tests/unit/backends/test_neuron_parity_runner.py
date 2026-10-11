"""Unit tests for the parity runner, saved references and reporting in
tests/numerical/neuron_parity.py, run with device="cpu"; no Neuron hardware needed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tests.numerical import neuron_parity as parity  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_passes_when_the_device_matches_cpu(dtype):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))
    result = parity.run_parity("mlp", model, (torch.randn(4, 16),), dtype=dtype, device="cpu")
    assert result["ok"], result["reasons"]
    assert (result["control"] is None) == (dtype == torch.float32)
    if dtype == torch.bfloat16:
        assert "vs_bf16" in result["metrics"] and "err_cos" in result["metrics"]


def test_low_precision_result_is_judged_against_the_control():
    calls = []

    def fn(x):
        calls.append(None)
        out = torch.sin(x)
        return out + 0.05 if len(calls) == 2 else out  # second call = the device run

    result = parity.run_parity("perturbed", fn, (torch.randn(256),), dtype=torch.bfloat16, device="cpu")
    assert not result["ok"]
    assert any("max_ratio_vs_control" in reason for reason in result["reasons"])


def test_fp32_result_must_match_the_reference():
    calls = []

    def fn(x):
        calls.append(None)
        return x * (1 + 1e-3) if len(calls) == 2 else x

    result = parity.run_parity("scaled", fn, (torch.randn(64),), dtype=torch.float32, device="cpu")
    assert not result["ok"] and any("fp32_max_rel" in reason for reason in result["reasons"])


def test_module_is_untouched_and_only_the_device_copy_is_prepared():
    model = nn.Linear(4, 4)
    prepared = []
    parity.run_parity("linear", model, (torch.randn(2, 4),), dtype=torch.bfloat16, device="cpu",
                      prepare=prepared.append)
    assert model.weight.dtype == torch.float32
    assert len(prepared) == 1 and prepared[0] is not model and prepared[0].weight.dtype == torch.bfloat16


def test_output_selection():
    class Labelled(nn.Module):
        def forward(self, x):
            return {"label": "aux", "out": x * 2}

    assert parity.run_parity("dict", Labelled(), (torch.randn(3),), dtype=torch.float32, device="cpu")["ok"]
    with pytest.raises(TypeError, match="no tensor"):
        parity.run_parity("none", lambda x: "text", (torch.randn(3),), dtype=torch.float32, device="cpu")


def test_reference_roundtrip(tmp_path):
    pytest.importorskip("safetensors")
    out = torch.randn(8, 8)
    parity.save_reference(tmp_path / "aot" / "step0", {"noise_pred": out}, {"backend": "trainium", "step": 0})
    tensors, record = parity.load_reference(tmp_path / "aot" / "step0")
    assert torch.equal(tensors["noise_pred"], out)
    assert record["meta"] == {"backend": "trainium", "step": 0} and "torch" in record["build"]


def test_report_exit_code_and_json(tmp_path, capsys):
    good = parity.run_parity("ok", nn.Identity(), (torch.ones(2),), dtype=torch.float32, device="cpu")
    bad = dict(good, name="bad", ok=False, reasons=["forced"])
    assert parity.report([good]) == 0
    assert parity.report([good, bad], json_path=tmp_path / "r.json") == 1
    assert "1/2 passed" in capsys.readouterr().out
    assert '"results"' in (tmp_path / "r.json").read_text()
