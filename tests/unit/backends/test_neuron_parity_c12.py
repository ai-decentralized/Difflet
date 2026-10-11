"""Unit tests for the neuron parity harness, run with device="cpu"; no Neuron hardware needed."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from difflet.backends.neuron import parity  # noqa: E402


def test_compare_identical_and_known_difference():
    ref = torch.tensor([1.0, -2.0, 4.0])
    same = parity.compare(ref.clone(), ref)
    assert (same.max_abs, same.mean_abs, same.max_rel) == (0.0, 0.0, 0.0)
    assert same.cosine == pytest.approx(1.0)
    diff = parity.compare(torch.tensor([1.0, -2.0, 5.0]), ref)
    assert diff.max_abs == pytest.approx(1.0)
    assert diff.mean_abs == pytest.approx(1 / 3)
    assert diff.max_rel == pytest.approx(0.25)


def test_compare_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        parity.compare(torch.zeros(3), torch.zeros(4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_check_passes_when_device_matches_cpu(dtype):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))
    result = parity.check("mlp", model, (torch.randn(4, 16),), dtype=dtype, device="cpu")
    assert result.passed, result
    assert (result.baseline is None) == (dtype == torch.float32)


def test_check_fails_when_the_device_run_differs():
    calls = []

    def fn(x):
        calls.append(None)
        out = torch.sin(x)
        return out + 0.5 if len(calls) == 2 else out  # second call = the device run

    result = parity.check("perturbed", fn, (torch.randn(64),), dtype=torch.bfloat16, device="cpu")
    assert not result.passed and "mean_abs above" in result.reason


def test_check_leaves_the_module_untouched_and_prepares_only_the_device_copy():
    model = nn.Linear(4, 4)
    prepared = []
    parity.check("linear", model, (torch.randn(2, 4),), dtype=torch.bfloat16, device="cpu",
                 prepare=lambda m: prepared.append(m))
    assert model.weight.dtype == torch.float32
    assert len(prepared) == 1 and prepared[0] is not model and prepared[0].weight.dtype == torch.bfloat16


def test_output_selection_for_tuples_and_dicts():
    class Pair(nn.Module):
        def forward(self, x):
            return {"aux": "label", "out": x * 2}

    result = parity.check("dict", Pair(), (torch.randn(3),), dtype=torch.float32, device="cpu")
    assert result.passed
    with pytest.raises(TypeError, match="no tensor"):
        parity.check("none", lambda x: "text", (torch.randn(3),), dtype=torch.float32, device="cpu")


def test_reference_roundtrip_and_comparison(tmp_path):
    pytest.importorskip("safetensors")
    out = torch.randn(8, 8)
    parity.save_reference(tmp_path / "aot" / "step0", {"noise_pred": out}, {"backend": "trainium", "step": 0})
    tensors, record = parity.load_reference(tmp_path / "aot" / "step0")
    assert torch.equal(tensors["noise_pred"], out)
    assert record["meta"] == {"backend": "trainium", "step": 0} and "torch" in record["build"]
    assert parity.compare_to_reference("same", out, tensors["noise_pred"]).passed
    assert not parity.compare_to_reference("other", -out, tensors["noise_pred"]).passed


def test_report_exit_code(capsys):
    good = parity.check("ok", nn.Identity(), (torch.ones(2),), dtype=torch.float32, device="cpu")
    bad = parity.compare_to_reference("bad", torch.ones(2), -torch.ones(2))
    assert parity.report([good]) == 0
    assert parity.report([good, bad]) == 1
    assert "1/2 passed" in capsys.readouterr().out
