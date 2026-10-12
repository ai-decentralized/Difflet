"""Argument handling of the CPU oracle reference script (tpu_oracle_reference.py).

The script's forwards are replaced by recorders, so these tests load no model and need no
weights: they check what the script would run, with how many threads, and that a missing
snapshot or output directory stops it before the minutes of loading and computing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).with_name("tpu_oracle_reference.py")


@pytest.fixture
def snapshot(tmp_path):
    """A directory shaped like a diffusers snapshot: ``transformer/config.json``."""
    root = tmp_path / "snapshot"
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text("{}")
    return root


@pytest.fixture
def oracle(monkeypatch, snapshot):
    """The script as a fresh module (it reads its environment on import), its forwards
    replaced by recorders, and the process's thread count restored afterwards."""
    monkeypatch.setenv("DIFFLET_WAN_SNAPSHOT", str(snapshot))
    monkeypatch.setenv("DIFFLET_QWEN_SNAPSHOT", str(snapshot))
    spec = importlib.util.spec_from_file_location("tpu_oracle_reference_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def recorder(name):
        def record(*args, **kwargs):
            calls.append({"fn": name, "args": args, "threads": torch.get_num_threads(),
                          **kwargs})
        return record

    monkeypatch.setattr(module, "wan", recorder("wan"))
    monkeypatch.setattr(module, "qwen_image", recorder("qwen_image"))
    module.calls = calls
    threads = torch.get_num_threads()
    yield module
    torch.set_num_threads(threads)


def run(monkeypatch, oracle, *argv):
    monkeypatch.setattr(sys, "argv", ["tpu_oracle_reference.py", *argv])
    oracle.main()
    return oracle.calls


def test_threads_default_to_the_host_cpu_count(monkeypatch, oracle, tmp_path):
    calls = run(monkeypatch, oracle, "wan", "--out", str(tmp_path / "ref.pt"))
    assert [c["threads"] for c in calls] == [os.cpu_count()]


def test_threads_flag_sets_the_thread_count(monkeypatch, oracle, tmp_path):
    calls = run(monkeypatch, oracle, "wan", "--threads", "4", "--out", str(tmp_path / "ref.pt"))
    assert [c["threads"] for c in calls] == [4]


@pytest.mark.parametrize("threads", ["0", "-2"])
def test_threads_flag_rejects_non_positive_counts(monkeypatch, oracle, tmp_path, threads):
    with pytest.raises(SystemExit):
        run(monkeypatch, oracle, "wan", "--threads", threads, "--out", str(tmp_path / "r.pt"))
    assert oracle.calls == []


@pytest.mark.parametrize("model", ["wan", "qwen_image"])
def test_snapshot_flag_overrides_the_environment(monkeypatch, oracle, tmp_path, model):
    other = tmp_path / "other"
    (other / "transformer").mkdir(parents=True)
    (other / "transformer" / "config.json").write_text("{}")
    calls = run(monkeypatch, oracle, model, "--snapshot", str(other),
                "--out", str(tmp_path / "ref.pt"))
    assert [(c["fn"], c["snapshot"], c["out"]) for c in calls] == [
        (model, str(other), tmp_path / "ref.pt")
    ]


def test_environment_still_selects_the_snapshot(monkeypatch, oracle, snapshot, tmp_path):
    calls = run(monkeypatch, oracle, "wan", "--frames", "5", "--out", str(tmp_path / "ref.pt"))
    assert [(c["snapshot"], c["frames"]) for c in calls] == [(str(snapshot), 5)]


def test_missing_snapshot_stops_before_loading(monkeypatch, oracle, tmp_path, capsys):
    monkeypatch.setenv("DIFFLET_WAN_SNAPSHOT", str(tmp_path / "no-such-snapshot"))
    with pytest.raises(SystemExit):
        run(monkeypatch, oracle, "wan", "--out", str(tmp_path / "ref.pt"))
    assert oracle.calls == []
    assert "--snapshot" in capsys.readouterr().err


def test_missing_output_directory_stops_before_loading(monkeypatch, oracle, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, oracle, "wan", "--out", str(tmp_path / "missing-dir" / "ref.pt"))
    assert oracle.calls == []
    assert "--out" in capsys.readouterr().err


# ---------------------------------------------------------------- --timesteps


def test_timesteps_default_to_500(monkeypatch, oracle, tmp_path):
    calls = run(monkeypatch, oracle, "wan", "--out", str(tmp_path / "ref.pt"))
    assert [c["timesteps"] for c in calls] == [[500.0]]


def test_timesteps_flag_takes_a_comma_separated_list(monkeypatch, oracle, tmp_path):
    calls = run(monkeypatch, oracle, "wan", "--timesteps", "996, 500",
                "--out", str(tmp_path / "ref.pt"))
    assert [c["timesteps"] for c in calls] == [[996.0, 500.0]]


@pytest.mark.parametrize("value", ["996,abc", "", "996,,500"])
def test_timesteps_flag_rejects_a_malformed_list(monkeypatch, oracle, tmp_path, value):
    with pytest.raises(SystemExit):
        run(monkeypatch, oracle, "wan", "--timesteps", value, "--out", str(tmp_path / "r.pt"))
    assert oracle.calls == []


def test_timesteps_flag_is_wan_only(monkeypatch, oracle, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, oracle, "qwen_image", "--timesteps", "0.5",
            "--out", str(tmp_path / "r.pt"))
    assert oracle.calls == []
    assert "--timesteps" in capsys.readouterr().err


class _FakeWan(torch.nn.Module):
    """Stands in for diffusers' WanTransformer3DModel: the output depends on the timestep,
    the latents and the text, and is computed in the module's dtype."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.seen = []

    def forward(self, hidden, timestep, text, return_dict=True):
        self.seen.append((float(timestep[0]), hidden.dtype, timestep.dtype, text.dtype))
        out = hidden * self.scale * timestep.view(-1, 1, 1, 1, 1) / 1000 + text.mean()
        return (out,)


def test_wan_saves_one_reference_and_one_bf16_control_per_timestep(monkeypatch, tmp_path):
    import diffusers

    spec = importlib.util.spec_from_file_location("tpu_oracle_reference_wan", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = _FakeWan()
    monkeypatch.setattr(diffusers.WanTransformer3DModel, "from_pretrained",
                        lambda path, torch_dtype: fake)
    out = tmp_path / "oracle.pt"

    module.wan(out=out, frames=1, snapshot=str(tmp_path), timesteps=[996.0, 500.0])

    blob = torch.load(out, weights_only=False)
    assert blob["timesteps"] == [996.0, 500.0]
    assert [float(r["timestep"][0]) for r in blob["runs"]] == [996.0, 500.0]
    # fp32 forwards first, then the bf16 control on the same inputs.
    assert fake.seen == [
        (996.0, torch.float32, torch.float32, torch.float32),
        (500.0, torch.float32, torch.float32, torch.float32),
        (996.0, torch.bfloat16, torch.bfloat16, torch.bfloat16),
        (500.0, torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ]
    hidden, text = blob["hidden_states"], blob["encoder_hidden_states"]
    assert hidden.shape == (1, 16, 1, 60, 104) and text.shape == (1, 512, 4096)
    for run_ in blob["runs"]:
        t = run_["timestep"]
        expected = hidden * 0.5 * t.view(-1, 1, 1, 1, 1) / 1000 + text.mean()
        torch.testing.assert_close(run_["reference"], expected)
        assert run_["reference_bf16"].dtype == torch.float32  # stored as fp32
        torch.testing.assert_close(run_["reference_bf16"], expected, atol=0.1, rtol=0.02)
        assert not torch.equal(run_["reference_bf16"], run_["reference"])
    # The single-timestep keys stay, holding the first timestep (what tpu_oracle_compare reads).
    assert torch.equal(blob["timestep"], blob["runs"][0]["timestep"])
    assert torch.equal(blob["reference"], blob["runs"][0]["reference"])
    assert torch.equal(blob["reference_bf16"], blob["runs"][0]["reference_bf16"])
    assert blob["frames"] == 1


def test_expert_selects_the_low_noise_subdirectory(monkeypatch, oracle, snapshot, tmp_path):
    (snapshot / "transformer_2").mkdir()
    (snapshot / "transformer_2" / "config.json").write_text("{}")
    calls = run(monkeypatch, oracle, "wan", "--expert", "transformer_2",
                "--out", str(tmp_path / "ref.pt"))
    assert [c["expert"] for c in calls] == ["transformer_2"]


def test_expert_defaults_to_the_high_noise_transformer(monkeypatch, oracle, tmp_path):
    calls = run(monkeypatch, oracle, "wan", "--out", str(tmp_path / "ref.pt"))
    assert [c["expert"] for c in calls] == ["transformer"]


def test_missing_low_noise_expert_stops_before_loading(monkeypatch, oracle, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, oracle, "wan", "--expert", "transformer_2", "--out", str(tmp_path / "r.pt"))
    assert oracle.calls == [] and "transformer_2/config.json" in capsys.readouterr().err


def test_expert_applies_to_wan_only(monkeypatch, oracle, tmp_path):
    with pytest.raises(SystemExit):
        run(monkeypatch, oracle, "qwen_image", "--expert", "transformer_2",
            "--out", str(tmp_path / "r.pt"))
    assert oracle.calls == []
