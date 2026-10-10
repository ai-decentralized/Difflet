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
