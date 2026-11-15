"""Host layer: CPU lists, the engine's environment, the neuron0 rule, the idle verdict."""
from __future__ import annotations

import pytest

from difflet.bench import host as H


def test_cpu_list_round_trip():
    assert H.parse_cpu_list("48-51,144, 146") == [48, 49, 50, 51, 144, 146]
    assert H.format_cpu_list([146, 48, 49, 50, 51, 144]) == "48-51,144,146"
    assert H.format_cpu_list([7]) == "7"


@pytest.mark.parametrize("spec", ["", "a-b", "5-3", "1,,x"])
def test_bad_cpu_lists_are_refused(spec):
    with pytest.raises(H.HostError):
        H.parse_cpu_list(spec)


@pytest.fixture(autouse=True)
def _no_sdk_lookup(monkeypatch):
    monkeypatch.setattr(H, "neuronx_cc_dir", lambda path: None)


def test_child_env_prepends_neuronx_cc_when_path_lacks_it(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "source_root", lambda: tmp_path)
    monkeypatch.setattr(H, "neuronx_cc_dir", lambda path: "/opt/sdk/bin")
    child = H.child_env(caller={"PATH": "/usr/bin"}, env_force={}, env_default={},
                        cores="0-3", cpus=[0], threads=None)
    assert child.env["PATH"] == "/opt/sdk/bin:/usr/bin"
    assert child.env["HF_HUB_OFFLINE"] == "1"
    assert child.env["NEURON_NUMA_PIN"] == "0"  # the lane's taskset, not torch-neuronx, pins


def test_child_env_scrubs_engine_knobs_and_records_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "source_root", lambda: tmp_path)
    caller = {"PATH": "/usr/bin", "HF_HUB_OFFLINE": "1", "DIFFLET_BACKEND": "trainium",
              "RANK": "3", "NKI_ENABLE_TRACE_CACHE": "1",
              "TORCH_NEURONX_NEFF_CACHE_DIR": "/x", "OMP_NUM_THREADS": "3",
              "NEURON_CC_FLAGS": "-O3", "KEEP_ME": "k", "FORCED": "caller"}
    child = H.child_env(caller=caller, env_force={"FORCED": "recipe", "NEURON_CC_FLAGS": "-O1"},
                        env_default={"KEEP_ME": "recipe", "NEW": "n"}, cores="20-23",
                        cpus=[0, 1, 2, 3], threads=None, extra={"DIFFLET_BENCH_TIMING_DIR": "/t"})
    env = child.env
    assert "DIFFLET_BACKEND" not in env and "TORCH_NEURONX_NEFF_CACHE_DIR" not in env
    assert "RANK" not in env and "NKI_ENABLE_TRACE_CACHE" not in env
    assert "OMP_NUM_THREADS" not in env  # unset: the engine's own defaults
    assert child.threads is None  # recorded as unset, not guessed
    assert env["NEURON_CC_FLAGS"] == "-O1" and env["FORCED"] == "recipe"
    assert env["KEEP_ME"] == "k" and env["NEW"] == "n"
    assert env["NEURON_RT_VISIBLE_CORES"] == "20-23"
    assert env["DIFFLET_BENCH_TIMING_DIR"] == "/t"
    assert env["PYTHONPATH"].split(":")[0] == str(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1" and env["PATH"] == "/usr/bin"
    assert "overrode" in child.applied["FORCED"]["source"]
    assert child.applied["KEEP_ME"]["source"].startswith("caller")
    assert "NEURON_CC_FLAGS" not in child.dropped  # set back by the recipe
    assert "DIFFLET_BACKEND" in child.dropped


def test_child_env_pins_threads_when_the_recipe_does(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "source_root", lambda: tmp_path)
    child = H.child_env(caller={}, env_force={}, env_default={}, cores="0-3",
                        cpus=list(range(96)), threads=48)
    assert child.env["OMP_NUM_THREADS"] == "48" and child.threads == 48


def test_env_snapshot_never_records_a_secret():
    snap = H.env_snapshot({"HF_TOKEN": "hf_xxx", "DIFFLET_S3_SECRET_ACCESS_KEY": "s",
                           "DIFFLET_S3_BUCKET": "b", "NEURON_RT_VISIBLE_CORES": "0-3",
                           "HOME": "/home/u"})
    assert snap == {"DIFFLET_S3_BUCKET": "<redacted>", "DIFFLET_S3_SECRET_ACCESS_KEY": "<redacted>",
                    "HF_TOKEN": "<redacted>", "NEURON_RT_VISIBLE_CORES": "0-3"}


@pytest.mark.parametrize("name, secret", [
    ("OPENAI_API_KEY", True), ("AUTH_HEADER", True), ("SSH_PRIVATE_FILE", True),
    ("AWS_SESSION_TOKEN", True), ("NEURON_RT_VISIBLE_CORES", False), ("KEYBOARD", False)])
def test_is_secret(name, secret):
    assert H.is_secret(name) is secret


def test_child_env_redacts_a_secret_the_recipe_sets(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "source_root", lambda: tmp_path)
    child = H.child_env(caller={}, env_force={"HF_TOKEN": "hf_x"}, env_default={}, cores="0-3",
                        cpus=[0], threads=None)
    assert child.env["HF_TOKEN"] == "hf_x" and child.applied["HF_TOKEN"]["value"] == "<redacted>"


@pytest.mark.parametrize("device, opened, cores, expect", [
    (5, {0, 5}, "20-23", True),      # holds device 5
    (5, {0}, "0-3", False),          # does not hold device 5 at all
    (0, {0, 5}, "20-23", False),     # every runtime opens neuron0; its cores are elsewhere
    (0, {0}, "20-23", True),         # neuron0 only: it is on device 0
    (0, {0, 5}, None, True),         # cores unknown: count it
    (0, {0, 5}, "2-3,20-21", True),  # overlaps device 0's cores
])
def test_uses_device_neuron0_rule(device, opened, cores, expect):
    assert H.uses_device(device, opened, cores, device0_cores=(0, 1, 2, 3)) is expect


def test_device_holders_ignore_neuron_monitor(monkeypatch):
    table = {"instance_type": "trn3.48xlarge",
             "devices": {0: H.Device(0, (0, 1, 2, 3), 1, "48-95"),
                         8: H.Device(8, (32, 33, 34, 35), 0, "0-47")}}
    monkeypatch.setattr(H, "topology", lambda: table)
    monkeypatch.setattr(H, "_proc_ids", lambda: [10, 11])
    monkeypatch.setattr(H, "_open_neuron_devices", lambda pid: {0, 8})
    monkeypatch.setattr(H, "_environ_value", lambda pid, name: "32-35" if pid == 11 else None)
    monkeypatch.setattr(H, "_who", lambda pid: {"pid": pid, "uid": 1000,
                                                "comm": "neuron-monitor" if pid == 10 else "python"})
    holders = H.device_holders([8])
    assert [h["pid"] for h in holders[8]] == [11]  # the sampler of another lane is not a holder


def test_busy_fraction():
    before = {0: (100, 1000), 1: (100, 1000)}
    after = {0: (150, 1100), 1: (100, 1100)}
    assert H.busy_fraction(before, after) == {0: 0.5, 1: 0.0}


def test_idle_report_verdict_and_reasons():
    ok = H.IdleReport(devices=[8], cpus=[0, 1], device_holders={8: []},
                      cpu_busy={0: 0.01, 1: 0.05})
    assert ok.idle and ok.reasons() == []
    busy = H.IdleReport(devices=[8], cpus=[0, 1],
                        device_holders={8: [{"pid": 7, "comm": "python", "uid": 1000,
                                             "NEURON_RT_VISIBLE_CORES": "32-35"}]},
                        cpu_busy={0: 0.9, 1: 0.05})
    assert not busy.idle
    text = "\n".join(busy.reasons())
    assert "device 8 is held by 1" in text and "pid 7" in text and "0=90%" in text
    assert busy.to_dict()["idle"] is False


def test_devices_to_cores_uses_neuron_ls(monkeypatch):
    table = {"instance_type": "trn3.48xlarge",
             "devices": {8: H.Device(8, (32, 33, 34, 35), 0, "0-47"),
                         9: H.Device(9, (36, 37, 38, 39), 0, "0-47")}}
    monkeypatch.setattr(H, "topology", lambda: table)
    assert H.devices_to_cores([8, 9]) == "32-39"
    with pytest.raises(H.HostError, match="no Neuron device 3"):
        H.devices_to_cores([3])
