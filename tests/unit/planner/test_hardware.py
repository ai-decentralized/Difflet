"""Hardware discovery: ``neuron-ls`` parsing, allocation precedence, fallbacks.

No device and no subprocess: every test injects a ``neuron-ls`` payload (or the
absence of one) so the same assertions hold on a laptop and on a trn2.
"""
from __future__ import annotations

import pytest

from difflet.planner import hardware
from difflet.planner.hardware import HardwareProfile, detect_hardware, detected_core_count

# The literal payload `neuron-ls -j` prints on the trn2.3xlarge Difflet is
# developed on. Kept verbatim so a format change in the Neuron SDK surfaces here.
TRN2_3XLARGE = [
    {
        "instance_type": "trn2.3xlarge",
        "instance_id": "i-08dc867eb640aad0d",
        "neuron_device": 0,
        "bdf": "0000:33:00.0",
        "cpu_affinity": "0-11",
        "numa_node": "0",
        "connected_to": None,
        "nc_count": 4,
        "logical_neuroncore_config": 2,
        "memory_size": 103079215104,
        "neuroncore_ids": [0, 1, 2, 3],
        "neuron_processes": [],
    }
]


def _trn2_48xlarge(num_devices: int = 16) -> list[dict]:
    return [
        {
            "instance_type": "trn2.48xlarge",
            "neuron_device": index,
            "nc_count": 4,
            "logical_neuroncore_config": 2,
            "memory_size": 103079215104,
            "neuroncore_ids": [index * 4 + offset for offset in range(4)],
            "neuron_processes": [],
        }
        for index in range(num_devices)
    ]


@pytest.fixture
def neuron_ls(monkeypatch):
    """Install a fake ``neuron-ls -j`` payload for the duration of a test."""

    def install(payload):
        hardware._probe_neuron_ls_cached.cache_clear()
        monkeypatch.setattr(hardware, "_run_neuron_ls", lambda: payload)

    yield install
    hardware._probe_neuron_ls_cached.cache_clear()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)


# ------------------------------------------------------------------ parsing


def test_parses_single_device_host(neuron_ls):
    neuron_ls(TRN2_3XLARGE)
    profile = detect_hardware()
    assert profile.instance_type == "trn2.3xlarge"
    assert profile.platform_target == "trn2"
    assert profile.num_devices == 1
    assert profile.cores_per_device == 4
    assert profile.machine_cores == 4
    assert profile.allocated_cores == 4
    assert profile.hbm_bytes_per_device == 103079215104
    assert profile.lnc == 2
    assert profile.source == "neuron-ls"
    assert not profile.is_multi_device


def test_parses_multi_device_host(neuron_ls):
    """The 64-core box the old hardcoded ``(0,1,2,3)`` silently under-reported."""

    neuron_ls(_trn2_48xlarge())
    profile = detect_hardware()
    assert profile.num_devices == 16
    assert profile.cores_per_device == 4
    assert profile.machine_cores == 64
    assert profile.allocated_cores == 64
    assert profile.is_multi_device


def test_busy_cores_surface_from_neuron_processes(neuron_ls):
    payload = [dict(TRN2_3XLARGE[0], neuron_processes=[{"pid": 42, "neuroncore_ids": [2, 3]}])]
    neuron_ls(payload)
    assert detect_hardware().busy_cores == (2, 3)


def test_unparseable_process_entry_does_not_break_the_probe(neuron_ls):
    payload = [dict(TRN2_3XLARGE[0], neuron_processes=[{"pid": 42, "mystery_field": "?"}])]
    neuron_ls(payload)
    profile = detect_hardware()
    assert profile.busy_cores == ()
    assert profile.allocated_cores == 4


def test_core_count_prefers_neuroncore_ids_over_nc_count(neuron_ls):
    """``nc_count`` and the id list disagree under some LNC settings; ids win."""

    payload = [dict(TRN2_3XLARGE[0], nc_count=8, neuroncore_ids=[0, 1, 2, 3])]
    neuron_ls(payload)
    assert detect_hardware().cores_per_device == 4


# ------------------------------------------------------- allocation precedence


def test_override_beats_everything(neuron_ls, monkeypatch):
    neuron_ls(TRN2_3XLARGE)
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0-1")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "2")
    profile = detect_hardware(allocated_cores_override=64)
    assert profile.allocated_cores == 64
    assert profile.machine_cores == 4
    assert "override" in profile.source


def test_visible_cores_beats_num_cores(neuron_ls, monkeypatch):
    neuron_ls(TRN2_3XLARGE)
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "2-3")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "4")
    profile = detect_hardware()
    assert profile.allocated_cores == 2
    assert "NEURON_RT_VISIBLE_CORES" in profile.source


def test_dp_worker_allocation_is_its_slice_not_the_box(neuron_ls, monkeypatch):
    """A DP worker inherits a core range; its budget is that range, not the host."""

    neuron_ls(_trn2_48xlarge())
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-11")
    profile = detect_hardware()
    assert profile.allocated_cores == 4
    assert profile.machine_cores == 64


def test_num_cores_used_when_no_visible_list(neuron_ls, monkeypatch):
    neuron_ls(TRN2_3XLARGE)
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "2")
    assert detect_hardware().allocated_cores == 2


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0,1,2,3", 4),
        ("0-3", 4),
        ("0-1,4-5", 4),
        ("  2 , 3 ", 2),
        ("0,0,1", 2),  # duplicates collapse
    ],
)
def test_visible_core_syntax(neuron_ls, monkeypatch, raw, expected):
    neuron_ls(TRN2_3XLARGE)
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", raw)
    assert detect_hardware().allocated_cores == expected


def test_malformed_visible_cores_falls_through_to_machine(neuron_ls, monkeypatch):
    neuron_ls(TRN2_3XLARGE)
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "not-a-core-list")
    assert detect_hardware().allocated_cores == 4


# ------------------------------------------------------------------ fallbacks


def test_missing_neuron_ls_yields_fallback_profile(neuron_ls):
    neuron_ls([])
    profile = detect_hardware()
    assert profile.source == "fallback"
    assert profile.num_devices == 1
    assert profile.allocated_cores >= 1


def test_fallback_is_distinguishable_from_measurement(neuron_ls):
    """The CLI keys its hard error off this: never fail a user on a guess."""

    neuron_ls([])
    assert not detect_hardware().source.startswith("neuron-ls")
    neuron_ls(TRN2_3XLARGE)
    assert detect_hardware().source.startswith("neuron-ls")


def test_detected_core_count_is_none_without_neuron_ls(neuron_ls):
    neuron_ls([])
    assert detected_core_count() is None


def test_detected_core_count_sums_devices(neuron_ls):
    neuron_ls(_trn2_48xlarge(num_devices=4))
    assert detected_core_count() == 16


def test_probe_survives_garbage_output(monkeypatch):
    hardware._probe_neuron_ls_cached.cache_clear()
    monkeypatch.setattr(hardware, "_run_neuron_ls", lambda: [])
    assert detect_hardware().source == "fallback"
    hardware._probe_neuron_ls_cached.cache_clear()


# ------------------------------------------------------------------ invariants


def test_rejects_nonsensical_profiles():
    with pytest.raises(ValueError):
        HardwareProfile(
            instance_type="x", platform_target="trn2", num_devices=0,
            cores_per_device=4, hbm_bytes_per_device=1, lnc=2, allocated_cores=4,
        )
    with pytest.raises(ValueError):
        HardwareProfile(
            instance_type="x", platform_target="trn2", num_devices=1,
            cores_per_device=4, hbm_bytes_per_device=1, lnc=2, allocated_cores=0,
        )


def test_describe_mentions_partial_allocation(neuron_ls, monkeypatch):
    neuron_ls(_trn2_48xlarge())
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0-3")
    text = detect_hardware().describe()
    assert "trn2.48xlarge" in text
    assert "4 of 64 cores allocated" in text
