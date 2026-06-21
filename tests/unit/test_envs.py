"""Unit tests for `difflet.envs`.

Cover the parser registry surface: lazy evaluation, type coercion, defaults,
dir() reporting, and unknown-name behavior.
"""

from __future__ import annotations

import importlib
import os

import pytest

from difflet import envs


def _unset(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_known_names_listed_in_dir() -> None:
    names = dir(envs)
    assert "DIFFLET_BACKEND" in names
    assert "DIFFLET_COMPILE_CACHE" in names
    assert "BASE_COMPILE_WORK_DIR" in names
    assert "RANK" in names
    assert "WORLD_SIZE" in names
    assert names == sorted(names), "dir() output should be sorted"


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="DOES_NOT_EXIST"):
        envs.DOES_NOT_EXIST


def test_lazy_evaluation_picks_up_env_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
    assert envs.DIFFLET_BACKEND == "trainium"
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    assert envs.DIFFLET_BACKEND == "cuda"


def test_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _unset(monkeypatch, "DIFFLET_BACKEND")
    assert envs.DIFFLET_BACKEND is None

    _unset(monkeypatch, "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    assert envs.RANK == 0
    assert envs.WORLD_SIZE == 1
    assert envs.LOCAL_RANK == 0
    assert envs.LOCAL_WORLD_SIZE == 1

    _unset(monkeypatch, "BASE_COMPILE_WORK_DIR")
    assert envs.BASE_COMPILE_WORK_DIR == "/tmp/nxd_model/"

    _unset(monkeypatch, "MASTER_ADDR", "MASTER_PORT")
    assert envs.MASTER_ADDR == "127.0.0.1"
    assert envs.MASTER_PORT == "29500"

    _unset(monkeypatch, "NEURON_RT_VIRTUAL_CORE_SIZE", "NEURON_LOGICAL_NC_CONFIG")
    assert envs.NEURON_RT_VIRTUAL_CORE_SIZE == 1
    assert envs.NEURON_LOGICAL_NC_CONFIG == -1


def test_int_coercion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
    assert envs.RANK == 3
    assert envs.WORLD_SIZE == 8
    assert envs.NEURON_RT_VIRTUAL_CORE_SIZE == 2


def test_compile_cache_expanduser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", "~/my-cache")
    expanded = envs.DIFFLET_COMPILE_CACHE
    assert "~" not in expanded
    assert expanded.endswith("/my-cache")


def test_compile_cache_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _unset(monkeypatch, "DIFFLET_COMPILE_CACHE")
    value = envs.DIFFLET_COMPILE_CACHE
    assert "~" not in value
    assert value.endswith("/.cache/difflet")


def test_nxd_inference_capture_snapshot_truthiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _unset(monkeypatch, "NXD_INFERENCE_CAPTURE_SNAPSHOT")
    assert envs.NXD_INFERENCE_CAPTURE_SNAPSHOT is False

    monkeypatch.setenv("NXD_INFERENCE_CAPTURE_SNAPSHOT", "1")
    assert envs.NXD_INFERENCE_CAPTURE_SNAPSHOT is True


def test_module_reimport_is_idempotent() -> None:
    reloaded = importlib.reload(envs)
    assert dir(reloaded) == dir(envs)
