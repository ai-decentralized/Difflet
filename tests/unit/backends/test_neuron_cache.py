"""Unit tests for the neuron backend's compile-cache layout (``neuron_cache_env`` in
difflet.backends.neuron.compile).

CPU only; nothing here imports torch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from difflet.backends.neuron import compile as neuron_compile

REPO_ROOT = Path(__file__).resolve().parents[3]
NEFF, NEFF_LOCAL, HLO = (
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR",
)


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path):
    """No cache variable in os.environ, DIFFLET_COMPILE_CACHE=tmp_path (neuron_cache_env
    writes nothing, so delenv needs nothing to restore)."""
    for var in neuron_compile.CACHE_ENV_DIRS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    return tmp_path


def test_layout_is_neuron_neff_neff_local_hlo():
    assert neuron_compile.CACHE_SUBDIR == "neuron"
    assert neuron_compile.CACHE_ENV_DIRS == {NEFF: "neff", NEFF_LOCAL: "neff_local", HLO: "hlo"}


def test_cache_env_defaults_under_difflet_cache(clean_cache_env):
    assert neuron_compile.neuron_cache_env() == {
        NEFF: str(clean_cache_env / "neuron" / "neff"),
        NEFF_LOCAL: str(clean_cache_env / "neuron" / "neff_local"),
        HLO: str(clean_cache_env / "neuron" / "hlo"),
    }
    assert not (clean_cache_env / "neuron").exists()  # nothing created


def test_cache_env_keeps_variables_already_set(clean_cache_env, monkeypatch):
    monkeypatch.setenv(NEFF, "/shared/neff")
    monkeypatch.setenv(NEFF_LOCAL, "")  # empty: not a choice, the default applies
    env = neuron_compile.neuron_cache_env(root=clean_cache_env / "other")
    assert env == {
        NEFF: "/shared/neff",
        NEFF_LOCAL: str(clean_cache_env / "other" / "neuron" / "neff_local"),
        HLO: str(clean_cache_env / "other" / "neuron" / "hlo"),
    }


def test_cache_env_reads_the_given_mapping_not_os_environ(clean_cache_env, monkeypatch):
    monkeypatch.setenv(NEFF, "/from/os/environ")
    given = {HLO: "relative/hlo"}
    before = dict(os.environ)
    env = neuron_compile.neuron_cache_env(env=given)
    assert env == {
        NEFF: str(clean_cache_env / "neuron" / "neff"),  # os.environ's value is not consulted
        NEFF_LOCAL: str(clean_cache_env / "neuron" / "neff_local"),
        HLO: "relative/hlo",  # verbatim, as the process that reads it will see it
    }
    assert given == {HLO: "relative/hlo"}
    assert dict(os.environ) == before


def test_cache_env_takes_the_root_from_the_given_mapping_first(clean_cache_env, monkeypatch):
    """A per-run environment (a benchmark row's) carries its own DIFFLET_COMPILE_CACHE: the
    locations follow it, not the calling process's; an explicit root still wins."""
    row = clean_cache_env / "row"
    given = {"DIFFLET_COMPILE_CACHE": str(row)}  # os.environ's is clean_cache_env
    assert neuron_compile.neuron_cache_env(env=given) == {
        NEFF: str(row / "neuron" / "neff"),
        NEFF_LOCAL: str(row / "neuron" / "neff_local"),
        HLO: str(row / "neuron" / "hlo"),
    }
    assert neuron_compile.neuron_cache_env(clean_cache_env / "explicit", env=given)[HLO] == str(
        clean_cache_env / "explicit" / "neuron" / "hlo"
    )
    monkeypatch.setenv("HOME", str(clean_cache_env))
    assert neuron_compile.neuron_cache_env(env={"DIFFLET_COMPILE_CACHE": "~/r"})[NEFF] == str(
        clean_cache_env / "r" / "neuron" / "neff"
    )
    # An empty value is no choice: the process's DIFFLET_COMPILE_CACHE applies.
    assert neuron_compile.neuron_cache_env(env={"DIFFLET_COMPILE_CACHE": ""})[NEFF] == str(
        clean_cache_env / "neuron" / "neff"
    )
    configured = {"DIFFLET_COMPILE_CACHE": str(row)}
    neuron_compile.configure_compile_cache(env=configured)
    assert configured["TORCH_NEURONX_HLO_CACHE_DIR"] == str(row / "neuron" / "hlo")


def test_cache_env_expands_the_home_directory_and_makes_paths_absolute(
    clean_cache_env, monkeypatch
):
    monkeypatch.setenv("HOME", str(clean_cache_env))
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", "~/dc")
    assert neuron_compile.neuron_cache_env()[HLO] == str(clean_cache_env / "dc" / "neuron" / "hlo")
    assert neuron_compile.neuron_cache_env(root="~/r")[NEFF] == str(
        clean_cache_env / "r" / "neuron" / "neff"
    )
    monkeypatch.chdir(clean_cache_env)
    assert neuron_compile.neuron_cache_env(root="rel")[NEFF_LOCAL] == str(
        clean_cache_env / "rel" / "neuron" / "neff_local"
    )


def test_default_neff_cache_dir_is_the_neff_entry(clean_cache_env, monkeypatch):
    assert neuron_compile.default_neff_cache_dir() == Path(neuron_compile.neuron_cache_env()[NEFF])
    assert neuron_compile.default_neff_cache_dir() == clean_cache_env / "neuron" / "neff"
    # The default location: a variable set in the environment does not move it.
    monkeypatch.setenv(NEFF, "/shared/neff")
    assert neuron_compile.default_neff_cache_dir() == clean_cache_env / "neuron" / "neff"


def test_compile_module_imports_without_torch():
    """A launcher computes and sets the locations before torch is imported, so the compile
    module must not import torch. -I ignores PYTHONPATH, so the child puts the worktree on
    sys.path itself (Difflet is not installed)."""
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "import difflet.backends.neuron.compile; "
        "sys.exit([n for n in ('torch', 'torch_neuronx') if n in sys.modules] or 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
