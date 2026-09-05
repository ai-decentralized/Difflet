"""Process-world guard: the two on-device crash signatures must become exceptions.

Neither test needs Neuron — the module is pure Python and the checks run at
load time, before any weight touches the device.
"""

from __future__ import annotations

import json

import pytest

from difflet.backends.trainium.core import world_check
from difflet.backends.trainium.core.world_check import (
    NeuronWorldMismatchError,
    check_artifact_world,
    check_component_worlds,
    check_process_world,
    process_world,
    read_saved_neuron_config,
    reset_process_world,
)


@pytest.fixture(autouse=True)
def _fresh_process_world():
    reset_process_world()
    yield
    reset_process_world()


# --------------------------------------------------------------------------- #
# artifact vs declared (the std::out_of_range signature)
# --------------------------------------------------------------------------- #
def test_artifact_world_mismatch_raises_with_both_numbers():
    # 2026-08-30: a VAE NEFF compiled for world 4 loaded by a world-2 config
    # aborted inside the runtime. It must be a Python error naming both.
    with pytest.raises(NeuronWorldMismatchError) as excinfo:
        check_artifact_world(
            "NeuronHunyuanVideoVAEDecoderApplication",
            declared_world=2,
            declared_tp=1,
            saved_neuron_config={"world_size": 4, "tp_degree": 1},
        )
    message = str(excinfo.value)
    assert "world_size: artifact=4 declared=2" in message
    assert "NeuronHunyuanVideoVAEDecoderApplication" in message


def test_artifact_tp_mismatch_raises():
    with pytest.raises(NeuronWorldMismatchError, match="tp_degree: artifact=4 declared=1"):
        check_artifact_world(
            "vae", declared_world=4, declared_tp=1,
            saved_neuron_config={"world_size": 4, "tp_degree": 4},
        )


def test_artifact_match_is_silent():
    check_artifact_world(
        "vae", declared_world=4, declared_tp=1,
        saved_neuron_config={"world_size": 4, "tp_degree": 1, "start_rank_id": None},
    )


def test_artifact_check_skips_when_no_saved_config():
    # Absence is the caller's "artifact not found" problem, not a mismatch.
    check_artifact_world("vae", declared_world=4, declared_tp=1, saved_neuron_config=None)
    check_artifact_world("vae", declared_world=4, declared_tp=1, saved_neuron_config={})


def test_read_saved_neuron_config(tmp_path):
    assert read_saved_neuron_config(tmp_path) is None
    (tmp_path / "neuron_config.json").write_text("not json", encoding="utf-8")
    assert read_saved_neuron_config(tmp_path) is None
    (tmp_path / "neuron_config.json").write_text(
        json.dumps({"neuron_config": {"world_size": 4, "tp_degree": 2}, "height": 320}),
        encoding="utf-8",
    )
    assert read_saved_neuron_config(tmp_path) == {"world_size": 4, "tp_degree": 2}
    # normalize_path-style trailing slash is accepted too
    assert read_saved_neuron_config(str(tmp_path) + "/") == {"world_size": 4, "tp_degree": 2}


# --------------------------------------------------------------------------- #
# declared vs process (the SIGSEGV signature)
# --------------------------------------------------------------------------- #
def test_first_multi_rank_component_establishes_the_process_world():
    assert process_world() is None
    check_process_world("dit", declared_world=4, local_ranks_size=None)
    assert process_world().world_size == 4
    assert process_world().established_by == "dit"


def test_same_world_components_coexist_regardless_of_tp():
    # Flux resident topology: CLIP tp1/w4 + T5 tp4/w4 + transformer + VAE tp1/w4.
    check_process_world("t5", declared_world=4, local_ranks_size=4)
    check_process_world("clip", declared_world=4, local_ranks_size=4)
    check_process_world("vae", declared_world=4, local_ranks_size=4)


def test_smaller_world_in_initialized_process_raises():
    # The exact HunyuanVideo signature: DiT initialized ranks 0..3, then a VAE
    # claiming world 2 in the same process.
    check_process_world("NeuronHunyuanVideoBackboneApplication", declared_world=4,
                        local_ranks_size=None)
    with pytest.raises(NeuronWorldMismatchError) as excinfo:
        check_process_world("NeuronHunyuanVideoVAEDecoderApplication", declared_world=2,
                            local_ranks_size=None)
    message = str(excinfo.value)
    assert "world_size=2" in message
    assert "world_size=4" in message
    assert "NeuronHunyuanVideoBackboneApplication" in message  # who established it
    assert "03_adaptation_assessment.md" in message  # where the constraint is documented


def test_larger_world_in_initialized_process_raises_too():
    check_process_world("small", declared_world=2, local_ranks_size=None)
    with pytest.raises(NeuronWorldMismatchError):
        check_process_world("big", declared_world=4, local_ranks_size=None)


def test_world_one_component_is_exempt_and_does_not_commit_the_process():
    # Standalone-stage convention (Wan/Qwen vae subprocess, HV 1.5 process mode).
    check_process_world("vae", declared_world=1, local_ranks_size=1)
    assert process_world() is None
    check_process_world("dit", declared_world=4, local_ranks_size=None)
    check_process_world("vae", declared_world=1, local_ranks_size=1)  # still fine after


def test_declared_world_must_equal_multi_rank_load_range():
    with pytest.raises(NeuronWorldMismatchError, match="local_ranks_size=4"):
        check_process_world("vae", declared_world=2, local_ranks_size=4)
    # and it must not have committed the process on the way out
    assert process_world() is None


def test_one_rank_per_process_torchrun_mode_is_allowed():
    # resolve_load_rank_range returns (rank, 1) under torchrun: the declared
    # world is the global one, the load range is this process's single core.
    check_process_world("dit", declared_world=4, local_ranks_size=1)
    check_process_world("vae", declared_world=4, local_ranks_size=1)
    assert process_world().world_size == 4


def test_reset_forgets_the_process_world():
    check_process_world("dit", declared_world=4, local_ranks_size=None)
    reset_process_world()
    check_process_world("dit", declared_world=2, local_ranks_size=None)
    assert process_world().world_size == 2


# --------------------------------------------------------------------------- #
# multi-component pre-flight
# --------------------------------------------------------------------------- #
def test_component_preflight_accepts_one_world_plus_standalone():
    assert check_component_worlds([("clip", 4), ("t5", 4), ("transformer", 4), ("vae", 1)]) == 4


def test_component_preflight_all_standalone_returns_none():
    assert check_component_worlds([("vae", 1)]) is None
    assert check_component_worlds([]) is None


def test_component_preflight_rejects_mixed_worlds_listing_every_component():
    with pytest.raises(NeuronWorldMismatchError) as excinfo:
        check_component_worlds([("transformer", 4), ("vae_decoder", 2), ("probe", 4)])
    message = str(excinfo.value)
    assert "transformer=w4" in message
    assert "vae_decoder=w2" in message
    assert "probe=w4" in message


def test_module_constants_point_at_the_topology_docs():
    # The error text is the only place a future reader meets this rule; keep
    # the pointers to the on-device evidence intact.
    assert "05_flux_runtime_validation.md" in world_check._TOPOLOGY_DOCS
