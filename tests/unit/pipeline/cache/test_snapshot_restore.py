from __future__ import annotations

import pytest
import torch

from difflet.pipeline.cache import (
    CacheRunner,
    CacheSession,
    LegacyResidualPredictor,
    PhasedStaticPolicy,
    QualityRecoveryConfig,
    QualityRecoveryGuard,
    TeaCacheControllerAdapter,
)


def _session(
    mask: tuple[bool, ...] = (True, True, False, True, True),
) -> CacheSession:
    runner = CacheRunner(
        PhasedStaticPolicy(mask),
        LegacyResidualPredictor(),
        recovery=QualityRecoveryGuard(QualityRecoveryConfig()),
    )
    session = CacheSession(
        runner,
        num_steps=len(mask),
        configuration_source="snapshot-test",
    )
    session.bind_schedule_coordinates(
        tuple(float(index) for index in range(len(mask))),
        tuple(float(index) / 10 for index in range(len(mask))),
    )
    return session


def _complete_step(session: CacheSession, step_index: int, value: float) -> bool:
    decision = session.decide_step(step_index)
    if decision.should_skip:
        session.estimate_output(step_index)
        return True
    session.record_anchor(step_index, torch.tensor([value]))
    return False


def test_restore_is_the_only_transition_that_reopens_earlier_steps():
    session = _session()
    assert not _complete_step(session, 0, 0.0)
    assert not _complete_step(session, 1, 1.0)
    checkpoint = session.snapshot()

    assert _complete_step(session, 2, 100.0)
    assert not _complete_step(session, 3, 100.0)
    assert session.anchor_error_trace()["entries"][-1]["estimate_step_indices"] == [2]
    with pytest.raises(ValueError, match="decided more than once"):
        session.decide_step(3)

    session.restore(checkpoint)
    replay = session.decide_step(2)
    assert replay.should_skip
    session.record_anchor(2, torch.tensor([4.0]))
    assert [anchor.step_index for anchor in session.runner.history] == [1, 2]
    trace = session.anchor_error_trace()["entries"]
    assert [entry["anchor_step_index"] for entry in trace] == [0, 1, 2]
    assert trace[-1]["estimate_step_indices"] == []
    assert all(entry["anchor_step_index"] != 3 for entry in trace)


def test_snapshot_clones_tensor_history_and_observation_without_aliasing():
    session = _session((True, True, True))
    original = torch.tensor([3.0])
    session.decide_step(0)
    session.record_anchor(0, original)
    checkpoint = session.snapshot()

    original.add_(100.0)
    session.runner.observation.last_output.add_(10.0)
    _complete_step(session, 1, 7.0)
    session.restore(checkpoint)

    restored_anchor = session.runner.history.latest.output
    restored_observation = session.runner.observation.last_output
    assert torch.equal(restored_anchor, torch.tensor([3.0]))
    assert torch.equal(restored_observation, torch.tensor([3.0]))
    assert restored_anchor.data_ptr() != original.data_ptr()
    assert restored_observation.data_ptr() != restored_anchor.data_ptr()


def test_snapshot_is_single_owner_bound_and_consumed_by_commit_or_restore():
    first = _session((True, True, True))
    second = _session((True, True, True))
    checkpoint = first.snapshot()

    with pytest.raises(RuntimeError, match="already has an active"):
        first.snapshot()
    with pytest.raises(ValueError, match="another CacheSession"):
        second.restore(checkpoint)

    first.commit_snapshot(checkpoint)
    assert not first.has_active_snapshot
    with pytest.raises(RuntimeError, match="not the active"):
        first.restore(checkpoint)

    next_checkpoint = first.snapshot()
    first.clear_request_state()
    with pytest.raises(RuntimeError, match="not the active"):
        first.restore(next_checkpoint)


def test_snapshot_and_restore_reject_incomplete_step_handshakes():
    session = _session((True, True, True))
    session.decide_step(0)
    with pytest.raises(RuntimeError, match="awaits output"):
        session.snapshot()
    session.record_anchor(0, torch.tensor([0.0]))

    checkpoint = session.snapshot()
    session.decide_step(1)
    with pytest.raises(RuntimeError, match="awaits output"):
        session.restore(checkpoint)
    session.record_anchor(1, torch.tensor([1.0]))
    session.restore(checkpoint)


def test_runner_restores_public_policy_and_recovery_state_and_counters():
    session = _session((True, True, False, True))
    _complete_step(session, 0, 0.0)
    _complete_step(session, 1, 1.0)
    session.runner.policy.public_runtime_counter = 7
    session.request_recovery("forced", steps=2, reset_history=False)
    checkpoint = session.snapshot()
    expected_stats = session.statistics()

    session.runner.policy.public_runtime_counter = 99
    _complete_step(session, 2, 2.0)
    assert session.runner.recovery.pending_steps == 1
    session.restore(checkpoint)

    assert session.runner.policy.public_runtime_counter == 7
    assert session.runner.recovery.pending_steps == 2
    assert session.statistics() == expected_stats


def test_adapter_snapshot_restores_legacy_delta_state():
    adapter = TeaCacheControllerAdapter(_session((True, True, True)))
    adapter.last_delta_estimate = 0.25
    checkpoint = adapter.snapshot()
    adapter.last_delta_estimate = 9.0

    adapter.restore(checkpoint)

    assert adapter.last_delta_estimate == 0.25
    assert not adapter.session.has_active_snapshot
