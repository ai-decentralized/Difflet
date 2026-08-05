from __future__ import annotations

from difflet.pipeline.cache import AdaptiveAnchorConfig
from difflet.pipeline.cache.types import CacheHistory, CacheStepContext, RuntimeObservation
from scripts.flux_cache_terminal_brake_sweep import TerminalStepPolicy


def _config() -> AdaptiveAnchorConfig:
    return AdaptiveAnchorConfig(
        initial_anchor_interval=4,
        minimum_anchor_interval=2,
        maximum_anchor_interval=6,
        warmup_steps=2,
        cooldown_steps=1,
        anchor_phase=1,
        tighten_error=2.0,
        recovery_error=3.0,
        acceleration_error=1.0,
        recovery_steps=1,
        disable_after_recoveries=1,
    )


def test_terminal_step_policy_permanently_disables_cache() -> None:
    policy = TerminalStepPolicy(_config(), terminal_step=4)
    history = CacheHistory(2)
    observation = RuntimeObservation()
    policy._next_anchor_step = 8

    assert policy.should_skip(CacheStepContext(3, 10), history, observation)
    assert not policy.should_skip(CacheStepContext(4, 10), history, observation)
    assert not policy.should_skip(CacheStepContext(5, 10), history, observation)
    policy.validate_complete()
    assert policy.stats()["terminal_applied"] is True
