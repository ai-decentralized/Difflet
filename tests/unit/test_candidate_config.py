"""M5.0.3.1 — candidate-axis abstraction.

Covers: world_size invariant (the candidate axis never touches the
parallel/communicator contract), additive-only cache-key behavior
(absent / trivial ≡ legacy byte-identical), max vs active semantics,
validation, and back-compat default.
"""

import pytest

from nova.pipeline.parallel_config import CandidateConfig, NovaParallelConfig


def test_default_is_trivial_and_back_compat():
    cfg = CandidateConfig()
    assert cfg.max_candidates == 1
    assert cfg.active_candidates == 1
    assert cfg.is_trivial is True


def test_active_defaults_to_max():
    cfg = CandidateConfig(max_candidates=4)
    assert cfg.active_candidates == 4
    assert cfg.is_trivial is False


@pytest.mark.parametrize(
    "parallel",
    [
        NovaParallelConfig(tp_degree=1),
        NovaParallelConfig(tp_degree=4),
        NovaParallelConfig(tp_degree=4, cp_degree=2),
        NovaParallelConfig(tp_degree=2, cfg_parallel_enabled=True),
    ],
)
@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_world_size_is_invariant_under_candidate_axis(parallel, n):
    # The candidate axis must NOT change world_size for any parallel
    # config (cclog 43 §3.1: reuse world_size math, no new collectives).
    cand = CandidateConfig(max_candidates=n)
    assert cand.world_size(parallel) == parallel.world_size


def test_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        CandidateConfig(max_candidates=0)
    with pytest.raises(ValueError):
        CandidateConfig(max_candidates=2, active_candidates=3)
    with pytest.raises(ValueError):
        CandidateConfig(max_candidates=4, active_candidates=0)


def test_to_cache_dict_excludes_active():
    # active_candidates is a runtime occupancy knob within the traced
    # max artifact; it must not be part of artifact identity.
    a = CandidateConfig(max_candidates=4, active_candidates=2)
    b = CandidateConfig(max_candidates=4, active_candidates=4)
    assert a.to_cache_dict() == b.to_cache_dict() == {"max_candidates": 4}


def _spec(candidate):
    from nova.pipeline.compile_cache import CacheSpec

    return CacheSpec(
        model_id="org/m",
        model_path="/tmp/m",
        model_name="unit_dummy",
        parallel=NovaParallelConfig(tp_degree=4),
        dtype="bf16",
        candidate=candidate,
    )


def test_cache_key_byte_identical_when_absent_or_trivial():
    from nova.pipeline.compile_cache import cache_key

    legacy = cache_key(_spec(None))
    trivial = cache_key(_spec(CandidateConfig()))
    trivial_active = cache_key(
        _spec(CandidateConfig(max_candidates=1, active_candidates=1))
    )
    assert legacy == trivial == trivial_active
    # And the raw cache_inputs must not even contain the key.
    assert "candidate" not in _spec(None).cache_inputs()
    assert "candidate" not in _spec(CandidateConfig()).cache_inputs()


def test_cache_key_differs_by_max_candidates():
    from nova.pipeline.compile_cache import cache_key

    k2 = cache_key(_spec(CandidateConfig(max_candidates=2)))
    k4 = cache_key(_spec(CandidateConfig(max_candidates=4)))
    legacy = cache_key(_spec(None))
    assert k2 != k4
    assert k2 != legacy and k4 != legacy


def test_active_within_max_is_a_cache_hit():
    # Compiling at max=4 then running active=2 must reuse the artifact.
    from nova.pipeline.compile_cache import cache_key

    compiled = cache_key(_spec(CandidateConfig(max_candidates=4)))
    runtime_n2 = cache_key(
        _spec(CandidateConfig(max_candidates=4, active_candidates=2))
    )
    assert compiled == runtime_n2


def test_exported_from_public_api():
    import nova
    from nova.pipeline import CandidateConfig as ViaPipeline

    assert nova.CandidateConfig is ViaPipeline
