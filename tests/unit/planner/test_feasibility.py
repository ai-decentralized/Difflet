"""Candidate enumeration: what survives, what is rejected, and why.

The load-bearing assertion is the last one: on four cores, the planner's feasible
set must equal the cells scripts/verify_cli.py actually runs on device. If those
diverge, one of them is lying about what Difflet supports.
"""
from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.planner.feasibility import (
    Candidate,
    config_label,
    divisors,
    enumerate_candidates,
)
from difflet.registry import resolve_model

MODELS = ("flux", "qwen_image", "wan", "hunyuan_video", "hunyuan_video_15", "ltx_2")


def _report(name: str, cores: int = 4, **kwargs):
    entry = resolve_model("", model_type=name)
    return enumerate_candidates(
        model_name=name,
        capabilities=entry.require_capabilities(),
        cores=cores,
        **kwargs,
    )


def _reason(report, label: str) -> tuple[str, str]:
    for rejection in report.rejected:
        if rejection.label == label:
            return rejection.kind, rejection.reason
    raise AssertionError(f"{label} was not rejected; feasible={report.labels()}")


# ------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    "parallel,expected",
    [
        (DiffletParallelConfig(tp_degree=4), "tp4"),
        (DiffletParallelConfig(tp_degree=2, cp_degree=2), "tp2cp2"),
        (DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True), "tp2cfg"),
        (DiffletParallelConfig(tp_degree=4, sp_enabled=True), "tp4sp"),
        (DiffletParallelConfig(tp_degree=2, dp_degree=2), "dp2tp2"),
        (
            DiffletParallelConfig(tp_degree=2, cp_degree=2, cp_mode="ulysses"),
            "tp2cp2ulysses",
        ),
    ],
)
def test_labels_match_the_verify_cli_matrix_keys(parallel, expected):
    assert config_label(parallel) == expected


def test_gather_kv_is_not_spelled_out_in_the_label():
    """The default mode stays implicit, matching the compile-cache key's elision."""

    assert config_label(DiffletParallelConfig(tp_degree=2, cp_degree=2)) == "tp2cp2"


def test_divisors():
    assert divisors(4) == (1, 2, 4)
    assert divisors(8) == (1, 2, 4, 8)
    assert divisors(1) == (1,)


# -------------------------------------------------------------- basic shape


@pytest.mark.parametrize("name", MODELS)
def test_every_feasible_candidate_fills_the_host(name):
    report = _report(name)
    for candidate in report.feasible:
        assert candidate.world_size == 4, candidate.label


@pytest.mark.parametrize("name", MODELS)
def test_every_feasible_candidate_constructs_a_real_config(name):
    """The planner must never propose something the runtime would refuse."""

    for candidate in _report(name).feasible:
        rebuilt = DiffletParallelConfig(
            tp_degree=candidate.parallel.tp_degree,
            cp_degree=candidate.parallel.cp_degree,
            cp_mode=candidate.parallel.cp_mode,
            cfg_parallel_enabled=candidate.parallel.cfg_parallel_enabled,
            sp_enabled=candidate.parallel.sp_enabled,
            dp_degree=candidate.parallel.dp_degree,
        )
        assert rebuilt == candidate.parallel


def test_labels_are_unique():
    for name in MODELS:
        report = _report(name)
        labels = [c.label for c in report.feasible] + [r.label for r in report.rejected]
        assert len(labels) == len(set(labels))


def test_rejects_zero_cores():
    with pytest.raises(ValueError, match="cores must be >= 1"):
        _report("flux", cores=0)


# ------------------------------------------------------------ rejection rules


def test_distilled_model_cannot_use_cfg_parallel():
    kind, reason = _reason(_report("flux"), "tp2cfg")
    assert kind == "capability"
    assert "guidance-distilled" in reason


def test_true_cfg_model_can_use_cfg_parallel():
    assert "tp2cfg" in _report("wan").labels()


def test_sp_and_cp_are_mutually_exclusive():
    kind, reason = _reason(_report("flux"), "tp2cp2sp")
    assert kind == "exclusivity"
    assert "mutually exclusive" in reason


def test_cp_and_cfg_are_mutually_exclusive():
    kind, reason = _reason(_report("wan"), "tp1cp2cfg")
    assert kind == "exclusivity"
    assert "mutually exclusive" in reason


def test_exclusivity_reason_comes_from_the_runtime_type():
    """The message is DiffletParallelConfig's own, not a planner paraphrase."""

    _, reason = _reason(_report("flux"), "tp2cp2sp")
    with pytest.raises(ValueError) as exc:
        DiffletParallelConfig(tp_degree=2, cp_degree=2, sp_enabled=True)
    assert reason == str(exc.value)


def test_model_without_sp_rejects_sp():
    kind, _ = _reason(_report("qwen_image"), "tp4sp")
    assert kind == "capability"


def test_model_without_cp_rejects_cp():
    kind, reason = _reason(_report("ltx_2"), "tp2cp2")
    assert kind == "capability"
    assert "context parallelism" in reason


def test_sp_at_tp1_is_rejected_as_a_no_op():
    kind, reason = _reason(_report("flux"), "dp4tp1sp")
    assert kind == "degenerate"
    assert "no-op" in reason


def test_known_bad_cells_are_reported_separately_from_infeasible_ones():
    kind, reason = _reason(_report("hunyuan_video"), "tp2cp2")
    assert kind == "known-bad"
    assert "neuronx-cc" in reason


def test_scaffold_model_denies_every_configuration():
    report = _report("hunyuan_video_15")
    assert report.feasible == ()
    assert any(r.kind == "known-bad" for r in report.rejected)


# ---------------------------------------------------- ulysses head divisibility


def test_ulysses_needs_heads_divisible_by_tp_times_cp():
    """24 heads with tp*cp = 16 is the case that used to fail at compile time."""

    kind, reason = _reason(_report("flux", cores=16), "tp8cp2ulysses")
    assert kind == "divisibility"
    assert "24 % 16 != 0" in reason


def test_gather_kv_survives_where_ulysses_does_not():
    """Only ulysses shards heads over cp; gather_kv has no such requirement."""

    labels = _report("flux", cores=16).labels()
    assert "tp8cp2" in labels
    assert "tp8cp2ulysses" not in labels


def test_tp_must_divide_the_head_count():
    kind, reason = _reason(_report("flux", cores=16), "tp16")
    assert kind == "divisibility"
    assert "24 attention heads" in reason


def test_head_counts_make_the_feasible_sets_differ_per_model():
    """flux has 24 heads and wan 40, so ulysses splits differ.

    On twelve cores, tp=4 cp=3 needs heads % 12 == 0: 24 qualifies, 40 does not.
    Both models pass the plain tp=4 check, so this isolates the ulysses rule.
    """

    assert "tp4cp3ulysses" in _report("flux", cores=12).labels()
    assert "tp4cp3ulysses" not in _report("wan", cores=12).labels()
    assert "tp4cp3" in _report("wan", cores=12).labels()  # gather_kv is unaffected


# --------------------------------------------------------------- serving mode


def test_serving_rejects_cfg_parallel():
    kind, _ = _reason(_report("wan", serving=True), "tp2cfg")
    assert kind == "serving"


def test_serving_rejects_data_parallel_replicas():
    kind, _ = _reason(_report("wan", serving=True), "dp2tp2")
    assert kind == "serving"


def test_serving_pins_qwen_to_cp1():
    kind, _ = _reason(_report("qwen_image", serving=True), "tp2cp2")
    assert kind == "serving"


# ------------------------------------------ agreement with the on-device matrix


def test_feasible_set_agrees_with_verify_cli_on_four_cores():
    """Planner feasibility must match what verify_cli actually runs.

    verify_cli covers six hand-picked four-core configurations; for each model it
    skips the ones the CLI rejects and marks compiler/HBM failures as expected.
    A cell it runs must be feasible here, and a cell it skips must not be --
    expected-failure cells are 'known-bad', which is rejected here by design.
    """

    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "scripts"))
    from verify_cli import EXPECTED_FAIL_CELLS, MODELS as VC_MODELS, PARALLEL_CONFIGS, skip_reason

    for model_key, spec in VC_MODELS.items():
        registry_name = resolve_model(spec.model_id).name
        labels = set(_report(registry_name).labels())
        for config_key in PARALLEL_CONFIGS:
            skipped = skip_reason(model_key, config_key) is not None
            expected_fail = (model_key, config_key) in EXPECTED_FAIL_CELLS
            planner_feasible = config_key in labels
            if skipped or expected_fail:
                assert not planner_feasible, (
                    f"{model_key}/{config_key}: verify_cli skips or expects failure, "
                    "but the planner calls it feasible"
                )
            else:
                assert planner_feasible, (
                    f"{model_key}/{config_key}: verify_cli runs this cell, but the "
                    "planner does not list it as feasible"
                )


def test_candidate_is_hashable_and_comparable():
    a = Candidate(DiffletParallelConfig(tp_degree=4))
    b = Candidate(DiffletParallelConfig(tp_degree=4))
    assert a == b
    assert len({a, b}) == 1
